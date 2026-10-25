//! Spin-waiting thread pool for the decode hot path. A decode step is ~80 small parallel
//! matvecs in a strict dependency chain; rayon parks idle workers, and waking them costs
//! ~100-200 us per dispatch on Windows, which dominated the step. Here workers busy-wait on a
//! generation counter between jobs (backing off to sleep when idle), the calling thread works
//! too, and chunks are claimed from an atomic counter so slower E-cores just take fewer.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Spin this long after the last chunk before backing off to short sleeps.
const SPIN_FOR: Duration = Duration::from_millis(20);

type Job = *const (dyn Fn(usize) + Sync);

// `state` packs generation | chunk count | next unclaimed chunk. A worker runs a chunk only after
// claiming it with a CAS on the current state, so it can never run a finished job's closure:
// the dispatcher does not return (or publish the next job) until every chunk has completed.
const IDX_BITS: u32 = 20;
const IDX_MASK: u64 = (1 << IDX_BITS) - 1;

struct Shared {
    state: AtomicU64,
    completed: AtomicUsize,
    job: UnsafeCell<Option<Job>>,
}

// `job` is written only by the dispatcher while no chunk is claimable, and read only by a thread
// that has just claimed a chunk of the current job (release on publish, acquire on claim).
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct Pool {
    workers: usize,
    sh: &'static Shared,
    lock: std::sync::Mutex<()>,
}

static POOL: OnceLock<Pool> = OnceLock::new();

/// The global pool, sized to rayon's thread count (so `--threads` controls both).
pub fn pool() -> &'static Pool {
    POOL.get_or_init(|| Pool::new(rayon::current_num_threads().max(1)))
}

impl Pool {
    fn new(threads: usize) -> Self {
        let sh: &'static Shared = Box::leak(Box::new(Shared {
            state: AtomicU64::new(0),
            completed: AtomicUsize::new(0),
            job: UnsafeCell::new(None),
        }));
        let workers = threads - 1;
        for _ in 0..workers {
            std::thread::spawn(move || worker(sh));
        }
        Pool { workers, sh, lock: std::sync::Mutex::new(()) }
    }

    /// Run `f(c)` for every chunk index `c` in `0..nchunks`, in parallel; returns when all are done.
    pub fn run(&self, nchunks: usize, f: &(dyn Fn(usize) + Sync)) {
        if nchunks <= 1 || self.workers == 0 {
            (0..nchunks).for_each(f);
            return;
        }
        assert!((nchunks as u64) <= IDX_MASK);
        let _g = self.lock.lock().unwrap();
        let sh = self.sh;
        // Erase the borrow's lifetime; sound because we wait for every chunk below.
        let job: Job = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(f) };
        unsafe { *sh.job.get() = Some(job) };
        sh.completed.store(0, Ordering::Relaxed);
        let gen = (sh.state.load(Ordering::Relaxed) >> (2 * IDX_BITS)) + 1;
        sh.state.store((gen << (2 * IDX_BITS)) | ((nchunks as u64) << IDX_BITS), Ordering::Release);
        while let Some(c) = claim(sh) {
            f(c);
            sh.completed.fetch_add(1, Ordering::Release);
        }
        while sh.completed.load(Ordering::Acquire) < nchunks {
            std::hint::spin_loop();
        }
    }
}

/// Claim the next chunk of the current job, if any is left.
#[inline]
fn claim(sh: &Shared) -> Option<usize> {
    let mut s = sh.state.load(Ordering::Acquire);
    loop {
        let n = (s >> IDX_BITS) & IDX_MASK;
        let i = s & IDX_MASK;
        if i >= n {
            return None;
        }
        match sh.state.compare_exchange_weak(s, s + 1, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return Some(i as usize),
            Err(cur) => s = cur,
        }
    }
}

fn worker(sh: &'static Shared) {
    let mut last = Instant::now();
    let mut spins = 0u32;
    loop {
        if let Some(c) = claim(sh) {
            let f = unsafe { &*(*sh.job.get()).unwrap() };
            f(c);
            sh.completed.fetch_add(1, Ordering::Release);
            last = Instant::now();
            spins = 0;
            continue;
        }
        spins = spins.wrapping_add(1);
        if spins % 1024 == 0 && last.elapsed() > SPIN_FOR {
            std::thread::sleep(Duration::from_micros(200));
        } else {
            std::hint::spin_loop();
        }
    }
}

/// Parallel `y.chunks_mut(ch).enumerate().for_each(f)` on the spin pool.
pub fn par_chunks_mut<T: Send>(y: &mut [T], ch: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let len = y.len();
    let nchunks = len.div_ceil(ch);
    let base = y.as_mut_ptr() as usize;
    pool().run(nchunks, &|c| {
        let start = c * ch;
        let end = (start + ch).min(len);
        // chunks are disjoint, and each index is claimed exactly once
        let part = unsafe { std::slice::from_raw_parts_mut((base as *mut T).add(start), end - start) };
        f(c, part);
    });
}
