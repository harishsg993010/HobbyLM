type IconProps = { className?: string };

export function PlayIcon({ className }: IconProps) {
  return <svg className={className} width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M5 3.25 12 8l-7 4.75V3.25Z" fill="currentColor" /></svg>;
}

export function PauseIcon({ className }: IconProps) {
  return <svg className={className} width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M4.5 3.25v9.5M11.5 3.25v9.5" stroke="currentColor" strokeWidth="1.5" /></svg>;
}

export function ReplayIcon({ className }: IconProps) {
  return <svg className={className} width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M5.1 4.1A5 5 0 1 1 3 8" stroke="currentColor" strokeWidth="1.35" strokeLinecap="round" /><path d="M2.8 3.1v3.6h3.6" stroke="currentColor" strokeWidth="1.35" strokeLinecap="round" strokeLinejoin="round" /></svg>;
}

export function PreviousIcon({ className }: IconProps) {
  return <svg className={className} width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="m10 3.5-4.5 4.5 4.5 4.5" stroke="currentColor" strokeWidth="1.35" strokeLinecap="round" strokeLinejoin="round" /></svg>;
}

export function NextIcon({ className }: IconProps) {
  return <svg className={className} width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="m6 3.5 4.5 4.5L6 12.5" stroke="currentColor" strokeWidth="1.35" strokeLinecap="round" strokeLinejoin="round" /></svg>;
}
