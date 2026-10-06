import type { Metadata } from "next";
import Link from "next/link";
import { researchArticles } from "@/content/site";
import { MotionFigure } from "@/components/figures/MotionFigure";
import { MlxTerminal } from "@/components/hobbylm/MlxTerminal";
import { SignalMark } from "@/components/typography/SignalHeading";
import styles from "./page.module.css";

export const metadata: Metadata = {
  title: "Model layout study",
  robots: { index: false, follow: false },
};

export default function LiquidModelsStudy() {
  return (
    <div className={styles.page}>
      <section className={styles.hero} aria-labelledby="study-title">
        <div className={`container ${styles.heroInner}`}>
          <div className={styles.heroLead}>
            <h1 id="study-title" className={styles.title}>
              A sparse model,<br /><span className={styles.heroLine}>built from scratch.</span>
            </h1>
            <div className={styles.actions}>
              <Link className={styles.primaryAction} href="/hobbylm">Meet HobbyLM</Link>
              <Link className={styles.textAction} href="/research">Read the research</Link>
            </div>
          </div>

          <p className={styles.heroAside}>
            Fuel Labs documents how HobbyLM is built, trained, and evaluated. Articles publish when their evidence and review are complete.
          </p>
        </div>
      </section>

      <section className={styles.index} aria-labelledby="model-index-title">
        <div className={`container ${styles.indexIntro}`}>
          <div>
            <h2 id="model-index-title">Explore the work</h2>
            <p>A model record and its research record, presented together.</p>
          </div>
          <Link className={styles.textAction} href="/research">All research</Link>
        </div>

        <div className={styles.ruledGrid}>
          <section className={styles.column} aria-labelledby="model-column-title">
            <header className={styles.columnHeader}>
              <h3 id="model-column-title">Model</h3>
              <p>The current Fuel Labs language-model project.</p>
            </header>
            <a className={styles.row} href="#selective-routing">
              <span>HobbyLM</span>
              <span className={styles.rowMeta}>Sparse mixture of experts</span>
            </a>
          </section>

          <section className={styles.column} aria-labelledby="research-column-title">
            <header className={styles.columnHeader}>
              <h3 id="research-column-title">Research record</h3>
              <p>Technical accounts publish as evidence and review are completed.</p>
            </header>
            <div className={styles.rows}>
              {researchArticles.map((article) => {
                const published = article.status === "Published";
                const content = (
                  <>
                    <span>{article.title}</span>
                    <span className={styles.rowMeta}>{article.status}</span>
                  </>
                );

                return published ? (
                  <Link className={styles.row} href={`/research/${article.slug}`} key={article.slug}>{content}</Link>
                ) : (
                  <div className={`${styles.row} ${styles.rowDisabled}`} key={article.slug}>{content}</div>
                );
              })}
            </div>
          </section>
        </div>
      </section>

      <section className={styles.chapters} aria-label="HobbyLM technical overview">
        <article className={styles.chapter} id="selective-routing">
          <div className={styles.chapterCopy}>
            <h2>Selective routing<SignalMark /></h2>
            <p>
              For each token, HobbyLM activates a selected subset of computational paths and combines their outputs before continuing through the model.
            </p>
          </div>
          <div className={styles.chapterMedia}>
            <MotionFigure
              animatedSrc="/diagrams/sparse-moe-motion-v1.svg"
              stillSrc="/diagrams/sparse-moe-frame-v1.png"
              width={1600}
              height={900}
              alt="A conceptual sparse mixture-of-experts routing trace showing a token entering a router, selected routes, a shared expert, and their combined output. No measured router scores or expert roles are shown."
              caption="A conceptual view of selective computation inside HobbyLM."
            />
          </div>
        </article>

        <article className={`${styles.chapter} ${styles.chapterReverse}`}>
          <div className={styles.chapterCopy}>
            <h2>Inside a<br /><span className={styles.signalTail}>sparse model<SignalMark /></span></h2>
            <p>
              HobbyLM combines one dense decoder layer with 19 sparse mixture-of-experts layers. Each sparse layer selects eight of 64 routed experts per token and evaluates one shared expert on every token.
            </p>
          </div>
          <div className={styles.chapterMedia}>
            <MotionFigure
              animatedSrc="/diagrams/figure-03-iso-motion.mp4"
              stillSrc="/diagrams/figure-03-iso-still.png"
              width={1920}
              height={1536}
              format="video"
              alt="System diagram of HobbyLM with one dense layer and 19 mixture-of-experts layers. An expanded layer shows eight selected paths among 64 routed experts plus one always-active shared expert."
              caption="One dense decoder layer followed by 19 sparse MoE layers."
            />
          </div>
        </article>

        <article className={`${styles.chapter} ${styles.localChapter}`}>
          <div className={styles.chapterCopy}>
            <h2><span className={styles.signalTail}>Run HobbyLM</span><br /><span className={styles.signalTail}>locally<SignalMark /></span></h2>
            <p>
              HobbyLM runs locally on Apple silicon through MLX, with sparse selected-expert computation and KV-cached decoding.
            </p>
          </div>
          <MlxTerminal />
        </article>
      </section>
    </div>
  );
}
