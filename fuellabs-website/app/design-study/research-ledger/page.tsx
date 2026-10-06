import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { researchArticles } from "@/content/site";
import { SignalMark } from "@/components/typography/SignalHeading";
import styles from "./page.module.css";

export const metadata: Metadata = {
  title: "Research layout study",
  robots: { index: false, follow: false },
};

const covers: Record<string, string> = {
  "pretraining-hobbylm": "/research/covers/pretraining-hobbylm-v1.jpg",
  "hobbylm-architecture": "/diagrams/figure-03-iso-still.png",
  "post-training-and-sft": "/research/covers/context-window-sft-v1.jpg",
};

export default function ResearchLedgerStudy() {
  return (
    <div className={styles.page}>
      <section className={styles.hero} aria-labelledby="research-study-title">
        <div className={`container ${styles.heroInner}`}>
          <h1 id="research-study-title">
            <span className={styles.signalTitle}>Research<SignalMark /></span>
          </h1>
          <p>
            Technical accounts of how HobbyLM is built, trained, and evaluated. Articles publish when their evidence and review are complete.
          </p>
        </div>
      </section>

      <section className={styles.archive} aria-label="Research articles">
        <div className={`container ${styles.grid}`}>
          {researchArticles.slice(0, 3).map((article) => {
            const published = article.status === "Published";
            const content = (
              <>
                <Image src={covers[article.slug]} fill sizes="(max-width: 800px) 100vw, 33vw" alt="" />
                <span className={styles.scrim} />
                <span className={styles.cardContent}>
                  <span className={styles.cardMeta}><span>{article.number}</span><span>{article.status}</span></span>
                  <strong>{article.title}</strong>
                  <small>{article.summary}</small>
                  {published ? <span className={styles.explore}>Explore the research</span> : null}
                </span>
              </>
            );

            return published ? (
              <Link className={styles.card} href={`/research/${article.slug}`} key={article.slug}>{content}</Link>
            ) : (
              <article className={`${styles.card} ${styles.inactive}`} key={article.slug}>{content}</article>
            );
          })}
        </div>
      </section>
    </div>
  );
}
