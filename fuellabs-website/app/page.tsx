import type { Metadata } from "next";
import Link from "next/link";
import { CurtainGate } from "@/components/home/CurtainGate";
import styles from "./page.module.css";

export const metadata: Metadata = { alternates: { canonical: "/" } };

export default function Home() {
  return <>
    <CurtainGate />
    <section className={`${styles.hero} home-viewport`} aria-labelledby="home-title">
      <div className={styles.heroInner}>
        <h1 className={styles.heroWord} id="home-title" aria-label="fuellabs.">
          <span aria-hidden="true">fuel<span className={styles.heroLabs}>labs</span></span>
          <span className={styles.heroSignal} aria-hidden="true" />
        </h1>
        <p className={styles.heroCopy}>An independent AI research lab.</p>
        <div className={styles.heroLower}>
          <span className={styles.lowerBrand}>fuellabs<span className={styles.lowerSignal} aria-hidden="true" /></span>
          <Link className="arrow-link" href="/hobbylm">Meet HobbyLM</Link>
        </div>
      </div>
    </section>
  </>;
}
