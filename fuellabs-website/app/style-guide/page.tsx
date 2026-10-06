import type { Metadata } from "next";
import { SignalHeading } from "@/components/typography/SignalHeading";
import styles from "./page.module.css";

export const metadata: Metadata = { title: "Style guide", alternates: { canonical: "/style-guide" }, robots: { index: false, follow: false } };

export default function StyleGuidePage() {
  return <section className={`section ${styles.guide}`}><div className="container"><SignalHeading as="h1" className="display">Style guide</SignalHeading><div className={styles.block}><h2>Color</h2><div className={styles.swatches}>{[["Paper","#F2EFE8"],["Ink","#11110F"],["Lime","#B7F34A"],["Muted","#66665F"],["Line","#D8D4C9"]].map(([name, hex]) => <div key={name}><i style={{ background: hex }} /><strong>{name}</strong><code>{hex}</code></div>)}</div></div><div className={styles.block}><h2>Typography</h2><div className={styles.type}><SignalHeading as="p" className="display">Display</SignalHeading><SignalHeading as="p">Section heading</SignalHeading><p className="intro-copy">Intro text explains one central idea with calm, readable measure.</p><p className="article-body-copy">Article body text uses Montserrat for sustained research reading. General Sans remains the display, title, deck, navigation, and interface voice.</p><span className="eyebrow">Technical metadata</span></div></div><div className={styles.block}><h2>Rules</h2><ul><li>Use the shared wordmark everywhere.</li><li>End page titles and major section headings with the lime square.</li><li>Use semantic type roles; never choose a size inside an individual page.</li><li>Keep internal review states out of public content.</li><li>Review at 390, 768, 1024, and 1440 pixels.</li></ul></div></div></section>;
}
