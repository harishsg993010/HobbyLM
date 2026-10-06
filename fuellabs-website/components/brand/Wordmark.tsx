import Link from "next/link";
import styles from "./Wordmark.module.css";

export function Wordmark({ large = false, tabIndex }: { large?: boolean; tabIndex?: number }) {
  return (
    <Link className={`${styles.wordmark} ${large ? styles.large : ""}`} href="/" aria-label="fuellabs. home" tabIndex={tabIndex}>
      <span>fuel</span><span className={styles.labs}>labs</span><span className={styles.square} aria-hidden="true" />
    </Link>
  );
}
