import Link from "next/link";
import { researchArticles } from "@/content/site";
import styles from "./ResearchList.module.css";

export function ResearchList({ preview = false }: { preview?: boolean }) {
  const articles = preview ? researchArticles.slice(0, 3) : researchArticles;
  return <div className={styles.list}>{articles.map((article) => {
    const content = <><span className={styles.number}>{article.number}</span><span><strong>{article.title}</strong><small>{article.summary}</small></span><span className={styles.status}>{article.status}</span></>;
    if (article.status !== "Published") return <div className={`${styles.item} ${styles.inactive}`} key={article.slug} aria-disabled="true">{content}</div>;
    return <Link className={styles.item} href={`/research/${article.slug}`} key={article.slug}>{content}</Link>;
  })}</div>;
}
