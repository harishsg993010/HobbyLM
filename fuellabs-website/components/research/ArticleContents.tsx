"use client";

import { useEffect, useState } from "react";
import styles from "./ArticleContents.module.css";

export type ArticleSectionLink = { id: string; label: string };

export function ArticleContents({ sections }: { sections: readonly ArticleSectionLink[] }) {
  const [active, setActive] = useState(sections[0]?.id ?? "");

  useEffect(() => {
    const nodes = sections.map(({ id }) => document.getElementById(id)).filter((node): node is HTMLElement => Boolean(node));
    const observer = new IntersectionObserver((entries) => {
      const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
      if (visible?.target.id) setActive(visible.target.id);
    }, { rootMargin: "-18% 0px -68% 0px", threshold: [0, 1] });
    nodes.forEach((node) => observer.observe(node));
    return () => observer.disconnect();
  }, [sections]);

  const links = <ol>{sections.map(({ id, label }) => <li key={id}><a href={`#${id}`} aria-current={active === id ? "location" : undefined}>{label}</a></li>)}</ol>;
  return <>
    <nav className={styles.desktop} aria-label="Article contents"><p>In this article</p>{links}</nav>
    <details className={styles.mobile}><summary>In this article</summary>{links}</details>
  </>;
}
