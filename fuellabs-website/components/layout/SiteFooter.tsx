"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Wordmark } from "@/components/brand/Wordmark";
import styles from "./SiteFooter.module.css";

export function SiteFooter() {
  const pathname = usePathname();
  if (pathname === "/") return null;

  return <footer className={styles.footer}><div className={styles.inner}><div className={styles.brand}><Wordmark large /></div><nav className={styles.links} aria-label="Footer navigation"><Link href="/hobbylm">HobbyLM</Link><Link href="/research">Research</Link><Link href="/team">Team</Link></nav><nav className={styles.social} aria-label="Social and contact links"><a href="https://x.com/prajhan" target="_blank" rel="noreferrer">X</a><a href="https://www.linkedin.com/in/prabhu-rajhan-18b79097" target="_blank" rel="noreferrer">LinkedIn</a><a href="mailto:hello@fuellabs.in">Contact</a></nav><div className={styles.note}><span>Independent AI research.</span><span>© {new Date().getFullYear()} fuellabs.</span></div></div></footer>;
}
