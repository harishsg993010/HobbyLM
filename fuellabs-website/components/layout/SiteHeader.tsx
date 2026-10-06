"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { Wordmark } from "@/components/brand/Wordmark";
import styles from "./SiteHeader.module.css";

const links = [{ href: "/hobbylm", label: "HobbyLM" }, { href: "/research", label: "Research" }, { href: "/team", label: "Team" }];

export function SiteHeader() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        <Wordmark />
        <button className={styles.menu} type="button" aria-expanded={open} aria-controls="main-navigation" onClick={() => setOpen(!open)}>{open ? "Close" : "Menu"}</button>
        <nav id="main-navigation" className={`${styles.nav} ${open ? styles.open : ""}`} aria-label="Main navigation">
          {links.map((link) => <Link key={link.href} href={link.href} aria-current={pathname.startsWith(link.href) ? "page" : undefined} onClick={() => setOpen(false)}>{link.label}</Link>)}
        </nav>
      </div>
    </header>
  );
}
