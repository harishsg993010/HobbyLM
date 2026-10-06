"use client";

import Image from "next/image";
import { useState } from "react";
import { founders } from "@/content/site";
import styles from "./FounderSelector.module.css";

export function FounderSelector() {
  const [active, setActive] = useState<(typeof founders)[number]["id"]>("prajhan");
  return <div className={styles.content}><section className={styles.portraits} aria-label="fuellabs. co-founders">{founders.map((founder) => <figure className={styles.portrait} key={founder.id}><Image src={`/portraits/${founder.id}${active === founder.id ? "-active" : ""}.png`} width={256} height={254} alt={founder.alt} priority /></figure>)}</section><section className={styles.selector} aria-label="Select a co-founder"><p>Co-founders</p>{founders.map((founder) => <button type="button" key={founder.id} aria-pressed={active === founder.id} onClick={() => setActive(founder.id)}><span className={styles.founderName}>{founder.name}</span></button>)}</section></div>;
}
