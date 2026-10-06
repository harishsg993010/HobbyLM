import type { Metadata } from "next";
import { SignalMark } from "@/components/typography/SignalHeading";
import { TeamProfileSelector } from "./TeamProfileSelector";
import styles from "./page.module.css";

export const metadata: Metadata = {
  title: "Team layout study",
  robots: { index: false, follow: false },
};

export default function TeamProfileStudy() {
  return (
    <div className={styles.page}>
      <header className={styles.heading}>
        <h1>Team<SignalMark /></h1>
      </header>
      <TeamProfileSelector />
    </div>
  );
}
