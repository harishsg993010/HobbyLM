import type { ElementType, ReactNode } from "react";

export function SignalMark() {
  return <span className="signal-square" aria-hidden="true" />;
}

export function SignalHeading({ as: Tag = "h2", children, className = "section-title", id, signal = true }: { as?: ElementType; children: ReactNode; className?: string; id?: string; signal?: boolean }) {
  return <Tag className={className} id={id}>{children}{signal && <SignalMark />}</Tag>;
}
