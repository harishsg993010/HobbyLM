import Link from "next/link";
import { SignalHeading } from "@/components/typography/SignalHeading";

export default function NotFound() {
  return (
    <section className="section">
      <div className="container not-found">
        <SignalHeading as="h1" className="display">Page not found</SignalHeading>
        <p className="intro-copy">The page you requested is not available.</p>
        <Link className="arrow-link" href="/research">Return to research</Link>
      </div>
    </section>
  );
}
