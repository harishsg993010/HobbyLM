/**
 * Production manifest for the six figures in "Pretraining HobbyLM from scratch".
 * Figure 7 (release boundary) is archived and intentionally excluded.
 *
 * Captions and alt text are copied verbatim from
 * docs/workstreams/02-PRETRAINING-VISUAL-PROMPTS.md. Figure 1 uses the
 * no-hardware wording because hardware/provider disclosure is intentionally
 * omitted from this release.
 * intendedSectionId values match the article route's current section ids.
 * See docs/workstreams/02-PRETRAINING-FIGURE-INTEGRATION-AUDIT.md.
 */

export type PretrainingFigureKind = "measured-result" | "system-diagram" | "conceptual-illustration";
export type PublicationState = "ready" | "approval-required";

export interface PretrainingFigure {
  /** Stable identifier, independent of numbering. */
  id: string;
  /** Production ID used during design (1–6). */
  productionNumber: number;
  /** Published number, following article reading order. */
  publishedNumber: number;
  title: string;
  kind: PretrainingFigureKind;
  desktopSrc: string;
  mobileSrc: string;
  caption: string;
  alt: string;
  intendedSectionId: string;
  /** Claim IDs from 02-PRETRAINING-EVIDENCE-REGISTRY.md. Empty means no registry claim yet. */
  evidenceClaimIds: readonly string[];
  publicationState: PublicationState;
}

export const pretrainingFigures = [
  {
    id: "pretraining-run-summary",
    productionNumber: 1,
    publishedNumber: 1,
    title: "Run summary",
    kind: "system-diagram",
    desktopSrc: "/research/pretraining/figure-01-run-summary-desktop.svg",
    mobileSrc: "/research/pretraining/figure-01-run-summary-mobile.svg",
    caption: "HobbyLM pretraining ran in two phases: approximately 66 hours of main pretraining followed by 10 hours of annealing, consuming approximately 100 billion tokens.",
    alt: "Two-phase training timeline showing approximately 66 hours of main pretraining and 10 hours of annealing, totaling approximately 76 hours and 100 billion tokens.",
    intendedSectionId: "introduction",
    evidenceClaimIds: ["RUN-05", "RUN-02", "DATA-04", "RELEASE-01", "RELEASE-03"],
    publicationState: "ready",
  },
  {
    id: "pretraining-data-delivery",
    productionNumber: 2,
    publishedNumber: 2,
    title: "Proxy-run data-delivery failure",
    kind: "conceptual-illustration",
    desktopSrc: "/research/pretraining/figure-02-data-delivery-desktop.svg",
    mobileSrc: "/research/pretraining/figure-02-data-delivery-mobile.svg",
    caption: "The proxy run exposed a mismatch between the intended data pool and the sequence delivered to the model. The corrected implementation remains internal; the comparison is conceptual.",
    alt: "Conceptual comparison between long sequential blocks from individual data families and a more varied stream verified before the full run. Exact proportions and ordering are not shown.",
    intendedSectionId: "derisking",
    evidenceClaimIds: ["FAIL-01"],
    publicationState: "ready",
  },
  {
    id: "pretraining-infrastructure-gates",
    productionNumber: 4,
    publishedNumber: 3,
    title: "Infrastructure gates",
    kind: "system-diagram",
    desktopSrc: "/research/pretraining/figure-04-infrastructure-gates-desktop.svg",
    mobileSrc: "/research/pretraining/figure-04-infrastructure-gates-mobile.svg",
    caption: "The full run launched only after progressively larger tests produced observable evidence: loss, validation, checkpoints and expected GPU activity.",
    alt: "Three-stage verification sequence moving from a single-GPU test to a multi-GPU test and then a full-configuration sanity run before the main launch.",
    intendedSectionId: "derisking",
    evidenceClaimIds: [],
    publicationState: "ready",
  },
  {
    id: "pretraining-architecture",
    productionNumber: 3,
    publishedNumber: 4,
    title: "Final HobbyLM architecture",
    kind: "system-diagram",
    desktopSrc: "/research/pretraining/figure-03-architecture-desktop.svg",
    mobileSrc: "/research/pretraining/figure-03-architecture-mobile.svg",
    caption: "HobbyLM uses one dense decoder layer followed by 19 sparse MoE layers. Each MoE layer selects eight of 64 routed experts per token and evaluates one shared expert on every token.",
    alt: "System diagram of HobbyLM with one dense layer and 19 mixture-of-experts layers. An expanded MoE layer shows eight selected paths among 64 routed experts plus one always-active shared expert.",
    intendedSectionId: "data-and-model",
    evidenceClaimIds: ["ARCH-01", "ARCH-02", "ARCH-03", "DATA-07", "RUN-03"],
    publicationState: "ready",
  },
  {
    id: "pretraining-validation",
    productionNumber: 5,
    publishedNumber: 5,
    title: "Main validation trajectory",
    kind: "measured-result",
    desktopSrc: "/research/pretraining/figure-05-validation-desktop.svg",
    mobileSrc: "/research/pretraining/figure-05-validation-mobile.svg",
    caption: "FineWeb validation loss across the main pretraining phase. The original log contains 324 step-labelled periodic measurements plus one final validation pass after step 81,060; loss moved from 4.3557 at the first recorded validation to 3.4112 at the end of the phase.",
    alt: "Raw FineWeb validation loss across 324 recorded steps, beginning at 4.3557 and ending at 3.4521 at step 81,000, with fluctuations. A separate final validation pass after logged step 81,060 reports 3.4112; its exact step is not recorded.",
    intendedSectionId: "training",
    evidenceClaimIds: ["LOSS-01"],
    publicationState: "ready",
  },
  {
    id: "pretraining-checkpoint-selection",
    productionNumber: 6,
    publishedNumber: 6,
    title: "Checkpoint-selection comparison",
    kind: "measured-result",
    desktopSrc: "/research/pretraining/figure-06-checkpoint-selection-desktop.svg",
    mobileSrc: "/research/pretraining/figure-06-checkpoint-selection-mobile.svg",
    caption: "FineWeb validation loss favored the main-phase checkpoint, while the reported seven-task average favored the annealed checkpoint. We selected the latter under the declared downstream protocol; BoolQ was the one reported task that declined.",
    alt: "Comparison showing the seven-task reported average rising from 44.71 to 47.61, BoolQ falling from 56.24 to 49.54, and FineWeb validation loss increasing from 3.4112 to 3.5487 after annealing.",
    intendedSectionId: "checkpoint",
    evidenceClaimIds: ["EVAL-01", "EVAL-02", "EVAL-04", "LOSS-01", "LOSS-02"],
    publicationState: "ready",
  },
] as const satisfies readonly PretrainingFigure[];
