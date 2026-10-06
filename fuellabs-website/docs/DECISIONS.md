# Decision log

## Working decisions

- Typography v2: Direction C is locked for implementation. General Sans remains the display, heading, title, deck, navigation and interface face; Montserrat 400 is used for research summaries and long-form article prose, with 500 reserved for limited semantic emphasis.
- The v1 implementation is preserved at Git tag `website-v1` (`661f5dc`).
- IBM Plex Sans remains the wordmark face.
- Decorative eyebrow labels and decorative section numbering are removed. IBM Plex Mono remains reserved for genuine technical data and metadata.
- Desktop hero copy is locked to two lines: `An independent` / `AI research lab.` followed by the proportional lime square.
- Home, HobbyLM, Research and Team are separate routes.
- Team treatment: `Fuel Labs Team — Signal Field v1`.
- Home structure: `Fuel Labs Home — Signal Gateway v1`.
- Opening treatment: `Curtain Reveal v3` with 1s appearance, 0.8s hold, 1s rise.
- Content and design primitives are centralized in this codebase.
- Long-form research layout: centred title and metadata, a centred left-aligned reading column of approximately 65–70 characters, and wide figure breakouts. Desktop uses a sticky contents rail with active-section indication; tablet and mobile use a collapsible `In this article` control.
- Pretraining article visuals: Figures 1–6 are approved by Prajhan for publication. Figure 1 intentionally omits hardware/provider disclosure for this release. Figure 7, `Release boundary`, is excluded from the article and remains only as an archived study.
- Final HobbyLM architecture visual: the isometric `Inside a sparse model` direction is selected. The `Context behind a result` direction remains open for clarification and refinement.
- Homepage/HobbyLM sparse-MoE motion: the main motion direction is provisionally locked after factual and visual cleanup. Remove review-only `FIG. 01` treatment and any illustrative router scores or labels that could read as measured telemetry before production use.

## Page directions

### Home

- Route: `/`.
- Direction: `Fuel Labs Home — Signal Gateway v1`.
- Two-line desktop hero: `An independent` / `AI research lab.` with the proportional lime square.
- Sections: hero, HobbyLM introduction with sparse-routing visual, research preview and shared footer.
- Opening treatment: `Curtain Reveal v3` with reduced-motion bypass and a visible skip action.

### HobbyLM

- Route: `/hobbylm`.
- Direction: `Fuel Labs HobbyLM — Sparse Ledger v1`, with an asymmetric editorial introduction, ruled model/research records, alternating technical chapters and a dark MLX chapter.
- The isometric `Inside a sparse model` architecture direction is selected.
- The sparse-MoE motion direction is provisionally locked subject to factual-label, mobile-composition and accessibility cleanup.
- Final GitHub, Hugging Face and Space destinations are added only after verification.

### Research index

- Route: `/research`.
- Direction: compact one-page editorial index with a Research masthead followed by three image-led article columns on desktop. Mobile stacks the same articles vertically.
- General Sans carries titles and interface text; Montserrat carries article summaries; IBM Plex Mono is limited to genuine numbering and publication status.
- Published and unpublished states must be visually clear. Unpublished rows must not lead to implementation-placeholder pages.

### Research article

- Route: `/research/[slug]`.
- Direction: compact centred title and metadata masthead, a centred left-aligned reading column of approximately 65–70 characters, and wide figure breakouts.
- Desktop uses a sticky contents rail with active-section indication. Tablet and mobile use a collapsible `In this article` control.
- The pretraining article integrates six visuals: production Figures 1–6. The release-boundary study is not published.
- Article prose uses Montserrat 400; display headings and interface copy remain General Sans; technical data uses IBM Plex Mono.

### Team

- Route: `/team`.
- Direction: `Fuel Labs Team — Signal Field v1`.
- Preserve the Team heading, two engraved portraits, lime active-founder treatment and compact co-founder selector within the shared ruled page grid.
- The portrait field uses the sparse Fuel signal pattern. Desktop and mobile use the same founder-selection behavior without biography filler.

### Style guide

- Route: `/style-guide`.
- Internal implementation reference for color, typography, signal-square, spacing and component review. It is not a primary public navigation destination.

## Pending decisions

- Final public research copy and article bodies.
- Approved GitHub and Hugging Face URLs.
- Final technical HobbyLM claims and evidence.
- Final long-form research typography review after real article bodies replace the current placeholder.
- Final treatment for the `Context behind a result` visual.
