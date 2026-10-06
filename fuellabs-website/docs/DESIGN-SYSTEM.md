# fuellabs. design system

This document is the visual source of truth. The live token values are in `styles/tokens.css`, and the rendered reference is available at `/style-guide`.

## Locked brand rules

- Background: Paper `#F2EFE8`
- Text: Ink `#11110F`
- Signal: Lime `#B7F34A`
- Secondary text: Muted `#66665F`
- Dividers: Line `#D8D4C9`
- Wordmark: black `fuel`, muted `labs`, lime square
- Page titles and major section headings end with a proportional lime square.
- Do not publish a location suffix with the brand.

## Typography

- Typography v2 is the locked implementation direction. The recoverable v1 baseline is Git tag `website-v1`.
- Editorial display, headings, article titles, article decks, navigation and interface copy: General Sans 400/500
- Research-list summaries and long-form article prose: Montserrat 400; use 500 only for limited semantic emphasis
- Wordmark: IBM Plex Sans 500
- Technical labels, status and code: IBM Plex Mono 400/500

Use the semantic CSS classes and tokens (`display`, `section-title`, `intro-copy`, `body-copy`, `article-body-copy`, `eyebrow`, `caption`). Page-specific font sizes require a written reason and screenshot review. Decorative eyebrows and decorative section numbering are not part of v2; monospace labels are reserved for genuine metadata and technical information.

General Sans, Montserrat and the IBM Plex faces are self-hosted through `next/font/local`. Keep the source files and their usage limited to the weights above.

## Desktop hero

- Copy is sentence case and intentionally breaks into two lines: `An independent` / `AI research lab.`
- The heading uses General Sans, not the wordmark face.
- The lime square follows the final period and uses the shared `0.22em` proportion.
- The shared wordmark remains unchanged in the header, opening curtain and footer.

## Layout

- Maximum container: 1360px
- Fluid gutters: 24–80px
- Section padding: 128px desktop, 72px mobile
- Body copy: maximum 72 characters
- Intro copy: maximum 42 characters
- Primary responsive breakpoint: 800px

## Motion

Motion supports hierarchy. The home introduction is 1 second appearance, 0.8 second hold, and 1 second curtain rise. Every motion has a `prefers-reduced-motion` alternative.
