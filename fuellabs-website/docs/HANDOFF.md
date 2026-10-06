# Handoff guide

The site is ordinary Next.js, TypeScript and CSS. A frontend developer can run it locally with `npm install` and `npm run dev`.

## Safe editing order

1. Change content in `content/` or the relevant page.
2. Reuse components from `components/`.
3. Change global typography or spacing only in `styles/tokens.css`.
4. Check `/style-guide` when introducing a new visual role.
5. Run `npm run lint` and `npm run build`.
6. Review 390, 768, 1024 and 1440 pixel screenshots.

## Figma

Figma is useful as a reference and exploration surface, but it is not the source of truth for the shipped interface. A designer can recreate Figma variables and text styles from `DESIGN-SYSTEM.md`, or use a browser-to-Figma importer for an initial editable capture. Imported layers still need cleanup and component naming.

Keep a small Figma library with the same names as the code tokens: colors, display, section title, article heading, intro, body, label, caption and spacing. Record any approved change in both Figma and `styles/tokens.css` in the same pull request.

## Versioning

- Protected snapshots remain in the existing `FuelLabs-Design-Versions` archives.
- Use Git branches for experiments and pull requests for review.
- Tag approved milestones, for example `website-v1`, `typography-v2`.
- Never edit an archived snapshot; create a new version.
