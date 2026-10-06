# fuellabs. website

The maintainable website implementation for fuellabs. It consolidates the approved design studies into shared components, tokens, structured content and four separate routes.

## Run locally

```bash
npm install
npm run dev
```

Open `http://localhost:3000`. The internal design reference is at `/style-guide`.

## Structure

- `app/` — Home, HobbyLM, Research, Team and article routes
- `components/` — shared brand, navigation, typography, research and team UI
- `content/` — structured content records
- `styles/tokens.css` — global design tokens
- `public/` — production assets
- `docs/` — design system, content, decisions and handoff guidance

Read `docs/HANDOFF.md` before creating a new visual variation. Locked studies and V1 archives outside this folder remain untouched.
