# Content guide

## Where content lives

- Shared site records: `content/site.ts`
- Route composition: `app/**/page.tsx`
- Images and diagrams: `public/portraits` and `public/diagrams`

Research article metadata is data, not layout. Add a record to `researchArticles`; the Research list and static route are generated from it. When articles become long-form, move their reviewed body content to MDX without changing the shared page shell.

## Publication rules

- Publish technical claims only after evidence and technical review.
- Keep draft status, approval notes, and internal instructions out of public copy.
- Do not publicly name the anonymous advisor.
- Use verified project assets for factual diagrams. Label conceptual illustrations clearly.
- Add GitHub and Hugging Face destinations only when the final links are approved.

## Adding an article

1. Add its slug, number, title, summary and status in `content/site.ts`.
2. Add reviewed body content to the matching route or MDX file.
3. Add diagrams in `public/diagrams` with descriptive alt text.
4. Review technical claims and known limitations.
5. Check desktop and mobile screenshots before publishing.
