# Website

Static site built with [Lume](https://lume.land/) that renders the main README.md.

## Commands

```bash
deno task serve   # Dev server at http://localhost:3000
deno task build   # Build to ../docs/
```

## Structure

- `_config.ts` - Lume configuration
- `_build.ts` - Copies README.md with front matter
- `_includes/layout.vto` - HTML template
- `styles.css` - Styling
- `main.js` - Section navigation and scroll handling
