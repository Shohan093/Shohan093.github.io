# Shohan — Research Portfolio

Personal academic and research portfolio built with Astro and deployed as a static website.

**Website:** [shohan093.github.io](https://shohan093.github.io/)

## Overview

The website presents my academic background, research interests, selected projects, publications, and professional contact information. It is designed with a minimal, dark visual style and responsive layouts.

## Technology Stack

- **Astro** — static site generation
- **Tailwind CSS** — styling
- **KaTeX** — mathematical notation
- **Shiki** — syntax highlighting
- **GitHub Pages** — website hosting

## Project Structure

```text
public/          Static assets and downloadable files
src/
├── components/  Shared interface components
├── content/     Content collections
├── layouts/     Page layouts
├── pages/       Website routes
├── styles/      Global styles
└── utils/       Reusable utilities
```

## Local Development

**Requirements:** Node.js and npm.

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open the local address printed by Astro, usually `http://localhost:4321`.

Build the production site:

```bash
npm run build
```

Preview the production build locally:

```bash
npm run preview
```

The generated static website is written to `dist/`.

## License

Unless a separate license is added, the website's source code and content remain subject to their respective copyright and licensing terms. Third-party dependencies retain their own licenses.