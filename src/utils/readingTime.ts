export function getReadingTime(body: string | undefined): number {
  if (!body) return 1;

  const prose = body
    // Exclude fenced code blocks.
    .replace(/```[\s\S]*?```/g, ' ')
    // Exclude display math.
    .replace(/\$\$[\s\S]*?\$\$/g, ' ')
    .replace(/\\\[[\s\S]*?\\\]/g, ' ')
    // Exclude inline math.
    .replace(/\$[^$\n]+\$/g, ' ')
    .replace(/\\\([\s\S]*?\\\)/g, ' ')
    // Keep link text but remove URLs and Markdown syntax.
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/[#>*_~`]/g, ' ');

  const words = prose.trim().split(/\s+/u).filter(Boolean).length;

  return Math.max(1, Math.ceil(words / 200));
}
