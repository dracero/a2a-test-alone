import katex from 'katex';

export function hasLatex(text: string): boolean {
  if (!text) return false;
  // Check for common LaTeX delimiters
  return (
    text.includes('$') ||
    text.includes('\\[') ||
    text.includes('\\(') ||
    /\\begin\{.*?\}/.test(text)
  );
}

/**
 * Pre-process LaTeX to handle common issues from LLM output:
 * - Normalize \begin{align}/\begin{equation} inside $...$ to $$...$$
 * - Convert unsupported environments to `aligned`
 * - Fix double-escaped backslashes (\\\\vec → \\vec)
 */
function preprocessLatex(text: string): string {
  let result = text;

  // Fix: LLMs sometimes put \begin{align} etc. inside single $...$
  // Convert $\begin{...}...\end{...}$ to $$\begin{...}...\end{...}$$
  result = result.replace(
    /(?<!\$)\$(?!\$)([\s\S]*?\\begin\{(?:align\*?|eqnarray\*?|equation\*?|gathered|split)\}[\s\S]*?\\end\{(?:align\*?|eqnarray\*?|equation\*?|gathered|split)\}[\s\S]*?)(?<!\$)\$(?!\$)/g,
    '$$$$1$$'  // $$ is escaped as $$$$ in replacement
  );

  // Convert unsupported block environments to supported `aligned`
  result = result.replace(/\\begin\{(align\*?|eqnarray\*?|equation\*?)\}/g, '\\begin{aligned}');
  result = result.replace(/\\end\{(align\*?|eqnarray\*?|equation\*?)\}/g, '\\end{aligned}');

  return result;
}

export function renderLatex(text: string): string {
  if (!text) return '';

  let result = preprocessLatex(text);

  // Render block math $$...$$
  result = result.replace(/\$\$([\s\S]*?)\$\$/g, (match, math) => {
    try {
      const rendered = katex.renderToString(math.trim(), { displayMode: true, throwOnError: false });
      return rendered;
    } catch (e) {
      console.error('KaTeX block error:', e);
      return match;
    }
  });

  // Render block math \[...\]
  result = result.replace(/\\\[([\s\S]*?)\\\]/g, (match, math) => {
    try {
      return katex.renderToString(math.trim(), { displayMode: true, throwOnError: false });
    } catch (e) {
      console.error('KaTeX block error:', e);
      return match;
    }
  });

  // Render inline math \(...\)
  result = result.replace(/\\\(([\s\S]*?)\\\)/g, (match, math) => {
    try {
      return katex.renderToString(math.trim(), { displayMode: false, throwOnError: false });
    } catch (e) {
      console.error('KaTeX inline error:', e);
      return match;
    }
  });

  // Render inline math $...$
  // Use a more robust regex that handles multi-character formulas
  // but avoids matching empty $ or dollar amounts like $100
  result = result.replace(/(?<![\\$])\$([^\$\n]+?)(?<!\\)\$/g, (match, math) => {
    try {
      const trimmed = math.trim();
      // Skip if it looks like a price (starts with a digit)
      if (/^\d/.test(trimmed) && !/[\\{}^_]/.test(trimmed)) {
        return match;
      }
      // Skip empty or whitespace-only
      if (trimmed === '') {
        return match;
      }
      return katex.renderToString(trimmed, { displayMode: false, throwOnError: false });
    } catch (e) {
      console.error('KaTeX inline error:', e);
      return match;
    }
  });

  return result;
}
