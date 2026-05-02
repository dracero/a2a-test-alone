'use client';

import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import { cn } from '@/lib/utils';

interface MarkdownWithLatexProps {
  content: string;
  className?: string;
}

export function MarkdownWithLatex({ content, className }: MarkdownWithLatexProps) {
  return (
    <div className={cn('prose prose-sm max-w-none dark:prose-invert', className)}>
      <ReactMarkdown
        remarkPlugins={[remarkMath, remarkGfm]}
        rehypePlugins={[rehypeKatex]}
        components={{
          p: ({ children }: any) => (
            <p className="mb-3 last:mb-0 leading-7">{children}</p>
          ),
          h1: ({ children }: any) => <h1 className="text-xl font-bold mb-3 mt-4">{children}</h1>,
          h2: ({ children }: any) => <h2 className="text-lg font-bold mb-2 mt-3">{children}</h2>,
          h3: ({ children }: any) => <h3 className="text-base font-semibold mb-2 mt-3">{children}</h3>,
          ul: ({ children }: any) => <ul className="list-disc list-outside ml-4 mb-3 space-y-1">{children}</ul>,
          ol: ({ children }: any) => <ol className="list-decimal list-outside ml-4 mb-3 space-y-1">{children}</ol>,
          li: ({ children }: any) => <li className="leading-6">{children}</li>,
          code: ({ className, children, ...props }: any) => {
            const isBlock = className?.includes('language-');
            return isBlock ? (
              <code className="block bg-slate-800 text-slate-100 p-3 rounded-lg text-xs font-mono overflow-x-auto mb-3" {...props}>
                {children}
              </code>
            ) : (
              <code className="bg-slate-200 dark:bg-slate-700 px-1 py-0.5 rounded text-xs font-mono" {...props}>
                {children}
              </code>
            );
          },
          pre: ({ children }: any) => <pre className="mb-3">{children}</pre>,
          blockquote: ({ children }: any) => (
            <blockquote className="border-l-4 border-slate-300 pl-3 italic text-slate-600 my-3">
              {children}
            </blockquote>
          ),
          table: ({ children }: any) => (
            <div className="overflow-x-auto mb-3">
              <table className="min-w-full border-collapse text-sm">{children}</table>
            </div>
          ),
          th: ({ children }: any) => (
            <th className="border border-slate-300 px-3 py-1.5 bg-slate-100 font-semibold text-left">{children}</th>
          ),
          td: ({ children }: any) => (
            <td className="border border-slate-300 px-3 py-1.5">{children}</td>
          ),
          hr: () => <hr className="my-4 border-slate-300" />,
          strong: ({ children }: any) => <strong className="font-semibold">{children}</strong>,
          em: ({ children }: any) => <em className="italic">{children}</em>,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
