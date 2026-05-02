'use client';

import { Message, Part } from '@/lib/api';
import { cn } from '@/lib/utils';
import { MarkdownWithLatex } from './MarkdownWithLatex';
import { User, Bot } from 'lucide-react';
import Image from 'next/image';
import { useState } from 'react';

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div
      className={cn(
        'flex items-start gap-3 mb-4 animate-in fade-in slide-in-from-bottom-2',
        isUser ? 'flex-row-reverse' : 'flex-row'
      )}
    >
      <div
        className={cn(
          'flex h-8 w-8 shrink-0 items-center justify-center rounded-full',
          isUser ? 'bg-blue-600' : 'bg-slate-800'
        )}
      >
        {isUser ? (
          <User className="h-5 w-5 text-white" />
        ) : (
          <Bot className="h-5 w-5 text-white" />
        )}
      </div>

      <div
        className={cn(
          'flex flex-col gap-2 max-w-[80%] rounded-2xl px-4 py-3',
          isUser
            ? 'bg-blue-600 text-white rounded-tr-sm'
            : 'bg-slate-100 text-slate-900 rounded-tl-sm'
        )}
      >
        {message.parts && message.parts.length > 0 ? (
          message.parts.map((part, index) => (
            <MessagePart key={index} part={part} isUser={isUser} />
          ))
        ) : (
          <p className="text-sm text-red-500">No content</p>
        )}
      </div>
    </div>
  );
}

function MessagePart({ part, isUser }: { part: Part; isUser: boolean }) {
  const [imageError, setImageError] = useState(false);

  if (!part || typeof part.kind === 'undefined') return null;

  // ── TEXT ──────────────────────────────────────────────────────────────────
  if (part.kind === 'text') {
    const text = part.text ?? '';

    // User messages: plain text, no markdown needed
    if (isUser) {
      return (
        <p className="text-sm leading-relaxed whitespace-pre-wrap break-words">
          {text}
        </p>
      );
    }

    // Agent messages: full Markdown + LaTeX via react-markdown + rehype-katex
    return (
      <MarkdownWithLatex
        content={text}
        className="text-sm text-slate-900"
      />
    );
  }

  // ── FILE / IMAGE ──────────────────────────────────────────────────────────
  if (part.kind === 'file') {
    if (!part.file) return null;

    const isImage = part.file.mime_type?.startsWith('image/');

    if (isImage) {
      const file = part.file;
      let imageSrc = '';

      if (file.uri) {
        imageSrc = file.uri;
      } else if (file.bytes) {
        const bytesStr = file.bytes.includes('base64,')
          ? file.bytes.split('base64,')[1]
          : file.bytes;
        imageSrc = `data:${file.mime_type || 'image/png'};base64,${bytesStr}`;
      }

      if (!imageSrc) {
        return <p className="text-sm text-red-500">Error: imagen sin datos</p>;
      }

      if (imageError) {
        return <p className="text-sm text-red-500">❌ Error cargando imagen</p>;
      }

      if (imageSrc.startsWith('data:')) {
        return (
          <div className="rounded-lg overflow-hidden">
            <img
              src={imageSrc}
              alt="Generated image"
              className="max-w-full h-auto max-h-96 object-contain"
              onError={() => setImageError(true)}
            />
          </div>
        );
      }

      return (
        <div className="rounded-lg overflow-hidden">
          <Image
            src={imageSrc}
            alt="Generated image"
            width={400}
            height={300}
            className="max-w-full h-auto max-h-96 object-contain"
            unoptimized
            onError={() => setImageError(true)}
          />
        </div>
      );
    }

    // Non-image file
    return (
      <div className={cn('text-sm px-3 py-2 rounded-lg', isUser ? 'bg-blue-700' : 'bg-slate-200')}>
        📎 {part.file.mime_type || 'File'}
        {part.file.uri && (
          <a
            href={part.file.uri}
            target="_blank"
            rel="noopener noreferrer"
            className="ml-2 underline hover:opacity-80"
          >
            Download
          </a>
        )}
      </div>
    );
  }

  return null;
}
