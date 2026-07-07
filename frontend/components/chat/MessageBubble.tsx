'use client';

import { Message, Part } from '@/lib/api';
import { cn } from '@/lib/utils';
import { MarkdownWithLatex } from './MarkdownWithLatex';
import { User, Bot } from 'lucide-react';
import Image from 'next/image';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { GraduationCap, BookOpen, XCircle } from 'lucide-react';

interface MessageBubbleProps {
  message: Message;
  onSend?: (text: string) => Promise<void>;
  isProfesor?: boolean;
  onCorrect?: (message: Message) => void;
}

export function MessageBubble({ message, onSend, isProfesor, onCorrect }: MessageBubbleProps) {
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
            <MessagePart key={index} part={part} isUser={isUser} onSend={onSend} />
          ))
        ) : (
          <p className="text-sm text-red-500">No content</p>
        )}

        {!isUser && isProfesor && onCorrect && (
          <div className="flex justify-end mt-2 pt-2 border-t border-slate-200">
            <Button
              onClick={() => onCorrect(message)}
              variant="outline"
              size="sm"
              className="text-purple-600 border-purple-200 hover:bg-purple-50 text-xs px-2.5 py-1 h-auto font-medium rounded-lg flex items-center gap-1.5"
            >
              <GraduationCap className="h-3.5 w-3.5" />
              Corregir Agente / Registrar Falencia
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}

function MessagePart({ part, isUser, onSend }: { part: Part; isUser: boolean; onSend?: (text: string) => Promise<void> }) {
  const [imageError, setImageError] = useState(false);

  if (!part || typeof part.kind === 'undefined') return null;

  // ── TEXT ──────────────────────────────────────────────────────────────────
  if (part.kind === 'text') {
    let text = part.text ?? '';

    // Detectar marcadores socráticos
    const hasSocraticChoice = text.includes('<!-- SOCRATIC_CHOICE -->');
    const hasSocraticExit = text.includes('<!-- SOCRATIC_EXIT -->');
    
    // Limpiar marcadores para no renderizarlos
    if (hasSocraticChoice) text = text.replace('<!-- SOCRATIC_CHOICE -->', '');
    if (hasSocraticExit) text = text.replace('<!-- SOCRATIC_EXIT -->', '');

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
      <div className="flex flex-col gap-4 w-full">
        <MarkdownWithLatex
          content={text}
          className="text-sm text-slate-900"
        />
        
        {/* Renderizar botones de elección inicial */}
        {hasSocraticChoice && onSend && (
          <div className="flex flex-wrap gap-2 mt-2 pt-2 border-t border-slate-200">
            <Button 
              onClick={() => onSend("[SOCRATIC]")} 
              className="bg-blue-600 hover:bg-blue-700 text-white gap-2"
              size="sm"
            >
              <GraduationCap className="h-4 w-4" />
              Modo Socrático (Guía Paso a Paso)
            </Button>
            <Button 
              onClick={() => onSend("[DIRECTO]")} 
              variant="outline" 
              className="gap-2"
              size="sm"
            >
              <BookOpen className="h-4 w-4" />
              Explicación Directa
            </Button>
          </div>
        )}

        {/* Renderizar botón de salida durante el modo socrático */}
        {hasSocraticExit && onSend && (
          <div className="flex justify-end mt-1">
            <Button 
              onClick={() => onSend("[DIRECTO]")} 
              variant="ghost" 
              size="sm"
              className="text-slate-500 hover:text-red-600 hover:bg-red-50 gap-1 text-xs"
            >
              <XCircle className="h-3 w-3" />
              Salir del modo socrático
            </Button>
          </div>
        )}
      </div>
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
