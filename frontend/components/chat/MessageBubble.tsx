'use client';

import { Message, Part } from '@/lib/api';
import { cn } from '@/lib/utils';
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
  
  // Guard clause
  if (!part || typeof part.kind === 'undefined') {
    console.warn('⚠️ Invalid part in MessagePart:', part);
    return null;
  }

  // Texto
  if (part.kind === 'text') {
    return (
      <p className="text-sm leading-relaxed whitespace-pre-wrap break-words">
        {part.text}
      </p>
    );
  }

  // Archivo
  if (part.kind === 'file') {
    if (!part.file) {
      console.error('❌ File part without file object:', part);
      return (
        <div className="text-sm text-red-500">
          Error: File part missing file data
        </div>
      );
    }

    // DEBUG CRÍTICO: Ver exactamente qué tiene el file
    console.log('🔍 File part details:', {
      file: part.file,
      mime_type: part.file.mime_type,
      mimeTypeType: typeof part.file.mime_type,
      hasBytes: !!part.file.bytes,
      hasUri: !!part.file.uri,
      bytesLength: part.file.bytes?.length || 0,
      keys: Object.keys(part.file)
    });

    const isImage = part.file.mime_type?.startsWith('image/');
    
    console.log('🔍 isImage check:', {
      isImage,
      mime_type: part.file.mime_type,
      startsWithImage: part.file.mime_type?.startsWith('image/')
    });

    if (isImage) {
      const file = part.file;
      let imageSrc = '';

      console.log('🖼️ Rendering image:', {
        mimeType: file.mime_type,
        hasUri: !!file.uri,
        hasBytes: !!file.bytes,
        bytesLength: file.bytes?.length || 0,
        uriPreview: file.uri?.substring(0, 50)
      });

      // Prioridad 1: URI (si viene del servidor después del cache)
      if (file.uri) {
        imageSrc = file.uri;
        console.log('✅ Using URI for image:', imageSrc);
      } 
      // Prioridad 2: Bytes (base64 directo)
      else if (file.bytes) {
        let bytesStr = file.bytes as string;

        // Limpiar el prefijo si viene incluido
        if (bytesStr.includes('base64,')) {
          bytesStr = bytesStr.split('base64,')[1];
          console.log('🔧 Cleaned base64 prefix from bytes');
        }

        const mime = file.mime_type || 'image/png';
        imageSrc = `data:${mime};base64,${bytesStr}`;
        console.log('✅ Using base64 for image:', {
          mimeType: mime,
          srcLength: imageSrc.length,
          srcPreview: imageSrc.substring(0, 100) + '...'
        });
      }

      if (!imageSrc) {
        console.error('❌ No image source available:', { file });
        return (
          <div className="text-sm text-red-500">
            Error: No se pudo cargar la imagen (sin URI ni bytes)
          </div>
        );
      }

      if (imageError) {
        return (
          <div className={cn('text-sm px-3 py-2 rounded-lg', isUser ? 'bg-blue-700' : 'bg-slate-200')}>
            <p className="text-red-500">❌ Error cargando imagen</p>
            <p className="text-xs mt-1 opacity-70">
              MIME: {file.mime_type}
              <br />
              Source: {imageSrc.substring(0, 50)}...
            </p>
          </div>
        );
      }

      // Si es data URL (base64), usar <img> regular
      if (imageSrc.startsWith('data:')) {
        return (
          <div className="rounded-lg overflow-hidden">
            <img
              src={imageSrc}
              alt="Uploaded content"
              className="max-w-full h-auto max-h-96 object-contain"
              onError={(e) => {
                console.error('❌ Error loading base64 image:', {
                  mimeType: file.mime_type,
                  srcPreview: imageSrc.substring(0, 100) + '...',
                  error: e
                });
                setImageError(true);
              }}
              onLoad={() => {
                console.log('✅ Image loaded successfully (base64)');
              }}
            />
          </div>
        );
      }

      // Si es URI, usar Next.js Image (con unoptimized para URIs relativas)
      return (
        <div className="rounded-lg overflow-hidden">
          <Image
            src={imageSrc}
            alt="Uploaded content"
            width={400}
            height={300}
            className="max-w-full h-auto max-h-96 object-contain"
            unoptimized={true}
            onError={(e) => {
              console.error('❌ Error loading image from URI:', {
                src: imageSrc,
                mimeType: file.mime_type,
                error: e
              });
              setImageError(true);
            }}
            onLoad={() => {
              console.log('✅ Image loaded successfully (URI)');
            }}
          />
        </div>
      );
    }

    // Archivo no-imagen
    return (
      <div className={cn('text-sm px-3 py-2 rounded-lg', isUser ? 'bg-blue-700' : 'bg-slate-200')}>
        📎 File: {part.file.mime_type}
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

  // Tipo desconocido
  console.error('❌ Unknown part type:', part);
  return (
    <div className="text-sm text-red-500">
      Error: Unknown content type
    </div>
  );
}