'use client';

import { useState, useRef, KeyboardEvent } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Send, Image as ImageIcon, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import Image from 'next/image';

interface MessageInputProps {
  onSend: (text: string, image?: { bytes: string; mimeType: string }) => void;
  disabled?: boolean;
}

export function MessageInput({ onSend, disabled }: MessageInputProps) {
  const [message, setMessage] = useState('');
  const [image, setImage] = useState<{
    bytes: string;
    mimeType: string;
    preview: string;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSend = () => {
    if ((!message.trim() && !image) || disabled) return;

    console.log('📤 MessageInput sending:', {
      hasText: !!message.trim(),
      hasImage: !!image,
      image: image ? {
        mimeType: image.mimeType,
        bytesLength: image.bytes.length,
        bytesPreview: image.bytes.substring(0, 50) + '...'
      } : null
    });

    onSend(
      message.trim(),
      image ? { bytes: image.bytes, mimeType: image.mimeType } : undefined
    );
    setMessage('');
    setImage(null);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleImageSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    console.log('📷 Image selected:', {
      name: file.name,
      type: file.type,
      size: file.size
    });

    if (!file.type.startsWith('image/')) {
      alert('Please select an image file');
      return;
    }

    const reader = new FileReader();
    reader.onloadend = () => {
      const base64 = reader.result as string;
      // Asegurarnos de que solo enviamos el base64 puro sin el prefijo
      const base64Data = base64.includes('base64,') 
        ? base64.split('base64,')[1]
        : base64;
      
      console.log('✅ Image processed:', {
        mimeType: file.type,
        base64Length: base64Data.length,
        base64Preview: base64Data.substring(0, 50) + '...',
        previewLength: base64.length
      });

      setImage({
        bytes: base64Data,
        mimeType: file.type,
        preview: base64, // Mantenemos el prefijo para la vista previa
      });
    };
    
    reader.onerror = (error) => {
      console.error('❌ Error reading file:', error);
      alert('Error reading file. Please try again.');
    };
    
    reader.readAsDataURL(file);

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="border-t bg-white p-4">
      {image && (
        <div className="mb-3 relative inline-block">
          <div className="relative">
            <Image
              src={image.preview}
              alt="Preview of uploaded image"
              width={80}
              height={80}
              className="h-20 w-20 object-cover rounded-lg border-2 border-slate-200"
              unoptimized={true}
            />
            <button
              onClick={() => {
                console.log('🗑️ Removing image');
                setImage(null);
              }}
              className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600 transition-colors"
              title="Remove image"
              aria-label="Remove uploaded image"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
          <p className="text-xs text-slate-500 mt-1">{image.mimeType}</p>
        </div>
      )}

      <div className="flex items-end gap-2">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleImageSelect}
          className="hidden"
          title="Upload image"
          aria-label="Upload image"
        />

        <Button
          type="button"
          variant="outline"
          size="icon"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          className="shrink-0"
        >
          <ImageIcon className="h-5 w-5" />
        </Button>

        <Textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message... (Shift+Enter for new line)"
          disabled={disabled}
          className="min-h-[60px] max-h-[200px] resize-none"
        />

        <Button
          onClick={handleSend}
          disabled={disabled || (!message.trim() && !image)}
          size="icon"
          className="shrink-0"
        >
          <Send className="h-5 w-5" />
        </Button>
      </div>
    </div>
  );
}