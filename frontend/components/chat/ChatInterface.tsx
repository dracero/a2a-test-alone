'use client';

import { useState, useEffect, useRef } from 'react';
import { chatAPI, Message, Part, Conversation } from '@/lib/api';
import { MessageBubble } from './MessageBubble';
import { MessageInput } from './MessageInput';
import { Button } from '@/components/ui/button';
import { Loader2, Plus, MessageSquare } from 'lucide-react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { cn } from '@/lib/utils';

export function ChatInterface() {
  const [conversationId, setConversationId] = useState<string>('');
  const [messages, setMessages] = useState<Message[] | undefined>(undefined);
  const [isLoading, setIsLoading] = useState(true);
  const [isSending, setIsSending] = useState(false);
  const [isCreatingChat, setIsCreatingChat] = useState(false);
  const [allConversations, setAllConversations] = useState<Conversation[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    let isMounted = true;
    
    const init = async () => {
      if (!isMounted) return;
      
      // Cargar conversaciones existentes
      const convs = await loadConversations();
      
      if (!isMounted) return;
      
      // Si hay conversaciones, usar la primera
      if (convs && convs.length > 0) {
        const firstConv = convs[0];
        setConversationId(firstConv.conversation_id);
        const msgs = await chatAPI.listMessages(firstConv.conversation_id);
        setMessages(normalizeMessages(msgs));
        startPolling(firstConv.conversation_id);
        setIsLoading(false);
      } else {
        // Solo crear una nueva si no hay ninguna
        await initializeChat();
      }
    };

    init();

    return () => {
      isMounted = false;
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, []);

  const loadConversations = async () => {
    try {
      const convs = await chatAPI.listConversations();
      const filtered = convs.filter((conv) => conv && conv.conversation_id).reverse();
      setAllConversations(filtered);
      return filtered;
    } catch (error) {
      console.error('Failed to load conversations:', error);
      return [];
    }
  };

  const initializeChat = async () => {
    setIsLoading(true);
    try {
      const convId = await chatAPI.createConversation();
      setConversationId(convId);
      const msgs = await chatAPI.listMessages(convId);
      setMessages(normalizeMessages(msgs));
      setAllConversations((prev) => [
        { conversation_id: convId, messages: msgs },
        ...prev,
      ]);
      startPolling(convId);
    } catch (error) {
      console.error('Failed to initialize chat:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const startPolling = (convId: string) => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
    }

    pollingRef.current = setInterval(async () => {
      if (!convId) return;
      try {
        const msgs = await chatAPI.listMessages(convId);
        const normalized = normalizeMessages(msgs);

        setMessages((prevMsgs) => {
          if (JSON.stringify(prevMsgs) !== JSON.stringify(normalized)) {
            setAllConversations((prevConvs) =>
              prevConvs.map((conv) =>
                conv.conversation_id === convId ? { ...conv, messages: normalized } : conv
              )
            );
            return normalized;
          }
          return prevMsgs;
        });
      } catch (error) {
        console.error('Failed to fetch messages:', error);
      }
    }, 1000);
  };

  const handleSendMessage = async (
    text: string,
    image?: { bytes: string; mimeType: string }
  ) => {
    if (!conversationId) return;

    setIsSending(true);

    try {
      const parts: Part[] = [];

      // Primero las imágenes (si hay)
      if (image?.bytes) {
        const cleanBytes = image.bytes.includes('base64,') 
          ? image.bytes.split('base64,')[1] 
          : image.bytes;
        
        console.log('📤 Adding image part:', {
          mimeType: image.mimeType,
          bytesLength: cleanBytes.length,
          bytesPreview: cleanBytes.substring(0, 50) + '...'
        });

        parts.push({
          kind: 'file',
          file: {
            mime_type: image.mimeType,
            bytes: cleanBytes
          },
        });
      }

      // Luego el texto (si hay)
      if (text.trim()) {
        parts.push({
          kind: 'text',
          text: text.trim(),
        });
      }

      const message: Message = {
        message_id: crypto.randomUUID(),
        context_id: conversationId,
        role: 'user',
        parts: parts,
      };

      console.log('📤 Sending message:', {
        messageId: message.message_id,
        partsCount: parts.length,
        parts: parts.map(p => ({ 
          kind: p.kind, 
          hasData: p.kind === 'file' ? !!p.file?.bytes : !!p.text 
        }))
      });

      // Agregar mensaje INMEDIATAMENTE a la UI
      setMessages((prevMsgs) => [...(prevMsgs || []), message]);
      
      await chatAPI.sendMessage(message);
    } catch (error) {
      console.error('Failed to send message:', error);
    } finally {
      setIsSending(false);
    }
  };

  const handleNewConversation = async () => {
    if (isCreatingChat) return;
    setIsCreatingChat(true);
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
    }
    setMessages(undefined);
    try {
      const convId = await chatAPI.createConversation();
      setConversationId(convId);
      setMessages([]);
      setAllConversations((prev) => [
        { conversation_id: convId, messages: [] },
        ...prev,
      ]);
      startPolling(convId);
    } catch (error) {
      console.error('Failed to create new conversation:', error);
    } finally {
      setIsCreatingChat(false);
    }
  };

  const handleSelectConversation = (convId: string) => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
    }
    setConversationId(convId);
    const selectedConv = allConversations.find(
      (c) => c.conversation_id === convId
    );
    setMessages(normalizeMessages(selectedConv?.messages || []));
    startPolling(convId);
  };

  // ✅ NORMALIZACIÓN MEJORADA
  function normalizeMessages(msgs: Message[] | undefined): Message[] {
    if (!msgs) return [];
    return msgs.map((m) => {
      const normalized = { 
        ...m, 
        parts: (m.parts || []).map(normalizePart).filter(Boolean) 
      };
      
      console.log('📥 Normalized message:', {
        messageId: m.message_id,
        role: m.role,
        originalPartsCount: m.parts?.length || 0,
        normalizedPartsCount: normalized.parts.length,
        parts: normalized.parts.map(p => ({
          kind: p.kind,
          hasBytes: p.kind === 'file' ? !!p.file?.bytes : undefined,
          hasUri: p.kind === 'file' ? !!p.file?.uri : undefined,
          hasText: p.kind === 'text' ? !!p.text : undefined
        }))
      });
      
      return normalized;
    });
  }

  function normalizePart(p: any): Part | null {
    if (!p) {
      console.warn('⚠️ Null part received');
      return null;
    }

    console.log('🔄 Normalizing part - RAW:', {
      raw: p,
      hasKind: !!p.kind,
      kind: p.kind,
      hasRoot: !!p.root,
      hasFile: !!p.file,
      hasText: p.text !== undefined,
      structure: Object.keys(p),
      // Ver TODOS los campos del file si existe
      fileKeys: p.file ? Object.keys(p.file) : null,
      // Ver qué hay en root si existe
      rootKeys: p.root ? Object.keys(p.root) : null
    });

    // 1. Ya está en la forma esperada
    if (p.kind === 'text' && p.text !== undefined) {
      console.log('✅ Text part (already normalized)');
      return { kind: 'text', text: p.text };
    }
    
    if (p.kind === 'file' && p.file) {
      // CRÍTICO: Verificar todas las posibles ubicaciones del mime_type
      const mimeType = p.file.mime_type || p.file.mimeType || p.mime_type || p.mimeType;
      
      console.log('✅ File part (already normalized):', {
        mimeType: mimeType,
        file_mime_type: p.file.mime_type,
        file_mimeType: p.file.mimeType,
        p_mime_type: p.mime_type,
        p_mimeType: p.mimeType,
        hasBytes: !!p.file.bytes,
        hasUri: !!p.file.uri,
        bytesLength: p.file.bytes?.length || 0,
        allFileKeys: Object.keys(p.file),
        allPKeys: Object.keys(p)
      });
      
      return { 
        kind: 'file', 
        file: {
          mime_type: mimeType,
          bytes: p.file.bytes,
          uri: p.file.uri
        }
      };
    }

    // 2. Formato con 'root' (del backend después de cache_content)
    if (p.root) {
      // root.file contiene el archivo
      if (p.root.file) {
        const mimeType = p.root.file.mime_type || p.root.file.mimeType || p.root.mime_type || p.mime_type;
        
        console.log('✅ File part (from root.file):', {
          mimeType: mimeType,
          root_file_mime_type: p.root.file.mime_type,
          root_file_mimeType: p.root.file.mimeType,
          root_mime_type: p.root.mime_type,
          hasBytes: !!p.root.file.bytes,
          hasUri: !!p.root.file.uri,
          rootFileKeys: Object.keys(p.root.file),
          rootKeys: Object.keys(p.root)
        });
        
        return { 
          kind: 'file', 
          file: {
            mime_type: mimeType,
            bytes: p.root.file.bytes,
            uri: p.root.file.uri
          }
        };
      }
      
      // root.text contiene el texto
      if (p.root.text !== undefined) {
        console.log('✅ Text part (from root.text)');
        return { kind: 'text', text: p.root.text };
      }
      
      // root es directamente el file (con mime_type, bytes o uri)
      if (p.root.mime_type || p.root.mimeType || p.root.bytes || p.root.uri) {
        const mimeType = p.root.mime_type || p.root.mimeType;
        
        console.log('✅ File part (root is file):', {
          mimeType: mimeType,
          hasBytes: !!p.root.bytes,
          hasUri: !!p.root.uri
        });
        
        return { 
          kind: 'file', 
          file: {
            mime_type: mimeType,
            bytes: p.root.bytes,
            uri: p.root.uri
          }
        };
      }
    }

    // 3. Tiene 'file' en el nivel superior sin 'kind'
    if (p.file && (p.file.mime_type || p.file.mimeType || p.file.bytes || p.file.uri)) {
      const mimeType = p.file.mime_type || p.file.mimeType || p.mime_type || p.mimeType;
      
      console.log('✅ File part (top level):', {
        mimeType: mimeType,
        hasBytes: !!p.file.bytes,
        hasUri: !!p.file.uri
      });
      
      return { 
        kind: 'file', 
        file: {
          mime_type: mimeType,
          bytes: p.file.bytes,
          uri: p.file.uri
        }
      };
    }

    // 4. Tiene 'text' en el nivel superior sin 'kind'
    if (p.text !== undefined && !p.file && !p.root) {
      console.log('✅ Text part (top level)');
      return { kind: 'text', text: p.text };
    }

    // Fallback
    console.error('❌ Could not normalize part:', p);
    return { kind: 'text', text: `[Error: Could not parse content: ${JSON.stringify(p)}]` };
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-slate-50">
        <div className="text-center">
          <Loader2 className="h-12 w-12 animate-spin text-blue-600 mx-auto mb-4" />
          <p className="text-slate-600">Initializing chat...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-slate-50">
      <header className="bg-white border-b px-6 py-4 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-3">
          <div className="bg-blue-600 p-2 rounded-lg">
            <MessageSquare className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-slate-900">
              AI Assistant
            </h1>
            <p className="text-sm text-slate-500">Powered by Agent SDK</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Select
            value={conversationId}
            onValueChange={handleSelectConversation}
            disabled={isCreatingChat}
          >
            <SelectTrigger className="w-[280px]">
              <SelectValue placeholder="Select a conversation" />
            </SelectTrigger>
            <SelectContent>
              {allConversations
                .filter(
                  (conv): conv is Conversation & { conversation_id: string } =>
                    conv && typeof conv.conversation_id === 'string'
                )
                .map((conv) => (
                  <SelectItem
                    key={conv.conversation_id}
                    value={conv.conversation_id}
                  >
                    {conv.conversation_id.slice(0, 8)}... (
                    {conv.messages?.length || 0} msgs)
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>

          <Button
            onClick={handleNewConversation}
            variant="outline"
            className="gap-2"
            disabled={isCreatingChat}
          >
            {isCreatingChat ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Plus className="h-4 w-4" />
            )}
            New Chat
          </Button>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-6">
        {(!messages || messages.length === 0) ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center max-w-md">
              <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <MessageSquare className="h-8 w-8 text-blue-600" />
              </div>
              <h2 className="text-2xl font-semibold text-slate-900 mb-2">
                Start a conversation
              </h2>
              <p className="text-slate-600">
                Send a message or upload an image to begin chatting with the AI
                assistant. The assistant can analyze images and provide detailed
                responses.
              </p>
            </div>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto">
            {messages.map((message) => (
              <MessageBubble key={message.message_id} message={message} />
            ))}
            {isSending && (
              <div className="flex items-start gap-3 mb-4">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-800">
                  <Loader2 className="h-5 w-5 text-white animate-spin" />
                </div>
                <div className="bg-slate-100 text-slate-900 rounded-2xl rounded-tl-sm px-4 py-3">
                  <p className="text-sm text-slate-600">Processing...</p>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <MessageInput onSend={handleSendMessage} disabled={isSending} />
    </div>
  );
}
