'use client';

import { useState, useEffect, useRef } from 'react';
import { chatAPI, Message, Part, Conversation } from '@/lib/api';
import { MessageBubble } from './MessageBubble';
import { MessageInput } from './MessageInput';
import { Button } from '@/components/ui/button';
import { Loader2, Plus, MessageSquare, Brain, GraduationCap } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { cn } from '@/lib/utils';

export function ChatInterface() {
  const [conversationId, setConversationId] = useState<string>('');
  const [messages, setMessages] = useState<Message[] | undefined>(undefined);
  const [isLoading, setIsLoading] = useState(true);
  const [isSending, setIsSending] = useState(false);
  const [isCreatingChat, setIsCreatingChat] = useState(false);
  const [allConversations, setAllConversations] = useState<Conversation[]>([]);
  const [isNamsOpen, setIsNamsOpen] = useState(false);
  const [namsConclusions, setNamsConclusions] = useState<string[]>([]);
  const [isLoadingNams, setIsLoadingNams] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  // New state variables
  const [role, setRole] = useState<'alumno' | 'profesor'>('alumno');
  const [studentName, setStudentName] = useState<string>('');
  const [isEditingName, setIsEditingName] = useState(false);
  const [nameInput, setNameInput] = useState('');
  const [namsDeficiencies, setNamsDeficiencies] = useState<string[]>([]);
  const [namsSelectedAgent, setNamsSelectedAgent] = useState<string>('Tutor Socrático de Física Multimodal');

  // Correction state variables
  const [isCorrectOpen, setIsCorrectOpen] = useState(false);
  const [selectedMessageToCorrect, setSelectedMessageToCorrect] = useState<Message | null>(null);
  const [correctTema, setCorrectTema] = useState('');
  const [correctExplanation, setCorrectExplanation] = useState('');
  const [isSubmittingCorrection, setIsSubmittingCorrection] = useState(false);

  // Sync student name when conversationId or allConversations change
  useEffect(() => {
    if (conversationId && allConversations.length > 0) {
      const activeConv = allConversations.find(c => c.conversation_id === conversationId);
      if (activeConv) {
        setStudentName(activeConv.name || '');
        setNameInput(activeConv.name || '');
      }
    }
  }, [conversationId, allConversations]);

  const handleOpenNams = async () => {
    setIsNamsOpen(true);
    setIsLoadingNams(true);
    
    // Detect active agent from messages
    let activeAgentName = 'Tutor Socrático de Física Multimodal';
    if (messages && messages.length > 0) {
      for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].role === 'agent' && (messages[i] as any).recipient) {
          activeAgentName = (messages[i] as any).recipient;
          break;
        }
      }
    }
    setNamsSelectedAgent(activeAgentName);

    try {
      const activeConv = allConversations.find(c => c.conversation_id === conversationId);
      const studentId = studentName || activeConv?.name || conversationId;
      const response = await chatAPI.getNamsConclusions(studentId, conversationId, activeAgentName);
      if (response.status === 'active') {
        setNamsConclusions(response.conclusions || []);
        setNamsDeficiencies(response.deficiencies || []);
      } else {
        setNamsConclusions([]);
        setNamsDeficiencies([]);
      }
    } catch (error) {
      console.error('Failed to fetch NAMS conclusions:', error);
      setNamsConclusions([]);
      setNamsDeficiencies([]);
    } finally {
      setIsLoadingNams(false);
    }
  };

  const handleNamsAgentChange = async (agentName: string) => {
    setNamsSelectedAgent(agentName);
    setIsLoadingNams(true);
    try {
      const activeConv = allConversations.find(c => c.conversation_id === conversationId);
      const studentId = studentName || activeConv?.name || conversationId;
      const response = await chatAPI.getNamsConclusions(studentId, conversationId, agentName);
      if (response.status === 'active') {
        setNamsConclusions(response.conclusions || []);
        setNamsDeficiencies(response.deficiencies || []);
      } else {
        setNamsConclusions([]);
        setNamsDeficiencies([]);
      }
    } catch (error) {
      console.error('Failed to fetch NAMS conclusions for agent:', error);
      setNamsConclusions([]);
      setNamsDeficiencies([]);
    } finally {
      setIsLoadingNams(false);
    }
  };

  const handleSaveStudentName = async () => {
    if (!conversationId || !nameInput.trim()) return;
    try {
      await chatAPI.updateConversationName(conversationId, nameInput.trim());
      setStudentName(nameInput.trim());
      setIsEditingName(false);
      setAllConversations(prev =>
        prev.map(c =>
          c.conversation_id === conversationId ? { ...c, name: nameInput.trim() } : c
        )
      );
    } catch (error) {
      console.error('Failed to update student name:', error);
    }
  };

  const handleOpenCorrectModal = (message: Message) => {
    setSelectedMessageToCorrect(message);
    setCorrectTema('');
    setCorrectExplanation('');
    setIsCorrectOpen(true);
  };

  const handleSubmitCorrection = async () => {
    const activeConv = allConversations.find(c => c.conversation_id === conversationId);
    const studentId = studentName || activeConv?.name || conversationId;
    if (!studentId || !correctTema.trim() || !correctExplanation.trim()) return;
    
    const agentName = (selectedMessageToCorrect as any)?.recipient;
    setIsSubmittingCorrection(true);
    try {
      const response = await chatAPI.correctAgent(
        studentId,
        correctTema.trim(),
        correctExplanation.trim(),
        agentName
      );
      if (response.status === 'success') {
        setIsCorrectOpen(false);
        alert('Corrección guardada con éxito como falencia del sistema.');
      } else {
        alert('Error al guardar corrección.');
      }
    } catch (error) {
      console.error('Failed to submit correction:', error);
      alert('Error de red al guardar la corrección.');
    } finally {
      setIsSubmittingCorrection(false);
    }
  };

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
        const msgsResponse = await chatAPI.listMessages(firstConv.conversation_id);
        setMessages(normalizeMessages(msgsResponse.result || []));
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
      const response = await chatAPI.listConversations();
      const convs = response.result || [];
      const filtered = (convs as Conversation[]).filter((conv) => conv && conv.conversation_id).reverse();
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
      const response = await chatAPI.createConversation();
      const convId = response.result.conversation_id;
      setConversationId(convId);
      const msgsResponse = await chatAPI.listMessages(convId);
      const msgs = msgsResponse.result || [];
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
        const msgsResponse = await chatAPI.listMessages(convId);
        const msgs = msgsResponse.result || [];
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
      const response = await chatAPI.createConversation();
      const convId = response.result.conversation_id;
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

  function normalizeMessages(msgs: Message[] | undefined): Message[] {
    if (!msgs) return [];
    return msgs.map((m) => ({
      ...m,
      parts: (m.parts || []).map(normalizePart).filter((p): p is Part => p !== null),
    }));
  }

  function normalizePart(p: any): Part | null {
    if (!p) return null;

    // Helper: detect base64 image string
    function asImagePart(text: string): Part | null {
      if (text.startsWith('data:image/')) {
        const match = text.match(/^data:(image\/[^;]+);base64,(.+)$/);
        if (match) return { kind: 'file', file: { mime_type: match[1], bytes: match[2] } };
      }
      if (text.length > 1000 && /^[A-Za-z0-9+/=]+$/.test(text)) {
        return { kind: 'file', file: { mime_type: 'image/png', bytes: text } };
      }
      return null;
    }

    // Already normalized
    if (p.kind === 'text' && p.text !== undefined) {
      return asImagePart(p.text) ?? { kind: 'text', text: p.text };
    }
    if (p.kind === 'file' && p.file) {
      const mime = p.file.mime_type || p.file.mimeType || p.mime_type || 'application/octet-stream';
      return { kind: 'file', file: { mime_type: mime, bytes: p.file.bytes, uri: p.file.uri } };
    }

    // Backend 'root' wrapper
    if (p.root) {
      if (p.root.file) {
        const mime = p.root.file.mime_type || p.root.file.mimeType || 'application/octet-stream';
        return { kind: 'file', file: { mime_type: mime, bytes: p.root.file.bytes, uri: p.root.file.uri } };
      }
      if (p.root.text !== undefined) {
        return asImagePart(p.root.text) ?? { kind: 'text', text: p.root.text };
      }
      if (p.root.mime_type || p.root.bytes || p.root.uri) {
        return { kind: 'file', file: { mime_type: p.root.mime_type, bytes: p.root.bytes, uri: p.root.uri } };
      }
    }

    // Top-level file/text without kind
    if (p.file) {
      const mime = p.file.mime_type || p.file.mimeType || 'application/octet-stream';
      return { kind: 'file', file: { mime_type: mime, bytes: p.file.bytes, uri: p.file.uri } };
    }
    if (p.text !== undefined) {
      return asImagePart(p.text) ?? { kind: 'text', text: p.text };
    }

    return { kind: 'text', text: '[Unrecognized content]' };
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
          {/* Selector de Rol */}
          <Select
            value={role}
            onValueChange={(val: any) => setRole(val)}
          >
            <SelectTrigger className="w-[145px] font-semibold text-slate-700 border-slate-200">
              <SelectValue placeholder="Rol" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="alumno">👨‍🎓 Alumno</SelectItem>
              <SelectItem value="profesor">👩‍🏫 Profesor</SelectItem>
            </SelectContent>
          </Select>

          {/* Nombre de Estudiante (Edición / Visualización) */}
          {role === 'alumno' ? (
            <div className="flex items-center gap-1.5 border rounded-lg px-2.5 py-1 bg-slate-50 border-slate-200 h-9 shadow-sm">
              {isEditingName ? (
                <>
                  <input
                    value={nameInput}
                    onChange={(e) => setNameInput(e.target.value)}
                    placeholder="Nombre alumno"
                    className="w-28 bg-transparent text-xs outline-none px-1 border-b border-blue-400"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleSaveStudentName();
                    }}
                  />
                  <Button size="sm" onClick={handleSaveStudentName} className="h-5 px-1.5 text-[10px]">
                    OK
                  </Button>
                  <Button size="sm" variant="ghost" onClick={() => setIsEditingName(false)} className="h-5 px-1 text-[10px]">
                    Cancel
                  </Button>
                </>
              ) : (
                <>
                  <span className="text-[10px] text-slate-500 font-bold uppercase">Alumno:</span>
                  <span className="text-xs font-semibold text-slate-700 max-w-24 truncate">
                    {studentName || 'Sin Nombre'}
                  </span>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      setNameInput(studentName);
                      setIsEditingName(true);
                    }}
                    className="h-5 w-5 p-0 text-slate-400 hover:text-slate-600 text-xs flex items-center justify-center"
                  >
                    ✏️
                  </Button>
                </>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-1.5 border border-purple-100 rounded-lg px-2.5 py-1 bg-purple-50/50 h-9 shadow-sm">
              <span className="text-[10px] text-purple-600 font-bold uppercase tracking-wider">Estudiante:</span>
              <span className="text-xs font-bold text-purple-900 max-w-24 truncate">
                {studentName || 'Conversación ' + conversationId.slice(0, 5)}
              </span>
            </div>
          )}

          {/* Selector de Conversación */}
          <Select
            value={conversationId}
            onValueChange={handleSelectConversation}
            disabled={isCreatingChat}
          >
            <SelectTrigger className="w-[220px]">
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
                    {conv.name ? `${conv.name} (${conv.conversation_id.slice(0, 5)})` : `${conv.conversation_id.slice(0, 8)}...`} ({conv.messages?.length || 0} msgs)
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

          <Button
            onClick={handleOpenNams}
            variant="outline"
            className="gap-2 border-purple-200 text-purple-700 hover:bg-purple-50 hover:text-purple-800"
          >
            <Brain className="h-4 w-4" />
            Memoria NAMS
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
              <MessageBubble 
                key={message.message_id} 
                message={message} 
                onSend={handleSendMessage}
                isProfesor={role === 'profesor'}
                onCorrect={handleOpenCorrectModal}
              />
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

      <Dialog open={isNamsOpen} onOpenChange={setIsNamsOpen}>
        <DialogContent className="max-w-3xl bg-white sm:rounded-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-xl font-bold text-purple-900 border-b pb-2">
              <Brain className="h-6 w-6 text-purple-600 animate-pulse" />
              Memoria a Largo Plazo (NAMS) - Estudiante: {studentName || 'Sin Nombre'}
            </DialogTitle>
          </DialogHeader>

          <div className="flex items-center gap-3 py-2 px-1 border-b">
            <span className="text-xs font-bold text-slate-500 uppercase">Seleccionar Agente:</span>
            <Select
              value={namsSelectedAgent}
              onValueChange={handleNamsAgentChange}
            >
              <SelectTrigger className="w-[280px] h-8 text-xs font-semibold text-slate-700 border-slate-200">
                <SelectValue placeholder="Seleccionar Agente" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Tutor Socrático de Física Multimodal">Tutor Socrático de Física Multimodal</SelectItem>
                <SelectItem value="Asistente Médico">Asistente Médico</SelectItem>
                <SelectItem value="Image Generator Agent">Image Generator Agent</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="mt-4 max-h-[65vh] overflow-y-auto pr-2">
            {isLoadingNams ? (
              <div className="flex flex-col items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-purple-600 mb-2" />
                <p className="text-sm text-slate-500 font-medium">Consultando base de datos de grafos Neo4j Aura DB...</p>
              </div>
            ) : namsConclusions.length === 0 && namsDeficiencies.length === 0 ? (
              <div className="text-center py-12 px-4 rounded-xl border border-dashed border-slate-200 bg-slate-50/50">
                <Brain className="h-10 w-10 text-slate-400 mx-auto mb-2 stroke-1" />
                <p className="font-semibold text-slate-700 text-sm">No se han registrado preferencias, insights ni falencias aún</p>
                <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
                  A medida que interactúes con el bot y el profesor registre correcciones del sistema, esta base de conocimiento se irá poblando.
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Columna de Preferencias de Estilo */}
                <div className="space-y-3">
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider border-b pb-1">
                    Preferencias e Insights del Alumno ({namsConclusions.length})
                  </h3>
                  {namsConclusions.length === 0 ? (
                    <p className="text-xs text-slate-400 italic">No hay preferencias ni insights registrados.</p>
                  ) : (
                    <ul className="space-y-2">
                      {namsConclusions.map((conclusion, idx) => (
                        <li 
                          key={idx} 
                          className="p-3 rounded-lg border border-slate-100 bg-slate-50 text-slate-700 text-xs leading-relaxed flex items-start gap-2 hover:border-slate-200 transition-all"
                        >
                          <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-slate-200 text-[9px] font-bold text-slate-600 mt-0.5">
                            {idx + 1}
                          </span>
                          <p className="flex-1">{conclusion}</p>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>

                {/* Columna de Falencias Registradas */}
                <div className="space-y-3">
                  <h3 className="text-xs font-bold text-purple-700 uppercase tracking-wider border-b pb-1">
                    Falencias del Sistema / Agente ({namsDeficiencies.length})
                  </h3>
                  {namsDeficiencies.length === 0 ? (
                    <p className="text-xs text-slate-400 italic">No hay falencias registradas.</p>
                  ) : (
                    <ul className="space-y-2">
                      {namsDeficiencies.map((deficiency, idx) => (
                        <li 
                          key={idx} 
                          className="p-3 rounded-lg border border-purple-100 bg-purple-50/30 text-slate-800 text-xs leading-relaxed flex items-start gap-2 hover:border-purple-200 transition-all"
                        >
                          <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-purple-100 text-[9px] font-bold text-purple-700 mt-0.5">
                            {idx + 1}
                          </span>
                          <p className="flex-1 font-medium">{deficiency}</p>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>

      {/* Dialog de Corrección de Profesor */}
      <Dialog open={isCorrectOpen} onOpenChange={setIsCorrectOpen}>
        <DialogContent className="max-w-md bg-white sm:rounded-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-lg font-bold text-purple-900 border-b pb-2">
              <GraduationCap className="h-5 w-5 text-purple-600" />
              Corregir Agente / Registrar Falencia
            </DialogTitle>
          </DialogHeader>
          <div className="mt-4 space-y-4">
            <p className="text-xs text-slate-500 leading-relaxed">
              Registra una corrección física/conceptual sobre las respuestas del agente. Esta corrección quedará asociada como una falencia del sistema para el agente y guiará al tutor en futuras conversaciones globales.
            </p>
            
            <div className="space-y-1.5">
              <label htmlFor="tema" className="text-xs font-bold text-slate-700">Tema o Concepto Físico</label>
              <Input
                id="tema"
                value={correctTema}
                onChange={(e) => setCorrectTema(e.target.value)}
                placeholder="Ej: Fuerza de rozamiento, Conservación de energía"
                className="text-sm"
              />
            </div>

            <div className="space-y-1.5">
              <label htmlFor="explicacion" className="text-xs font-bold text-slate-700">Corrección / Concepto a Corregir</label>
              <Textarea
                id="explicacion"
                value={correctExplanation}
                onChange={(e) => setCorrectExplanation(e.target.value)}
                placeholder="Ej: En rodadura pura sin deslizar, el rozamiento no es μ*N. Se debe calcular a partir de las ecuaciones de Newton y la relación de aceleraciones."
                rows={4}
                className="text-sm"
              />
            </div>

            <div className="flex justify-end gap-2 pt-2 border-t">
              <Button
                variant="outline"
                onClick={() => setIsCorrectOpen(false)}
                disabled={isSubmittingCorrection}
                className="text-xs"
              >
                Cancelar
              </Button>
              <Button
                onClick={handleSubmitCorrection}
                disabled={isSubmittingCorrection || !correctTema.trim() || !correctExplanation.trim()}
                className="bg-purple-600 hover:bg-purple-700 text-white text-xs gap-1.5"
              >
                {isSubmittingCorrection ? (
                  <>
                    <Loader2 className="h-3 w-3 animate-spin" />
                    Guardando...
                  </>
                ) : (
                  <>
                    Guardar Corrección
                  </>
                )}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
