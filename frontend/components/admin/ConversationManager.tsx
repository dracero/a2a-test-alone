'use client';

import { useState, useEffect } from 'react';
import { chatAPI, Conversation } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Loader2, Plus, MessageCircle } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';

export function ConversationManager() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    fetchConversations();
  }, []);

  const fetchConversations = async () => {
    setIsLoading(true);
    try {
      const data = await chatAPI.listConversations();
      setConversations(data);
    } catch (error) {
      console.error('Failed to fetch conversations:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreateConversation = async () => {
    try {
      await chatAPI.createConversation();
      await fetchConversations();
    } catch (error) {
      console.error('Failed to create conversation:', error);
    }
  };

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="bg-blue-100 p-2 rounded-lg">
            <MessageCircle className="h-6 w-6 text-blue-600" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-slate-900">
              Conversations
            </h2>
            <p className="text-sm text-slate-500">
              {conversations.length} conversation{conversations.length !== 1 ? 's' : ''}
            </p>
          </div>
        </div>
        <Button onClick={handleCreateConversation} disabled={isLoading}>
          {isLoading ? (
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
          ) : (
            <Plus className="h-4 w-4 mr-2" />
          )}
          New
        </Button>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin text-slate-400" />
        </div>
      ) : conversations.length === 0 ? (
        <div className="text-center py-8 text-slate-500">
          No conversations yet
        </div>
      ) : (
        <div className="space-y-2">
          {conversations.map((conv) => (
            <Dialog key={conv.conversation_id}>
              <DialogTrigger asChild>
                <button className="w-full text-left p-3 rounded-lg hover:bg-slate-50 transition-colors border border-slate-200">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <MessageCircle className="h-4 w-4 text-slate-400" />
                      <div>
                        <p className="text-sm font-medium text-slate-900">
                          {conv.conversation_id.slice(0, 8)}...
                        </p>
                        <p className="text-xs text-slate-500">
                          {conv.messages?.length || 0} messages
                        </p>
                      </div>
                    </div>
                  </div>
                </button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl max-h-96 overflow-y-auto">
                <DialogHeader>
                  <DialogTitle>Conversation Details</DialogTitle>
                </DialogHeader>
                <div className="space-y-3">
                  <div>
                    <p className="text-sm font-medium text-slate-600">ID:</p>
                    <p className="text-xs font-mono text-slate-500 break-all">
                      {conv.conversation_id}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-slate-600 mb-2">
                      Messages ({conv.messages?.length || 0}):
                    </p>
                    <div className="space-y-2">
                      {conv.messages?.map((msg, idx) => (
                        <div
                          key={idx}
                          className="p-2 rounded bg-slate-50 border border-slate-200"
                        >
                          <p className="text-xs font-mono text-slate-600">
                            {msg.sender}
                          </p>
                          <p className="text-xs text-slate-500 mt-1">
                            {msg.parts
                              ?.map((p) =>
                                p.root?.kind === 'text' ? p.root.text : 'File'
                              )
                              .join(' ')}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          ))}
        </div>
      )}
    </Card>
  );
}
