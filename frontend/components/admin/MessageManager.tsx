'use client';

import { useState, useEffect } from 'react';
import { chatAPI, Message } from '@/lib/api';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2, RotateCw, AlertCircle } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';

export function MessageManager() {
  const [pendingMessages, setPendingMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    fetchPendingMessages();
    const interval = setInterval(fetchPendingMessages, 2000);
    return () => clearInterval(interval);
  }, []);

  const fetchPendingMessages = async () => {
    try {
      const data = await chatAPI.getPendingMessages();
      setPendingMessages(data);
    } catch (error) {
      console.error('Failed to fetch pending messages:', error);
    }
  };

  const handleRefresh = async () => {
    setIsLoading(true);
    await fetchPendingMessages();
    setIsLoading(false);
  };

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="bg-amber-100 p-2 rounded-lg">
            <AlertCircle className="h-6 w-6 text-amber-600" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-slate-900">
              Pending Messages
            </h2>
            <p className="text-sm text-slate-500">
              {pendingMessages.length} pending
            </p>
          </div>
        </div>
        <Button
          onClick={handleRefresh}
          variant="outline"
          disabled={isLoading}
          size="sm"
        >
          {isLoading ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <RotateCw className="h-4 w-4 mr-2" />
          )}
          Refresh
        </Button>
      </div>

      {pendingMessages.length === 0 ? (
        <div className="text-center py-8 text-slate-500">
          No pending messages
        </div>
      ) : (
        <div className="space-y-2">
          {pendingMessages.map((msg, idx) => (
            <Dialog key={idx}>
              <DialogTrigger asChild>
                <button className="w-full text-left p-3 rounded-lg hover:bg-slate-50 transition-colors border border-slate-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-slate-900">
                        {msg.sender}
                      </p>
                      <p className="text-xs text-slate-500">
                        Context: {msg.context_id ? `${msg.context_id.slice(0, 12)}...` : '—'}
                      </p>
                    </div>
                    <span className="px-2 py-1 bg-amber-100 text-amber-700 text-xs rounded-full font-medium">
                      Pending
                    </span>
                  </div>
                </button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle>Message Details</DialogTitle>
                </DialogHeader>
                <div className="space-y-3">
                  <div>
                    <p className="text-sm font-medium text-slate-600">ID:</p>
                    <p className="text-xs font-mono text-slate-500 break-all">
                      {msg.message_id}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-slate-600">
                      Context:
                    </p>
                    <p className="text-xs font-mono text-slate-500 break-all">
                      {msg.context_id}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-slate-600 mb-2">
                      Parts:
                    </p>
                    <div className="space-y-2">
                      {msg.parts?.map((part, idx) => {
                        const root: any = (part as any)?.root ?? part;
                        return (
                          <div
                            key={idx}
                            className="p-2 rounded bg-slate-50 border border-slate-200"
                          >
                            {root?.kind === 'text' ? (
                              <p className="text-xs text-slate-600">
                                {root?.text}
                              </p>
                            ) : (
                              <p className="text-xs text-slate-500">
                                File: {root?.file?.mime_type || 'unknown'}
                              </p>
                            )}
                          </div>
                        );
                      })}
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
