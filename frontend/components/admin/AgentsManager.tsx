'use client';

import { useState, useEffect } from 'react';
import { chatAPI, Agent } from '@/lib/api';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Loader2, Plus, RotateCw, Cpu } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';

export function AgentsManager() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showRegisterDialog, setShowRegisterDialog] = useState(false);
  const [agentUrl, setAgentUrl] = useState('');
  const [isRegistering, setIsRegistering] = useState(false);

  useEffect(() => {
    fetchAgents();
  }, []);

  const fetchAgents = async () => {
    try {
      const data = await chatAPI.listAgents();
      setAgents(data);
    } catch (error) {
      console.error('Failed to fetch agents:', error);
    }
  };

  const handleRefresh = async () => {
    setIsLoading(true);
    await fetchAgents();
    setIsLoading(false);
  };

  const handleRegisterAgent = async () => {
    if (!agentUrl.trim()) return;

    setIsRegistering(true);
    try {
      await chatAPI.registerAgent(agentUrl);
      setAgentUrl('');
      setShowRegisterDialog(false);
      await fetchAgents();
    } catch (error) {
      console.error('Failed to register agent:', error);
    } finally {
      setIsRegistering(false);
    }
  };

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="bg-cyan-100 p-2 rounded-lg">
            <Cpu className="h-6 w-6 text-cyan-600" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-slate-900">Agents</h2>
            <p className="text-sm text-slate-500">
              {agents.length} agent{agents.length !== 1 ? 's' : ''}
            </p>
          </div>
        </div>
        <div className="flex gap-2">
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
          <Dialog open={showRegisterDialog} onOpenChange={setShowRegisterDialog}>
            <DialogTrigger asChild>
              <Button size="sm">
                <Plus className="h-4 w-4 mr-2" />
                Register
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Register New Agent</DialogTitle>
              </DialogHeader>
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-slate-700 block mb-2">
                    Agent URL
                  </label>
                  <Input
                    placeholder="http://localhost:8000"
                    value={agentUrl}
                    onChange={(e) => setAgentUrl(e.target.value)}
                    disabled={isRegistering}
                  />
                </div>
                <Button
                  onClick={handleRegisterAgent}
                  disabled={isRegistering || !agentUrl.trim()}
                  className="w-full"
                >
                  {isRegistering ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Plus className="h-4 w-4 mr-2" />
                  )}
                  Register Agent
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {agents.length === 0 ? (
        <div className="text-center py-8 text-slate-500">No agents yet</div>
      ) : (
        <div className="space-y-2">
          {agents.map((agent, idx) => (
            <Dialog key={idx}>
              <DialogTrigger asChild>
                <button className="w-full text-left p-3 rounded-lg hover:bg-slate-50 transition-colors border border-slate-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-slate-900">
                        {agent.name}
                      </p>
                      <p className="text-xs text-slate-500 truncate">
                        {agent.url}
                      </p>
                    </div>
                    <span className="px-2 py-1 bg-cyan-100 text-cyan-700 text-xs rounded-full font-medium">
                      Active
                    </span>
                  </div>
                </button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle>Agent Details</DialogTitle>
                </DialogHeader>
                <div className="space-y-3">
                  <div>
                    <p className="text-sm font-medium text-slate-600">ID:</p>
                    <p className="text-xs font-mono text-slate-500 break-all">
                      {agent.id}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-slate-600">Name:</p>
                    <p className="text-xs text-slate-500">{agent.name}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-slate-600">URL:</p>
                    <p className="text-xs text-slate-500 break-all">{agent.url}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-slate-600">
                      Registered:
                    </p>
                    <p className="text-xs text-slate-500">
                      {new Date(agent.created_at).toLocaleString()}
                    </p>
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
