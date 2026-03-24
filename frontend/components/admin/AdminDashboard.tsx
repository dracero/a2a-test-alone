'use client';

import { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ConversationManager } from './ConversationManager';
import { MessageManager } from './MessageManager';
import { EventsManager } from './EventsManager';
import { TasksManager } from './TasksManager';
import { AgentsManager } from './AgentsManager';
import { ApiKeyManager } from './ApiKeyManager';
import {
  MessageCircle,
  AlertCircle,
  Activity,
  CheckSquare,
  Cpu,
  Key,
  Settings,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ChatInterface } from '../chat/ChatInterface';

export function AdminDashboard() {
  const [currentTab, setCurrentTab] = useState('chat');

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="bg-white border-b px-6 py-4 shadow-sm sticky top-0 z-10">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-slate-900 p-2 rounded-lg">
              <Settings className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-slate-900">
                Agent Control Panel
              </h1>
              <p className="text-sm text-slate-500">
                Manage conversations, agents, and system configuration
              </p>
            </div>
          </div>
        </div>
      </header>

      <Tabs value={currentTab} onValueChange={setCurrentTab} className="w-full">
        <TabsList className="w-full justify-start rounded-none border-b bg-white px-6 py-0">
          <TabsTrigger
            value="chat"
            className="flex items-center gap-2 rounded-none border-b-2 border-transparent px-4 py-4 hover:border-slate-300"
          >
            <MessageCircle className="h-4 w-4" />
            Chat
          </TabsTrigger>
          <TabsTrigger
            value="conversations"
            className="flex items-center gap-2 rounded-none border-b-2 border-transparent px-4 py-4 hover:border-slate-300"
          >
            <MessageCircle className="h-4 w-4" />
            Conversations
          </TabsTrigger>
          <TabsTrigger
            value="messages"
            className="flex items-center gap-2 rounded-none border-b-2 border-transparent px-4 py-4 hover:border-slate-300"
          >
            <AlertCircle className="h-4 w-4" />
            Messages
          </TabsTrigger>
          <TabsTrigger
            value="events"
            className="flex items-center gap-2 rounded-none border-b-2 border-transparent px-4 py-4 hover:border-slate-300"
          >
            <Activity className="h-4 w-4" />
            Events
          </TabsTrigger>
          <TabsTrigger
            value="tasks"
            className="flex items-center gap-2 rounded-none border-b-2 border-transparent px-4 py-4 hover:border-slate-300"
          >
            <CheckSquare className="h-4 w-4" />
            Tasks
          </TabsTrigger>
          <TabsTrigger
            value="agents"
            className="flex items-center gap-2 rounded-none border-b-2 border-transparent px-4 py-4 hover:border-slate-300"
          >
            <Cpu className="h-4 w-4" />
            Agents
          </TabsTrigger>
          <TabsTrigger
            value="settings"
            className="flex items-center gap-2 rounded-none border-b-2 border-transparent px-4 py-4 hover:border-slate-300"
          >
            <Key className="h-4 w-4" />
            Settings
          </TabsTrigger>
        </TabsList>

        <div className="p-6">
          <TabsContent value="chat" className="mt-0">
            <ChatInterface />
          </TabsContent>

          <TabsContent value="conversations" className="space-y-6 mt-0">
            <ConversationManager />
          </TabsContent>

          <TabsContent value="messages" className="space-y-6 mt-0">
            <MessageManager />
          </TabsContent>

          <TabsContent value="events" className="space-y-6 mt-0">
            <EventsManager />
          </TabsContent>

          <TabsContent value="tasks" className="space-y-6 mt-0">
            <TasksManager />
          </TabsContent>

          <TabsContent value="agents" className="space-y-6 mt-0">
            <AgentsManager />
          </TabsContent>

          <TabsContent value="settings" className="space-y-6 mt-0">
            <div className="grid grid-cols-1 gap-6">
              <ApiKeyManager />
              <div className="rounded-lg border border-slate-200 bg-white p-6">
                <h3 className="text-lg font-semibold text-slate-900 mb-4">
                  System Information
                </h3>
                <div className="space-y-3">
                  <div>
                    <p className="text-sm font-medium text-slate-600">
                      Backend URL:
                    </p>
                    <p className="text-xs text-slate-500 font-mono">
                      http://localhost:12000
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-slate-600">
                      Frontend Version:
                    </p>
                    <p className="text-xs text-slate-500">1.0.0</p>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}
