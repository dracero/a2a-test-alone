'use client';

import { useState, useEffect } from 'react';
import { chatAPI, Task } from '@/lib/api';
import { Card } from '@/components/ui/card';
import { Loader2, RotateCw, CheckSquare } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';

export function TasksManager() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    fetchTasks();
  }, []);

  const fetchTasks = async () => {
    try {
      const data = await chatAPI.listTasks();
      setTasks(data);
    } catch (error) {
      console.error('Failed to fetch tasks:', error);
    }
  };

  const handleRefresh = async () => {
    setIsLoading(true);
    await fetchTasks();
    setIsLoading(false);
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return 'bg-green-100 text-green-700';
      case 'running':
        return 'bg-blue-100 text-blue-700';
      case 'failed':
        return 'bg-red-100 text-red-700';
      default:
        return 'bg-slate-100 text-slate-700';
    }
  };

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="bg-purple-100 p-2 rounded-lg">
            <CheckSquare className="h-6 w-6 text-purple-600" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-slate-900">Tasks</h2>
            <p className="text-sm text-slate-500">
              {tasks.length} task{tasks.length !== 1 ? 's' : ''}
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

      {tasks.length === 0 ? (
        <div className="text-center py-8 text-slate-500">No tasks yet</div>
      ) : (
        <div className="space-y-2">
          {tasks.map((task, idx) => (
            <Dialog key={idx}>
              <DialogTrigger asChild>
                <button className="w-full text-left p-3 rounded-lg hover:bg-slate-50 transition-colors border border-slate-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-slate-900">
                        {task.name}
                      </p>
                      <p className="text-xs text-slate-500">
                        Created: {new Date(task.created_at).toLocaleString()}
                      </p>
                    </div>
                    <span
                      className={`px-2 py-1 text-xs rounded-full font-medium ${getStatusColor(task.status)}`}
                    >
                      {task.status}
                    </span>
                  </div>
                </button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle>Task Details</DialogTitle>
                </DialogHeader>
                <div className="space-y-3">
                  <div>
                    <p className="text-sm font-medium text-slate-600">ID:</p>
                    <p className="text-xs font-mono text-slate-500 break-all">
                      {task.id}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-slate-600">Name:</p>
                    <p className="text-xs text-slate-500">{task.name}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-slate-600">Status:</p>
                    <span
                      className={`inline-block px-2 py-1 text-xs rounded-full font-medium ${getStatusColor(task.status)}`}
                    >
                      {task.status}
                    </span>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-slate-600">
                      Created:
                    </p>
                    <p className="text-xs text-slate-500">
                      {new Date(task.created_at).toLocaleString()}
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
