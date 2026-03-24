'use client';

import { useState, useEffect } from 'react';
import { chatAPI, Event } from '@/lib/api';
import { Card } from '@/components/ui/card';
import { Loader2, RotateCw, Activity } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';

export function EventsManager() {
  const [events, setEvents] = useState<Event[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    fetchEvents();
    const interval = setInterval(fetchEvents, 3000);
    return () => clearInterval(interval);
  }, []);

  const fetchEvents = async () => {
    try {
      const data = await chatAPI.getEvents();
      setEvents(data);
    } catch (error) {
      console.error('Failed to fetch events:', error);
    }
  };

  const handleRefresh = async () => {
    setIsLoading(true);
    await fetchEvents();
    setIsLoading(false);
  };

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="bg-emerald-100 p-2 rounded-lg">
            <Activity className="h-6 w-6 text-emerald-600" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-slate-900">Events</h2>
            <p className="text-sm text-slate-500">
              {events.length} event{events.length !== 1 ? 's' : ''}
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

      {events.length === 0 ? (
        <div className="text-center py-8 text-slate-500">No events yet</div>
      ) : (
        <div className="space-y-2 max-h-80 overflow-y-auto">
          {events.map((event, idx) => (
            <Dialog key={idx}>
              <DialogTrigger asChild>
                <button className="w-full text-left p-3 rounded-lg hover:bg-slate-50 transition-colors border border-slate-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-slate-900">
                        {event.type}
                      </p>
                      <p className="text-xs text-slate-500">
                        {new Date(event.timestamp).toLocaleString()}
                      </p>
                    </div>
                    <span className="px-2 py-1 bg-emerald-100 text-emerald-700 text-xs rounded-full font-medium">
                      {event.type}
                    </span>
                  </div>
                </button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl max-h-96 overflow-y-auto">
                <DialogHeader>
                  <DialogTitle>Event Details</DialogTitle>
                </DialogHeader>
                <div className="space-y-3">
                  <div>
                    <p className="text-sm font-medium text-slate-600">ID:</p>
                    <p className="text-xs font-mono text-slate-500 break-all">
                      {event.id}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-slate-600">Type:</p>
                    <p className="text-xs text-slate-500">{event.type}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-slate-600">
                      Timestamp:
                    </p>
                    <p className="text-xs text-slate-500">
                      {new Date(event.timestamp).toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-slate-600 mb-2">
                      Data:
                    </p>
                    <pre className="p-2 rounded bg-slate-50 border border-slate-200 text-xs overflow-x-auto">
                      {JSON.stringify(event.data, null, 2)}
                    </pre>
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
