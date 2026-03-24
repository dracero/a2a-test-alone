'use client';

import { useState } from 'react';
import { chatAPI } from '@/lib/api';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Loader2, Key, Eye, EyeOff, Check } from 'lucide-react';

export function ApiKeyManager() {
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);

  const handleUpdateApiKey = async () => {
    if (!apiKey.trim()) return;

    setIsLoading(true);
    try {
      await chatAPI.updateApiKey(apiKey);
      setShowSuccess(true);
      setTimeout(() => setShowSuccess(false), 3000);
    } catch (error) {
      console.error('Failed to update API key:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="bg-rose-100 p-2 rounded-lg">
          <Key className="h-6 w-6 text-rose-600" />
        </div>
        <div>
          <h2 className="text-xl font-semibold text-slate-900">API Key</h2>
          <p className="text-sm text-slate-500">Manage your API key</p>
        </div>
      </div>

      <div className="space-y-4">
        <div>
          <label className="text-sm font-medium text-slate-700 block mb-2">
            Google API Key
          </label>
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Input
                type={showKey ? 'text' : 'password'}
                placeholder="Enter your API key"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                disabled={isLoading}
                className="pr-10"
              />
              <button
                onClick={() => setShowKey(!showKey)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
              >
                {showKey ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </button>
            </div>
            <Button
              onClick={handleUpdateApiKey}
              disabled={isLoading || !apiKey.trim()}
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                'Update'
              )}
            </Button>
          </div>
        </div>

        {showSuccess && (
          <div className="p-3 bg-green-50 border border-green-200 rounded-lg flex items-center gap-2">
            <Check className="h-4 w-4 text-green-600" />
            <p className="text-sm text-green-700 font-medium">
              API key updated successfully
            </p>
          </div>
        )}

        <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-xs text-blue-700">
            The API key is used for authentication with the backend service. Keep
            it secure and never share it publicly.
          </p>
        </div>
      </div>
    </Card>
  );
}
