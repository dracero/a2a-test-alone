export interface Agent {
  id: string;
  name: string;
  url: string;
  created_at?: string;
}

export interface Event {
  id: string;
  type: string;
  timestamp: string;
  data: any;
}

export interface Task {
  id: string;
  name?: string;
  title?: string;
  status: string;
  created_at?: string;
}

export interface TextPart {
  kind: 'text';
  text: string;
}

export interface FilePart {
  kind: 'file';
  file: {
    mime_type: string;
    bytes?: string;
    uri?: string;
  };
}

export type Part = TextPart | FilePart;

export interface Message {
  message_id: string;
  context_id: string;
  role: string;
  parts: Part[];
}

export interface Conversation {
  conversation_id: string;
  messages?: Message[];
}

async function postJSON<T>(url: string, body: any = {}): Promise<T> {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export const chatAPI = {
  createConversation: async (): Promise<{ result: { conversation_id: string } }> => {
    return postJSON<{ result: { conversation_id: string } }>('/conversation/create');
  },

  listConversations: async (): Promise<{ result: Conversation[] }> => {
    return postJSON<{ result: Conversation[] }>('/conversation/list');
  },

  sendMessage: async (message: Message): Promise<{ result: { message_id: string; context_id: string } }> => {
    return postJSON<{ result: { message_id: string; context_id: string } }>('/message/send', {
      params: message,
    });
  },

  listMessages: async (conversationId: string): Promise<{ result: Message[] }> => {
    return postJSON<{ result: Message[] }>('/message/list', {
      params: conversationId,
    });
  },

  getPendingMessages: async (): Promise<{ result: Message[] }> => {
    return postJSON<{ result: Message[] }>('/message/pending');
  },

  getEvents: async (): Promise<{ result: Event[] }> => {
    return postJSON<{ result: Event[] }>('/events/get');
  },

  listTasks: async (): Promise<{ result: Task[] }> => {
    return postJSON<{ result: Task[] }>('/task/list');
  },

  registerAgent: async (agentUrl: string): Promise<void> => {
    await postJSON<void>('/agent/register', {
      params: agentUrl,
    });
  },

  getAgents: async (): Promise<{ result: Agent[] }> => {
    return postJSON<{ result: Agent[] }>('/agent/list');
  },

  updateApiKey: async (apiKey: string): Promise<{ status: string }> => {
    return postJSON<{ status: string }>('/api_key/update', {
      api_key: apiKey,
    });
  },

  getNamsConclusions: async (): Promise<{ status: string; conclusions: string[]; message?: string }> => {
    return postJSON<{ status: string; conclusions: string[]; message?: string }>('/nams/conclusions');
  },
};
