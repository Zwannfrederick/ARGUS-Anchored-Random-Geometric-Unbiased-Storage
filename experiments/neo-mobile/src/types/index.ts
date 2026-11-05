export type NeoEventPriority = 'low' | 'normal' | 'urgent';

export type NeoEventType =
  | 'message'
  | 'task_progress'
  | 'task_completed'
  | 'approval_required'
  | 'reminder'
  | 'urgent_attention'
  | 'screenshot_ready'
  | 'session_updated'
  | 'error';

export interface NeoEvent {
  id: string;
  session_id: string;
  type: NeoEventType;
  priority: NeoEventPriority;
  title: string;
  body: string;
  timestamp: number;
  payload?: any;
  image_ref?: string;
}

export interface ToolAction {
  id?: string;
  name: string;
  status: 'pending' | 'success' | 'failure';
  label: string;
  arguments?: Record<string, any>;
  result?: any;
  timestamp?: number;
}

export interface ChatMessage {
  id: string;
  session_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  reasoning_content?: string | null;
  thinking_enabled?: boolean;
  tool_calls?: ToolAction[];
  screenshot_url?: string;
  timestamp: number;
  client_msg_id?: string;
  turn_id?: string;
  isPending?: boolean;
}

export interface ApprovalRequest {
  approval_id: string;
  session_id: string;
  command: string;
  action_description: string;
  risk_level: 'medium' | 'high' | 'critical';
  created_at: number;
  status: 'pending' | 'approved' | 'rejected';
  action_hash?: string;
}

export interface Session {
  session_id: string;
  title: string;
  created_at: number;
  updated_at: number;
  last_message?: string;
  hermes_session_id?: string;
}

export type ConnectionStatus = 'connected' | 'connecting' | 'disconnected' | 'reconnecting';

export interface ArgusKvStats {
  live_bytes: number;
  resident_budget_bytes: number;
  resident_bytes: number;
  peak_resident_bytes: number;
  read_bytes: number;
  paged_out_bytes: number;
  attention_calls: number;
}

export interface ArgusStatus {
  enabled: boolean;
  max_bytes: number | null;
  resident_budget_bytes: number | null;
  stats: ArgusKvStats | null;
  generating: boolean;
}

export interface PlainChatMessage {
  role: 'user' | 'assistant';
  content: string;
  reasoning?: string;
  error?: boolean;
}
