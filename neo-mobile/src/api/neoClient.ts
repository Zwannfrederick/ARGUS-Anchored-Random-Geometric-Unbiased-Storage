import AsyncStorage from '@react-native-async-storage/async-storage';
import * as SecureStore from 'expo-secure-store';
import { consumeSse, StreamedReply } from './chatStream';
import {
  ApprovalRequest,
  ArgusStatus,
  ChatMessage,
  ConnectionStatus,
  NeoEvent,
  PlainChatMessage,
  Session,
  ToolAction,
} from '../types';
import { notificationService } from '../notifications/notificationService';

type MessageHandler = (message: ChatMessage) => void;
type DeltaHandler = (data: { session_id: string; role: string; delta: string; type: 'content' | 'reasoning' }) => void;
type ToolProgressHandler = (data: { session_id: string; tool_action: ToolAction }) => void;
type ApprovalHandler = (approval: ApprovalRequest) => void;
type StatusHandler = (status: ConnectionStatus) => void;
type SessionUpdateHandler = (session: Session) => void;

export const STORAGE_KEY_GATEWAY_URL = '@neo_gateway_url';
export const STORAGE_KEY_TAILSCALE_URL = '@neo_tailscale_url';
export const STORAGE_KEY_LAN_URL = '@neo_lan_url';
export const SECURE_KEY_AUTH_TOKEN = 'neo_auth_token_secure';
export const LEGACY_STORAGE_KEY_AUTH_TOKEN = '@neo_auth_token';

// Default endpoints
export const DEFAULT_LAN_GATEWAY_URL = 'http://192.168.8.5:8765';
export const DEFAULT_TAILSCALE_GATEWAY_URL = 'http://100.98.181.117:8765';
export const DEFAULT_MAGICDNS_GATEWAY_URL = 'http://devshub.tailaa2b98.ts.net:8765';

export class NeoClient {
  private gatewayUrl: string = DEFAULT_LAN_GATEWAY_URL;
  private tailscaleUrl: string = DEFAULT_TAILSCALE_GATEWAY_URL;
  private lanUrl: string = DEFAULT_LAN_GATEWAY_URL;
  // Paired from the gateway's ~/.hermes/neo_auth_token.txt; never shipped in the bundle.
  private authToken: string = '';
  private candidateUrls: string[] = [
    DEFAULT_LAN_GATEWAY_URL,
    DEFAULT_TAILSCALE_GATEWAY_URL,
    DEFAULT_MAGICDNS_GATEWAY_URL,
  ];
  private candidateIndex: number = 0;
  private connectionTimeoutTimer: any = null;
  private activeConnectedUrl: string = '';

  private ws: WebSocket | null = null;
  private status: ConnectionStatus = 'disconnected';
  private reconnectAttempts = 0;
  private reconnectTimer: any = null;
  private pingInterval: any = null;

  private messageListeners: Set<MessageHandler> = new Set();
  private deltaListeners: Set<DeltaHandler> = new Set();
  private toolProgressListeners: Set<ToolProgressHandler> = new Set();
  private approvalListeners: Set<ApprovalHandler> = new Set();
  private statusListeners: Set<StatusHandler> = new Set();
  private sessionUpdateListeners: Set<SessionUpdateHandler> = new Set();

  constructor() {
    this.loadCredentials();
  }

  public updateCandidateList(): void {
    const list: string[] = [];
    const addClean = (u: string) => {
      if (!u) return;
      const clean = u.trim().replace(/\/+$/, '');
      if (clean && !list.includes(clean)) {
        list.push(clean);
      }
    };
    // Priority order:
    // 1. Preferred/active gateway URL
    addClean(this.gatewayUrl);
    // 2. Tailscale endpoint (IP or custom)
    addClean(this.tailscaleUrl);
    // 3. Tailscale MagicDNS endpoint
    addClean(DEFAULT_MAGICDNS_GATEWAY_URL);
    // 4. Local LAN endpoint
    addClean(this.lanUrl);

    if (list.length === 0) {
      list.push(DEFAULT_LAN_GATEWAY_URL);
    }
    this.candidateUrls = list;
  }

  public async loadCredentials(): Promise<{
    url: string;
    tailscaleUrl: string;
    lanUrl: string;
    token: string;
  }> {
    try {
      const storedUrl = await AsyncStorage.getItem(STORAGE_KEY_GATEWAY_URL);
      if (storedUrl) this.gatewayUrl = storedUrl;

      const storedTailscale = await AsyncStorage.getItem(STORAGE_KEY_TAILSCALE_URL);
      if (storedTailscale) this.tailscaleUrl = storedTailscale;

      const storedLan = await AsyncStorage.getItem(STORAGE_KEY_LAN_URL);
      if (storedLan) this.lanUrl = storedLan;

      // 1. Attempt retrieval from OS Secure Store (Keychain on iOS, Keystore on Android)
      let secureToken: string | null = null;
      try {
        secureToken = await SecureStore.getItemAsync(SECURE_KEY_AUTH_TOKEN);
      } catch (e) {
        // Fallback for non-native environments (e.g. unit tests)
      }

      // 2. Migration: check if legacy AsyncStorage holds token, migrate and erase from AsyncStorage
      if (!secureToken) {
        const legacyToken = await AsyncStorage.getItem(LEGACY_STORAGE_KEY_AUTH_TOKEN);
        if (legacyToken) {
          secureToken = legacyToken;
          try {
            await SecureStore.setItemAsync(SECURE_KEY_AUTH_TOKEN, legacyToken);
            await AsyncStorage.removeItem(LEGACY_STORAGE_KEY_AUTH_TOKEN);
          } catch (e) {
            // Ignored
          }
        }
      }

      if (secureToken) {
        this.authToken = secureToken;
      }
    } catch (e) {
      console.warn('Failed to load credentials from secure storage', e);
    }
    this.updateCandidateList();
    return {
      url: this.gatewayUrl,
      tailscaleUrl: this.tailscaleUrl,
      lanUrl: this.lanUrl,
      token: this.authToken,
    };
  }

  public async setCredentials(
    url: string,
    token: string,
    options?: { tailscaleUrl?: string; lanUrl?: string }
  ): Promise<void> {
    this.gatewayUrl = url.trim().replace(/\/+$/, '');
    this.authToken = token.trim();

    if (options?.tailscaleUrl !== undefined) {
      this.tailscaleUrl = options.tailscaleUrl.trim().replace(/\/+$/, '');
      await AsyncStorage.setItem(STORAGE_KEY_TAILSCALE_URL, this.tailscaleUrl);
    }
    if (options?.lanUrl !== undefined) {
      this.lanUrl = options.lanUrl.trim().replace(/\/+$/, '');
      await AsyncStorage.setItem(STORAGE_KEY_LAN_URL, this.lanUrl);
    }

    // Gateway URL is non-sensitive configuration
    await AsyncStorage.setItem(STORAGE_KEY_GATEWAY_URL, this.gatewayUrl);

    // Auth Token saved strictly to hardware-backed OS secure storage
    try {
      await SecureStore.setItemAsync(SECURE_KEY_AUTH_TOKEN, this.authToken);
      await AsyncStorage.removeItem(LEGACY_STORAGE_KEY_AUTH_TOKEN);
    } catch (e) {
      // Fallback for non-native environments
      await AsyncStorage.setItem(LEGACY_STORAGE_KEY_AUTH_TOKEN, this.authToken);
    }

    this.candidateIndex = 0;
    this.updateCandidateList();
    this.reconnect();
  }

  public getGatewayUrl(): string {
    return this.gatewayUrl;
  }

  public getTailscaleUrl(): string {
    return this.tailscaleUrl;
  }

  public getLanUrl(): string {
    return this.lanUrl;
  }

  public getActiveConnectedUrl(): string {
    return this.activeConnectedUrl || this.gatewayUrl;
  }

  public getAuthToken(): string {
    return this.authToken;
  }

  public getStatus(): ConnectionStatus {
    return this.status;
  }

  public onMessage(listener: MessageHandler): () => void {
    this.messageListeners.add(listener);
    return () => this.messageListeners.delete(listener);
  }

  public onDelta(listener: DeltaHandler): () => void {
    this.deltaListeners.add(listener);
    return () => this.deltaListeners.delete(listener);
  }

  public onToolProgress(listener: ToolProgressHandler): () => void {
    this.toolProgressListeners.add(listener);
    return () => this.toolProgressListeners.delete(listener);
  }

  public onApproval(listener: ApprovalHandler): () => void {
    this.approvalListeners.add(listener);
    return () => this.approvalListeners.delete(listener);
  }

  public onStatus(listener: StatusHandler): () => void {
    this.statusListeners.add(listener);
    listener(this.status);
    return () => this.statusListeners.delete(listener);
  }

  public onSessionUpdated(listener: SessionUpdateHandler): () => void {
    this.sessionUpdateListeners.add(listener);
    return () => this.sessionUpdateListeners.delete(listener);
  }

  private setStatus(status: ConnectionStatus) {
    this.status = status;
    this.statusListeners.forEach((l) => l(status));
  }

  private clearConnectionTimeout(): void {
    if (this.connectionTimeoutTimer) {
      clearTimeout(this.connectionTimeoutTimer);
      this.connectionTimeoutTimer = null;
    }
  }

  private rotateCandidateAndRetry(): void {
    this.clearConnectionTimeout();
    this.stopHeartbeat();
    if (this.ws) {
      try {
        this.ws.close();
      } catch (e) {}
      this.ws = null;
    }
    this.candidateIndex++;
    this.setStatus('reconnecting');
    this.scheduleReconnect();
  }

  public connect(): void {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.updateCandidateList();
    const candidate = this.candidateUrls[this.candidateIndex % this.candidateUrls.length];
    this.setStatus(this.reconnectAttempts > 0 ? 'reconnecting' : 'connecting');

    const wsProto = candidate.startsWith('https') ? 'wss' : 'ws';
    const host = candidate.replace(/^https?:\/\//, '');
    // Safe authenticated handshake: token is NEVER leaked in the URL query string!
    const wsUrl = `${wsProto}://${host}/ws`;

    // Fast connection timeout per candidate (4000ms) to allow quick fallback between LAN and Tailscale
    this.clearConnectionTimeout();
    this.connectionTimeoutTimer = setTimeout(() => {
      if (this.status !== 'connected') {
        console.warn(`[NeoClient] Connection attempt to ${candidate} timed out; trying next candidate.`);
        this.rotateCandidateAndRetry();
      }
    }, 4000);

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        // Immediately perform authenticated handshake over established socket
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ type: 'auth', token: this.authToken }));
        }
      };

      this.ws.onmessage = (event) => {
        this.handleInboundMessage(event.data, candidate);
      };

      this.ws.onclose = () => {
        this.clearConnectionTimeout();
        this.stopHeartbeat();
        this.setStatus('disconnected');
        this.rotateCandidateAndRetry();
      };

      this.ws.onerror = (err) => {
        console.warn(`[NeoClient] WebSocket error on ${candidate}:`, err);
      };
    } catch (e) {
      console.error('Failed to create WebSocket:', e);
      this.rotateCandidateAndRetry();
    }
  }

  public disconnect(): void {
    this.clearConnectionTimeout();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.stopHeartbeat();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.setStatus('disconnected');
  }

  public reconnect(): void {
    this.disconnect();
    this.reconnectAttempts = 0;
    this.candidateIndex = 0;
    this.connect();
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    // Exponential backoff with jitter
    const delay = Math.min(1000 * Math.pow(1.5, Math.min(this.reconnectAttempts, 6)), 10000);
    this.reconnectAttempts++;
    this.setStatus('reconnecting');
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.pingInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 20000);
  }

  private stopHeartbeat(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  private handleInboundMessage(dataStr: string, sourceCandidate?: string): void {
    try {
      const msg = JSON.parse(dataStr);

      switch (msg.type) {
        case 'auth_ok': {
          // Handshake verified by server!
          this.clearConnectionTimeout();
          if (sourceCandidate) {
            this.activeConnectedUrl = sourceCandidate;
            this.gatewayUrl = sourceCandidate;
            AsyncStorage.setItem(STORAGE_KEY_GATEWAY_URL, this.gatewayUrl).catch(() => {});
          }
          this.setStatus('connected');
          this.reconnectAttempts = 0;
          this.startHeartbeat();
          break;
        }

        case 'auth_error': {
          console.warn('Neo WebSocket auth rejected:', msg.error);
          this.setStatus('disconnected');
          this.stopHeartbeat();
          break;
        }

        case 'pong':
          // Keepalive response
          break;

        case 'event': {
          const event: NeoEvent = msg.event;
          notificationService.handleEvent(event);
          if (event.type === 'approval_required' && event.payload) {
            this.approvalListeners.forEach((l) => l(event.payload));
          }
          break;
        }

        case 'message': {
          const chatMsg: ChatMessage = msg.message;
          this.messageListeners.forEach((l) => l(chatMsg));
          break;
        }

        case 'message.delta': {
          this.deltaListeners.forEach((l) =>
            l({
              session_id: msg.session_id,
              role: 'assistant',
              delta: msg.delta,
              type: 'content',
            })
          );
          break;
        }

        case 'thinking.delta': {
          this.deltaListeners.forEach((l) =>
            l({
              session_id: msg.session_id,
              role: 'assistant',
              delta: msg.delta,
              type: 'reasoning',
            })
          );
          break;
        }

        case 'tool_progress': {
          this.toolProgressListeners.forEach((l) =>
            l({
              session_id: msg.session_id,
              tool_action: msg.tool_action,
            })
          );
          break;
        }

        case 'approval_required': {
          const approval: ApprovalRequest = msg.approval;
          this.approvalListeners.forEach((l) => l(approval));
          break;
        }

        case 'session_updated': {
          const sess: Session = msg.session;
          if (sess) {
            this.sessionUpdateListeners.forEach((l) => l(sess));
          }
          break;
        }

        default:
          break;
      }
    } catch (e) {
      console.warn('Failed to parse inbound WS frame:', e);
    }
  }

  private getAuthHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    }
    return headers;
  }

  public async fetchSessions(): Promise<Session[]> {
    const res = await fetch(`${this.gatewayUrl}/api/sessions`, {
      headers: this.getAuthHeaders(),
    });
    if (!res.ok) {
      throw new Error(`Failed to fetch sessions: HTTP ${res.status}`);
    }
    const data = await res.json();
    return data.sessions || [];
  }

  public async createSession(title?: string): Promise<Session> {
    const res = await fetch(`${this.gatewayUrl}/api/sessions`, {
      method: 'POST',
      headers: this.getAuthHeaders(),
      body: JSON.stringify({ title }),
    });
    if (!res.ok) {
      throw new Error(`Failed to create session: HTTP ${res.status}`);
    }
    const data = await res.json();
    return data.session;
  }

  public async deleteSession(sessionId: string): Promise<void> {
    const res = await fetch(`${this.gatewayUrl}/api/sessions/${sessionId}`, {
      method: 'DELETE',
      headers: this.getAuthHeaders(),
    });
    if (!res.ok) {
      throw new Error(`Failed to delete session: HTTP ${res.status}`);
    }
  }

  public async fetchMessages(sessionId: string): Promise<ChatMessage[]> {
    const res = await fetch(`${this.gatewayUrl}/api/sessions/${sessionId}/messages`, {
      headers: this.getAuthHeaders(),
    });
    if (!res.ok) {
      throw new Error(`Failed to fetch messages: HTTP ${res.status}`);
    }
    const data = await res.json();
    return data.messages || [];
  }

  public async sendMessage(sessionId: string, text: string, clientMsgId: string): Promise<void> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(
        JSON.stringify({
          type: 'send_message',
          session_id: sessionId,
          text,
          client_msg_id: clientMsgId,
        })
      );
      return;
    }

    // Fallback to HTTP POST if WebSocket is temporarily down
    const res = await fetch(`${this.gatewayUrl}/api/sessions/${sessionId}/messages`, {
      method: 'POST',
      headers: this.getAuthHeaders(),
      body: JSON.stringify({
        text,
        client_msg_id: clientMsgId,
      }),
    });
    if (!res.ok) {
      throw new Error(`Failed to send message: HTTP ${res.status}`);
    }
  }

  public async decideApproval(
    approvalId: string,
    decision: 'approve' | 'reject',
    actionHash?: string
  ): Promise<{ success: boolean; message?: string }> {
    const payload: Record<string, any> = { decision };
    if (actionHash) {
      payload.action_hash = actionHash;
    }
    const res = await fetch(`${this.gatewayUrl}/api/approvals/${approvalId}/decide`, {
      method: 'POST',
      headers: this.getAuthHeaders(),
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || err.detail || `Approval decision failed: HTTP ${res.status}`);
    }
    return await res.json();
  }

  public async fetchArgusStatus(): Promise<ArgusStatus> {
    const res = await fetch(`${this.gatewayUrl}/api/argus/status`, { headers: this.getAuthHeaders() });
    if (!res.ok) {
      throw new Error(`Failed to fetch ARGUS status: HTTP ${res.status}`);
    }
    return await res.json();
  }

  /**
   * Plain chat without tools. React Native's fetch has no streaming body, so the
   * SSE response is read incrementally through XMLHttpRequest progress events.
   * Returns a function that aborts the request.
   */
  public streamChat(
    messages: PlainChatMessage[],
    onUpdate: (reply: StreamedReply) => void,
    onDone: (error?: Error) => void,
  ): () => void {
    const xhr = new XMLHttpRequest();
    const reply: StreamedReply = { content: '', reasoning: '' };
    let consumed = 0;
    let buffer = '';
    let finished = false;
    const finish = (error?: Error) => {
      if (finished) return;
      finished = true;
      onDone(error);
    };
    const drain = () => {
      buffer += xhr.responseText.slice(consumed);
      consumed = xhr.responseText.length;
      try {
        buffer = consumeSse(buffer, reply);
        onUpdate({ ...reply });
      } catch (error) {
        xhr.abort();
        finish(error as Error);
      }
    };
    xhr.open('POST', `${this.gatewayUrl}/api/chat/completions`);
    Object.entries(this.getAuthHeaders()).forEach(([key, value]) => xhr.setRequestHeader(key, value));
    xhr.onprogress = () => {
      if (xhr.status === 200) drain();
    };
    xhr.onload = () => {
      if (xhr.status !== 200) {
        let message = `HTTP ${xhr.status}`;
        try {
          message = JSON.parse(xhr.responseText).error || message;
        } catch (e) {
          // Non-JSON error body; keep the status text.
        }
        finish(new Error(message));
        return;
      }
      drain();
      finish();
    };
    xhr.onerror = () => finish(new Error('Gateway is unreachable'));
    xhr.onabort = () => finish();
    xhr.send(JSON.stringify({ messages: messages.map(({ role, content }) => ({ role, content })) }));
    return () => xhr.abort();
  }

  public async triggerTestUrgentAlert(): Promise<any> {
    const res = await fetch(`${this.gatewayUrl}/api/events/test_urgent`, {
      method: 'POST',
      headers: this.getAuthHeaders(),
    });
    return await res.json();
  }
}

export const neoClient = new NeoClient();
