import React, { useEffect, useRef, useState } from 'react';
import {
  FlatList,
  Image,
  KeyboardAvoidingView,
  Platform,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import Markdown from 'react-native-markdown-display';
import { ArrowLeft, Bell, BellRing, ShieldAlert, Sparkles, User } from 'lucide-react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../navigation/AppNavigator';
import { neoClient } from '../api/neoClient';
import { notificationService } from '../notifications/notificationService';

const neoLogo = require('../../assets/logo.png');
import {
  ApprovalRequest,
  ChatMessage,
  ConnectionStatus,
  NeoEvent,
  Session,
  ToolAction,
} from '../types';
import { ThinkingBlock } from '../components/ThinkingBlock';
import { ToolActionBadge } from '../components/ToolActionBadge';
import { ScreenshotModal } from '../components/ScreenshotModal';
import { ApprovalCard } from '../components/ApprovalCard';
import { ChatInput } from '../components/ChatInput';

type ChatScreenProps = NativeStackScreenProps<RootStackParamList, 'Chat'>;

export const ChatScreen: React.FC<ChatScreenProps> = ({ route, navigation }) => {
  const { session } = route.params;
  const insets = useSafeAreaInsets();
  const [sessionTitle, setSessionTitle] = useState<string>(session.title || 'Conversation');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pendingApprovals, setPendingApprovals] = useState<ApprovalRequest[]>([]);
  const [activeUrgentBanner, setActiveUrgentBanner] = useState<NeoEvent | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>(neoClient.getStatus());
  const [isSending, setIsSending] = useState<boolean>(false);

  const flatListRef = useRef<FlatList>(null);

  useEffect(() => {
    // 1. Subscribe to status
    const unsubStatus = neoClient.onStatus(setStatus);

    // 2. Load message history
    loadMessages();

    // 3. Subscribe to incoming messages
    const unsubMessage = neoClient.onMessage((newMsg) => {
      if (newMsg.session_id === session.session_id) {
        setMessages((prev) => {
          // 1. Direct match by id
          const idIdx = prev.findIndex((m) => m.id === newMsg.id);
          if (idIdx !== -1) {
            const next = [...prev];
            next[idIdx] = newMsg;
            return next;
          }

          // 2. Correlation by client_msg_id or turn_id
          if (newMsg.client_msg_id) {
            // The user message carries the SAME client_msg_id as its pending assistant
            // bubble, and it is added first — so matching on client_msg_id alone made an
            // incoming assistant reply overwrite the user's own bubble, leaving the
            // streamed pending bubble behind as a duplicate. Match the role too.
            const clientMatchIdx = prev.findIndex(
              (m) =>
                m.role === newMsg.role &&
                (m.client_msg_id === newMsg.client_msg_id ||
                  m.id === `pending_${newMsg.client_msg_id}` ||
                  m.turn_id === newMsg.client_msg_id)
            );
            if (clientMatchIdx !== -1) {
              const next = [...prev];
              next[clientMatchIdx] = newMsg;
              return next;
            }
          }

          // 3. Fallback for assistant message: replace pending assistant message
          if (newMsg.role === 'assistant') {
            const pendingIdx = prev.findIndex((m) => m.role === 'assistant' && m.isPending);
            if (pendingIdx !== -1) {
              const next = [...prev];
              next[pendingIdx] = newMsg;
              return next;
            }
          }

          // 4. Otherwise append
          return [...prev, newMsg];
        });
        setIsSending(false);
        scrollToBottom();
      }
    });

    // 4. Subscribe to delta streams
    const unsubDelta = neoClient.onDelta(({ session_id, delta, type }) => {
      if (session_id === session.session_id) {
        setMessages((prev) => {
          const pendingIdx = prev.findIndex((m) => m.role === 'assistant' && m.isPending);
          const targetIdx = pendingIdx !== -1 ? pendingIdx : prev.length - 1;
          const target = prev[targetIdx];
          if (target && target.role === 'assistant' && target.isPending) {
            const updated = { ...target };
            if (type === 'content') {
              updated.content = (updated.content || '') + delta;
            } else if (type === 'reasoning') {
              updated.reasoning_content = (updated.reasoning_content || '') + delta;
            }
            const next = [...prev];
            next[targetIdx] = updated;
            return next;
          }
          return prev;
        });
      }
    });

    // 5. Subscribe to tool progress updates
    const unsubToolProgress = neoClient.onToolProgress(({ session_id, tool_action }) => {
      if (session_id === session.session_id) {
        setMessages((prev) => {
          const pendingIdx = prev.findIndex((m) => m.role === 'assistant' && m.isPending);
          const targetIdx = pendingIdx !== -1 ? pendingIdx : prev.length - 1;
          const target = prev[targetIdx];
          if (target && target.role === 'assistant') {
            const currentTools = target.tool_calls ? [...target.tool_calls] : [];
            const idx = currentTools.findIndex(
              (t) => (t.id && tool_action.id && t.id === tool_action.id) || t.name === tool_action.name
            );
            if (idx >= 0) {
              currentTools[idx] = tool_action;
            } else {
              currentTools.push(tool_action);
            }
            const updated = { ...target, tool_calls: currentTools };
            const next = [...prev];
            next[targetIdx] = updated;
            return next;
          }
          return prev;
        });
        scrollToBottom();
      }
    });

    // 6. Subscribe to approval events
    const unsubApproval = neoClient.onApproval((approval) => {
      if (approval.session_id === session.session_id) {
        setPendingApprovals((prev) => {
          if (prev.some((a) => a.approval_id === approval.approval_id)) return prev;
          return [...prev, approval];
        });
      }
    });

    // 7. Subscribe to notification alerts
    const unsubNotifications = notificationService.subscribe((event) => {
      if (event.priority === 'urgent') {
        setActiveUrgentBanner(event);
      }
    });

    // 8. Subscribe to dynamic session title updates
    const unsubSessionUpdated = neoClient.onSessionUpdated((updatedSession) => {
      if (updatedSession.session_id === session.session_id && updatedSession.title) {
        setSessionTitle(updatedSession.title);
      }
    });

    return () => {
      unsubStatus();
      unsubMessage();
      unsubDelta();
      unsubToolProgress();
      unsubApproval();
      unsubNotifications();
      unsubSessionUpdated();
    };
  }, [session.session_id]);

  const loadMessages = async () => {
    try {
      const history = await neoClient.fetchMessages(session.session_id);
      setMessages(history);
      scrollToBottom();
    } catch (e) {
      console.warn('Failed to fetch message history', e);
    }
  };

  const scrollToBottom = () => {
    setTimeout(() => {
      flatListRef.current?.scrollToEnd({ animated: true });
    }, 150);
  };

  const handleSendMessage = async (text: string) => {
    const clientMsgId = `client-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`;
    const userMessage: ChatMessage = {
      id: clientMsgId,
      session_id: session.session_id,
      role: 'user',
      content: text,
      timestamp: Date.now() / 1000,
      client_msg_id: clientMsgId,
    };

    const pendingAssistantMessage: ChatMessage = {
      id: `pending_${clientMsgId}`,
      session_id: session.session_id,
      role: 'assistant',
      content: '',
      timestamp: Date.now() / 1000,
      isPending: true,
      client_msg_id: clientMsgId,
      turn_id: clientMsgId,
      tool_calls: [],
    };

    setMessages((prev) => [...prev, userMessage, pendingAssistantMessage]);
    setIsSending(true);
    scrollToBottom();

    try {
      await neoClient.sendMessage(session.session_id, text, clientMsgId);
    } catch (e: any) {
      alert(`Message failed to deliver: ${e.message}`);
      setIsSending(false);
    }
  };

  const handleDecideApproval = async (
    approvalId: string,
    decision: 'approve' | 'reject',
    actionHash?: string
  ) => {
    await neoClient.decideApproval(approvalId, decision, actionHash);
    setPendingApprovals((prev) =>
      prev.map((a) =>
        a.approval_id === approvalId
          ? { ...a, status: decision === 'approve' ? 'approved' : 'rejected' }
          : a
      )
    );
  };

  const handleDismissBanner = () => {
    notificationService.cancelVibration();
    setActiveUrgentBanner(null);
  };

  const renderMessageItem = ({ item }: { item: ChatMessage }) => {
    const isUser = item.role === 'user';

    return (
      <View
        style={[
          styles.messageRow,
          isUser ? styles.userRow : styles.assistantRow,
        ]}
      >
        {!isUser && (
          <View style={styles.assistantAvatar}>
            <Image source={neoLogo} style={styles.assistantAvatarImage} resizeMode="contain" />
          </View>
        )}

        <View
          style={[
            styles.messageBubble,
            isUser ? styles.userBubble : styles.assistantBubble,
          ]}
        >
          {/* Reasoning / Thinking Drawer (Collapsible) */}
          {!isUser && (
            <ThinkingBlock
              reasoning={item.reasoning_content}
              thinkingEnabled={item.thinking_enabled}
            />
          )}

          {/* Tool Action Progress Badges */}
          {!isUser && item.tool_calls && item.tool_calls.length > 0 && (
            <View style={styles.toolCallsContainer}>
              {item.tool_calls.map((tool, index) => (
                <ToolActionBadge key={tool.id || `${tool.name}-${index}`} action={tool} />
              ))}
            </View>
          )}

          {/* In-Chat Screenshot Preview */}
          {item.screenshot_url ? (
            <ScreenshotModal
              imageUrl={
                item.screenshot_url.startsWith('http')
                  ? item.screenshot_url
                  : `${neoClient.getGatewayUrl()}${item.screenshot_url}`
              }
              caption="Desktop Capture"
            />
          ) : null}

          {/* Markdown Content */}
          {item.content ? (
            <Markdown style={markdownStyles}>{item.content}</Markdown>
          ) : item.isPending && !item.reasoning_content && (!item.tool_calls || item.tool_calls.length === 0) ? (
            <Text style={styles.thinkingPlaceholder}>Neo is processing...</Text>
          ) : null}
        </View>

        {isUser && (
          <View style={styles.userAvatar}>
            <User size={16} color="#ffffff" />
          </View>
        )}
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <TouchableOpacity
            style={styles.backButton}
            onPress={() => navigation.goBack()}
            activeOpacity={0.7}
          >
            <ArrowLeft size={20} color="#f4f4f5" />
          </TouchableOpacity>

          <View style={styles.headerAvatarBadge}>
            <Image source={neoLogo} style={styles.headerLogoImage} resizeMode="contain" />
          </View>
        </View>

        <View style={styles.headerCenter}>
          <Text style={styles.sessionTitle} numberOfLines={1}>
            {sessionTitle}
          </Text>
          <Text style={styles.sessionStatus}>
            {status === 'connected' ? 'Gemma-4 E4B • 128K' : status}
          </Text>
        </View>

        <TouchableOpacity
          style={styles.testAlertBtn}
          onPress={() => neoClient.triggerTestUrgentAlert()}
          activeOpacity={0.7}
        >
          <BellRing size={18} color="#f59e0b" />
        </TouchableOpacity>
      </View>

      {/* Urgent Notification Banner */}
      {activeUrgentBanner && (
        <View style={styles.urgentBanner}>
          <View style={styles.bannerContent}>
            <ShieldAlert size={18} color="#f87171" />
            <View style={styles.bannerTexts}>
              <Text style={styles.bannerTitle}>{activeUrgentBanner.title}</Text>
              <Text style={styles.bannerBody} numberOfLines={2}>
                {activeUrgentBanner.body}
              </Text>
            </View>
          </View>
          <TouchableOpacity
            style={styles.bannerDismiss}
            onPress={handleDismissBanner}
          >
            <Text style={styles.dismissText}>Dismiss</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Message List */}
      <KeyboardAvoidingView
        style={[
          styles.chatArea,
          { paddingBottom: Platform.OS === 'ios' ? 0 : Math.max(insets.bottom, 4) },
        ]}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
      >
        <FlatList
          ref={flatListRef}
          data={messages}
          keyExtractor={(item) => item.id}
          renderItem={renderMessageItem}
          contentContainerStyle={styles.listContent}
          keyboardShouldPersistTaps="handled"
          keyboardDismissMode="interactive"
          ListFooterComponent={
            pendingApprovals.length > 0 ? (
              <View style={styles.approvalsSection}>
                {pendingApprovals.map((approval) => (
                  <ApprovalCard
                    key={approval.approval_id}
                    approval={approval}
                    onDecide={handleDecideApproval}
                  />
                ))}
              </View>
            ) : null
          }
        />

        {/* Input Bar */}
        <ChatInput
          onSend={handleSendMessage}
          status={status}
          disabled={status !== 'connected' || isSending}
        />
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#09090b',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#18181b',
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  headerAvatarBadge: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#18181b',
    borderWidth: 1,
    borderColor: '#27272a',
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerLogoImage: {
    width: 24,
    height: 24,
    borderRadius: 12,
  },
  backButton: {
    padding: 6,
    borderRadius: 8,
  },
  headerCenter: {
    flex: 1,
    marginHorizontal: 10,
  },
  sessionTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: '#f4f4f5',
  },
  sessionStatus: {
    fontSize: 11,
    color: '#10b981',
    marginTop: 2,
  },
  testAlertBtn: {
    padding: 6,
    borderRadius: 8,
    backgroundColor: '#18181b',
  },
  urgentBanner: {
    backgroundColor: '#450a0a',
    borderBottomWidth: 1,
    borderBottomColor: '#7f1d1d',
    paddingHorizontal: 16,
    paddingVertical: 10,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  bannerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
    gap: 10,
  },
  bannerTexts: {
    flex: 1,
  },
  bannerTitle: {
    color: '#fca5a5',
    fontWeight: '700',
    fontSize: 13,
  },
  bannerBody: {
    color: '#e4e4e7',
    fontSize: 12,
    marginTop: 2,
  },
  bannerDismiss: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    backgroundColor: '#7f1d1d',
    borderRadius: 6,
    marginLeft: 8,
  },
  dismissText: {
    color: '#ffffff',
    fontSize: 11,
    fontWeight: '600',
  },
  chatArea: {
    flex: 1,
  },
  listContent: {
    paddingHorizontal: 12,
    paddingVertical: 16,
  },
  messageRow: {
    flexDirection: 'row',
    marginVertical: 6,
    alignItems: 'flex-start',
  },
  userRow: {
    justifyContent: 'flex-end',
  },
  assistantRow: {
    justifyContent: 'flex-start',
  },
  assistantAvatar: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: '#18181b',
    borderWidth: 1,
    borderColor: '#27272a',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 8,
    marginTop: 2,
  },
  assistantAvatarImage: {
    width: 24,
    height: 24,
    borderRadius: 12,
  },
  userAvatar: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: '#27272a',
    borderWidth: 1,
    borderColor: '#3f3f46',
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: 8,
    marginTop: 2,
  },
  messageBubble: {
    maxWidth: '82%',
    borderRadius: 14,
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  userBubble: {
    backgroundColor: '#27272a',
    borderTopRightRadius: 4,
    borderWidth: 1,
    borderColor: '#3f3f46',
  },
  assistantBubble: {
    backgroundColor: '#121215',
    borderTopLeftRadius: 4,
    borderWidth: 1,
    borderColor: '#27272a',
  },
  toolCallsContainer: {
    marginVertical: 6,
  },
  thinkingPlaceholder: {
    color: '#71717a',
    fontSize: 13,
    fontStyle: 'italic',
  },
  approvalsSection: {
    marginTop: 10,
  },
});

const markdownStyles = {
  body: {
    color: '#e4e4e7',
    fontSize: 14,
    lineHeight: 22,
  },
  heading1: {
    color: '#f4f4f5',
    fontSize: 18,
    fontWeight: '700',
    marginTop: 10,
    marginBottom: 6,
  },
  heading2: {
    color: '#f4f4f5',
    fontSize: 16,
    fontWeight: '600',
    marginTop: 8,
    marginBottom: 4,
  },
  heading3: {
    color: '#f4f4f5',
    fontSize: 15,
    fontWeight: '600',
    marginTop: 6,
    marginBottom: 4,
  },
  bullet_list: {
    marginVertical: 4,
  },
  ordered_list: {
    marginVertical: 4,
  },
  code_inline: {
    backgroundColor: '#18181b',
    color: '#38bdf8',
    fontFamily: 'monospace',
    paddingHorizontal: 5,
    paddingVertical: 2,
    borderRadius: 4,
    fontSize: 13,
  },
  code_block: {
    backgroundColor: '#000000',
    borderColor: '#27272a',
    borderWidth: 1,
    borderRadius: 8,
    padding: 10,
    marginVertical: 8,
    fontFamily: 'monospace',
    fontSize: 12,
    color: '#a1a1aa',
  },
  fence: {
    backgroundColor: '#000000',
    borderColor: '#27272a',
    borderWidth: 1,
    borderRadius: 8,
    padding: 10,
    marginVertical: 8,
    fontFamily: 'monospace',
    fontSize: 12,
    color: '#a1a1aa',
  },
  link: {
    color: '#38bdf8',
    textDecorationLine: 'underline' as const,
  },
  table: {
    borderWidth: 1,
    borderColor: '#27272a',
    borderRadius: 6,
    marginVertical: 8,
  },
  tr: {
    borderBottomWidth: 1,
    borderColor: '#27272a',
    flexDirection: 'row' as const,
  },
  th: {
    padding: 6,
    backgroundColor: '#18181b',
    color: '#f4f4f5',
    fontWeight: '700',
  },
  td: {
    padding: 6,
    color: '#d4d4d8',
  },
};
