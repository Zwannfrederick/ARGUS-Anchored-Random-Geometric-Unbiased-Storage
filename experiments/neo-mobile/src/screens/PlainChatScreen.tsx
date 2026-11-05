import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  FlatList,
  KeyboardAvoidingView,
  Platform,
  SafeAreaView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Markdown from 'react-native-markdown-display';
import { ArrowLeft, Square, Trash2 } from 'lucide-react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { neoClient } from '../api/neoClient';
import { ArgusKvCard } from '../components/ArgusKvCard';
import { ChatInput } from '../components/ChatInput';
import { ThinkingBlock } from '../components/ThinkingBlock';
import { RootStackParamList } from '../navigation/AppNavigator';
import { ConnectionStatus, PlainChatMessage } from '../types';

type Props = NativeStackScreenProps<RootStackParamList, 'PlainChat'>;

const STORAGE_KEY = '@neo_plain_chat';

/** Tool-free chat with the local model; history stays on the phone. */
export const PlainChatScreen: React.FC<Props> = ({ navigation }) => {
  const [messages, setMessages] = useState<PlainChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [status, setStatus] = useState<ConnectionStatus>(neoClient.getStatus());
  const abortRef = useRef<(() => void) | null>(null);
  const listRef = useRef<FlatList<PlainChatMessage>>(null);

  useEffect(() => {
    AsyncStorage.getItem(STORAGE_KEY)
      .then((raw) => raw && setMessages(JSON.parse(raw)))
      .catch((e) => console.warn('Failed to load plain chat history', e));
    const unsubscribe = neoClient.onStatus(setStatus);
    return () => {
      unsubscribe();
      abortRef.current?.();
    };
  }, []);

  const persist = useCallback((next: PlainChatMessage[]) => {
    AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(next)).catch((e) =>
      console.warn('Failed to save plain chat history', e),
    );
  }, []);

  const handleSend = (text: string) => {
    const history: PlainChatMessage[] = [...messages, { role: 'user', content: text }];
    setMessages([...history, { role: 'assistant', content: '' }]);
    setStreaming(true);
    const replace = (reply: PlainChatMessage) =>
      setMessages((current) => [...current.slice(0, -1), reply]);
    abortRef.current = neoClient.streamChat(
      history,
      (reply) => replace({ role: 'assistant', content: reply.content, reasoning: reply.reasoning }),
      (error) => {
        abortRef.current = null;
        setStreaming(false);
        setMessages((current) => {
          const last = current[current.length - 1];
          const next = error
            ? [...current.slice(0, -1), { ...last, error: true, content: last.content || `Hata: ${error.message}` }]
            : current;
          persist(next);
          return next;
        });
      },
    );
  };

  const clearHistory = () => {
    abortRef.current?.();
    setMessages([]);
    persist([]);
  };

  const renderItem = ({ item }: { item: PlainChatMessage }) =>
    item.role === 'user' ? (
      <View style={styles.userBubble}>
        <Text style={styles.userText}>{item.content}</Text>
      </View>
    ) : (
      <View style={styles.assistant}>
        <ThinkingBlock reasoning={item.reasoning} thinkingEnabled />
        {item.error ? (
          <Text style={styles.errorText}>{item.content}</Text>
        ) : (
          <Markdown style={markdownStyles}>{item.content || '…'}</Markdown>
        )}
      </View>
    );

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.iconButton} accessibilityLabel="Geri">
          <ArrowLeft size={20} color="#f4f4f5" />
        </TouchableOpacity>
        <View style={styles.headerTitle}>
          <Text style={styles.title}>Sohbet</Text>
          <Text style={styles.subtitle}>Araçsız · yerel model</Text>
        </View>
        {streaming ? (
          <TouchableOpacity onPress={() => abortRef.current?.()} style={styles.iconButton} accessibilityLabel="Yanıtı durdur">
            <Square size={18} color="#f97316" />
          </TouchableOpacity>
        ) : (
          <TouchableOpacity onPress={clearHistory} style={styles.iconButton} accessibilityLabel="Geçmişi temizle">
            <Trash2 size={18} color="#a1a1aa" />
          </TouchableOpacity>
        )}
      </View>
      <ArgusKvCard />
      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 8 : 0}
      >
        <FlatList
          ref={listRef}
          data={messages}
          keyExtractor={(_, index) => String(index)}
          renderItem={renderItem}
          contentContainerStyle={styles.list}
          onContentSizeChange={() => listRef.current?.scrollToEnd({ animated: true })}
          ListEmptyComponent={
            <View style={styles.empty}>
              <Text style={styles.emptyTitle}>Yerel modelinle konuş</Text>
              <Text style={styles.emptyText}>Masaüstü kontrolü yok; yalnız sohbet. KV belleğini ARGUS yönetir.</Text>
            </View>
          }
        />
        <ChatInput onSend={handleSend} status={status} disabled={streaming} />
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#09090b' },
  flex: { flex: 1 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#27272a',
    marginBottom: 8,
  },
  headerTitle: { flex: 1, marginLeft: 6 },
  title: { color: '#f4f4f5', fontSize: 17, fontWeight: '700' },
  subtitle: { color: '#71717a', fontSize: 12 },
  iconButton: { padding: 8, borderRadius: 10 },
  list: { paddingHorizontal: 16, paddingBottom: 12, flexGrow: 1 },
  userBubble: {
    alignSelf: 'flex-end',
    backgroundColor: '#10b981',
    borderRadius: 18,
    borderBottomRightRadius: 4,
    paddingHorizontal: 14,
    paddingVertical: 9,
    marginVertical: 6,
    maxWidth: '85%',
  },
  userText: { color: '#ffffff', fontSize: 15, lineHeight: 21 },
  assistant: { marginVertical: 6 },
  errorText: { color: '#f97316', fontSize: 14 },
  empty: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 32, paddingTop: 80 },
  emptyTitle: { color: '#f4f4f5', fontSize: 18, fontWeight: '700', marginBottom: 6 },
  emptyText: { color: '#71717a', fontSize: 14, textAlign: 'center', lineHeight: 20 },
});

const markdownStyles = {
  body: { color: '#e4e4e7', fontSize: 15, lineHeight: 22 },
  code_inline: { backgroundColor: '#1c1c1f', color: '#f4f4f5', borderRadius: 4 },
  fence: { backgroundColor: '#121214', borderColor: '#27272a', color: '#e4e4e7', borderRadius: 8 },
  link: { color: '#34d399' },
};
