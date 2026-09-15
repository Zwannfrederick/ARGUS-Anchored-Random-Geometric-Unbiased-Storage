import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Image,
  KeyboardAvoidingView,
  Modal,
  Platform,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import {
  ChevronRight,
  Clock,
  Trash2,
  MessageSquarePlus,
  Radio,
  MessageCircle,
  Settings,
  X,
} from 'lucide-react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../navigation/AppNavigator';
import { neoClient } from '../api/neoClient';
import { ConnectionStatus, Session } from '../types';

const neoLogo = require('../../assets/logo.png');

type ConversationListScreenProps = NativeStackScreenProps<
  RootStackParamList,
  'ConversationList'
>;

export const ConversationListScreen: React.FC<ConversationListScreenProps> = ({
  navigation,
}) => {
  const [sessions, setSessions] = useState<Session[]>([]);

  const handleDeleteSession = (session: Session) => {
    Alert.alert(
      'Sohbeti sil',
      `"${session.title || 'Untitled Conversation'}" ve tüm mesajları kalıcı olarak silinecek.`,
      [
        { text: 'Vazgeç', style: 'cancel' },
        {
          text: 'Sil',
          style: 'destructive',
          onPress: async () => {
            const previous = sessions;
            setSessions((prev) => prev.filter((s) => s.session_id !== session.session_id));
            try {
              await neoClient.deleteSession(session.session_id);
            } catch (e: any) {
              setSessions(previous);
              Alert.alert('Silinemedi', e?.message || 'Bilinmeyen hata');
            }
          },
        },
      ]
    );
  };
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [status, setStatus] = useState<ConnectionStatus>(neoClient.getStatus());

  // Settings modal state
  const [settingsOpen, setSettingsOpen] = useState<boolean>(false);
  const [gatewayUrl, setGatewayUrl] = useState<string>(neoClient.getGatewayUrl());
  const [tailscaleUrl, setTailscaleUrl] = useState<string>(neoClient.getTailscaleUrl());
  const [lanUrl, setLanUrl] = useState<string>(neoClient.getLanUrl());
  const [authToken, setAuthToken] = useState<string>(neoClient.getAuthToken());

  useEffect(() => {
    const unsubStatus = neoClient.onStatus((newStatus) => {
      setStatus(newStatus);
    });

    const unsubSessionUpdated = neoClient.onSessionUpdated((updatedSession) => {
      setSessions((prev) =>
        prev.map((s) =>
          s.session_id === updatedSession.session_id
            ? { ...s, ...updatedSession }
            : s
        )
      );
    });

    loadInitialData();

    return () => {
      unsubStatus();
      unsubSessionUpdated();
    };
  }, []);

  const loadInitialData = async () => {
    setLoading(true);
    try {
      const creds = await neoClient.loadCredentials();
      setGatewayUrl(creds.url);
      setTailscaleUrl(creds.tailscaleUrl);
      setLanUrl(creds.lanUrl);
      setAuthToken(creds.token);
      neoClient.connect();
      await fetchSessionsList();
    } catch (e) {
      console.warn('Failed to load initial sessions', e);
    } finally {
      setLoading(false);
    }
  };

  const fetchSessionsList = async () => {
    try {
      const list = await neoClient.fetchSessions();
      setSessions(list);
    } catch (e) {
      console.warn('Failed to fetch sessions', e);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchSessionsList();
    setRefreshing(false);
  };

  const handleNewSession = async () => {
    try {
      setLoading(true);
      const newSession = await neoClient.createSession('New Session');
      await fetchSessionsList();
      navigation.navigate('Chat', { session: newSession });
    } catch (e: any) {
      alert(`Could not create session: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveSettings = async () => {
    await neoClient.setCredentials(gatewayUrl, authToken, {
      tailscaleUrl,
      lanUrl,
    });
    setSettingsOpen(false);
    await fetchSessionsList();
  };

  const formatTime = (timestamp: number) => {
    if (!timestamp) return '';
    const date = new Date(timestamp * 1000);
    const now = new Date();
    if (date.toDateString() === now.toDateString()) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
  };

  const getStatusColor = () => {
    return status === 'connected'
      ? '#10b981'
      : status === 'reconnecting'
      ? '#f59e0b'
      : '#ef4444';
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <View style={styles.avatar}>
            <Image source={neoLogo} style={styles.logoImage} resizeMode="contain" />
            <View style={[styles.statusIndicatorOrb, { backgroundColor: getStatusColor() }]} />
          </View>
          <View>
            <View style={styles.titleRow}>
              <Text style={styles.appName}>Neo</Text>
              <View style={styles.proBadge}>
                <Text style={styles.proBadgeText}>HERMES</Text>
              </View>
            </View>
            <Text style={styles.statusText}>
              {status === 'connected' ? 'Gemma-4 E4B • 128K' : status}
            </Text>
          </View>
        </View>

        <View style={styles.headerRight}>
          <TouchableOpacity
            style={styles.iconButton}
            onPress={() => navigation.navigate('PlainChat')}
            activeOpacity={0.7}
            accessibilityLabel="Araçsız sohbet"
          >
            <MessageCircle size={20} color="#a1a1aa" />
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.iconButton}
            onPress={() => setSettingsOpen(true)}
            activeOpacity={0.7}
          >
            <Settings size={20} color="#a1a1aa" />
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.newChatButton}
            onPress={handleNewSession}
            activeOpacity={0.7}
          >
            <MessageSquarePlus size={18} color="#ffffff" />
            <Text style={styles.newChatText}>New</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Session List */}
      {loading ? (
        <View style={styles.centerBox}>
          <ActivityIndicator size="large" color="#10b981" />
          <Text style={styles.loadingText}>Loading conversations...</Text>
        </View>
      ) : (
        <FlatList
          data={sessions}
          keyExtractor={(item) => item.session_id}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              tintColor="#10b981"
            />
          }
          ListEmptyComponent={
            <View style={styles.emptyContainer}>
              <View style={styles.emptyLogoWrapper}>
                <Image source={neoLogo} style={styles.emptyLogoImage} resizeMode="contain" />
              </View>
              <Text style={styles.emptyTitle}>Ready to Assist</Text>
              <Text style={styles.emptySubtitle}>
                Connected to Neo on your PC with full Hermes autonomous execution.
              </Text>
              <TouchableOpacity style={styles.emptyButton} onPress={handleNewSession}>
                <Text style={styles.emptyButtonText}>Start Conversation</Text>
              </TouchableOpacity>
            </View>
          }
          renderItem={({ item }) => (
            <TouchableOpacity
              style={styles.sessionItem}
              onPress={() => navigation.navigate('Chat', { session: item })}
              onLongPress={() => handleDeleteSession(item)}
              delayLongPress={400}
              activeOpacity={0.7}
            >
              <View style={styles.sessionContent}>
                <Text style={styles.sessionTitle} numberOfLines={1}>
                  {item.title || 'Untitled Conversation'}
                </Text>
                {item.last_message ? (
                  <Text style={styles.sessionPreview} numberOfLines={1}>
                    {item.last_message}
                  </Text>
                ) : null}
                <View style={styles.sessionMeta}>
                  <Clock size={11} color="#71717a" />
                  <Text style={styles.sessionDate}>{formatTime(item.updated_at)}</Text>
                  {item.hermes_session_id ? (
                    <Text style={styles.hermesBadge}>
                      Hermes: {item.hermes_session_id.slice(0, 6)}
                    </Text>
                  ) : null}
                </View>
              </View>
              <TouchableOpacity
                onPress={() => handleDeleteSession(item)}
                hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
                style={styles.deleteButton}
              >
                <Trash2 size={17} color="#71717a" />
              </TouchableOpacity>
              <ChevronRight size={18} color="#52525b" />
            </TouchableOpacity>
          )}
        />
      )}

      {/* Settings Modal */}
      <Modal
        visible={settingsOpen}
        transparent={true}
        animationType="slide"
        onRequestClose={() => setSettingsOpen(false)}
      >
        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
          style={styles.modalBackdrop}
        >
          <View style={styles.settingsSheet}>
            <ScrollView
              contentContainerStyle={{ paddingBottom: 20 }}
              keyboardShouldPersistTaps="handled"
              showsVerticalScrollIndicator={false}
            >
              <View style={styles.modalHeader}>
                <Text style={styles.modalTitle}>Gateway Connection</Text>
                <TouchableOpacity onPress={() => setSettingsOpen(false)}>
                  <X size={20} color="#a1a1aa" />
                </TouchableOpacity>
              </View>

              <View style={styles.activeUrlBox}>
                <Text style={styles.activeUrlLabel}>Active Connection:</Text>
                <Text style={styles.activeUrlValue}>{neoClient.getActiveConnectedUrl()}</Text>
              </View>

              <Text style={styles.inputLabel}>Primary Gateway URL</Text>
              <TextInput
                style={styles.modalInput}
                value={gatewayUrl}
                onChangeText={setGatewayUrl}
                autoCapitalize="none"
                autoCorrect={false}
                placeholder="http://192.168.8.5:8765"
                placeholderTextColor="#71717a"
              />

              <View style={styles.presetRow}>
                <TouchableOpacity
                  style={styles.presetButton}
                  onPress={() => setGatewayUrl('http://192.168.8.5:8765')}
                >
                  <Text style={styles.presetButtonText}>LAN</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.presetButton}
                  onPress={() => {
                    if (tailscaleUrl) setGatewayUrl(tailscaleUrl);
                    else setGatewayUrl('http://100.98.181.117:8765');
                  }}
                >
                  <Text style={styles.presetButtonText}>Tailscale IP</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.presetButton}
                  onPress={() => setGatewayUrl('http://devshub.tailaa2b98.ts.net:8765')}
                >
                  <Text style={styles.presetButtonText}>MagicDNS</Text>
                </TouchableOpacity>
              </View>

              <Text style={styles.inputLabel}>Tailscale Remote Endpoint (VPN)</Text>
              <TextInput
                style={styles.modalInput}
                value={tailscaleUrl}
                onChangeText={setTailscaleUrl}
                autoCapitalize="none"
                autoCorrect={false}
                placeholder="http://100.98.181.117:8765 or devshub.tailaa2b98.ts.net:8765"
                placeholderTextColor="#71717a"
              />

              <Text style={styles.inputLabel}>Local LAN Endpoint (Home Wi-Fi)</Text>
              <TextInput
                style={styles.modalInput}
                value={lanUrl}
                onChangeText={setLanUrl}
                autoCapitalize="none"
                autoCorrect={false}
                placeholder="http://192.168.8.5:8765"
                placeholderTextColor="#71717a"
              />

              <Text style={styles.inputLabel}>Auth Token (Hardware Keystore)</Text>
              <TextInput
                style={styles.modalInput}
                value={authToken}
                onChangeText={setAuthToken}
                autoCapitalize="none"
                autoCorrect={false}
                placeholder="Paste Bearer Token"
                placeholderTextColor="#71717a"
                secureTextEntry={true}
              />

              <View style={styles.modalActionRow}>
                <TouchableOpacity
                  style={styles.saveSettingsButton}
                  onPress={handleSaveSettings}
                >
                  <Text style={styles.saveSettingsText}>Connect & Save Endpoints</Text>
                </TouchableOpacity>
              </View>
            </ScrollView>
          </View>
        </KeyboardAvoidingView>
      </Modal>
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
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#18181b',
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  avatar: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: '#18181b',
    borderWidth: 1,
    borderColor: '#27272a',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
  },
  logoImage: {
    width: 30,
    height: 30,
    borderRadius: 15,
  },
  statusIndicatorOrb: {
    position: 'absolute',
    bottom: -1,
    right: -1,
    width: 10,
    height: 10,
    borderRadius: 5,
    borderWidth: 2,
    borderColor: '#09090b',
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  appName: {
    fontSize: 16,
    fontWeight: '700',
    color: '#f4f4f5',
  },
  proBadge: {
    backgroundColor: '#18181b',
    borderWidth: 1,
    borderColor: '#27272a',
    paddingHorizontal: 5,
    paddingVertical: 1,
    borderRadius: 4,
  },
  proBadgeText: {
    fontSize: 9,
    fontWeight: '700',
    color: '#10b981',
    letterSpacing: 0.5,
  },
  statusText: {
    fontSize: 11,
    color: '#a1a1aa',
    marginTop: 1,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  iconButton: {
    padding: 8,
    borderRadius: 8,
    backgroundColor: '#18181b',
  },
  newChatButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#10b981',
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: 8,
    gap: 6,
  },
  newChatText: {
    color: '#ffffff',
    fontSize: 13,
    fontWeight: '600',
  },
  centerBox: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    gap: 12,
  },
  loadingText: {
    color: '#a1a1aa',
    fontSize: 14,
  },
  sessionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: '#18181b',
  },
  deleteButton: {
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  sessionContent: {
    flex: 1,
    marginRight: 12,
  },
  sessionTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#f4f4f5',
    marginBottom: 4,
  },
  sessionPreview: {
    fontSize: 13,
    color: '#71717a',
    marginBottom: 6,
  },
  sessionMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  sessionDate: {
    fontSize: 11,
    color: '#71717a',
  },
  hermesBadge: {
    fontSize: 10,
    color: '#10b981',
    backgroundColor: '#064e3b',
    paddingHorizontal: 5,
    paddingVertical: 1,
    borderRadius: 3,
    marginLeft: 6,
  },
  emptyContainer: {
    paddingTop: 80,
    alignItems: 'center',
    paddingHorizontal: 32,
    gap: 10,
  },
  emptyLogoWrapper: {
    width: 68,
    height: 68,
    borderRadius: 34,
    backgroundColor: '#121215',
    borderWidth: 1,
    borderColor: '#27272a',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  emptyLogoImage: {
    width: 44,
    height: 44,
  },
  emptyTitle: {
    fontSize: 17,
    fontWeight: '700',
    color: '#d4d4d8',
    marginTop: 4,
  },
  emptySubtitle: {
    fontSize: 13,
    color: '#71717a',
    textAlign: 'center',
    lineHeight: 18,
  },
  emptyButton: {
    marginTop: 16,
    backgroundColor: '#10b981',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 8,
  },
  emptyButtonText: {
    color: '#ffffff',
    fontWeight: '600',
    fontSize: 14,
  },
  modalBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.88)',
    justifyContent: 'flex-end',
  },
  settingsSheet: {
    backgroundColor: '#18181b',
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: '#27272a',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  modalTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#f4f4f5',
  },
  inputLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#a1a1aa',
    marginBottom: 6,
    marginTop: 10,
  },
  modalInput: {
    backgroundColor: '#09090b',
    borderWidth: 1,
    borderColor: '#27272a',
    borderRadius: 8,
    color: '#f4f4f5',
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 14,
  },
  modalActionRow: {
    marginTop: 20,
    marginBottom: 10,
  },
  saveSettingsButton: {
    backgroundColor: '#10b981',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  saveSettingsText: {
    color: '#ffffff',
    fontWeight: '700',
    fontSize: 14,
  },
  activeUrlBox: {
    backgroundColor: '#09090b',
    padding: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#27272a',
    marginBottom: 8,
  },
  activeUrlLabel: {
    fontSize: 11,
    color: '#71717a',
    fontWeight: '600',
  },
  activeUrlValue: {
    fontSize: 13,
    color: '#10b981',
    fontWeight: '700',
    marginTop: 2,
  },
  presetRow: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 6,
  },
  presetButton: {
    backgroundColor: '#27272a',
    paddingVertical: 5,
    paddingHorizontal: 10,
    borderRadius: 6,
  },
  presetButtonText: {
    fontSize: 11,
    color: '#d4d4d8',
    fontWeight: '600',
  },
});
