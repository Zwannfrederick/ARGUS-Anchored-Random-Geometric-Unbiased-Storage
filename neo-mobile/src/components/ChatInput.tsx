import React, { useState } from 'react';
import {
  ActivityIndicator,
  Platform,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { ArrowUp } from 'lucide-react-native';
import { ConnectionStatus } from '../types';

interface ChatInputProps {
  onSend: (text: string) => void;
  status: ConnectionStatus;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSend, status, disabled }) => {
  const [text, setText] = useState<string>('');

  const handleSend = () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText('');
  };

  const isSendDisabled = !text.trim() || disabled;

  return (
    <View style={styles.container}>
      <View style={styles.inputRow}>
        <View style={styles.inputWrapper}>
          <TextInput
            style={styles.textInput}
            placeholder="Message Neo..."
            placeholderTextColor="#71717a"
            value={text}
            onChangeText={setText}
            multiline={true}
            maxLength={4000}
            editable={!disabled}
            blurOnSubmit={false}
            returnKeyType="default"
          />
        </View>

        <TouchableOpacity
          style={[
            styles.sendBtn,
            isSendDisabled ? styles.sendBtnDisabled : styles.sendBtnActive,
          ]}
          onPress={handleSend}
          disabled={isSendDisabled}
          activeOpacity={0.8}
        >
          {disabled ? (
            <ActivityIndicator size="small" color="#ffffff" />
          ) : (
            <ArrowUp size={18} color={isSendDisabled ? '#71717a' : '#ffffff'} />
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: 12,
    paddingTop: 6,
    paddingBottom: 8,
    backgroundColor: '#09090b',
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 8,
  },
  inputWrapper: {
    flex: 1,
    backgroundColor: '#18181b',
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#27272a',
    paddingHorizontal: 14,
    paddingVertical: Platform.OS === 'ios' ? 8 : 4,
    minHeight: 40,
    maxHeight: 120,
    justifyContent: 'center',
  },
  textInput: {
    color: '#f4f4f5',
    fontSize: 15,
    lineHeight: 20,
    padding: 0,
    margin: 0,
  },
  sendBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 0,
  },
  sendBtnDisabled: {
    backgroundColor: '#18181b',
    borderWidth: 1,
    borderColor: '#27272a',
  },
  sendBtnActive: {
    backgroundColor: '#10b981',
  },
});
