import React, { useState } from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { Brain, ChevronDown, ChevronRight, Sparkles } from 'lucide-react-native';

interface ThinkingBlockProps {
  reasoning: string | null | undefined;
  thinkingEnabled?: boolean;
}

export const ThinkingBlock: React.FC<ThinkingBlockProps> = ({ reasoning, thinkingEnabled }) => {
  const [isExpanded, setIsExpanded] = useState<boolean>(false);

  // Requirement: LOW-risk fast actions naturally have no thinking panel
  if (!reasoning || reasoning.trim() === '') {
    return null;
  }

  const wordCount = reasoning.trim().split(/\s+/).length;

  return (
    <View style={styles.container}>
      <TouchableOpacity
        activeOpacity={0.7}
        onPress={() => setIsExpanded(!isExpanded)}
        style={styles.header}
      >
        <View style={styles.headerLeft}>
          <Brain size={15} color="#a1a1aa" />
          <Text style={styles.title}>Thinking</Text>
          <Text style={styles.metaBadge}>{wordCount} words</Text>
        </View>
        <View style={styles.headerRight}>
          {isExpanded ? (
            <ChevronDown size={15} color="#71717a" />
          ) : (
            <ChevronRight size={15} color="#71717a" />
          )}
        </View>
      </TouchableOpacity>

      {isExpanded && (
        <View style={styles.contentContainer}>
          <Text style={styles.reasoningText}>{reasoning.trim()}</Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#18181b',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#27272a',
    marginVertical: 6,
    overflow: 'hidden',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 8,
    paddingHorizontal: 12,
    backgroundColor: '#18181b',
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  title: {
    fontSize: 13,
    fontWeight: '600',
    color: '#d4d4d8',
  },
  metaBadge: {
    fontSize: 11,
    color: '#71717a',
    backgroundColor: '#27272a',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    marginLeft: 4,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  contentContainer: {
    padding: 12,
    borderTopWidth: 1,
    borderTopColor: '#27272a',
    backgroundColor: '#09090b',
  },
  reasoningText: {
    fontSize: 12,
    lineHeight: 18,
    color: '#a1a1aa',
    fontFamily: 'monospace',
  },
});
