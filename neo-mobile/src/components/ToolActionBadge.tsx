import React, { useState } from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { ActivityIndicator } from 'react-native';
import { CheckCircle2, AlertCircle, Wrench, ChevronDown, ChevronUp } from 'lucide-react-native';
import { ToolAction } from '../types';

interface ToolActionBadgeProps {
  action: ToolAction;
}

export const ToolActionBadge: React.FC<ToolActionBadgeProps> = ({ action }) => {
  const [showDetails, setShowDetails] = useState<boolean>(false);

  const getStatusIcon = () => {
    switch (action.status) {
      case 'pending':
        return <ActivityIndicator size="small" color="#38bdf8" />;
      case 'success':
        return <CheckCircle2 size={14} color="#10b981" />;
      case 'failure':
        return <AlertCircle size={14} color="#ef4444" />;
    }
  };

  const getBorderColor = () => {
    switch (action.status) {
      case 'pending':
        return '#0284c7';
      case 'success':
        return '#065f46';
      case 'failure':
        return '#991b1b';
    }
  };

  return (
    <View style={[styles.container, { borderColor: getBorderColor() }]}>
      <TouchableOpacity
        activeOpacity={0.7}
        onPress={() => setShowDetails(!showDetails)}
        style={styles.badgeHeader}
      >
        <View style={styles.badgeLeft}>
          {getStatusIcon()}
          <Text style={styles.label}>{action.label || action.name}</Text>
        </View>
        <View style={styles.badgeRight}>
          {action.arguments || action.result ? (
            showDetails ? (
              <ChevronUp size={12} color="#71717a" />
            ) : (
              <ChevronDown size={12} color="#71717a" />
            )
          ) : null}
        </View>
      </TouchableOpacity>

      {showDetails && (
        <View style={styles.detailsBox}>
          {action.arguments && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Arguments:</Text>
              <Text style={styles.codeSnippet}>
                {typeof action.arguments === 'object'
                  ? JSON.stringify(action.arguments, null, 2)
                  : String(action.arguments)}
              </Text>
            </View>
          )}
          {action.result && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Result:</Text>
              <Text style={styles.codeSnippet}>
                {typeof action.result === 'object'
                  ? JSON.stringify(action.result, null, 2)
                  : String(action.result)}
              </Text>
            </View>
          )}
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#18181b',
    borderRadius: 6,
    borderWidth: 1,
    marginVertical: 3,
    overflow: 'hidden',
  },
  badgeHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 6,
    paddingHorizontal: 10,
  },
  badgeLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  label: {
    fontSize: 12,
    fontWeight: '500',
    color: '#e4e4e7',
  },
  badgeRight: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  detailsBox: {
    padding: 8,
    borderTopWidth: 1,
    borderTopColor: '#27272a',
    backgroundColor: '#09090b',
  },
  section: {
    marginBottom: 4,
  },
  sectionTitle: {
    fontSize: 10,
    fontWeight: '600',
    color: '#71717a',
    textTransform: 'uppercase',
    marginBottom: 2,
  },
  codeSnippet: {
    fontSize: 11,
    color: '#a1a1aa',
    fontFamily: 'monospace',
    backgroundColor: '#18181b',
    padding: 6,
    borderRadius: 4,
  },
});
