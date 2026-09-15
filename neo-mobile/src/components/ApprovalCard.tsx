import React, { useState } from 'react';
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { AlertTriangle, Check, ShieldAlert, X } from 'lucide-react-native';
import { ApprovalRequest } from '../types';

interface ApprovalCardProps {
  approval: ApprovalRequest;
  onDecide: (approvalId: string, decision: 'approve' | 'reject', actionHash?: string) => Promise<void>;
}

export const ApprovalCard: React.FC<ApprovalCardProps> = ({ approval, onDecide }) => {
  const [isDeciding, setIsDeciding] = useState<boolean>(false);
  const [localStatus, setLocalStatus] = useState<'pending' | 'approved' | 'rejected'>(
    approval.status
  );
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleDecision = async (decision: 'approve' | 'reject') => {
    if (localStatus !== 'pending' || isDeciding) return;
    setIsDeciding(true);
    setErrorMsg(null);
    try {
      await onDecide(approval.approval_id, decision, approval.action_hash);
      setLocalStatus(decision === 'approve' ? 'approved' : 'rejected');
    } catch (e: any) {
      setErrorMsg(e.message || 'Decision failed');
    } finally {
      setIsDeciding(false);
    }
  };

  const isPending = localStatus === 'pending';

  const getRiskBadgeStyle = () => {
    switch (approval.risk_level) {
      case 'critical':
        return styles.badge_critical;
      case 'high':
        return styles.badge_high;
      case 'medium':
      default:
        return styles.badge_medium;
    }
  };

  return (
    <View style={[styles.card, !isPending && styles.resolvedCard]}>
      <View style={styles.header}>
        <View style={styles.titleGroup}>
          <ShieldAlert size={18} color="#f59e0b" />
          <Text style={styles.title}>High-Risk Action Approval</Text>
        </View>
        <View style={[styles.badge, getRiskBadgeStyle()]}>
          <Text style={styles.badgeText}>{approval.risk_level.toUpperCase()}</Text>
        </View>
      </View>

      <View style={styles.body}>
        <Text style={styles.descriptionLabel}>Action:</Text>
        <Text style={styles.descriptionText}>{approval.action_description}</Text>

        <Text style={styles.commandLabel}>Target Command / Payload:</Text>
        <View style={styles.commandBox}>
          <Text style={styles.commandText} selectable={true}>
            {approval.command}
          </Text>
        </View>

        <View style={styles.idRow}>
          <Text style={styles.idText}>Action ID: {approval.approval_id.slice(0, 8)}...</Text>
        </View>

        {errorMsg && <Text style={styles.errorText}>{errorMsg}</Text>}
      </View>

      {isPending ? (
        <View style={styles.actionRow}>
          <TouchableOpacity
            style={[styles.btn, styles.rejectBtn, isDeciding && styles.btnDisabled]}
            onPress={() => handleDecision('reject')}
            disabled={isDeciding}
            activeOpacity={0.7}
          >
            {isDeciding ? (
              <ActivityIndicator size="small" color="#fca5a5" />
            ) : (
              <>
                <X size={16} color="#fca5a5" />
                <Text style={styles.rejectBtnText}>Reject</Text>
              </>
            )}
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.btn, styles.approveBtn, isDeciding && styles.btnDisabled]}
            onPress={() => handleDecision('approve')}
            disabled={isDeciding}
            activeOpacity={0.7}
          >
            {isDeciding ? (
              <ActivityIndicator size="small" color="#ffffff" />
            ) : (
              <>
                <Check size={16} color="#ffffff" />
                <Text style={styles.approveBtnText}>Approve</Text>
              </>
            )}
          </TouchableOpacity>
        </View>
      ) : (
        <View style={styles.resolvedFooter}>
          <Text
            style={[
              styles.resolvedText,
              localStatus === 'approved' ? styles.approvedColor : styles.rejectedColor,
            ]}
          >
            {localStatus === 'approved' ? '✓ Action Authorized' : '✗ Action Blocked / Rejected'}
          </Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#18181b',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#d97706',
    marginVertical: 10,
    overflow: 'hidden',
  },
  resolvedCard: {
    borderColor: '#3f3f46',
    opacity: 0.85,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 14,
    paddingVertical: 10,
    backgroundColor: '#27272a',
  },
  titleGroup: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  title: {
    fontSize: 14,
    fontWeight: '700',
    color: '#fbbf24',
  },
  badge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
    backgroundColor: '#b45309',
  },
  badge_medium: {
    backgroundColor: '#b45309',
  },
  badge_high: {
    backgroundColor: '#dc2626',
  },
  badge_critical: {
    backgroundColor: '#7f1d1d',
  },
  badgeText: {
    fontSize: 10,
    fontWeight: '700',
    color: '#ffffff',
  },
  body: {
    padding: 14,
  },
  descriptionLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#a1a1aa',
    marginBottom: 4,
  },
  descriptionText: {
    fontSize: 14,
    color: '#f4f4f5',
    lineHeight: 20,
    marginBottom: 10,
  },
  commandLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#a1a1aa',
    marginBottom: 4,
  },
  commandBox: {
    backgroundColor: '#09090b',
    borderRadius: 6,
    padding: 10,
    borderWidth: 1,
    borderColor: '#27272a',
  },
  commandText: {
    fontFamily: 'monospace',
    fontSize: 12,
    color: '#f87171',
  },
  idRow: {
    marginTop: 8,
  },
  idText: {
    fontSize: 10,
    color: '#71717a',
  },
  errorText: {
    color: '#ef4444',
    fontSize: 12,
    marginTop: 6,
  },
  actionRow: {
    flexDirection: 'row',
    borderTopWidth: 1,
    borderTopColor: '#27272a',
    backgroundColor: '#18181b',
  },
  btn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    gap: 6,
  },
  rejectBtn: {
    backgroundColor: '#450a0a',
    borderRightWidth: 1,
    borderRightColor: '#27272a',
  },
  rejectBtnText: {
    color: '#fca5a5',
    fontWeight: '600',
    fontSize: 14,
  },
  approveBtn: {
    backgroundColor: '#065f46',
  },
  approveBtnText: {
    color: '#ffffff',
    fontWeight: '700',
    fontSize: 14,
  },
  btnDisabled: {
    opacity: 0.6,
  },
  resolvedFooter: {
    paddingVertical: 10,
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: '#27272a',
    backgroundColor: '#18181b',
  },
  resolvedText: {
    fontSize: 13,
    fontWeight: '600',
  },
  approvedColor: {
    color: '#10b981',
  },
  rejectedColor: {
    color: '#ef4444',
  },
});
