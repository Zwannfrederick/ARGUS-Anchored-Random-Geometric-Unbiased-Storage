import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { HardDrive } from 'lucide-react-native';
import { neoClient } from '../api/neoClient';
import { ArgusStatus } from '../types';

const POLL_MS = 3000;

const formatBytes = (value: number | null | undefined): string => {
  if (value == null) return '—';
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(0)} KB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  return `${(value / 1024 ** 3).toFixed(2)} GB`;
};

/** Live ARGUS KV residency published by the patched llama-server. */
export const ArgusKvCard: React.FC = () => {
  const [status, setStatus] = useState<ArgusStatus | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const next = await neoClient.fetchArgusStatus();
        if (alive) {
          setStatus(next);
          setFailed(false);
        }
      } catch (e) {
        if (alive) setFailed(true);
      }
    };
    poll();
    const timer = setInterval(poll, POLL_MS);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  const stats = status?.stats;
  const budget = status?.resident_budget_bytes ?? null;
  const ratio = stats ? Math.min(1, stats.resident_bytes / (budget || stats.live_bytes || 1)) : 0;
  const summary = failed
    ? 'Durum alınamadı'
    : !status
      ? 'Bağlanıyor…'
      : !status.enabled
        ? 'Kapalı — stok llama.cpp KV'
        : stats
          ? `RAM'de ${formatBytes(stats.resident_bytes)}${budget ? ` / ${formatBytes(budget)}` : ''}`
          : 'İlk attention bekleniyor';

  return (
    <View style={styles.card} accessible accessibilityLabel={`ARGUS KV: ${summary}`}>
      <View style={styles.row}>
        <HardDrive size={14} color="#10b981" />
        <Text style={styles.title}>ARGUS KV</Text>
        <Text style={styles.summary} numberOfLines={1}>{summary}</Text>
      </View>
      {stats ? (
        <>
          <View style={styles.track}>
            <View style={[styles.fill, { width: `${Math.round(ratio * 100)}%` as `${number}%` }]} />
          </View>
          <Text style={styles.meta}>
            Ayrılan {formatBytes(stats.live_bytes)} · okunan {formatBytes(stats.read_bytes)} · diske verilen{' '}
            {formatBytes(stats.paged_out_bytes)}
          </Text>
        </>
      ) : null}
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    marginHorizontal: 16,
    marginBottom: 8,
    padding: 10,
    borderRadius: 12,
    backgroundColor: '#121214',
    borderWidth: 1,
    borderColor: '#27272a',
  },
  row: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  title: { color: '#f4f4f5', fontSize: 12, fontWeight: '700', letterSpacing: 0.4 },
  summary: { color: '#a1a1aa', fontSize: 12, flex: 1, textAlign: 'right' },
  track: { height: 5, borderRadius: 3, backgroundColor: '#27272a', marginTop: 8, overflow: 'hidden' },
  fill: { height: '100%', backgroundColor: '#10b981' },
  meta: { color: '#71717a', fontSize: 11, marginTop: 6 },
});
