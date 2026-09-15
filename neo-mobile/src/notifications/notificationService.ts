import { Vibration } from 'react-native';
import * as Haptics from 'expo-haptics';
import { NeoEvent, NeoEventPriority } from '../types';

type NotificationListener = (event: NeoEvent) => void;

class NotificationService {
  private listeners: Set<NotificationListener> = new Set();
  private activeAlerts: NeoEvent[] = [];

  public subscribe(listener: NotificationListener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  public handleEvent(event: NeoEvent): void {
    this.activeAlerts.unshift(event);
    if (this.activeAlerts.length > 50) {
      this.activeAlerts.pop();
    }

    // Trigger tactile haptic/vibration feedback according to priority
    this.triggerHaptics(event.priority);

    // Notify all active in-app listeners
    this.listeners.forEach((listener) => {
      try {
        listener(event);
      } catch (err) {
        console.error('Error dispatching notification to listener:', err);
      }
    });
  }

  public triggerHaptics(priority: NeoEventPriority): void {
    switch (priority) {
      case 'urgent':
        // 1. Hardware Taptic Engine / haptic motor on physical iOS/Android device
        try {
          Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
        } catch (e) {
          // Ignored if haptics unsupported
        }
        // 2. Heavy physical vibration pattern (Wait 0ms, buzz 500ms, pause 200ms, buzz 500ms, pause 200ms, buzz 800ms)
        Vibration.vibrate([0, 500, 200, 500, 200, 800], false);
        break;

      case 'normal':
        try {
          Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        } catch (e) {}
        Vibration.vibrate(250);
        break;

      case 'low':
      default:
        // Silent
        break;
    }
  }

  public cancelVibration(): void {
    Vibration.cancel();
  }

  public getRecentAlerts(): NeoEvent[] {
    return [...this.activeAlerts];
  }
}

export const notificationService = new NotificationService();
