import React, { useRef, useState } from 'react';
import {
  Animated,
  Dimensions,
  Image,
  Modal,
  PanResponder,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { Maximize2, X, Monitor, RotateCw, Minimize2 } from 'lucide-react-native';

interface ScreenshotModalProps {
  imageUrl: string;
  timestamp?: number;
  caption?: string;
}

export const ScreenshotModal: React.FC<ScreenshotModalProps> = ({
  imageUrl,
  timestamp,
  caption,
}) => {
  const [modalVisible, setModalVisible] = useState<boolean>(false);
  const [rotation, setRotation] = useState<number>(0);

  // Pinch-zoom / pan with RN core only — no gesture-handler or reanimated dependency.
  const scale = useRef(new Animated.Value(1)).current;
  const translate = useRef(new Animated.ValueXY({ x: 0, y: 0 })).current;
  const gesture = useRef({ scale: 1, x: 0, y: 0, startDist: 0, startScale: 1 }).current;

  const resetView = () => {
    gesture.scale = 1;
    gesture.x = 0;
    gesture.y = 0;
    setRotation(0);
    scale.setValue(1);
    translate.setValue({ x: 0, y: 0 });
  };

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,
      onPanResponderGrant: () => {
        gesture.startDist = 0;
        gesture.startScale = gesture.scale;
      },
      onPanResponderMove: (evt, gs) => {
        const touches = evt.nativeEvent.touches;
        if (touches.length >= 2) {
          const dx = touches[0].pageX - touches[1].pageX;
          const dy = touches[0].pageY - touches[1].pageY;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (gesture.startDist === 0) {
            gesture.startDist = dist;
            gesture.startScale = gesture.scale;
            return;
          }
          const next = Math.min(6, Math.max(1, (dist / gesture.startDist) * gesture.startScale));
          gesture.scale = next;
          scale.setValue(next);
        } else if (touches.length === 1) {
          translate.setValue({ x: gesture.x + gs.dx, y: gesture.y + gs.dy });
        }
      },
      onPanResponderRelease: (_evt, gs) => {
        gesture.x += gs.dx;
        gesture.y += gs.dy;
        gesture.startDist = 0;
        if (gesture.scale <= 1.02) {
          // Snap back so a zoomed-out image can never be left stranded off-screen.
          gesture.x = 0;
          gesture.y = 0;
          Animated.spring(translate, { toValue: { x: 0, y: 0 }, useNativeDriver: true }).start();
        }
      },
    })
  ).current;

  const openViewer = () => {
    resetView();
    setModalVisible(true);
  };

  return (
    <View style={styles.thumbnailContainer}>
      <TouchableOpacity
        activeOpacity={0.85}
        onPress={openViewer}
        style={styles.imageWrapper}
      >
        <Image
          source={{ uri: imageUrl }}
          style={styles.thumbnailImage}
          resizeMode="cover"
        />
        <View style={styles.overlayBar}>
          <View style={styles.captionGroup}>
            <Monitor size={12} color="#f4f4f5" />
            <Text style={styles.captionText} numberOfLines={1}>
              {caption || 'Desktop Screenshot'}
            </Text>
          </View>
          <Maximize2 size={12} color="#f4f4f5" />
        </View>
      </TouchableOpacity>

      <Modal
        visible={modalVisible}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.fullscreenBackdrop}>
          <TouchableOpacity
            style={styles.closeButton}
            onPress={() => setModalVisible(false)}
            activeOpacity={0.8}
          >
            <X size={22} color="#ffffff" />
          </TouchableOpacity>

          <View style={styles.toolbar}>
            <TouchableOpacity
              style={styles.toolButton}
              onPress={() => setRotation((r) => (r + 90) % 360)}
              activeOpacity={0.8}
            >
              <RotateCw size={18} color="#ffffff" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.toolButton} onPress={resetView} activeOpacity={0.8}>
              <Minimize2 size={18} color="#ffffff" />
            </TouchableOpacity>
          </View>

          <View style={styles.fullscreenImageContainer} {...panResponder.panHandlers}>
            <Animated.Image
              source={{ uri: imageUrl }}
              style={[
                styles.fullscreenImage,
                {
                  transform: [
                    { translateX: translate.x },
                    { translateY: translate.y },
                    { scale },
                    { rotate: `${rotation}deg` },
                  ],
                },
              ]}
              resizeMode="contain"
            />
          </View>

          <View style={styles.modalFooter}>
            <Text style={styles.footerText}>
              {caption ? `${caption} — ` : ''}iki parmakla yakınlaştır, sürükle, döndür
            </Text>
          </View>
        </View>
      </Modal>
    </View>
  );
};

const { width, height } = Dimensions.get('window');

const styles = StyleSheet.create({
  thumbnailContainer: {
    marginVertical: 8,
    borderRadius: 8,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#27272a',
    backgroundColor: '#09090b',
  },
  imageWrapper: {
    position: 'relative',
    height: 180,
    width: '100%',
  },
  thumbnailImage: {
    width: '100%',
    height: '100%',
  },
  overlayBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(9, 9, 11, 0.85)',
    paddingHorizontal: 10,
    paddingVertical: 6,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: '#27272a',
  },
  captionGroup: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  captionText: {
    color: '#f4f4f5',
    fontSize: 11,
    fontWeight: '500',
  },
  fullscreenBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.95)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  closeButton: {
    position: 'absolute',
    top: 50,
    right: 20,
    zIndex: 10,
    padding: 10,
    backgroundColor: '#27272a',
    borderRadius: 20,
  },
  fullscreenImageContainer: {
    width: width,
    height: height * 0.85,
    justifyContent: 'center',
    alignItems: 'center',
  },
  fullscreenImage: {
    width: '100%',
    height: '100%',
  },
  toolbar: {
    position: 'absolute',
    top: 44,
    left: 16,
    zIndex: 10,
    flexDirection: 'row',
    gap: 10,
  },
  toolButton: {
    width: 38,
    height: 38,
    borderRadius: 19,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(39, 39, 42, 0.9)',
  },
  modalFooter: {
    position: 'absolute',
    bottom: 30,
    backgroundColor: '#18181b',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#27272a',
  },
  footerText: {
    color: '#d4d4d8',
    fontSize: 13,
  },
});
