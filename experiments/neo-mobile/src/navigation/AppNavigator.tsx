import React from 'react';
import { NavigationContainer, DarkTheme } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { ConversationListScreen } from '../screens/ConversationListScreen';
import { ChatScreen } from '../screens/ChatScreen';
import { PlainChatScreen } from '../screens/PlainChatScreen';
import { Session } from '../types';

export type RootStackParamList = {
  ConversationList: undefined;
  Chat: { session: Session };
  PlainChat: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

const customDarkTheme = {
  ...DarkTheme,
  dark: true,
  colors: {
    ...DarkTheme.colors,
    primary: '#10b981',
    background: '#09090b',
    card: '#121214',
    text: '#f4f4f5',
    border: '#27272a',
    notification: '#ec4899',
  },
};

export const AppNavigator: React.FC = () => {
  return (
    <NavigationContainer theme={customDarkTheme}>
      <Stack.Navigator
        initialRouteName="ConversationList"
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: '#09090b' },
          animation: 'slide_from_right',
        }}
      >
        <Stack.Screen name="ConversationList" component={ConversationListScreen} />
        <Stack.Screen name="Chat" component={ChatScreen} />
        <Stack.Screen name="PlainChat" component={PlainChatScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};
