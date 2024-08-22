// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export type ConversationRequest = {
  conversationId: string;
  userPrompt: Message;
  messages: Message[];
  model: string;
};
export enum MessageRole {
  Assistant = "assistant",
  User = "user",
  System = "system",
}

export interface Message {
  role: MessageRole;
  content: string;
  time?: string;
}

export interface Conversation {
  id: string;
  first_query?: string;
}

type file = {
  name: string;
};

export interface ConversationReducer {
  selectedConversationId: string;
  conversations: Conversation[];
  selectedConversationHistory: Message[];
  onGoingResult: string;
  filesInDataSource: file[];
}