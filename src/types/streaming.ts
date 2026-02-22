export interface StreamingMessage {
  id: string;
  type: StreamingMessageType;
  payload: any;
  timestamp: number;
  parentId?: string;
  metadata?: StreamingMetadata;
}

export interface StreamingMetadata {
  source: 'local_llm' | 'backend_llm' | 'voice_service' | 'orchestration' | 'intent_evaluation';
  provider?: string;
  priority?: number;
  sessionId?: string;
  userId?: string;
  clientId?: string;
  confidence?: number;
}

export enum StreamingMessageType {
  VOICE_STT_START = 'voice_stt_start',
  VOICE_STT_CHUNK = 'voice_stt_chunk',
  VOICE_STT_END = 'voice_stt_end',
  VOICE_TTS_REQUEST = 'voice_tts_request',
  VOICE_TTS_CHUNK = 'voice_tts_chunk',
  VOICE_TTS_END = 'voice_tts_end',

  LLM_REQUEST = 'llm_request',
  LLM_STREAM_START = 'llm_stream_start',
  LLM_STREAM_CHUNK = 'llm_stream_chunk',
  LLM_STREAM_END = 'llm_stream_end',
  LLM_ERROR = 'llm_error',

  INTENT_CLASSIFICATION = 'intent_classification',

  CONVERSATION_START = 'conversation_start',
  CONVERSATION_CHUNK = 'conversation_chunk',
  CONVERSATION_END = 'conversation_end',
  CONVERSATION_INTERRUPT = 'conversation_interrupt',

  HEARTBEAT = 'heartbeat',
  ERROR = 'error',
  STATUS = 'status',
  CONNECTION_STATUS = 'connection_status',

  INTERRUPT = 'interrupt',
  CANCEL = 'cancel',
  PAUSE = 'pause',
  RESUME = 'resume',
}

export interface VoiceSTTChunk {
  audioData: string;
  format: 'wav' | 'mp3' | 'webm';
  sampleRate: number;
  channels: number;
  duration: number;
}

export interface VoiceSTTResult {
  text: string;
  confidence: number;
  isFinal: boolean;
  language?: string;
  timestamp: number;
}

export interface VoiceTTSRequest {
  text: string;
  voice: string;
  speed: number;
  pitch: number;
  provider: 'elevenlabs' | 'openai' | 'local';
  options?: {
    stability?: number;
    similarity_boost?: number;
    style?: number;
  };
}

export interface VoiceTTSChunk {
  audioData: string;
  format: 'wav' | 'mp3' | 'webm';
  sampleRate: number;
  channels: number;
  duration: number;
  isLast: boolean;
}

export interface LLMStreamRequest {
  prompt: string;
  provider?: string;
  options?: {
    temperature?: number;
    maxTokens?: number;
    stream?: boolean;
    taskType?: string;
    responseLength?: 'short' | 'medium' | 'long';
    enableWebSearch?: boolean;
  };
  context?: {
    recentContext?: Array<{ role: string; content: string; timestamp?: string; messageId?: string }>;
    sessionFacts?: Array<{ fact: string; confidence: number; timestamp?: string }>;
    sessionEntities?: Array<{ entity: string; type: string; value?: any }>;
    memories?: Array<{ content: string; relevance?: number; timestamp?: string }>;
    webSearchResults?: Array<{ title: string; snippet: string; url: string }>;
    systemInstructions?: string;
    sessionId?: string;
    userId?: string;
  };
}

export interface LLMStreamChunk {
  text: string;
  provider: string;
  tokenCount?: number;
  finishReason?: 'stop' | 'length' | 'content_filter' | null;
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

export interface LLMStreamResult {
  fullText: string;
  provider: string;
  processingTime: number;
  tokenUsage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  fallbackChain?: Array<{
    provider: string;
    success: boolean;
    error?: string;
    latencyMs: number;
  }>;
}

export interface ConversationContext {
  sessionId: string;
  userId?: string;
  conversationHistory: ConversationMessage[];
  currentTopic?: string;
  userPreferences?: Record<string, any>;
  localLLMCapabilities: string[];
  backendLLMCapabilities: string[];
}

export interface ConversationMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  source: 'voice' | 'text' | 'system';
  metadata?: {
    provider?: string;
    processingTime?: number;
    confidence?: number;
  };
}

export interface StreamingError {
  code: string;
  message: string;
  details?: any;
  recoverable: boolean;
  provider?: string;
}

export interface ConnectionStatus {
  connected: boolean;
  lastHeartbeat: number;
  reconnectAttempts: number;
  latency: number;
  activeStreams: number;
}

export interface StreamingResponse<T = any> {
  success: boolean;
  data?: T;
  error?: StreamingError;
  metadata: StreamingMetadata;
}
