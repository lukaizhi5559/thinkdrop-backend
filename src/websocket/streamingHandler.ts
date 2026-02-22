/**
 * WebSocket streaming handler — LLM streaming with context enrichment.
 * Receives stategraph-built prompts with context (memories, history, entities, web search)
 * and embeds them into the prompt body before forwarding to the LLM provider.
 * systemInstructions from stategraph are passed as the LLM system message.
 * No persona injection, no web search decisions beyond what stategraph provides.
 */

import WebSocket from 'ws';
import { v4 as uuidv4 } from 'uuid';
import { llmStreamingRouter } from '../utils/llmStreamingRouter';
import {
  StreamingMessage,
  StreamingMessageType,
  LLMStreamRequest,
  VoiceSTTChunk,
  VoiceTTSRequest,
  ConversationContext,
  StreamingMetadata,
  StreamingError,
} from '../types/streaming';
import { logger } from '../utils/logger';

export class StreamingHandler {
  private ws: WebSocket;
  private sessionId: string;
  private userId?: string;
  private clientId?: string;
  private conversationContext: ConversationContext;
  private activeRequests: Map<string, AbortController> = new Map();

  constructor(ws: WebSocket, sessionId: string, userId?: string, clientId?: string) {
    this.ws = ws;
    this.sessionId = sessionId;
    this.userId = userId;
    this.clientId = clientId;

    this.conversationContext = {
      sessionId,
      userId,
      conversationHistory: [],
      localLLMCapabilities: ['simple_qa', 'intent_routing', 'context_switching'],
      backendLLMCapabilities: ['orchestration', 'code_generation', 'complex_reasoning', 'api_calls'],
    };
  }

  async handleMessage(message: StreamingMessage): Promise<void> {
    const { type, id, payload } = message;

    try {
      switch (type) {
        case StreamingMessageType.LLM_REQUEST:
          await this.handleLLMRequest(id, payload as LLMStreamRequest, message.metadata);
          break;

        case StreamingMessageType.VOICE_STT_CHUNK:
          await this.handleVoiceSTTChunk(id, payload as VoiceSTTChunk, message.metadata);
          break;

        case StreamingMessageType.VOICE_TTS_REQUEST:
          await this.handleVoiceTTSRequest(id, payload as VoiceTTSRequest, message.metadata);
          break;

        case StreamingMessageType.CONVERSATION_START:
          await this.handleConversationStart(id, payload, message.metadata);
          break;

        case StreamingMessageType.INTERRUPT:
          await this.handleInterrupt(id, payload);
          break;

        case StreamingMessageType.CANCEL:
          await this.handleCancel(id, payload);
          break;

        case StreamingMessageType.HEARTBEAT:
          await this.handleHeartbeat(id);
          break;

        default:
          this.sendError(id, `Unknown message type: ${type}`);
      }
    } catch (error) {
      logger.error('Error handling streaming message:', { error: error instanceof Error ? error.message : String(error) });
      this.sendError(id, error instanceof Error ? error.message : String(error));
    }
  }

  /**
   * Build enriched prompt embedding all context (memories, history, entities, web search)
   * into the prompt body — mirrors bibscrip-backend buildThinkdropAIPrompt pattern.
   * systemInstructions from stategraph is preserved as the system message.
   */
  private buildEnrichedPrompt(message: string, context?: any): string {
    const recentContext = context?.recentContext || [];
    const sessionFacts = context?.sessionFacts || [];
    const sessionEntities = context?.sessionEntities || [];
    const memories = context?.memories || [];
    const webSearchResults = context?.webSearchResults || [];
    const systemInstructions = context?.systemInstructions || '';

    logger.info('🔍 [StreamingHandler] Context for prompt:', {
      recentContextCount: recentContext.length,
      sessionFactsCount: sessionFacts.length,
      sessionEntitiesCount: sessionEntities.length,
      memoriesCount: memories.length,
      webSearchResultsCount: webSearchResults.length,
      hasSystemInstructions: !!systemInstructions,
    });

    if (recentContext.length === 0 && sessionFacts.length === 0 && memories.length === 0) {
      logger.warn(`⚠️ [StreamingHandler] Minimal context for: ${message.substring(0, 60)}`);
    }

    const historyContext = recentContext.length > 0
      ? `\n\nRecent Conversation History:\n${recentContext.slice(-8).map((h: any) => `${h.role}: ${h.content}`).join('\n')}`
      : '';

    const factsContext = sessionFacts.length > 0
      ? `\n\nSession Facts:\n${sessionFacts.map((f: any) => `- ${f.fact}${f.confidence !== undefined ? ` (confidence: ${f.confidence})` : ''}`).join('\n')}`
      : '';

    const entitiesContext = sessionEntities.length > 0
      ? `\n\nSession Entities:\n${sessionEntities.map((e: any) => `- ${e.entity} (${e.type})${e.value ? `: ${e.value}` : ''}`).join('\n')}`
      : '';

    const memoriesContext = memories.length > 0
      ? `\n\nScreen Activity & User Memories:\n${memories.map((m: any, i: number) => {
          const text = m.content || m.text || '';
          const score = m.relevance ?? m.similarity;
          const meta = typeof m.metadata === 'string'
            ? (() => { try { return JSON.parse(m.metadata); } catch { return {}; } })()
            : (m.metadata || {});
          const app = meta.appName;
          const title = meta.windowTitle;
          const files = Array.isArray(meta.files) && meta.files.length > 0 ? meta.files.slice(0, 5).join(', ') : '';
          const capturedAt = meta.capturedAt
            ? new Date(meta.capturedAt).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
            : (m.created_at ? new Date(m.created_at).toLocaleString() : '');
          let line = `[${i + 1}]`;
          if (app) line += ` App: ${app}`;
          if (title) line += ` | Window: ${title}`;
          if (files) line += ` | Files: ${files}`;
          if (capturedAt) line += ` | At: ${capturedAt}`;
          if (score !== undefined) line += ` (similarity: ${typeof score === 'number' ? score.toFixed(2) : score})`;
          if (!app && text) line += ` ${text.substring(0, 200)}`;
          return line;
        }).join('\n')}`
      : '';

    const webSearchContext = webSearchResults.length > 0
      ? `\n\nWeb Search Results:\n${webSearchResults.map((r: any, i: number) => `${i + 1}. ${r.title}\n   ${r.snippet}\n   Source: ${r.url}`).join('\n\n')}`
      : '';

    const systemInstructionsContext = systemInstructions
      ? `\n\nSystem Instructions:\n${systemInstructions}`
      : '';

    return `${message}${historyContext}${factsContext}${entitiesContext}${memoriesContext}${webSearchContext}${systemInstructionsContext}`.trim();
  }

  private async handleLLMRequest(
    requestId: string,
    request: LLMStreamRequest,
    metadata?: StreamingMetadata
  ): Promise<void> {
    const abortController = new AbortController();
    this.activeRequests.set(requestId, abortController);

    try {
      const streamingMetadata: StreamingMetadata = {
        source: 'backend_llm',
        sessionId: this.sessionId,
        userId: this.userId,
        clientId: this.clientId,
        ...metadata,
      };

      // Build enriched prompt with all context embedded in the message body
      const enrichedPrompt = this.buildEnrichedPrompt(request.prompt, request.context);

      // Track original user message in session history
      this.conversationContext.conversationHistory.push({
        id: requestId,
        role: 'user',
        content: request.prompt,
        timestamp: Date.now(),
        source: 'text',
      });

      logger.info(`🚀 [StreamingHandler] LLM request ${requestId}`, {
        originalLength: request.prompt.length,
        enrichedLength: enrichedPrompt.length,
        provider: request.provider || 'auto',
        enrichedPreview: enrichedPrompt.substring(0, 500),
      });

      // Send enriched prompt — context is embedded in the prompt body
      const enrichedRequest: LLMStreamRequest = {
        ...request,
        prompt: enrichedPrompt,
      };

      const result = await llmStreamingRouter.processPromptWithStreaming(
        enrichedRequest,
        (chunk: StreamingMessage) => {
          if (this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(chunk));
          }
        },
        streamingMetadata
      );

      logger.info(`✅ [StreamingHandler] LLM streaming completed for ${requestId}`, {
        provider: result.provider,
        processingTime: result.processingTime,
        textLength: result.fullText.length,
      });

      // Track assistant response in session history
      this.conversationContext.conversationHistory.push({
        id: `${requestId}_response`,
        role: 'assistant',
        content: result.fullText,
        timestamp: Date.now(),
        source: 'text',
        metadata: { provider: result.provider, processingTime: result.processingTime },
      });

      // Keep session history bounded (last 20 messages)
      if (this.conversationContext.conversationHistory.length > 20) {
        this.conversationContext.conversationHistory =
          this.conversationContext.conversationHistory.slice(-20);
      }
    } catch (error) {
      this.sendError(requestId, error instanceof Error ? error.message : String(error));
    } finally {
      this.activeRequests.delete(requestId);
    }
  }

  private async handleVoiceSTTChunk(
    requestId: string,
    chunk: VoiceSTTChunk,
    metadata?: StreamingMetadata
  ): Promise<void> {
    try {
      const transcription = await this.processSTTChunk(chunk);

      if (transcription) {
        this.send({
          id: `${requestId}_stt_result`,
          type: StreamingMessageType.VOICE_STT_CHUNK,
          payload: { text: transcription.text, confidence: transcription.confidence, isFinal: transcription.isFinal },
          timestamp: Date.now(),
          parentId: requestId,
          metadata: { source: 'voice_service', ...metadata },
        });

        if (transcription.isFinal && transcription.text.trim()) {
          await this.handleLLMRequest(`${requestId}_llm`, { prompt: transcription.text, options: { taskType: 'conversation' } }, metadata);
        }
      }
    } catch (error) {
      this.sendError(requestId, error instanceof Error ? error.message : String(error));
    }
  }

  private async handleVoiceTTSRequest(
    requestId: string,
    request: VoiceTTSRequest,
    metadata?: StreamingMetadata
  ): Promise<void> {
    try {
      const audioChunks = await this.processTTSRequest(request);

      for (const chunk of audioChunks) {
        this.send({
          id: `${requestId}_tts_chunk_${Date.now()}`,
          type: StreamingMessageType.VOICE_TTS_CHUNK,
          payload: chunk,
          timestamp: Date.now(),
          parentId: requestId,
          metadata: { source: 'voice_service', ...metadata },
        });
      }

      this.send({
        id: `${requestId}_tts_end`,
        type: StreamingMessageType.VOICE_TTS_END,
        payload: { completed: true },
        timestamp: Date.now(),
        parentId: requestId,
        metadata: { source: 'voice_service', ...metadata },
      });
    } catch (error) {
      this.sendError(requestId, error instanceof Error ? error.message : String(error));
    }
  }

  private async handleConversationStart(requestId: string, payload: any, metadata?: StreamingMetadata): Promise<void> {
    try {
      this.conversationContext.currentTopic = payload.topic;
      this.conversationContext.userPreferences = payload.preferences || {};

      this.send({
        id: `${requestId}_conversation_ready`,
        type: StreamingMessageType.CONVERSATION_START,
        payload: {
          sessionId: this.sessionId,
          capabilities: {
            local: this.conversationContext.localLLMCapabilities,
            backend: this.conversationContext.backendLLMCapabilities,
          },
          ready: true,
        },
        timestamp: Date.now(),
        metadata: { source: 'local_llm', ...metadata },
      });
    } catch (error) {
      this.sendError(requestId, error instanceof Error ? error.message : String(error));
    }
  }

  private async handleInterrupt(requestId: string, payload: any): Promise<void> {
    try {
      const targetId = payload.targetId;

      if (targetId && this.activeRequests.has(targetId)) {
        this.activeRequests.get(targetId)?.abort();
        this.activeRequests.delete(targetId);
        this.send({ id: `${requestId}_interrupt_success`, type: StreamingMessageType.INTERRUPT, payload: { interrupted: targetId, success: true }, timestamp: Date.now(), metadata: { source: 'local_llm' } });
      } else {
        for (const [, controller] of this.activeRequests) controller.abort();
        this.activeRequests.clear();
        this.send({ id: `${requestId}_interrupt_all`, type: StreamingMessageType.INTERRUPT, payload: { interrupted: 'all', success: true }, timestamp: Date.now(), metadata: { source: 'local_llm' } });
      }
    } catch (error) {
      this.sendError(requestId, error instanceof Error ? error.message : String(error));
    }
  }

  private async handleCancel(requestId: string, payload: any): Promise<void> {
    try {
      const targetId = payload.targetId;

      if (targetId && this.activeRequests.has(targetId)) {
        this.activeRequests.get(targetId)?.abort();
        this.activeRequests.delete(targetId);
        this.send({ id: `${requestId}_cancel_success`, type: StreamingMessageType.CANCEL, payload: { cancelled: targetId, success: true }, timestamp: Date.now(), metadata: { source: 'local_llm' } });
      } else {
        this.sendError(requestId, `Request ${targetId} not found or already completed`);
      }
    } catch (error) {
      this.sendError(requestId, error instanceof Error ? error.message : String(error));
    }
  }

  private async handleHeartbeat(requestId: string): Promise<void> {
    this.send({
      id: `${requestId}_heartbeat`,
      type: StreamingMessageType.HEARTBEAT,
      payload: { timestamp: Date.now(), activeRequests: this.activeRequests.size, sessionId: this.sessionId },
      timestamp: Date.now(),
      metadata: { source: 'local_llm' },
    });
  }

  private send(message: StreamingMessage): void {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  private sendError(requestId: string, message: string, code: string = 'STREAMING_ERROR'): void {
    const error: StreamingError = { code, message, recoverable: true };
    this.send({ id: `${requestId}_error`, type: StreamingMessageType.ERROR, payload: error, timestamp: Date.now(), metadata: { source: 'local_llm' } });
  }

  private async processSTTChunk(_chunk: VoiceSTTChunk): Promise<{ text: string; confidence: number; isFinal: boolean } | null> {
    return { text: 'STT not yet implemented', confidence: 0.0, isFinal: false };
  }

  private async processTTSRequest(_request: VoiceTTSRequest): Promise<Array<{ audioData: string; format: string; sampleRate: number; channels: number; duration: number; isLast: boolean }>> {
    return [];
  }

  getConversationContext(): ConversationContext {
    return this.conversationContext;
  }

  cleanup(): void {
    for (const [, controller] of this.activeRequests) controller.abort();
    this.activeRequests.clear();
    llmStreamingRouter.cleanup();
  }
}
