/**
 * Streaming LLM router — dispatches prompts to LLM providers with streaming support.
 * Pure pass-through: does NOT inject personas or build prompts.
 * The stategraph is responsible for all prompt construction.
 */

import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { Mistral } from '@mistralai/mistralai';
import axios from 'axios';
import { LLMRouter } from './llmRouter';
import { providerCircuitBreaker } from './providerCircuitBreaker';
import {
  StreamingMessage,
  StreamingMessageType,
  LLMStreamRequest,
  LLMStreamChunk,
  LLMStreamResult,
  StreamingError,
  StreamingMetadata,
} from '../types/streaming';
import { logger } from './logger';

export class LLMStreamingRouter extends LLMRouter {
  private activeStreams: Map<string, AbortController> = new Map();

  async processPromptWithStreaming(
    request: LLMStreamRequest,
    onChunk: (chunk: StreamingMessage) => void,
    metadata: StreamingMetadata
  ): Promise<LLMStreamResult> {
    const { prompt, provider: preferredProvider, options = {}, context } = request;

    // Context serialization is done upstream in streamingHandler.buildEnrichedPrompt().
    // The prompt already contains all context (memories, history, etc.) embedded.
    // systemInstructions from context is passed as the system message to each provider.
    const enrichedSystemInstructions = context?.systemInstructions?.trim() || undefined;

    const streamId = `stream_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const startTime = performance.now();

    const abortController = new AbortController();
    this.activeStreams.set(streamId, abortController);

    try {
      onChunk({
        id: streamId,
        type: StreamingMessageType.LLM_STREAM_START,
        payload: {
          prompt: prompt.substring(0, 100) + '...',
          preferredProvider,
          options,
        },
        timestamp: Date.now(),
        metadata,
      });

      let streamResult: LLMStreamResult | undefined;

      const handleChunk = (chunk: LLMStreamChunk) => {
        if (abortController.signal.aborted) return;

        onChunk({
          id: `${streamId}_chunk_${Date.now()}`,
          type: StreamingMessageType.LLM_STREAM_CHUNK,
          payload: chunk,
          timestamp: Date.now(),
          parentId: streamId,
          metadata: { ...metadata, provider: chunk.provider },
        });
      };

      if (preferredProvider && this.isProviderConfigured(preferredProvider)) {
        if (providerCircuitBreaker.isOpen(preferredProvider)) {
          logger.warn(`[StreamingRouter] Preferred provider ${preferredProvider} is circuit-broken (rate-limited) — skipping to fallback`);
        } else {
          try {
            streamResult = await this.callProviderWithStreaming(
              preferredProvider,
              prompt,
              enrichedSystemInstructions,
              handleChunk,
              abortController.signal,
              startTime
            );
          } catch (err) {
            const errMsg = err instanceof Error ? err.message : String(err);
            providerCircuitBreaker.recordFailure(preferredProvider, errMsg);
            logger.warn(`[StreamingRouter] Preferred provider ${preferredProvider} failed`, { error: errMsg });
            onChunk({
              id: `${streamId}_fallback`,
              type: StreamingMessageType.LLM_STREAM_FALLBACK,
              payload: { failedProvider: preferredProvider, reason: errMsg },
              timestamp: Date.now(),
              parentId: streamId,
              metadata,
            });
          }
        }
      } else if (preferredProvider && !this.isProviderConfigured(preferredProvider)) {
        logger.warn(`[StreamingRouter] Preferred provider ${preferredProvider} not configured, going to fallback`);
      }

      if (!streamResult) {
        const fallbackChain = ['groq', 'openai', 'claude', 'gemini', 'mistral', 'deepseek', 'grok'];
        for (const provider of fallbackChain) {
          if (provider === preferredProvider) continue;
          if (abortController.signal.aborted) break;
          if (!this.isProviderConfigured(provider)) {
            logger.debug(`[StreamingRouter] Skipping unconfigured provider: ${provider}`);
            continue;
          }
          if (providerCircuitBreaker.isOpen(provider)) {
            logger.debug(`[StreamingRouter] Skipping circuit-broken provider: ${provider}`);
            continue;
          }

          try {
            logger.info(`[StreamingRouter] Trying provider: ${provider}`);
            streamResult = await this.callProviderWithStreaming(
              provider,
              prompt,
              enrichedSystemInstructions,
              handleChunk,
              abortController.signal,
              startTime
            );
            break;
          } catch (err) {
            const errMsg = err instanceof Error ? err.message : String(err);
            providerCircuitBreaker.recordFailure(provider, errMsg);
            logger.warn(`[StreamingRouter] Provider ${provider} failed`, { error: errMsg });
          }
        }
      }

      if (!streamResult) {
        throw new Error('All LLM providers failed for streaming request');
      }

      onChunk({
        id: `${streamId}_end`,
        type: StreamingMessageType.LLM_STREAM_END,
        payload: {
          fullText: streamResult.fullText,
          provider: streamResult.provider,
          processingTime: streamResult.processingTime,
          tokenUsage: streamResult.tokenUsage,
          fallbackChain: streamResult.fallbackChain,
        },
        timestamp: Date.now(),
        parentId: streamId,
        metadata: { ...metadata, provider: streamResult.provider },
      });

      return streamResult;
    } catch (error) {
      const streamingError: StreamingError = {
        code: 'STREAMING_ERROR',
        message: error instanceof Error ? error.message : String(error),
        recoverable: true,
        provider: preferredProvider,
      };

      onChunk({
        id: `${streamId}_error`,
        type: StreamingMessageType.LLM_ERROR,
        payload: streamingError,
        timestamp: Date.now(),
        parentId: streamId,
        metadata,
      });

      throw error;
    } finally {
      this.activeStreams.delete(streamId);
    }
  }

  private async callProviderWithStreaming(
    provider: string,
    prompt: string,
    systemInstructions: string | undefined,
    onChunk: (chunk: LLMStreamChunk) => void,
    abortSignal: AbortSignal,
    startTime: number
  ): Promise<LLMStreamResult> {
    switch (provider) {
      case 'groq':
        return this.callGroqWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime);
      case 'claude':
        return this.callClaudeWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime);
      case 'openai':
        return this.callOpenAIWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime);
      case 'grok':
        return this.callGrokWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime);
      case 'gemini':
        return this.callGeminiWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime);
      case 'mistral':
        return this.callMistralWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime);
      case 'deepseek':
        return this.callDeepseekWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime);
      case 'lambda':
        throw new Error('Lambda provider removed from fallback chain');
      default:
        throw new Error(`Unknown provider: ${provider}`);
    }
  }

  private async callClaudeWithStreaming(
    prompt: string,
    systemInstructions: string | undefined,
    onChunk: (chunk: LLMStreamChunk) => void,
    abortSignal: AbortSignal,
    startTime: number
  ): Promise<LLMStreamResult> {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) throw new Error('ANTHROPIC_API_KEY not configured');

    const anthropic = new Anthropic({ apiKey });
    let fullText = '';
    let tokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    const stream = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 4096,
      ...(systemInstructions ? { system: systemInstructions } : {}),
      messages: [{ role: 'user', content: prompt }],
      stream: true,
    });

    for await (const chunk of stream) {
      if (abortSignal.aborted) break;

      if (chunk.type === 'content_block_delta' && chunk.delta && 'text' in chunk.delta) {
        const text = (chunk.delta as any).text;
        fullText += text;
        onChunk({ text, provider: 'claude', tokenCount: text.split(' ').length, finishReason: null });
      }

      if (chunk.type === 'message_delta' && chunk.usage) {
        tokenUsage = {
          promptTokens: 0,
          completionTokens: chunk.usage.output_tokens || 0,
          totalTokens: chunk.usage.output_tokens || 0,
        };
      }
    }

    return { fullText, provider: 'claude', processingTime: performance.now() - startTime, tokenUsage };
  }

  private async callOpenAIWithStreaming(
    prompt: string,
    systemInstructions: string | undefined,
    onChunk: (chunk: LLMStreamChunk) => void,
    abortSignal: AbortSignal,
    startTime: number
  ): Promise<LLMStreamResult> {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) throw new Error('OPENAI_API_KEY not configured');

    const openai = new OpenAI({ apiKey });
    let fullText = '';
    let tokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    const messages: Array<{ role: 'system' | 'user'; content: string }> = [];
    if (systemInstructions) messages.push({ role: 'system', content: systemInstructions });
    messages.push({ role: 'user', content: prompt });

    const stream = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages,
      stream: true,
      temperature: 0.7,
      max_tokens: 4096,
    });

    for await (const chunk of stream) {
      if (abortSignal.aborted) break;

      const content = chunk.choices[0]?.delta?.content;
      if (content) {
        fullText += content;
        onChunk({
          text: content,
          provider: 'openai',
          tokenCount: content.split(' ').length,
          finishReason: (chunk.choices[0]?.finish_reason as any) || null,
        });
      }

      if (chunk.usage) {
        tokenUsage = {
          promptTokens: chunk.usage.prompt_tokens,
          completionTokens: chunk.usage.completion_tokens,
          totalTokens: chunk.usage.total_tokens,
        };
      }
    }

    return { fullText, provider: 'openai', processingTime: performance.now() - startTime, tokenUsage };
  }

  private async callGroqWithStreaming(
    prompt: string,
    systemInstructions: string | undefined,
    onChunk: (chunk: LLMStreamChunk) => void,
    abortSignal: AbortSignal,
    startTime: number
  ): Promise<LLMStreamResult> {
    const apiKey = process.env.GROQ_API_KEY;
    if (!apiKey) throw new Error('GROQ_API_KEY not configured');

    const groq = new OpenAI({ apiKey, baseURL: 'https://api.groq.com/openai/v1' });
    let fullText = '';
    let tokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    const messages: Array<{ role: 'system' | 'user'; content: string }> = [];
    if (systemInstructions) messages.push({ role: 'system', content: systemInstructions });
    messages.push({ role: 'user', content: prompt });

    const stream = await groq.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages,
      stream: true,
      temperature: 0.7,
      max_tokens: 4096,
    });

    for await (const chunk of stream) {
      if (abortSignal.aborted) break;

      const content = chunk.choices[0]?.delta?.content;
      if (content) {
        fullText += content;
        onChunk({
          text: content,
          provider: 'groq',
          tokenCount: content.split(' ').length,
          finishReason: (chunk.choices[0]?.finish_reason as any) || null,
        });
      }

      if (chunk.usage) {
        tokenUsage = {
          promptTokens: chunk.usage.prompt_tokens,
          completionTokens: chunk.usage.completion_tokens,
          totalTokens: chunk.usage.total_tokens,
        };
      }
    }

    return { fullText, provider: 'groq', processingTime: performance.now() - startTime, tokenUsage };
  }

  private async callGeminiWithStreaming(
    prompt: string,
    systemInstructions: string | undefined,
    onChunk: (chunk: LLMStreamChunk) => void,
    abortSignal: AbortSignal,
    startTime: number
  ): Promise<LLMStreamResult> {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) throw new Error('GEMINI_API_KEY not configured');

    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({
      model: 'gemini-1.5-flash',
      ...(systemInstructions ? { systemInstruction: systemInstructions } : {}),
    });
    let fullText = '';
    const tokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    const result = await model.generateContentStream({
      contents: [{ role: 'user', parts: [{ text: prompt }] }],
    });

    for await (const chunk of result.stream) {
      if (abortSignal.aborted) break;
      const chunkText = chunk.text();
      if (chunkText) {
        fullText += chunkText;
        onChunk({ text: chunkText, provider: 'gemini', tokenCount: chunkText.split(' ').length, finishReason: null });
      }
    }

    return { fullText, provider: 'gemini', processingTime: performance.now() - startTime, tokenUsage };
  }

  private async callMistralWithStreaming(
    prompt: string,
    systemInstructions: string | undefined,
    onChunk: (chunk: LLMStreamChunk) => void,
    abortSignal: AbortSignal,
    startTime: number
  ): Promise<LLMStreamResult> {
    const apiKey = process.env.MISTRAL_API_KEY;
    if (!apiKey) throw new Error('MISTRAL_API_KEY not configured');

    const client = new Mistral({ apiKey });
    let fullText = '';
    let tokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    const mistralMessages: Array<{ role: string; content: string }> = [];
    if (systemInstructions) mistralMessages.push({ role: 'system', content: systemInstructions });
    mistralMessages.push({ role: 'user', content: prompt });

    const stream = await client.chat.stream({
      model: 'mistral-medium',
      messages: mistralMessages as any,
    });

    for await (const chunk of stream) {
      if (abortSignal.aborted) break;
      const rawContent = chunk.data.choices[0]?.delta?.content;
      const content = typeof rawContent === 'string' ? rawContent : undefined;
      if (content) {
        fullText += content;
        onChunk({
          text: content,
          provider: 'mistral',
          tokenCount: content.split(' ').length,
          finishReason: (chunk.data.choices[0]?.finishReason as any) || null,
        });
      }
      if (chunk.data.usage) {
        tokenUsage = {
          promptTokens: (chunk.data.usage as any).promptTokens ?? (chunk.data.usage as any).prompt_tokens ?? 0,
          completionTokens: (chunk.data.usage as any).completionTokens ?? (chunk.data.usage as any).completion_tokens ?? 0,
          totalTokens: (chunk.data.usage as any).totalTokens ?? (chunk.data.usage as any).total_tokens ?? 0,
        };
      }
    }

    return { fullText, provider: 'mistral', processingTime: performance.now() - startTime, tokenUsage };
  }

  private async callGrokWithStreaming(
    prompt: string,
    systemInstructions: string | undefined,
    onChunk: (chunk: LLMStreamChunk) => void,
    abortSignal: AbortSignal,
    startTime: number
  ): Promise<LLMStreamResult> {
    const apiKey = process.env.GROK_API_KEY;
    if (!apiKey) throw new Error('GROK_API_KEY not configured');

    let fullText = '';
    let tokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    const grokMessages: Array<{ role: string; content: string }> = [];
    if (systemInstructions) grokMessages.push({ role: 'system', content: systemInstructions });
    grokMessages.push({ role: 'user', content: prompt });

    const response = await axios.post(
      'https://api.x.ai/v1/chat/completions',
      { model: 'grok-beta', messages: grokMessages, stream: true, temperature: 0.7, max_tokens: 4096 },
      { headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' }, responseType: 'stream', signal: abortSignal }
    );

    return new Promise((resolve, reject) => {
      response.data.on('data', (chunk: Buffer) => {
        if (abortSignal.aborted) return;
        const lines = chunk.toString().split('\n').filter((l: string) => l.trim());
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.substring(6);
          if (data === '[DONE]') { resolve({ fullText, provider: 'grok', processingTime: performance.now() - startTime, tokenUsage }); return; }
          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices[0]?.delta?.content;
            if (content) { fullText += content; onChunk({ text: content, provider: 'grok', tokenCount: content.split(' ').length, finishReason: parsed.choices[0]?.finish_reason || null }); }
            if (parsed.usage) tokenUsage = { promptTokens: parsed.usage.prompt_tokens, completionTokens: parsed.usage.completion_tokens, totalTokens: parsed.usage.total_tokens };
          } catch { /* skip malformed */ }
        }
      });
      response.data.on('error', reject);
      response.data.on('end', () => resolve({ fullText, provider: 'grok', processingTime: performance.now() - startTime, tokenUsage }));
    });
  }

  private async callDeepseekWithStreaming(
    prompt: string,
    systemInstructions: string | undefined,
    onChunk: (chunk: LLMStreamChunk) => void,
    abortSignal: AbortSignal,
    startTime: number
  ): Promise<LLMStreamResult> {
    const apiKey = process.env.DEEPSEEK_API_KEY;
    if (!apiKey) throw new Error('DEEPSEEK_API_KEY not configured');

    let fullText = '';
    let tokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    const deepseekMessages: Array<{ role: string; content: string }> = [];
    if (systemInstructions) deepseekMessages.push({ role: 'system', content: systemInstructions });
    deepseekMessages.push({ role: 'user', content: prompt });

    const response = await axios.post(
      'https://api.deepseek.com/v1/chat/completions',
      { model: 'deepseek-chat', messages: deepseekMessages, stream: true, temperature: 0.7, max_tokens: 4096 },
      { headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' }, responseType: 'stream', signal: abortSignal }
    );

    return new Promise((resolve, reject) => {
      response.data.on('data', (chunk: Buffer) => {
        if (abortSignal.aborted) return;
        const lines = chunk.toString().split('\n').filter((l: string) => l.trim());
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.substring(6);
          if (data === '[DONE]') { resolve({ fullText, provider: 'deepseek', processingTime: performance.now() - startTime, tokenUsage }); return; }
          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices[0]?.delta?.content;
            if (content) { fullText += content; onChunk({ text: content, provider: 'deepseek', tokenCount: content.split(' ').length, finishReason: parsed.choices[0]?.finish_reason || null }); }
          } catch { /* skip malformed */ }
        }
      });
      response.data.on('error', reject);
      response.data.on('end', () => resolve({ fullText, provider: 'deepseek', processingTime: performance.now() - startTime, tokenUsage }));
    });
  }

  private async callLambdaWithStreaming(
    prompt: string,
    systemInstructions: string | undefined,
    onChunk: (chunk: LLMStreamChunk) => void,
    abortSignal: AbortSignal,
    startTime: number
  ): Promise<LLMStreamResult> {
    const apiKey = process.env.LAMBDA_AI;
    if (!apiKey) throw new Error('LAMBDA_AI not configured');

    let fullText = '';
    const tokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    const response = await axios.post(
      'https://api.lambda.ai/v1/generate/stream',
      { prompt, model: 'lambda-large', stream: true },
      { headers: { 'x-api-key': apiKey, 'Content-Type': 'application/json' }, responseType: 'stream', signal: abortSignal }
    );

    return new Promise((resolve, reject) => {
      response.data.on('data', (chunk: Buffer) => {
        if (abortSignal.aborted) return;
        try {
          const data = JSON.parse(chunk.toString());
          const text = data.text || '';
          if (text) { fullText += text; onChunk({ text, provider: 'lambda', tokenCount: text.split(' ').length, finishReason: data.finish_reason || null }); }
          if (data.done) resolve({ fullText, provider: 'lambda', processingTime: performance.now() - startTime, tokenUsage });
        } catch { /* partial JSON */ }
      });
      response.data.on('error', reject);
      response.data.on('end', () => resolve({ fullText, provider: 'lambda', processingTime: performance.now() - startTime, tokenUsage }));
    });
  }

  private isProviderConfigured(provider: string): boolean {
    switch (provider) {
      case 'openai':   return !!process.env.OPENAI_API_KEY;
      case 'claude':   return !!process.env.ANTHROPIC_API_KEY;
      case 'gemini':   return !!process.env.GEMINI_API_KEY;
      case 'mistral':  return !!process.env.MISTRAL_API_KEY;
      case 'deepseek': return !!process.env.DEEPSEEK_API_KEY;
      case 'groq':     return !!process.env.GROQ_API_KEY;
      case 'grok':     return !!process.env.GROK_API_KEY;
      case 'lambda':   return false;
      default:         return false;
    }
  }

  interruptStream(streamId: string): boolean {
    const controller = this.activeStreams.get(streamId);
    if (controller) {
      controller.abort();
      this.activeStreams.delete(streamId);
      return true;
    }
    return false;
  }

  getActiveStreamCount(): number {
    return this.activeStreams.size;
  }

  cleanup(): void {
    for (const [, controller] of this.activeStreams) {
      controller.abort();
    }
    this.activeStreams.clear();
  }
}

export const llmStreamingRouter = new LLMStreamingRouter();
