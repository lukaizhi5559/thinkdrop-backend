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
  PROVIDER_CONFIG,
  detectTaskType,
  getFallbackChain,
  getProviderModels,
  getProviderBaseURL,
  TaskType,
} from './providerConfig';
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

/**
 * Stream watchdog — wraps an async iterable and throws if no value is yielded
 * within `timeoutMs` of the previous value (or the first value). This catches
 * stalled streams where the connection is open but no data flows (e.g. SambaNova
 * returning 0 chars after 56s). The caller's catch block triggers fallback.
 */
async function* withStreamWatchdog<T>(
  iterable: AsyncIterable<T>,
  timeoutMs = 15_000
): AsyncIterable<T> {
  const iterator = iterable[Symbol.asyncIterator]();
  while (true) {
    const result = await Promise.race([
      iterator.next(),
      new Promise<IteratorResult<T>>((_, reject) =>
        setTimeout(() => reject(new Error(`Stream watchdog: no chunk in ${timeoutMs / 1000}s`)), timeoutMs)
      ),
    ]);
    if (result.done) break;
    yield result.value;
  }
}

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

    // Detect task type for adaptive routing
    const taskType = detectTaskType(
      metadata?.clientId,
      (metadata as any)?.source,
      prompt.length,
      (options as any)?.taskType
    );
    logger.info(`[StreamingRouter] Task type: ${taskType} (prompt ${prompt.length} chars)`);

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

      const callerTemperature = typeof options.temperature === 'number' ? options.temperature : undefined;
      const callerMaxTokens = typeof options.maxTokens === 'number' ? options.maxTokens : undefined;

      // Handle 'auto' provider — use adaptive routing
      // Handle explicit preferred provider (backward compatibility)
      const effectiveProvider = preferredProvider === 'auto' ? undefined : preferredProvider;

      if (effectiveProvider && this.isProviderConfigured(effectiveProvider)) {
        if (providerCircuitBreaker.isOpen(effectiveProvider)) {
          logger.warn(`[StreamingRouter] Preferred provider ${effectiveProvider} is circuit-broken (rate-limited) — skipping to fallback`);
        } else {
          try {
            const result = await this.callProviderWithStreaming(
              effectiveProvider,
              prompt,
              enrichedSystemInstructions,
              handleChunk,
              abortController.signal,
              startTime,
              callerTemperature,
              callerMaxTokens,
              taskType
            );
            logger.info(`[StreamingRouter] Preferred provider ${effectiveProvider} succeeded`, {
              provider: effectiveProvider,
              inputChars: prompt.length,
              outputChars: result.fullText.length,
              processingTimeMs: Math.round(result.processingTime),
              tokensPerSec: result.processingTime > 0
                ? Math.round((result.fullText.length / 4) / (result.processingTime / 1000))
                : 0,
              tokenUsage: result.tokenUsage,
            });
            // Empty response detection — validate BEFORE assigning to streamResult
            // so a stale empty result is never returned as success after fallback
            if (!result.fullText.trim()) {
              throw new Error(`${effectiveProvider} returned empty response`);
            }
            streamResult = result;
            const estTokensPref = result.tokenUsage.totalTokens || Math.ceil((prompt.length + result.fullText.length) / 4);
            providerCircuitBreaker.recordSuccess(effectiveProvider, estTokensPref);
          } catch (err) {
            const errMsg = err instanceof Error ? err.message : String(err);
            providerCircuitBreaker.recordFailure(effectiveProvider, errMsg);
            logger.warn(`[StreamingRouter] Preferred provider ${effectiveProvider} failed`, { error: errMsg });
            onChunk({
              id: `${streamId}_fallback`,
              type: StreamingMessageType.LLM_STREAM_FALLBACK,
              payload: { failedProvider: effectiveProvider, reason: errMsg },
              timestamp: Date.now(),
              parentId: streamId,
              metadata,
            });
          }
        }
      } else if (effectiveProvider && !this.isProviderConfigured(effectiveProvider)) {
        logger.warn(`[StreamingRouter] Preferred provider ${effectiveProvider} not configured, going to fallback`);
      }

      if (!streamResult) {
        // Use adaptive fallback chain based on task type, with round-robin rotation
        const baseChain = getFallbackChain(taskType);
        const fallbackChain = providerCircuitBreaker.getRotatedChain(baseChain);
        for (const provider of fallbackChain) {
          if (provider === effectiveProvider) continue;
          if (abortController.signal.aborted) break;
          if (!this.isProviderConfigured(provider)) {
            logger.debug(`[StreamingRouter] Skipping unconfigured provider: ${provider}`);
            continue;
          }
          if (providerCircuitBreaker.isOpen(provider)) {
            logger.debug(`[StreamingRouter] Skipping circuit-broken provider: ${provider}`);
            continue;
          }
          if (providerCircuitBreaker.isTpdExhausted(provider)) {
            logger.debug(`[StreamingRouter] Skipping TPD-exhausted provider: ${provider}`, providerCircuitBreaker.getTpdUsage(provider));
            continue;
          }

          // Intra-provider model fallback: try each model in the provider's chain
          const models = getProviderModels(provider, taskType);
          let providerSucceeded = false;
          for (const model of models) {
            if (abortController.signal.aborted) break;
            try {
              logger.info(`[StreamingRouter] Trying provider: ${provider} (model: ${model.id})`);
              const result = await this.callProviderWithStreaming(
                provider,
                prompt,
                enrichedSystemInstructions,
                handleChunk,
                abortController.signal,
                startTime,
                callerTemperature,
                callerMaxTokens,
                taskType,
                model.id
              );
              logger.info(`[StreamingRouter] Provider ${provider} succeeded`, {
                provider,
                model: model.id,
                inputChars: prompt.length,
                outputChars: result.fullText.length,
                processingTimeMs: Math.round(result.processingTime),
                tokensPerSec: result.processingTime > 0
                  ? Math.round((result.fullText.length / 4) / (result.processingTime / 1000))
                  : 0,
                tokenUsage: result.tokenUsage,
              });
              // Empty response detection — validate BEFORE assigning to streamResult
              if (!result.fullText.trim()) {
                throw new Error(`${provider}/${model.id} returned empty response`);
              }
              streamResult = result;
              const estTokens = result.tokenUsage.totalTokens || Math.ceil((prompt.length + result.fullText.length) / 4);
              providerCircuitBreaker.recordSuccess(provider, estTokens);
              providerSucceeded = true;
              break;
            } catch (err) {
              const errMsg = err instanceof Error ? err.message : String(err);
              providerCircuitBreaker.recordFailure(provider, errMsg);
              logger.warn(`[StreamingRouter] Provider ${provider} model ${model.id} failed`, { error: errMsg });
              // Try next model in this provider
            }
          }
          if (providerSucceeded) break;
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
    startTime: number,
    temperature?: number,
    maxTokens?: number,
    taskType: TaskType = 'heavy',
    modelId?: string
  ): Promise<LLMStreamResult> {
    // Resolve model ID: explicit override > first model for task type
    const resolvedModel = modelId || getProviderModels(provider, taskType)[0]?.id;
    if (!resolvedModel) throw new Error(`No models configured for provider: ${provider}`);

    switch (provider) {
      case 'glm':
        return this.callGLMWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime, temperature, maxTokens, resolvedModel);
      case 'groq':
        return this.callGroqWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime, temperature, maxTokens, resolvedModel);
      case 'sambanova':
        return this.callSambanovaWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime, temperature, maxTokens, resolvedModel);
      case 'nvidia':
        return this.callNvidiaWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime, temperature, maxTokens, resolvedModel);
      case 'cloudflare':
        return this.callCloudflareWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime, temperature, maxTokens, resolvedModel);
      case 'claude':
        return this.callClaudeWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime, temperature, maxTokens, resolvedModel);
      case 'openai':
        return this.callOpenAIWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime, temperature, maxTokens, resolvedModel);
      case 'grok':
        return this.callGrokWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime, temperature, maxTokens, resolvedModel);
      case 'gemini-free':
      case 'gemini-paid':
        return this.callGeminiWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime, temperature, maxTokens, resolvedModel, provider);
      case 'mistral':
        return this.callMistralWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime, temperature, maxTokens, resolvedModel);
      case 'deepseek':
        return this.callDeepseekWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime, temperature, maxTokens, resolvedModel);
      default:
        throw new Error(`Unknown provider: ${provider}`);
    }
  }

  private async callGLMWithStreaming(
    prompt: string,
    systemInstructions: string | undefined,
    onChunk: (chunk: LLMStreamChunk) => void,
    abortSignal: AbortSignal,
    startTime: number,
    temperature?: number,
    maxTokens?: number,
    modelId: string = 'glm-4.7-flash'
  ): Promise<LLMStreamResult> {
    const apiKey = process.env.GLM_API_KEY;
    if (!apiKey) throw new Error('GLM_API_KEY not configured');

    const glm = new OpenAI({ apiKey, baseURL: getProviderBaseURL('glm'), timeout: 30_000, maxRetries: 0 });
    let fullText = '';
    let tokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    const messages: Array<{ role: 'system' | 'user'; content: string }> = [];
    if (systemInstructions) messages.push({ role: 'system', content: systemInstructions });
    messages.push({ role: 'user', content: prompt });

    // GLM-4.7 uses forced thinking mode by default — reasoning tokens go to
    // delta.reasoning_content (which we don't read) and can exhaust max_tokens
    // before any content is produced. These are instruction-following tasks;
    // disable thinking for direct, fast answers.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const glmParams: any = {
      model: modelId,
      messages,
      stream: true,
      temperature: temperature ?? 0.7,
      max_tokens: maxTokens ?? 4096,
      thinking: { type: 'disabled' },
    };
    const stream = await glm.chat.completions.create(glmParams) as unknown as AsyncIterable<import('openai/resources/chat/completions').ChatCompletionChunk>;

    for await (const chunk of withStreamWatchdog(stream)) {
      if (abortSignal.aborted) break;

      const content = chunk.choices[0]?.delta?.content;
      if (content) {
        fullText += content;
        onChunk({
          text: content,
          provider: 'glm',
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

    return { fullText, provider: 'glm', processingTime: performance.now() - startTime, tokenUsage };
  }

  private async callClaudeWithStreaming(
    prompt: string,
    systemInstructions: string | undefined,
    onChunk: (chunk: LLMStreamChunk) => void,
    abortSignal: AbortSignal,
    startTime: number,
    temperature?: number,
    maxTokens?: number,
    modelId: string = 'claude-sonnet-4-20250514'
  ): Promise<LLMStreamResult> {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) throw new Error('ANTHROPIC_API_KEY not configured');

    const anthropic = new Anthropic({ apiKey, timeout: 30_000, maxRetries: 0 });
    let fullText = '';
    let tokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    const stream = await anthropic.messages.create({
      model: modelId,
      max_tokens: maxTokens ?? 4096,
      ...(systemInstructions ? { system: systemInstructions } : {}),
      ...(temperature !== undefined ? { temperature } : {}),
      messages: [{ role: 'user', content: prompt }],
      stream: true,
    });

    for await (const chunk of withStreamWatchdog(stream)) {
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
    startTime: number,
    temperature?: number,
    maxTokens?: number,
    modelId: string = 'gpt-4o'
  ): Promise<LLMStreamResult> {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) throw new Error('OPENAI_API_KEY not configured');

    const openai = new OpenAI({ apiKey, timeout: 30_000, maxRetries: 0 });
    let fullText = '';
    let tokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    const messages: Array<{ role: 'system' | 'user'; content: string }> = [];
    if (systemInstructions) messages.push({ role: 'system', content: systemInstructions });
    messages.push({ role: 'user', content: prompt });

    const stream = await openai.chat.completions.create({
      model: modelId,
      messages,
      stream: true,
      temperature: temperature ?? 0.7,
      max_tokens: maxTokens ?? 4096,
    });

    for await (const chunk of withStreamWatchdog(stream)) {
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
    startTime: number,
    temperature?: number,
    maxTokens?: number,
    modelId: string = 'llama-3.3-70b-versatile'
  ): Promise<LLMStreamResult> {
    const apiKey = process.env.GROQ_API_KEY;
    if (!apiKey) throw new Error('GROQ_API_KEY not configured');

    const groq = new OpenAI({ apiKey, baseURL: getProviderBaseURL('groq'), timeout: 30_000, maxRetries: 0 });
    let fullText = '';
    let tokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    const messages: Array<{ role: 'system' | 'user'; content: string }> = [];
    if (systemInstructions) messages.push({ role: 'system', content: systemInstructions });
    messages.push({ role: 'user', content: prompt });

    const stream = await groq.chat.completions.create({
      model: modelId,
      messages,
      stream: true,
      temperature: temperature ?? 0.7,
      max_tokens: maxTokens ?? 4096,
    });

    for await (const chunk of withStreamWatchdog(stream)) {
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

  private async callSambanovaWithStreaming(
    prompt: string,
    systemInstructions: string | undefined,
    onChunk: (chunk: LLMStreamChunk) => void,
    abortSignal: AbortSignal,
    startTime: number,
    temperature?: number,
    maxTokens?: number,
    modelId: string = 'gpt-oss-120b'
  ): Promise<LLMStreamResult> {
    const apiKey = process.env.SAMBANOVA_API_KEY;
    if (!apiKey) throw new Error('SAMBANOVA_API_KEY not configured');

    const client = new OpenAI({ apiKey, baseURL: getProviderBaseURL('sambanova'), timeout: 30_000, maxRetries: 0 });
    let fullText = '';
    let tokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    const messages: Array<{ role: 'system' | 'user'; content: string }> = [];
    if (systemInstructions) messages.push({ role: 'system', content: systemInstructions });
    messages.push({ role: 'user', content: prompt });

    const stream = await client.chat.completions.create({
      model: modelId,
      messages,
      stream: true,
      temperature: temperature ?? 0.7,
      max_tokens: maxTokens ?? 4096,
    });

    for await (const chunk of withStreamWatchdog(stream)) {
      if (abortSignal.aborted) break;

      const content = chunk.choices[0]?.delta?.content;
      if (content) {
        fullText += content;
        onChunk({
          text: content,
          provider: 'sambanova',
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

    return { fullText, provider: 'sambanova', processingTime: performance.now() - startTime, tokenUsage };
  }

  private async callNvidiaWithStreaming(
    prompt: string,
    systemInstructions: string | undefined,
    onChunk: (chunk: LLMStreamChunk) => void,
    abortSignal: AbortSignal,
    startTime: number,
    temperature?: number,
    maxTokens?: number,
    modelId: string = 'z-ai/glm-5.2'
  ): Promise<LLMStreamResult> {
    const apiKey = process.env.NVIDIA_API_KEY;
    if (!apiKey) throw new Error('NVIDIA_API_KEY not configured');

    const client = new OpenAI({ apiKey, baseURL: getProviderBaseURL('nvidia'), timeout: 30_000, maxRetries: 0 });
    let fullText = '';
    let tokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    const messages: Array<{ role: 'system' | 'user'; content: string }> = [];
    if (systemInstructions) messages.push({ role: 'system', content: systemInstructions });
    messages.push({ role: 'user', content: prompt });

    const stream = await client.chat.completions.create({
      model: modelId,
      messages,
      stream: true,
      temperature: temperature ?? 0.7,
      max_tokens: maxTokens ?? 4096,
    });

    for await (const chunk of withStreamWatchdog(stream)) {
      if (abortSignal.aborted) break;

      const content = chunk.choices[0]?.delta?.content;
      if (content) {
        fullText += content;
        onChunk({
          text: content,
          provider: 'nvidia',
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

    return { fullText, provider: 'nvidia', processingTime: performance.now() - startTime, tokenUsage };
  }

  private async callCloudflareWithStreaming(
    prompt: string,
    systemInstructions: string | undefined,
    onChunk: (chunk: LLMStreamChunk) => void,
    abortSignal: AbortSignal,
    startTime: number,
    temperature?: number,
    maxTokens?: number,
    modelId: string = '@cf/zai-org/glm-4.7-flash'
  ): Promise<LLMStreamResult> {
    const apiToken = process.env.CLOUDFLARE_API_TOKEN;
    const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;
    if (!apiToken) throw new Error('CLOUDFLARE_API_TOKEN not configured');
    if (!accountId) throw new Error('CLOUDFLARE_ACCOUNT_ID not configured');

    const baseURL = getProviderBaseURL('cloudflare');
    const client = new OpenAI({ apiKey: apiToken, baseURL, timeout: 30_000, maxRetries: 0 });
    let fullText = '';
    const tokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    const messages: Array<{ role: 'system' | 'user'; content: string }> = [];
    if (systemInstructions) messages.push({ role: 'system', content: systemInstructions });
    messages.push({ role: 'user', content: prompt });

    const stream = await client.chat.completions.create({
      model: modelId,
      messages,
      stream: true,
      temperature: temperature ?? 0.7,
      max_tokens: maxTokens ?? 4096,
    });

    for await (const chunk of withStreamWatchdog(stream)) {
      if (abortSignal.aborted) break;

      const content = chunk.choices[0]?.delta?.content;
      if (content) {
        fullText += content;
        onChunk({
          text: content,
          provider: 'cloudflare',
          tokenCount: content.split(' ').length,
          finishReason: (chunk.choices[0]?.finish_reason as any) || null,
        });
      }

      // Cloudflare Workers AI does not return token usage; values stay 0.
    }

    return { fullText, provider: 'cloudflare', processingTime: performance.now() - startTime, tokenUsage };
  }

  private async callGeminiWithStreaming(
    prompt: string,
    systemInstructions: string | undefined,
    onChunk: (chunk: LLMStreamChunk) => void,
    abortSignal: AbortSignal,
    startTime: number,
    temperature?: number,
    maxTokens?: number,
    modelId: string = 'gemini-3.5-flash-lite',
    provider: string = 'gemini-free'
  ): Promise<LLMStreamResult> {
    const envKey = PROVIDER_CONFIG[provider]?.envKey || 'GEMINI_API_KEY_FREE';
    const apiKey = process.env[envKey];
    if (!apiKey) throw new Error(`${envKey} not configured`);

    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({
      model: modelId,
      ...(systemInstructions ? { systemInstruction: systemInstructions } : {}),
    });
    let fullText = '';
    const tokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    // GoogleGenerativeAI SDK doesn't support timeout natively — race against a 30s timer
    const result = await Promise.race([
      model.generateContentStream({
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: {
          ...(temperature !== undefined ? { temperature } : {}),
          ...(maxTokens !== undefined ? { maxOutputTokens: maxTokens } : {}),
        },
      }),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error('Gemini request timed out after 30s')), 30_000)
      ),
    ]);

    for await (const chunk of withStreamWatchdog(result.stream)) {
      if (abortSignal.aborted) break;
      const chunkText = chunk.text();
      if (chunkText) {
        fullText += chunkText;
        onChunk({ text: chunkText, provider, tokenCount: chunkText.split(' ').length, finishReason: null });
      }
    }

    return { fullText, provider, processingTime: performance.now() - startTime, tokenUsage };
  }

  private async callMistralWithStreaming(
    prompt: string,
    systemInstructions: string | undefined,
    onChunk: (chunk: LLMStreamChunk) => void,
    abortSignal: AbortSignal,
    startTime: number,
    temperature?: number,
    maxTokens?: number,
    modelId: string = 'mistral-medium'
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
      model: modelId,
      messages: mistralMessages as any,
      ...(temperature !== undefined ? { temperature } : {}),
      ...(maxTokens !== undefined ? { maxTokens } : {}),
    });

    for await (const chunk of withStreamWatchdog(stream)) {
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
    startTime: number,
    temperature?: number,
    maxTokens?: number,
    modelId: string = 'grok-4.20-0309-non-reasoning'
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
      { model: modelId, messages: grokMessages, stream: true, temperature: temperature ?? 0.7, max_tokens: maxTokens ?? 4096 },
      { headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' }, responseType: 'stream', signal: abortSignal, timeout: 30_000 }
    );

    return new Promise((resolve, reject) => {
      let watchdog = setTimeout(() => reject(new Error('Grok stream watchdog: no data in 15s')), 15_000);
      response.data.on('data', (chunk: Buffer) => {
        clearTimeout(watchdog);
        watchdog = setTimeout(() => reject(new Error('Grok stream watchdog: no data in 15s')), 15_000);
        if (abortSignal.aborted) return;
        const lines = chunk.toString().split('\n').filter((l: string) => l.trim());
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.substring(6);
          if (data === '[DONE]') { clearTimeout(watchdog); resolve({ fullText, provider: 'grok', processingTime: performance.now() - startTime, tokenUsage }); return; }
          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices[0]?.delta?.content;
            if (content) { fullText += content; onChunk({ text: content, provider: 'grok', tokenCount: content.split(' ').length, finishReason: parsed.choices[0]?.finish_reason || null }); }
            if (parsed.usage) tokenUsage = { promptTokens: parsed.usage.prompt_tokens, completionTokens: parsed.usage.completion_tokens, totalTokens: parsed.usage.total_tokens };
          } catch { /* skip malformed */ }
        }
      });
      response.data.on('error', (err: Error) => { clearTimeout(watchdog); reject(err); });
      response.data.on('end', () => { clearTimeout(watchdog); resolve({ fullText, provider: 'grok', processingTime: performance.now() - startTime, tokenUsage }); });
    });
  }

  private async callDeepseekWithStreaming(
    prompt: string,
    systemInstructions: string | undefined,
    onChunk: (chunk: LLMStreamChunk) => void,
    abortSignal: AbortSignal,
    startTime: number,
    temperature?: number,
    maxTokens?: number,
    modelId: string = 'deepseek-chat'
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
      { model: modelId, messages: deepseekMessages, stream: true, temperature: temperature ?? 0.7, max_tokens: maxTokens ?? 4096 },
      { headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' }, responseType: 'stream', signal: abortSignal, timeout: 30_000 }
    );

    return new Promise((resolve, reject) => {
      let watchdog = setTimeout(() => reject(new Error('DeepSeek stream watchdog: no data in 15s')), 15_000);
      response.data.on('data', (chunk: Buffer) => {
        clearTimeout(watchdog);
        watchdog = setTimeout(() => reject(new Error('DeepSeek stream watchdog: no data in 15s')), 15_000);
        if (abortSignal.aborted) return;
        const lines = chunk.toString().split('\n').filter((l: string) => l.trim());
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.substring(6);
          if (data === '[DONE]') { clearTimeout(watchdog); resolve({ fullText, provider: 'deepseek', processingTime: performance.now() - startTime, tokenUsage }); return; }
          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices[0]?.delta?.content;
            if (content) { fullText += content; onChunk({ text: content, provider: 'deepseek', tokenCount: content.split(' ').length, finishReason: parsed.choices[0]?.finish_reason || null }); }
          } catch { /* skip malformed */ }
        }
      });
      response.data.on('error', (err: Error) => { clearTimeout(watchdog); reject(err); });
      response.data.on('end', () => { clearTimeout(watchdog); resolve({ fullText, provider: 'deepseek', processingTime: performance.now() - startTime, tokenUsage }); });
    });
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
