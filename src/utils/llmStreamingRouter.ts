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
import { catalogManager } from './catalogManager';
import {
  PROVIDER_CONFIG,
  detectTaskType,
  getFallbackChain,
  getProviderModels,
  getProviderBaseURL,
  getProviderAPIType,
  getProviderEnvKeyDynamic,
  PAID_CHAIN,
  TaskType,
  ProviderModel,
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
 * NVIDIA reasoning models that produce hidden reasoning_content tokens,
 * causing 10-40s delays before visible output. Disable thinking via
 * chat_template_kwargs when possible.
 */
const NVIDIA_REASONING_MODELS = new Set([
  'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',
  'thinkingmachines/inkling',
  'nvidia/nemotron-3-super-120b-a12b',
  'openai/gpt-oss-120b', // very slow on NVIDIA, reasoning model
]);

/**
 * Returns provider-specific params to disable thinking/reasoning mode.
 * - GLM (z.ai): `thinking: { type: 'disabled' }`
 * - NVIDIA reasoning models: `chat_template_kwargs: { enable_thinking: false }`
 * - Others: no params (no reasoning or can't disable)
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getDisableThinkingParams(provider: string, modelId: string): Record<string, any> {
  if (provider === 'glm') {
    return { thinking: { type: 'disabled' } };
  }
  if (provider === 'nvidia' && NVIDIA_REASONING_MODELS.has(modelId)) {
    return { chat_template_kwargs: { enable_thinking: false } };
  }
  return {};
}

/**
 * Per-provider timeout — free providers should be fast, short timeout for quick
 * fallback. Paid providers get more time since they're more reliable but slower.
 */
function getProviderTimeout(provider: string): number {
  const timeouts: Record<string, number> = {
    groq: 10_000,       // Fast — 10s max
    sambanova: 15_000,  // Medium — 15s
    nvidia: 20_000,     // Can be slow — 20s
    glm: 15_000,        // Should be fast — 15s
    cloudflare: 15_000, // Can be slow — 15s
    mistral: 15_000,    // Medium — 15s
  };
  return timeouts[provider] ?? 30_000; // Paid providers get 30s
}

/**
 * Per-provider stream watchdog timeout — how long to wait for the first chunk
 * (or between chunks) before declaring the stream stalled. NVIDIA needs more
 * time because it can take 10-15s to send the first token even for non-reasoning
 * models. Groq is fast and should send the first chunk within 5s.
 */
function getProviderWatchdogTimeout(provider: string): number {
  const timeouts: Record<string, number> = {
    groq: 10_000,       // Fast — 10s watchdog
    sambanova: 15_000,  // Medium — 15s
    nvidia: 20_000,     // Slow first token — 20s
    glm: 15_000,        // 15s
    cloudflare: 15_000, // 15s
    mistral: 15_000,    // 15s
  };
  return timeouts[provider] ?? 15_000; // Default 15s
}

/**
 * Stream watchdog — wraps an async iterable and throws if no value is yielded
 * within `timeoutMs` of the previous value (or the first value). This catches
 * stalled streams where the connection is open but no data flows (e.g. SambaNova
 * returning 0 chars after 56s). The caller's catch block triggers fallback.
 */
async function* withStreamWatchdog<T>(
  iterable: AsyncIterable<T>,
  timeoutMs = 10_000
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
    const isHeartbeat = metadata?.clientId?.startsWith('hb_');
    logger.debug(`[StreamingRouter] Task type: ${taskType} (prompt ${prompt.length} chars)`);

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
          // Circuit-breaker open/close state is logged by providerCircuitBreaker itself —
          // no need to repeat on every request.
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
            // Only measure speed for responses >100 chars — short responses give
            // misleadingly low t/s (e.g. "ok" in 300ms = 1.7 t/s) that would drag
            // down the EMA even for fast models. The probe already sets good initial
            // speed values; runtime updates should only come from substantial outputs.
            const measuredSpeedPref = result.processingTime > 0 && result.fullText.length > 100
              ? Math.round((result.fullText.length / 4) / (result.processingTime / 1000))
              : 0;
            catalogManager.markSuccess(effectiveProvider, result.modelId || '', measuredSpeedPref, result.processingTime);
            providerCircuitBreaker.recordSuccess(effectiveProvider, estTokensPref);
          } catch (err) {
            const errMsg = err instanceof Error ? err.message : String(err);
            catalogManager.markFailure(effectiveProvider, '', errMsg);
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
        // Use live catalog if loaded, else static config
        const baseChain = catalogManager.isLoaded()
          ? catalogManager.getRankedFallbackChain(taskType)
          : getFallbackChain(taskType);
        // Split at the PAID_CHAIN boundary — rotate free providers only, keep paid as fallback
        const paidStart = baseChain.findIndex(p => (PAID_CHAIN as readonly string[]).includes(p));
        const freeChain = paidStart >= 0 ? baseChain.slice(0, paidStart) : baseChain;
        const paidChain = paidStart >= 0 ? baseChain.slice(paidStart) : [];
        const fallbackChain = [...providerCircuitBreaker.getRotatedChain(freeChain), ...paidChain];
        for (const provider of fallbackChain) {
          if (provider === effectiveProvider) continue;
          if (abortController.signal.aborted) break;
          if (!this.isProviderConfigured(provider)) {
            continue;
          }
          if (providerCircuitBreaker.isOpen(provider)) {
            continue;
          }
          if (providerCircuitBreaker.isTpdExhausted(provider)) {
            continue;
          }

          // Intra-provider model fallback: try each model in the provider's chain
          // Use ranked models from catalog if loaded, else static config
          const models: ProviderModel[] = catalogManager.isLoaded()
            ? catalogManager.getRankedModels(provider, taskType).map(r => r.model)
            : getProviderModels(provider, taskType);
          const estimatedPromptTokens = Math.ceil(prompt.length / 4);
          const maxTok = callerMaxTokens ?? 4096;
          let providerSucceeded = false;
          for (const model of models) {
            if (abortController.signal.aborted) break;
            // Skip models with insufficient context window
            if (model.contextWindow && estimatedPromptTokens + maxTok > model.contextWindow) {
              continue;
            }
            try {
              if (isHeartbeat) {
                logger.debug(`[StreamingRouter] Trying provider: ${provider} (model: ${model.id})`);
              } else {
                logger.info(`[StreamingRouter] Trying provider: ${provider} (model: ${model.id})`);
              }
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
              const successMeta: Record<string, any> = {
                provider,
                model: model.id,
                inputChars: prompt.length,
                outputChars: result.fullText.length,
                processingTimeMs: Math.round(result.processingTime),
                tokensPerSec: result.processingTime > 0
                  ? Math.round((result.fullText.length / 4) / (result.processingTime / 1000))
                  : 0,
              };
              // Only include tokenUsage if it has non-zero values (Gemini streaming always returns 0)
              if (result.tokenUsage && result.tokenUsage.totalTokens > 0) {
                successMeta.tokenUsage = result.tokenUsage;
              }
              if (isHeartbeat) {
                logger.debug(`[StreamingRouter] Provider ${provider} succeeded`, successMeta);
              } else {
                logger.info(`[StreamingRouter] Provider ${provider} succeeded`, successMeta);
              }
              // Empty response detection — validate BEFORE assigning to streamResult
              if (!result.fullText.trim()) {
                throw new Error(`${provider}/${model.id} returned empty response`);
              }
              streamResult = result;
              const estTokens = result.tokenUsage.totalTokens || Math.ceil((prompt.length + result.fullText.length) / 4);
              // Only measure speed for responses >100 chars (see comment above)
              const measuredSpeed = result.processingTime > 0 && result.fullText.length > 100
                ? Math.round((result.fullText.length / 4) / (result.processingTime / 1000))
                : 0;
              catalogManager.markSuccess(provider, model.id, measuredSpeed, result.processingTime);
              providerCircuitBreaker.recordSuccess(provider, estTokens);
              providerSucceeded = true;
              break;
            } catch (err) {
              const errMsg = err instanceof Error ? err.message : String(err);
              catalogManager.markFailure(provider, model.id, errMsg);
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
    // Resolve model ID: explicit override > best ranked model from catalog > first static model
    const resolvedModel = modelId
      || (catalogManager.isLoaded() ? catalogManager.getRankedModels(provider, taskType)[0]?.model.id : undefined)
      || getProviderModels(provider, taskType)[0]?.id;
    if (!resolvedModel) throw new Error(`No models configured for provider: ${provider}`);

    let result: LLMStreamResult;
    switch (provider) {
      case 'claude':
        result = await this.callClaudeWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime, temperature, maxTokens, resolvedModel);
        break;
      case 'gemini-free':
      case 'gemini-paid':
        result = await this.callGeminiWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime, temperature, maxTokens, resolvedModel, provider);
        break;
      case 'mistral':
        result = await this.callMistralWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime, temperature, maxTokens, resolvedModel);
        break;
      case 'glm':
        result = await this.callGLMWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime, temperature, maxTokens, resolvedModel);
        break;
      default: {
        // All openai-compatible providers (groq, sambanova, nvidia, cloudflare, glm,
        // deepseek, grok, openai, and any dynamically added providers)
        const apiType = getProviderAPIType(provider);
        if (apiType === 'openai-compatible') {
          result = await this.callOpenAICompatibleWithStreaming(provider, prompt, systemInstructions, onChunk, abortSignal, startTime, temperature, maxTokens, resolvedModel);
          break;
        }
        throw new Error(`Unknown provider or unsupported apiType: ${provider} (${apiType})`);
      }
    }
    return { ...result, modelId: resolvedModel };
  }

  /**
   * Generic OpenAI-compatible streaming handler.
   * Works with any provider that uses the OpenAI API format (chat.completions.create).
   * Reads baseURL + envKey dynamically from config or catalog — supports dynamically added providers.
   */
  private async callOpenAICompatibleWithStreaming(
    provider: string,
    prompt: string,
    systemInstructions: string | undefined,
    onChunk: (chunk: LLMStreamChunk) => void,
    abortSignal: AbortSignal,
    startTime: number,
    temperature?: number,
    maxTokens?: number,
    modelId: string = ''
  ): Promise<LLMStreamResult> {
    const envKey = getProviderEnvKeyDynamic(provider);
    const apiKey = process.env[envKey];
    if (!apiKey) throw new Error(`${envKey} not configured for provider: ${provider}`);

    const baseURL = getProviderBaseURL(provider);
    if (!baseURL) throw new Error(`No baseURL for provider: ${provider}`);

    const client = new OpenAI({ apiKey, baseURL, timeout: getProviderTimeout(provider), maxRetries: 0 });
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
      ...getDisableThinkingParams(provider, modelId),
    });

    for await (const chunk of withStreamWatchdog(stream, getProviderWatchdogTimeout(provider))) {
      if (abortSignal.aborted) break;

      const content = chunk.choices[0]?.delta?.content;
      if (content) {
        fullText += content;
        onChunk({ text: content, provider });
      }
      if (chunk.choices[0]?.finish_reason) {
        tokenUsage = {
          promptTokens: chunk.usage?.prompt_tokens || 0,
          completionTokens: chunk.usage?.completion_tokens || 0,
          totalTokens: chunk.usage?.total_tokens || 0,
        };
      }
    }

    if (!fullText.trim()) {
      throw new Error(`${provider}/${modelId} returned empty response`);
    }

    return { fullText, provider, processingTime: performance.now() - startTime, tokenUsage };
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

    const glm = new OpenAI({ apiKey, baseURL: getProviderBaseURL('glm'), timeout: getProviderTimeout('glm'), maxRetries: 0 });
    let fullText = '';
    let tokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    const messages: Array<{ role: 'system' | 'user'; content: string }> = [];
    if (systemInstructions) messages.push({ role: 'system', content: systemInstructions });
    messages.push({ role: 'user', content: prompt });

    // GLM-4.7 uses forced thinking mode by default — reasoning tokens go to
    // delta.reasoning_content (which we don't read) and can exhaust max_tokens
    // before any content is produced. Disable thinking for direct, fast answers.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const glmParams: any = {
      model: modelId,
      messages,
      stream: true,
      temperature: temperature ?? 0.7,
      max_tokens: maxTokens ?? 4096,
      ...getDisableThinkingParams('glm', modelId),
    };
    const stream = await glm.chat.completions.create(glmParams) as unknown as AsyncIterable<import('openai/resources/chat/completions').ChatCompletionChunk>;

    for await (const chunk of withStreamWatchdog(stream, getProviderWatchdogTimeout('glm'))) {
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

    const groq = new OpenAI({ apiKey, baseURL: getProviderBaseURL('groq'), timeout: getProviderTimeout('groq'), maxRetries: 0 });
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

    for await (const chunk of withStreamWatchdog(stream, getProviderWatchdogTimeout('groq'))) {
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

    const client = new OpenAI({ apiKey, baseURL: getProviderBaseURL('sambanova'), timeout: getProviderTimeout('sambanova'), maxRetries: 0 });
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

    for await (const chunk of withStreamWatchdog(stream, getProviderWatchdogTimeout('sambanova'))) {
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

    const client = new OpenAI({ apiKey, baseURL: getProviderBaseURL('nvidia'), timeout: getProviderTimeout('nvidia'), maxRetries: 0 });
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
      ...getDisableThinkingParams('nvidia', modelId),
    });

    for await (const chunk of withStreamWatchdog(stream, getProviderWatchdogTimeout('nvidia'))) {
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
    const client = new OpenAI({ apiKey: apiToken, baseURL, timeout: getProviderTimeout('cloudflare'), maxRetries: 0 });
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

    for await (const chunk of withStreamWatchdog(stream, getProviderWatchdogTimeout('cloudflare'))) {
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

    for await (const chunk of withStreamWatchdog(stream, getProviderWatchdogTimeout('mistral'))) {
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
