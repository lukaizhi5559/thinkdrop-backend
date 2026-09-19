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
  sanitizePrompt,
  resolveReasoningParams,
  getDisableThinkingParams,
  ThinkStripper,
  isCannedRefusal,
  couldBeRefusalPrefix,
  REFUSAL_MAX_LEN,
  RefusalError,
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
 * Best-effort HTTP status extraction from a provider error — SDK errors carry
 * .status/.statusCode/.response.status; message-only errors embed "404" /
 * "status code 429" style text. markFailure needs the real code so its 404/403
 * immediate-dead logic actually fires (it previously always got undefined).
 */
function statusCodeOf(err: unknown, errMsg: string): number | undefined {
  const e = err as { status?: number; statusCode?: number; response?: { status?: number } } | undefined;
  const s = e?.status ?? e?.statusCode ?? e?.response?.status;
  if (typeof s === 'number' && s >= 100 && s < 600) return s;
  const m = /\b(?:HTTP|status|code|error)[\s:_-]*(\d{3})\b/i.exec(errMsg) || /\b([45]\d{2})\b/.exec(errMsg);
  return m ? parseInt(m[1], 10) : undefined;
}

/**
 * Per-provider timeout — free providers should be fast, short timeout for quick
 * fallback. Paid providers get more time since they're more reliable but slower.
 * Scales by taskType: complex/super-heavy get 3-4x more time for large models.
 */
function getProviderTimeout(provider: string, taskType: string = 'heavy'): number {
  const timeouts: Record<string, number> = {
    groq: 10_000,       // Fast — 10s max
    sambanova: 15_000,  // Medium — 15s
    nvidia: 15_000,     // Can be slow — 15s (22.5s for heavy tasks)
    glm: 15_000,        // Should be fast — 15s
    cloudflare: 15_000, // Can be slow — 15s
    mistral: 15_000,    // Medium — 15s
    openrouter: 30_000, // Routes to various backends — can be slow
  };
  const base = timeouts[provider] ?? 30_000; // Paid providers get 30s
  // Scale by task type — heavy/super-heavy/complex models need more time
  const multiplier = taskType === 'complex' ? 4      // 240s max for paid chain
    : taskType === 'super-heavy' ? 3                  // 180s for 70B+ free models
    : taskType === 'heavy' ? 1.5                      // 22s for planning/synthesis
    : taskType === 'conversational' ? 0.8             // 8s for groq, 12s for others — fail fast for real-time chat
    : 1;                                              // light — keep base
  return Math.min(Math.round(base * multiplier), 240_000); // hard cap at 240s
}

/**
 * Per-provider stream watchdog timeout — how long to wait for the first chunk
 * (or between chunks) before declaring the stream stalled. NVIDIA needs more
 * time because it can take 10-15s to send the first token even for non-reasoning
 * models. Groq is fast and should send the first chunk within 5s.
 * Scales by taskType for large models that take longer to produce first token.
 */
function getProviderWatchdogTimeout(provider: string, taskType: string = 'heavy'): number {
  const timeouts: Record<string, number> = {
    groq: 10_000,       // Fast — 10s watchdog
    sambanova: 15_000,  // Medium — 15s
    nvidia: 15_000,     // Slow first token — 15s (was 20s — reduce wasted time on broken models)
    glm: 15_000,        // 15s
    cloudflare: 15_000, // 15s
    mistral: 15_000,    // 15s
    openrouter: 30_000, // Routes to various backends — can be slow
  };
  const base = timeouts[provider] ?? 15_000; // Default 15s
  // Scale by task type — large models can take 30-60s for first token
  const multiplier = taskType === 'complex' ? 4      // 60-120s watchdog
    : taskType === 'super-heavy' ? 3                  // 45-60s for 70B+ models
    : taskType === 'heavy' ? 1.5                      // 22s default
    : taskType === 'conversational' ? 0.3             // 3s — fail fast for real-time chat
    : 1;                                              // light — keep base
  return Math.min(Math.round(base * multiplier), 120_000); // hard cap at 120s
}

/**
 * Stream watchdog — wraps an async iterable and throws if no value is yielded
 * within `timeoutMs` of the previous value (or the first value). This catches
 * stalled streams where the connection is open but no data flows (e.g. SambaNova
 * returning 0 chars after 56s). The caller's catch block triggers fallback.
 *
 * `totalTimeoutMs` (optional) is a hard cap on total streaming time, regardless
 * of chunk flow. This catches slow-trickling streams (e.g. NVIDIA at 9 t/s
 * running for 145s) that never trigger the per-chunk watchdog.
 */
async function* withStreamWatchdog<T>(
  iterable: AsyncIterable<T>,
  timeoutMs = 10_000,
  totalTimeoutMs?: number,
): AsyncIterable<T> {
  const iterator = iterable[Symbol.asyncIterator]();
  const startTime = Date.now();
  while (true) {
    const elapsed = Date.now() - startTime;
    if (totalTimeoutMs && elapsed >= totalTimeoutMs) {
      throw new Error(`Stream total timeout: ${totalTimeoutMs / 1000}s exceeded (elapsed ${elapsed / 1000}s)`);
    }
    // If total timeout is set, shrink per-chunk timeout to not exceed remaining time
    const remaining = totalTimeoutMs ? totalTimeoutMs - elapsed : undefined;
    const perChunkTimeout = remaining !== undefined ? Math.min(timeoutMs, remaining) : timeoutMs;
    const result = await Promise.race([
      iterator.next(),
      new Promise<IteratorResult<T>>((_, reject) =>
        setTimeout(() => reject(new Error(`Stream watchdog: no chunk in ${perChunkTimeout / 1000}s`)), perChunkTimeout)
      ),
    ]);
    if (result.done) break;
    yield result.value;
  }
}

/**
 * Refusal hold-back — wraps a chunk sink and buffers leading text chunks while
 * the accumulated text could still be a canned refusal (see providerConfig).
 * If the text diverges from refusal phrasing or grows past REFUSAL_MAX_LEN,
 * all held chunks flush and the handler switches to pass-through.
 *
 * If the completed response turns out to be a canned refusal, the caller
 * throws RefusalError BEFORE calling release() — held chunks are dropped and
 * never reach the client, so refusal text can't leak into the stream.
 *
 * Non-text chunks (reasoning, empty) always pass through unheld.
 */
export function createRefusalHold(sink: (chunk: LLMStreamChunk) => void): {
  feed: (chunk: LLMStreamChunk) => void;
  release: () => void;
} {
  const held: LLMStreamChunk[] = [];
  let heldText = '';
  let released = false;
  return {
    feed(chunk: LLMStreamChunk) {
      if (released || !chunk.text) {
        sink(chunk);
        return;
      }
      held.push(chunk);
      heldText += chunk.text;
      if (heldText.length > REFUSAL_MAX_LEN || !couldBeRefusalPrefix(heldText)) {
        released = true;
        for (const c of held) sink(c);
        held.length = 0;
      }
    },
    release() {
      if (released) return;
      released = true;
      for (const c of held) sink(c);
      held.length = 0;
    },
  };
}

export class LLMStreamingRouter extends LLMRouter {
  private activeStreams: Map<string, AbortController> = new Map();

  async processPromptWithStreaming(
    request: LLMStreamRequest,
    onChunk: (chunk: StreamingMessage) => void,
    metadata: StreamingMetadata
  ): Promise<LLMStreamResult> {
    const { prompt: rawPrompt, provider: preferredProvider, options = {}, context } = request;

    // Sanitize prompt to remove malformed UTF-8 that causes 400 errors on all providers
    const prompt = sanitizePrompt(rawPrompt);

    // Context serialization is done upstream in streamingHandler.buildEnrichedPrompt().
    // The prompt already contains all context (memories, history, etc.) embedded.
    // systemInstructions from context is passed as the system message to each provider.
    const enrichedSystemInstructions = sanitizePrompt(context?.systemInstructions?.trim() || '') || undefined;

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
      // Last-refusal grace: if every provider fails but at least one produced a
      // canned refusal, return that refusal text instead of erroring — keeps a
      // polite-refusal UX for genuinely out-of-scope requests.
      let lastRefusal: { text: string; provider: string } | undefined;

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
      // Structured-output request (OpenAI response_format shape) — passed verbatim
      // to OpenAI-compatible providers. Ignored by providers that don't accept it.
      const callerResponseFormat = options.responseFormat;

      // Handle 'auto' provider — use adaptive routing
      // Handle explicit preferred provider (backward compatibility)
      const effectiveProvider = preferredProvider === 'auto' ? undefined : preferredProvider;

      if (effectiveProvider && this.isProviderConfigured(effectiveProvider)) {
        if (providerCircuitBreaker.isOpen(effectiveProvider)) {
          // Circuit-breaker open/close state is logged by providerCircuitBreaker itself —
          // no need to repeat on every request.
        } else {
          try {
            // Hold back leading text until a canned refusal can be ruled out —
            // refusal chunks are dropped on RefusalError, never reaching the client.
            const hold = createRefusalHold(handleChunk);
            const result = await this.callProviderWithStreaming(
              effectiveProvider,
              prompt,
              enrichedSystemInstructions,
              hold.feed,
              abortController.signal,
              startTime,
              callerTemperature,
              callerMaxTokens,
              taskType,
              undefined,
              callerResponseFormat
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
            // Canned-refusal detection — treated as a provider failure so the
            // fallback chain tries the next provider (held chunks are dropped).
            if (isCannedRefusal(result.fullText)) {
              throw new RefusalError(`${effectiveProvider} returned canned refusal`, result.fullText);
            }
            hold.release();
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
            const errHeaders = (err as { headers?: Record<string, string> })?.headers;
            const elapsedMs = performance.now() - startTime;
            if (err instanceof RefusalError) lastRefusal = { text: err.refusalText, provider: effectiveProvider };
            catalogManager.markFailure(effectiveProvider, '', errMsg, statusCodeOf(err, errMsg), elapsedMs);
            providerCircuitBreaker.recordFailure(effectiveProvider, errMsg, errHeaders);
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

      // Cross-taskType escalation: when the detected taskType's entire chain is
      // exhausted, retry on a heavier taskType's chain before giving up. This
      // recovers from "triple failures" (e.g. GLM 429 → Cerebras empty → Cerebras
      // 404, with Groq TPD-exhausted) by reaching providers (DeepSeek, Claude)
      // that the lighter chain's paid tail didn't get to. Escalation only fires
      // on genuine chain exhaustion — steady-state routing is unchanged.
      if (!streamResult) {
        // Inner helper: iterate one taskType's fallback chain. Returns the
        // result on success, undefined on exhaustion. Captures the outer
        // closure (prompt, handleChunk, abortController, etc.) so the chain
        // logic is parameterized only by taskType.
        const tryFallbackChain = async (tt: TaskType): Promise<LLMStreamResult | undefined> => {
          // Use adaptive fallback chain based on task type, with round-robin rotation
          // Use live catalog if loaded, else static config
          const baseChain = catalogManager.isLoaded()
            ? catalogManager.getRankedFallbackChain(tt)
            : getFallbackChain(tt);
          // Split at the PAID_CHAIN boundary — rotate free providers only, keep paid as fallback
          const paidStart = baseChain.findIndex(p => (PAID_CHAIN as readonly string[]).includes(p));
          const freeChain = paidStart >= 0 ? baseChain.slice(0, paidStart) : baseChain;
          const paidChain = paidStart >= 0 ? baseChain.slice(paidStart) : [];
          // Build provider→score map for weighted round-robin (fast providers get more slots)
          const providerScores = new Map<string, number>();
          if (catalogManager.isLoaded()) {
            for (const p of freeChain) {
              const ranked = catalogManager.getRankedModels(p, tt);
              if (ranked[0]) providerScores.set(p, ranked[0].score);
            }
          }
          const fallbackChain = [...providerCircuitBreaker.getRotatedChain(freeChain, providerScores), ...paidChain];
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
              ? catalogManager.getRankedModels(provider, tt).map(r => r.model)
              : getProviderModels(provider, tt);
            const estimatedPromptTokens = Math.ceil(prompt.length / 4);
            const maxTok = callerMaxTokens ?? 4096;
            for (const model of models) {
              if (abortController.signal.aborted) break;
              // Skip models with insufficient context window
              if (model.contextWindow && estimatedPromptTokens + maxTok > model.contextWindow) {
                continue;
              }
              // Skip models that don't support streaming (from preflight profiler)
              if (model.supportsStreaming === false) {
                logger.debug(`[StreamingRouter] Skipping ${provider}/${model.id} — no streaming support`);
                continue;
              }
              try {
                if (isHeartbeat) {
                  logger.debug(`[StreamingRouter] Trying provider: ${provider} (model: ${model.id})`);
                } else {
                  logger.info(`[StreamingRouter] Trying provider: ${provider} (model: ${model.id})`);
                }
                // Hold back leading text until a canned refusal can be ruled
                // out — refusal chunks are dropped on RefusalError, never
                // reaching the client.
                const hold = createRefusalHold(handleChunk);
                const result = await this.callProviderWithStreaming(
                  provider,
                  prompt,
                  enrichedSystemInstructions,
                  hold.feed,
                  abortController.signal,
                  startTime,
                  callerTemperature,
                  callerMaxTokens,
                  tt,
                  model.id,
                  callerResponseFormat
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
                // Canned-refusal detection — treated as a provider failure so
                // the fallback chain tries the next model/provider.
                if (isCannedRefusal(result.fullText)) {
                  throw new RefusalError(`${provider}/${model.id} returned canned refusal`, result.fullText);
                }
                hold.release();
                const estTokens = result.tokenUsage.totalTokens || Math.ceil((prompt.length + result.fullText.length) / 4);
                // Only measure speed for responses >100 chars (see comment above)
                const measuredSpeed = result.processingTime > 0 && result.fullText.length > 100
                  ? Math.round((result.fullText.length / 4) / (result.processingTime / 1000))
                  : 0;
                catalogManager.markSuccess(provider, model.id, measuredSpeed, result.processingTime);
                providerCircuitBreaker.recordSuccess(provider, estTokens);
                return result;
              } catch (err) {
                const errMsg = err instanceof Error ? err.message : String(err);
                const errHeaders = (err as { headers?: Record<string, string> })?.headers;
                const elapsedMs = performance.now() - startTime;
                if (err instanceof RefusalError) lastRefusal = { text: err.refusalText, provider };
                catalogManager.markFailure(provider, model.id, errMsg, statusCodeOf(err, errMsg), elapsedMs);
                providerCircuitBreaker.recordFailure(provider, errMsg, errHeaders);
                logger.warn(`[StreamingRouter] Provider ${provider} model ${model.id} failed`, { error: errMsg });
                // Try next model in this provider
              }
            }
            // return above exits on success; fall through to next provider
          }
          return undefined;
        };

        // Escalation order: detected taskType first, then heavier taskTypes.
        // light → heavy → complex. complex already appends COMPLEX_CHAIN +
        // HEAVY_CHAIN via getFallbackChain, so it reaches paid + free providers.
        const escalationOrder: TaskType[] = (['light', 'heavy', 'complex'] as TaskType[])
          .filter(t => t !== taskType);
        escalationOrder.unshift(taskType);
        const triedTaskTypes = new Set<TaskType>();
        for (const tt of escalationOrder) {
          if (triedTaskTypes.has(tt)) continue;
          triedTaskTypes.add(tt);
          if (abortController.signal.aborted) break;
          streamResult = await tryFallbackChain(tt);
          if (streamResult) break;
          if (tt !== escalationOrder[escalationOrder.length - 1]) {
            logger.warn(`[StreamingRouter] Chain exhausted for ${tt} — escalating`);
            onChunk({
              id: `${streamId}_fallback`,
              type: StreamingMessageType.LLM_STREAM_FALLBACK,
              payload: { exhaustedTaskType: tt, escalating: true },
              timestamp: Date.now(),
              parentId: streamId,
              metadata,
            });
          }
        }
      }

      if (!streamResult) {
        // Last-refusal grace: every provider failed, but at least one produced a
        // canned refusal — surface that refusal text instead of a bare error so
        // genuinely out-of-scope requests still get a polite response.
        if (lastRefusal) {
          logger.info(`[StreamingRouter] All providers failed; returning last canned refusal from ${lastRefusal.provider}`);
          handleChunk({ text: lastRefusal.text, provider: lastRefusal.provider });
          streamResult = {
            fullText: lastRefusal.text,
            provider: lastRefusal.provider,
            processingTime: performance.now() - startTime,
            tokenUsage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
          };
        } else {
          throw new Error('All LLM providers failed for streaming request');
        }
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
    modelId?: string,
    responseFormat?: { type: 'json_schema' | 'json_object'; json_schema?: { name: string; schema: object; strict?: boolean } }
  ): Promise<LLMStreamResult> {
    // Resolve model ID: explicit override > best ranked model from catalog > first static model
    const resolvedModel = modelId
      || (catalogManager.isLoaded() ? catalogManager.getRankedModels(provider, taskType)[0]?.model.id : undefined)
      || getProviderModels(provider, taskType)[0]?.id;
    if (!resolvedModel) throw new Error(`No models configured for provider: ${provider}`);

    let result: LLMStreamResult;
    switch (provider) {
      case 'claude':
        result = await this.callClaudeWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime, temperature, maxTokens, resolvedModel, taskType);
        break;
      case 'gemini-free':
      case 'gemini-paid':
        result = await this.callGeminiWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime, temperature, maxTokens, resolvedModel, provider, taskType);
        break;
      case 'mistral':
        result = await this.callMistralWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime, temperature, maxTokens, resolvedModel, taskType);
        break;
      case 'glm':
        result = await this.callGLMWithStreaming(prompt, systemInstructions, onChunk, abortSignal, startTime, temperature, maxTokens, resolvedModel, taskType);
        break;
      default: {
        // All openai-compatible providers (groq, sambanova, nvidia, cloudflare, glm,
        // deepseek, grok, openai, and any dynamically added providers)
        const apiType = getProviderAPIType(provider);
        if (apiType === 'openai-compatible') {
          result = await this.callOpenAICompatibleWithStreaming(provider, prompt, systemInstructions, onChunk, abortSignal, startTime, temperature, maxTokens, resolvedModel, taskType, responseFormat);
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
    modelId: string = '',
    taskType: TaskType = 'heavy',
    responseFormat?: { type: 'json_schema' | 'json_object'; json_schema?: { name: string; schema: object; strict?: boolean } }
  ): Promise<LLMStreamResult> {
    const envKey = getProviderEnvKeyDynamic(provider);
    const apiKey = process.env[envKey];
    if (!apiKey) throw new Error(`${envKey} not configured for provider: ${provider}`);

    const baseURL = getProviderBaseURL(provider);
    if (!baseURL) throw new Error(`No baseURL for provider: ${provider}`);

    // Look up catalog model for reasoning profile data
    const catalogModel = catalogManager.isLoaded()
      ? catalogManager.getProvider(provider)?.models.find(m => m.id === modelId)
      : undefined;

    // Resolve reasoning request params + which delta field carries reasoning.
    // Profiled models trust the catalog; unprofiled models fall back to the
    // provider registry (optimistic separate-params) or disable-thinking logic.
    const { params: reasoningParams, responseField } = resolveReasoningParams(provider, modelId, catalogModel);

    const client = new OpenAI({ apiKey, baseURL, timeout: getProviderTimeout(provider, taskType), maxRetries: 0 });
    let fullText = '';
    let tokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    const messages: Array<{ role: 'system' | 'user'; content: string }> = [];
    if (systemInstructions) messages.push({ role: 'system', content: systemInstructions });
    messages.push({ role: 'user', content: prompt });

    const baseCreateParams: any = {
      model: modelId,
      messages,
      stream: true,
      temperature: temperature ?? 0.7,
      max_tokens: maxTokens ?? 4096,
      ...reasoningParams,
    };

    // Structured output — passed verbatim as OpenAI response_format.
    // Only applied when caller requests it; calls without responseFormat are unchanged.
    if (responseFormat) {
      baseCreateParams.response_format = responseFormat;
      logger.info(`[StreamingRouter] structured output: ${provider}/${modelId} ${responseFormat.type}`);
    }

    let stream: AsyncIterable<any>;
    try {
      stream = await client.chat.completions.create(baseCreateParams) as unknown as AsyncIterable<any>;
    } catch (err: any) {
      // Graceful degrade: if the provider rejects response_format OR a
      // reasoning param (e.g. reasoning_format on a model that doesn't accept
      // it), retry once without the offending param so the request still
      // succeeds. The inline-think safety net below handles any leakage.
      const msg = err?.message || '';
      const isResponseFormatErr = responseFormat && err?.status === 400 && /response_format/i.test(msg);
      const isReasoningParamErr = err?.status === 400 && /reasoning_format|unknown (parameter|field)/i.test(msg);
      if (isResponseFormatErr || isReasoningParamErr) {
        if (isResponseFormatErr) logger.warn(`[StreamingRouter] ${provider}/${modelId} rejected response_format — retrying without it`);
        if (isReasoningParamErr) logger.warn(`[StreamingRouter] ${provider}/${modelId} rejected reasoning param — retrying without it`);
        const { response_format: _rf, reasoning_format: _rfmt, ...degradedParams } = baseCreateParams;
        stream = await client.chat.completions.create(degradedParams) as unknown as AsyncIterable<any>;
      } else {
        throw err;
      }
    }

    // Stream-aware inline-think safety net. Catches Mattis blocks that leak
    // into delta.content even when separate-reasoning params are sent
    // (unprofiled models, models that ignore reasoning_format, etc.).
    const stripper = new ThinkStripper();

    for await (const chunk of withStreamWatchdog(stream, getProviderWatchdogTimeout(provider, taskType), getProviderTimeout(provider, taskType))) {
      if (abortSignal.aborted) break;

      const delta = chunk.choices[0]?.delta;
      // Read reasoning from the dedicated field (if configured) and forward it
      // separately — never append it to fullText or put it in `text`.
      if (responseField && delta) {
        const reasoning = (delta as any)?.[responseField];
        if (reasoning) {
          onChunk({ text: '', reasoning, provider });
        }
      }
      // Normal answer content — run through the safety net before emitting.
      const rawContent = delta?.content;
      if (rawContent) {
        const content = stripper.feed(rawContent);
        if (content) {
          fullText += content;
          onChunk({ text: content, provider });
        }
      }
      if (chunk.choices[0]?.finish_reason) {
        tokenUsage = {
          promptTokens: chunk.usage?.prompt_tokens || 0,
          completionTokens: chunk.usage?.completion_tokens || 0,
          totalTokens: chunk.usage?.total_tokens || 0,
        };
      }
    }

    // Flush any buffered carry (a partial tag that turned out not to be one).
    const tail = stripper.flush();
    if (tail) {
      fullText += tail;
      onChunk({ text: tail, provider });
    }

    // Warn if inline think was stripped — indicates a model that needs
    // profiling (reasoningMode) or a registry entry.
    if (stripper.didStrip()) {
      logger.warn(`[StreamingRouter] ${provider}/${modelId} emitted inline Mattis in content — set reasoningMode via profiler or add REASONING_PROVIDER_CONFIG entry`);
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
    modelId: string = 'glm-4.7-flash',
    taskType: TaskType = 'heavy'
  ): Promise<LLMStreamResult> {
    const apiKey = process.env.GLM_API_KEY;
    if (!apiKey) throw new Error('GLM_API_KEY not configured');

    // Look up catalog model for thinking-disable profile data
    const catalogModel = catalogManager.isLoaded()
      ? catalogManager.getProvider('glm')?.models.find(m => m.id === modelId)
      : undefined;

    const glm = new OpenAI({ apiKey, baseURL: getProviderBaseURL('glm'), timeout: getProviderTimeout('glm', taskType), maxRetries: 0 });
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
      ...getDisableThinkingParams('glm', modelId, catalogModel),
    };
    const stream = await glm.chat.completions.create(glmParams) as unknown as AsyncIterable<import('openai/resources/chat/completions').ChatCompletionChunk>;

    // Safety net: GLM thinking is disabled above, but if a model ignores the
    // disable param or emits inline Mattis, strip it. Also read
    // delta.reasoning_content separately when present (forwarded as reasoning).
    const stripper = new ThinkStripper();

    for await (const chunk of withStreamWatchdog(stream, getProviderWatchdogTimeout('glm', taskType), getProviderTimeout('glm', taskType))) {
      if (abortSignal.aborted) break;

      const delta = chunk.choices[0]?.delta as any;
      // Forward any reasoning_content separately (present when thinking isn't
      // disabled — e.g. a model that ignores the disable param).
      if (delta?.reasoning_content) {
        onChunk({ text: '', reasoning: delta.reasoning_content, provider: 'glm' });
      }
      const rawContent = delta?.content;
      if (rawContent) {
        const content = stripper.feed(rawContent);
        if (content) {
          fullText += content;
          onChunk({
            text: content,
            provider: 'glm',
            tokenCount: content.split(' ').length,
            finishReason: (chunk.choices[0]?.finish_reason as any) || null,
          });
        }
      }

      if (chunk.usage) {
        tokenUsage = {
          promptTokens: chunk.usage.prompt_tokens,
          completionTokens: chunk.usage.completion_tokens,
          totalTokens: chunk.usage.total_tokens,
        };
      }
    }

    const glmTail = stripper.flush();
    if (glmTail) {
      fullText += glmTail;
      onChunk({ text: glmTail, provider: 'glm' });
    }
    if (stripper.didStrip()) {
      logger.warn(`[StreamingRouter] glm/${modelId} emitted inline Mattis in content — set reasoningMode via profiler`);
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
    modelId: string = 'claude-sonnet-4-20250514',
    taskType: TaskType = 'heavy'
  ): Promise<LLMStreamResult> {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) throw new Error('ANTHROPIC_API_KEY not configured');

    const anthropic = new Anthropic({ apiKey, timeout: getProviderTimeout('claude', taskType), maxRetries: 0 });
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

    for await (const chunk of withStreamWatchdog(stream, getProviderWatchdogTimeout('claude', taskType), getProviderTimeout('claude', taskType))) {
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

    for await (const chunk of withStreamWatchdog(stream, getProviderWatchdogTimeout('openai'), getProviderTimeout('openai'))) {
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

    for await (const chunk of withStreamWatchdog(stream, getProviderWatchdogTimeout('groq'), getProviderTimeout('groq'))) {
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

    for await (const chunk of withStreamWatchdog(stream, getProviderWatchdogTimeout('sambanova'), getProviderTimeout('sambanova'))) {
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
    modelId: string = 'deepseek-ai/deepseek-v4-flash-0731'
  ): Promise<LLMStreamResult> {
    const apiKey = process.env.NVIDIA_API_KEY;
    if (!apiKey) throw new Error('NVIDIA_API_KEY not configured');

    // Look up catalog model for thinking-disable profile data
    const catalogModel = catalogManager.isLoaded()
      ? catalogManager.getProvider('nvidia')?.models.find(m => m.id === modelId)
      : undefined;

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
      ...getDisableThinkingParams('nvidia', modelId, catalogModel),
    });

    for await (const chunk of withStreamWatchdog(stream, getProviderWatchdogTimeout('nvidia'), getProviderTimeout('nvidia'))) {
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

    for await (const chunk of withStreamWatchdog(stream, getProviderWatchdogTimeout('cloudflare'), getProviderTimeout('cloudflare'))) {
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
    provider: string = 'gemini-free',
    taskType: TaskType = 'heavy'
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

    // GoogleGenerativeAI SDK doesn't support timeout natively — race against a dynamic timer
    const geminiTimeoutMs = getProviderTimeout(provider, taskType);
    const result = await Promise.race([
      model.generateContentStream({
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: {
          ...(temperature !== undefined ? { temperature } : {}),
          ...(maxTokens !== undefined ? { maxOutputTokens: maxTokens } : {}),
        },
      }),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error(`Gemini request timed out after ${geminiTimeoutMs / 1000}s`)), geminiTimeoutMs)
      ),
    ]);

    for await (const chunk of withStreamWatchdog(result.stream, getProviderWatchdogTimeout(provider, taskType), getProviderTimeout(provider, taskType))) {
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
    modelId: string = 'mistral-medium',
    taskType: TaskType = 'heavy'
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

    // Safety net: magistral reasoning models can leak inline Mattis into
    // delta.content. Strip it; Mistral has no separate-reasoning param to send.
    const stripper = new ThinkStripper();

    for await (const chunk of withStreamWatchdog(stream, getProviderWatchdogTimeout('mistral', taskType), getProviderTimeout('mistral', taskType))) {
      if (abortSignal.aborted) break;
      const delta = chunk.data.choices[0]?.delta as any;
      // Forward any reasoning_content separately if the SDK exposes it.
      if (delta?.reasoning_content) {
        onChunk({ text: '', reasoning: delta.reasoning_content, provider: 'mistral' });
      }
      const rawContent = delta?.content;
      const content = typeof rawContent === 'string' ? stripper.feed(rawContent) : '';
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

    const mistralTail = stripper.flush();
    if (mistralTail) {
      fullText += mistralTail;
      onChunk({ text: mistralTail, provider: 'mistral' });
    }
    if (stripper.didStrip()) {
      logger.warn(`[StreamingRouter] mistral/${modelId} emitted inline Mattis in content — set reasoningMode via profiler`);
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
