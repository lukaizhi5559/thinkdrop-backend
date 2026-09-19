/**
 * Base LLM router — handles non-streaming prompt dispatch with provider fallback chain.
 * Used by LLMElementMatcher for element matching calls.
 */

import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenerativeAI } from '@google/generative-ai';
import axios from 'axios';
import { logger } from './logger';
import { providerCircuitBreaker } from './providerCircuitBreaker';
import { catalogManager } from './catalogManager';
import {
  PROVIDER_CONFIG,
  HEAVY_CHAIN,
  LIGHT_CHAIN,
  SUPER_HEAVY_CHAIN,
  COMPLEX_CHAIN,
  CONVERSATIONAL_CHAIN,
  FREE_PREMIUM_CHAIN,
  PAID_CHAIN,
  detectTaskType,
  getProviderModels,
  getProviderBaseURL,
  getProviderAPIType,
  getProviderEnvKeyDynamic,
  isProviderConfiguredStatic,
  sanitizePrompt,
  resolveReasoningParams,
  getDisableThinkingParams,
  stripInlineThinking,
  isCannedRefusal,
  RefusalError,
  isProviderExcludedForTaskType,
  TaskType,
  ProviderModel,
} from './providerConfig';

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
 */
function getProviderTimeout(provider: string): number {
  const timeouts: Record<string, number> = {
    groq: 10_000,       // Fast — 10s max
    sambanova: 15_000,  // Medium — 15s
    nvidia: 20_000,     // Can be slow — 20s
    glm: 15_000,        // Should be fast — 15s
    cloudflare: 15_000, // Can be slow — 15s
    mistral: 15_000,    // Medium — 15s
    openrouter: 30_000, // Routes to various backends — can be slow
  };
  return timeouts[provider] ?? 30_000; // Paid providers get 30s
}

export interface LLMRouterOptions {
  skipCache?: boolean;
  taskType?: string;
  preferredProvider?: string;
  /**
   * Structured-output request — passed verbatim as OpenAI `response_format`
   * to OpenAI-compatible providers. Ignored by providers that don't accept it
   * (graceful degrade on 400). Only applied when present.
   */
  responseFormat?: {
    type: 'json_schema' | 'json_object';
    json_schema?: { name: string; schema: object; strict?: boolean };
  };
}

export interface LLMRouterResult {
  text: string;
  provider: string;
  processingTime: number;
}

export class LLMRouter {
  async processPrompt(prompt: string, options: LLMRouterOptions = {}): Promise<LLMRouterResult> {
    const startTime = performance.now();
    const preferred = options.preferredProvider === 'auto' ? undefined : options.preferredProvider;
    const callerResponseFormat = options.responseFormat;

    // Sanitize prompt to remove malformed UTF-8 that causes 400 errors on all providers
    prompt = sanitizePrompt(prompt);

    // Detect task type for adaptive routing
    const taskType: TaskType = (options.taskType === 'heartbeat' || options.taskType === 'classification')
      ? 'light'
      : (options.taskType === 'conversational' || options.taskType === 'chat')
        ? 'conversational'
        : (options.taskType === 'complex' || options.taskType === 'command_automate')
          ? 'complex'
          : options.taskType === 'super-heavy'
            ? 'super-heavy'
            : 'heavy';

    // Cross-taskType escalation: when the detected taskType's entire chain is
    // exhausted, retry on a heavier taskType's chain before giving up. Mirrors
    // the streaming router's escalation so non-streaming callers (e.g.
    // LLMElementMatcher) also recover from total chain failures.
    // Inner helper: iterate one taskType's ordered chain. Returns the result on
    // success, undefined on exhaustion. Captures the outer closure (prompt,
    // preferred, callerResponseFormat, startTime) so the chain logic is
    // parameterized only by taskType.
    // Last-refusal grace: if every provider fails but at least one produced a
    // canned refusal, that refusal text is returned instead of a generic error.
    let lastRefusal: { text: string; provider: string } | undefined;
    const tryChain = async (tt: TaskType): Promise<LLMRouterResult | undefined> => {
      // Build ordered chain: use live catalog if loaded, else static config.
      // Filter out providers excluded for this task type (e.g. mistral for
      // conversational — its safety layer over-refuses under persona prompts).
      const baseChain = (catalogManager.isLoaded()
        ? catalogManager.getRankedFallbackChain(tt)
        : (tt === 'complex'
            ? [...COMPLEX_CHAIN]
            : tt === 'conversational'
              ? [...CONVERSATIONAL_CHAIN, ...FREE_PREMIUM_CHAIN, ...PAID_CHAIN]
              : tt === 'light'
                ? [...LIGHT_CHAIN, ...FREE_PREMIUM_CHAIN, ...PAID_CHAIN]
                : tt === 'super-heavy'
                  ? [...SUPER_HEAVY_CHAIN, ...FREE_PREMIUM_CHAIN, ...PAID_CHAIN]
                  : [...HEAVY_CHAIN, ...FREE_PREMIUM_CHAIN, ...PAID_CHAIN])
      ).filter(p => !isProviderExcludedForTaskType(p, tt));
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
      const rotatedChain = [...providerCircuitBreaker.getRotatedChain(freeChain, providerScores), ...paidChain];
      const ordered = preferred
        ? [preferred, ...rotatedChain.filter((p) => p !== preferred)]
        : rotatedChain;

      for (const provider of ordered) {
        if (!this.isProviderConfigured(provider)) {
          logger.debug(`[LLMRouter] Skipping unconfigured provider: ${provider}`);
          continue;
        }
        if (providerCircuitBreaker.isOpen(provider)) {
          logger.debug(`[LLMRouter] Skipping circuit-broken provider: ${provider}`);
          continue;
        }
        if (providerCircuitBreaker.isTpdExhausted(provider)) {
          logger.debug(`[LLMRouter] Skipping TPD-exhausted provider: ${provider}`);
          continue;
        }

        // Intra-provider model fallback — use ranked models from catalog if loaded
        const models: ProviderModel[] = catalogManager.isLoaded()
          ? catalogManager.getRankedModels(provider, tt).map(r => r.model)
          : getProviderModels(provider, tt);
        const estimatedPromptTokens = Math.ceil(prompt.length / 4);
        for (const model of models) {
          // Skip models with insufficient context window (assume 4096 maxTokens if not specified)
          if (model.contextWindow && estimatedPromptTokens + 4096 > model.contextWindow) {
            continue;
          }
          try {
            const text = await this.callProvider(provider, prompt, model.id, callerResponseFormat);
            if (!text || !text.trim()) {
              throw new Error(`${provider}/${model.id} returned empty response`);
            }
            // Canned-refusal detection — treated as a provider failure so the
            // fallback chain tries the next model/provider.
            if (isCannedRefusal(text)) {
              throw new RefusalError(`${provider}/${model.id} returned canned refusal`, text);
            }
            const estTokens = Math.ceil((prompt.length + text.length) / 4);
            const processingTime = performance.now() - startTime;
            // Only measure speed for responses >100 chars — short responses give
            // misleadingly low t/s that would drag down the EMA for fast models.
            const measuredSpeed = processingTime > 0 && text.length > 100
              ? Math.round((text.length / 4) / (processingTime / 1000))
              : 0;
            catalogManager.markSuccess(provider, model.id, measuredSpeed, processingTime);
            providerCircuitBreaker.recordSuccess(provider, estTokens);
            return {
              text,
              provider,
              processingTime,
            };
          } catch (err) {
            const errMsg = err instanceof Error ? err.message : String(err);
            const errHeaders = (err as { headers?: Record<string, string> })?.headers;
            const elapsedMs = performance.now() - startTime;
            if (err instanceof RefusalError) lastRefusal = { text: err.refusalText, provider };
            catalogManager.markFailure(provider, model.id, errMsg, statusCodeOf(err, errMsg), elapsedMs);
            providerCircuitBreaker.recordFailure(provider, errMsg, errHeaders);
            logger.warn(`[LLMRouter] Provider ${provider} model ${model.id} failed`, { error: errMsg });
          }
        }
      }
      return undefined;
    };

    // Escalation order: detected taskType first, then heavier taskTypes.
    // light → heavy → super-heavy → complex. complex appends COMPLEX_CHAIN +
    // HEAVY_CHAIN via getFallbackChain, so it reaches paid + free providers.
    const escalationOrder: TaskType[] = (['light', 'heavy', 'super-heavy', 'complex'] as TaskType[])
      .filter(t => t !== taskType);
    escalationOrder.unshift(taskType);
    const triedTaskTypes = new Set<TaskType>();
    let result: LLMRouterResult | undefined;
    for (const tt of escalationOrder) {
      if (triedTaskTypes.has(tt)) continue;
      triedTaskTypes.add(tt);
      result = await tryChain(tt);
      if (result) break;
      if (tt !== escalationOrder[escalationOrder.length - 1]) {
        logger.warn(`[LLMRouter] Chain exhausted for ${tt} — escalating`);
      }
    }

    if (!result) {
      // Last-refusal grace: surface the canned refusal rather than a bare error
      // so genuinely out-of-scope requests still get a polite response.
      if (lastRefusal) {
        logger.info(`[LLMRouter] All providers failed; returning last canned refusal from ${lastRefusal.provider}`);
        return {
          text: lastRefusal.text,
          provider: lastRefusal.provider,
          processingTime: performance.now() - startTime,
        };
      }
      throw new Error('[LLMRouter] All providers failed');
    }
    return result;
  }

  private async callProvider(
    provider: string,
    prompt: string,
    modelId: string,
    responseFormat?: { type: 'json_schema' | 'json_object'; json_schema?: { name: string; schema: object; strict?: boolean } }
  ): Promise<string> {
    switch (provider) {
      case 'claude':
        return this.callClaude(prompt, modelId);
      case 'gemini-free':
      case 'gemini-paid':
        return this.callGemini(prompt, modelId, provider);
      case 'mistral':
        return this.callMistral(prompt, modelId);
      case 'glm':
        return this.callGLM(prompt, modelId);
      default: {
        // All openai-compatible providers (groq, sambanova, nvidia, cloudflare, glm,
        // deepseek, grok, openai, and any dynamically added providers)
        const apiType = getProviderAPIType(provider);
        if (apiType === 'openai-compatible') {
          return this.callOpenAICompatible(provider, prompt, modelId, responseFormat);
        }
        throw new Error(`Unknown provider or unsupported apiType: ${provider} (${apiType})`);
      }
    }
  }

  /**
   * Generic OpenAI-compatible non-streaming handler.
   * Works with any provider that uses the OpenAI API format.
   * Reads baseURL + envKey dynamically — supports dynamically added providers.
   */
  private async callOpenAICompatible(
    provider: string,
    prompt: string,
    modelId: string,
    responseFormat?: { type: 'json_schema' | 'json_object'; json_schema?: { name: string; schema: object; strict?: boolean } }
  ): Promise<string> {
    const envKey = getProviderEnvKeyDynamic(provider);
    const apiKey = process.env[envKey];
    if (!apiKey) throw new Error(`${envKey} not configured for provider: ${provider}`);

    const baseURL = getProviderBaseURL(provider);
    if (!baseURL) throw new Error(`No baseURL for provider: ${provider}`);

    // Look up catalog model for reasoning profile data
    const catalogModel = catalogManager.isLoaded()
      ? catalogManager.getProvider(provider)?.models.find(m => m.id === modelId)
      : undefined;
    const { params: reasoningParams } = resolveReasoningParams(provider, modelId, catalogModel);

    const client = new OpenAI({ apiKey, baseURL, timeout: getProviderTimeout(provider), maxRetries: 0 });
    const baseCreateParams: any = {
      model: modelId,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.7,
      max_tokens: 4096,
      ...reasoningParams,
    };

    // Structured output — passed verbatim as OpenAI response_format.
    if (responseFormat) {
      baseCreateParams.response_format = responseFormat;
      logger.info(`[LLMRouter] structured output: ${provider}/${modelId} ${responseFormat.type}`);
    }

    let response;
    try {
      response = await client.chat.completions.create(baseCreateParams);
    } catch (err: any) {
      // Graceful degrade: if the provider rejects response_format OR a
      // reasoning param (e.g. reasoning_format), retry once without it.
      const msg = err?.message || '';
      const isResponseFormatErr = responseFormat && err?.status === 400 && /response_format/i.test(msg);
      const isReasoningParamErr = err?.status === 400 && /reasoning_format|unknown (parameter|field)/i.test(msg);
      if (isResponseFormatErr || isReasoningParamErr) {
        if (isResponseFormatErr) logger.warn(`[LLMRouter] ${provider}/${modelId} rejected response_format — retrying without it`);
        if (isReasoningParamErr) logger.warn(`[LLMRouter] ${provider}/${modelId} rejected reasoning param — retrying without it`);
        const { response_format: _rf, reasoning_format: _rfmt, ...degradedParams } = baseCreateParams;
        response = await client.chat.completions.create(degradedParams);
      } else {
        throw err;
      }
    }

    const rawText = response.choices[0]?.message?.content || '';
    // Safety net: strip any inline Mattis that leaked into content (groq
    // gpt-oss raw format, unprofiled models, models that ignore reasoning_format).
    const text = stripInlineThinking(rawText);
    if (text !== rawText) {
      logger.warn(`[LLMRouter] ${provider}/${modelId} emitted inline Mattis in content — set reasoningMode via profiler or add REASONING_PROVIDER_CONFIG entry`);
    }
    if (!text.trim()) throw new Error(`${provider}/${modelId} returned empty response`);
    return text;
  }

  private async callGLM(prompt: string, modelId: string): Promise<string> {
    const apiKey = process.env.GLM_API_KEY;
    if (!apiKey) throw new Error('GLM_API_KEY not set');

    const glm = new OpenAI({ apiKey, baseURL: getProviderBaseURL('glm'), timeout: getProviderTimeout('glm'), maxRetries: 0 });
    const response = await glm.chat.completions.create({
      model: modelId,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 512,
      ...getDisableThinkingParams('glm', modelId),
    } as any);

    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error('GLM returned empty content');
    return content;
  }

  private async callOpenAI(prompt: string, modelId: string): Promise<string> {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) throw new Error('OPENAI_API_KEY not set');

    const openai = new OpenAI({ apiKey, timeout: 30_000, maxRetries: 0 });
    const response = await openai.chat.completions.create({
      model: modelId,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 512,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error('OpenAI returned empty content');
    return content;
  }

  private async callGroq(prompt: string, modelId: string): Promise<string> {
    const apiKey = process.env.GROQ_API_KEY;
    if (!apiKey) throw new Error('GROQ_API_KEY not set');

    const groq = new OpenAI({ apiKey, baseURL: getProviderBaseURL('groq'), timeout: getProviderTimeout('groq'), maxRetries: 0 });
    const response = await groq.chat.completions.create({
      model: modelId,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 512,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error('Groq returned empty content');
    return content;
  }

  private async callSambanova(prompt: string, modelId: string): Promise<string> {
    const apiKey = process.env.SAMBANOVA_API_KEY;
    if (!apiKey) throw new Error('SAMBANOVA_API_KEY not set');

    const client = new OpenAI({ apiKey, baseURL: getProviderBaseURL('sambanova'), timeout: getProviderTimeout('sambanova'), maxRetries: 0 });
    const response = await client.chat.completions.create({
      model: modelId,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 512,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error('SambaNova returned empty content');
    return content;
  }

  private async callNvidia(prompt: string, modelId: string): Promise<string> {
    const apiKey = process.env.NVIDIA_API_KEY;
    if (!apiKey) throw new Error('NVIDIA_API_KEY not set');

    const client = new OpenAI({ apiKey, baseURL: getProviderBaseURL('nvidia'), timeout: getProviderTimeout('nvidia'), maxRetries: 0 });
    const response = await client.chat.completions.create({
      model: modelId,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 512,
      ...getDisableThinkingParams('nvidia', modelId),
    });

    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error('NVIDIA NIM returned empty content');
    return content;
  }

  private async callCloudflare(prompt: string, modelId: string): Promise<string> {
    const apiToken = process.env.CLOUDFLARE_API_TOKEN;
    const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;
    if (!apiToken) throw new Error('CLOUDFLARE_API_TOKEN not set');
    if (!accountId) throw new Error('CLOUDFLARE_ACCOUNT_ID not set');

    const baseURL = getProviderBaseURL('cloudflare');
    const client = new OpenAI({ apiKey: apiToken, baseURL, timeout: getProviderTimeout('cloudflare'), maxRetries: 0 });
    const response = await client.chat.completions.create({
      model: modelId,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 512,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error('Cloudflare Workers AI returned empty content');
    return content;
  }

  private async callClaude(prompt: string, modelId: string): Promise<string> {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) throw new Error('ANTHROPIC_API_KEY not set');

    const anthropic = new Anthropic({ apiKey, timeout: 30_000, maxRetries: 0 });
    const response = await anthropic.messages.create({
      model: modelId,
      max_tokens: 512,
      messages: [{ role: 'user', content: prompt }],
    });

    const block = response.content[0];
    return block && 'text' in block ? block.text : '';
  }

  private async callGemini(prompt: string, modelId: string, provider: string): Promise<string> {
    const envKey = PROVIDER_CONFIG[provider]?.envKey || 'GEMINI_API_KEY_FREE';
    const apiKey = process.env[envKey];
    if (!apiKey) throw new Error(`${envKey} not set`);

    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ model: modelId });
    const result = await model.generateContent(prompt);
    return result.response.text();
  }

  private async callMistral(prompt: string, modelId: string): Promise<string> {
    const apiKey = process.env.MISTRAL_API_KEY;
    if (!apiKey) throw new Error('MISTRAL_API_KEY not set');

    const response = await axios.post(
      'https://api.mistral.ai/v1/chat/completions',
      {
        model: modelId,
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.1,
        max_tokens: 512,
      },
      {
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        timeout: 30000,
      }
    );

    return response.data.choices[0]?.message?.content || '';
  }

  private async callGrok(prompt: string, modelId: string): Promise<string> {
    const apiKey = process.env.GROK_API_KEY;
    if (!apiKey) throw new Error('GROK_API_KEY not set');

    const response = await axios.post(
      'https://api.x.ai/v1/chat/completions',
      {
        model: modelId,
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.1,
        max_tokens: 512,
      },
      {
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        timeout: 30000,
      }
    );

    return response.data.choices[0]?.message?.content || '';
  }

  private async callDeepseek(prompt: string, modelId: string): Promise<string> {
    const apiKey = process.env.DEEPSEEK_API_KEY;
    if (!apiKey) throw new Error('DEEPSEEK_API_KEY not set');

    const response = await axios.post(
      'https://api.deepseek.com/v1/chat/completions',
      {
        model: modelId,
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.1,
        max_tokens: 512,
      },
      {
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        timeout: 30000,
      }
    );

    return response.data.choices[0]?.message?.content || '';
  }

  protected isProviderConfigured(provider: string): boolean {
    return isProviderConfiguredStatic(provider);
  }
}
