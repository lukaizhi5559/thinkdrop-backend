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
  PAID_CHAIN,
  detectTaskType,
  getProviderModels,
  getProviderBaseURL,
  getProviderAPIType,
  getProviderEnvKeyDynamic,
  isProviderConfiguredStatic,
  TaskType,
  ProviderModel,
} from './providerConfig';

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

export interface LLMRouterOptions {
  skipCache?: boolean;
  taskType?: string;
  preferredProvider?: string;
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

    // Detect task type for adaptive routing
    const taskType: TaskType = (options.taskType === 'heartbeat' || options.taskType === 'classification')
      ? 'light'
      : (options.taskType === 'complex' || options.taskType === 'command_automate')
        ? 'complex'
        : options.taskType === 'super-heavy'
          ? 'super-heavy'
          : 'heavy';

    // Build ordered chain: use live catalog if loaded, else static config
    const baseChain = catalogManager.isLoaded()
      ? catalogManager.getRankedFallbackChain(taskType)
      : (taskType === 'complex'
          ? [...COMPLEX_CHAIN]
          : taskType === 'light'
            ? [...LIGHT_CHAIN, ...PAID_CHAIN]
            : taskType === 'super-heavy'
              ? [...SUPER_HEAVY_CHAIN, ...PAID_CHAIN]
              : [...HEAVY_CHAIN, ...PAID_CHAIN]);
    // Split at the PAID_CHAIN boundary — rotate free providers only, keep paid as fallback
    const paidStart = baseChain.findIndex(p => (PAID_CHAIN as readonly string[]).includes(p));
    const freeChain = paidStart >= 0 ? baseChain.slice(0, paidStart) : baseChain;
    const paidChain = paidStart >= 0 ? baseChain.slice(paidStart) : [];
    // Build provider→score map for weighted round-robin (fast providers get more slots)
    const providerScores = new Map<string, number>();
    if (catalogManager.isLoaded()) {
      for (const p of freeChain) {
        const ranked = catalogManager.getRankedModels(p, taskType);
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
        ? catalogManager.getRankedModels(provider, taskType).map(r => r.model)
        : getProviderModels(provider, taskType);
      const estimatedPromptTokens = Math.ceil(prompt.length / 4);
      for (const model of models) {
        // Skip models with insufficient context window (assume 4096 maxTokens if not specified)
        if (model.contextWindow && estimatedPromptTokens + 4096 > model.contextWindow) {
          continue;
        }
        try {
          const text = await this.callProvider(provider, prompt, model.id);
          if (!text || !text.trim()) {
            throw new Error(`${provider}/${model.id} returned empty response`);
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
          catalogManager.markFailure(provider, model.id, errMsg);
          providerCircuitBreaker.recordFailure(provider, errMsg, errHeaders);
          logger.warn(`[LLMRouter] Provider ${provider} model ${model.id} failed`, { error: errMsg });
        }
      }
    }

    throw new Error('[LLMRouter] All providers failed');
  }

  private async callProvider(provider: string, prompt: string, modelId: string): Promise<string> {
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
          return this.callOpenAICompatible(provider, prompt, modelId);
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
  private async callOpenAICompatible(provider: string, prompt: string, modelId: string): Promise<string> {
    const envKey = getProviderEnvKeyDynamic(provider);
    const apiKey = process.env[envKey];
    if (!apiKey) throw new Error(`${envKey} not configured for provider: ${provider}`);

    const baseURL = getProviderBaseURL(provider);
    if (!baseURL) throw new Error(`No baseURL for provider: ${provider}`);

    const client = new OpenAI({ apiKey, baseURL, timeout: getProviderTimeout(provider), maxRetries: 0 });
    const response = await client.chat.completions.create({
      model: modelId,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.7,
      max_tokens: 4096,
      ...getDisableThinkingParams(provider, modelId),
    });

    const text = response.choices[0]?.message?.content || '';
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
