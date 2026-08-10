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

/**
 * Centralized model IDs & base URLs — kept in sync with llmStreamingRouter.ts.
 * Order: FREE providers (by speed, fastest first), then PAID providers (cheapest first).
 */
const PROVIDERS = [
  // --- FREE (permanent free tiers, no credit card) ---
  'sambanova',   // fast RDU, Llama 3.3 70B Instruct — most reliable free provider
  'groq',        // ~400-500 t/s, Llama 3.3 70B — TPD limited (100K/day)
  'gemini',      // Google AI Studio free tier — fast on large prompts
  'glm',         // z.ai free tier (glm-4.7-flash, 200K context, thinking disabled)
  'nvidia',      // NVIDIA NIM, Llama 3.3 70B Instruct — frequently times out
  'cloudflare',  // ~40-60 t/s, Workers AI edge
  'openrouter',  // free :free variants (Llama 3.3 70B Instruct)
  'github',      // GitHub Models (currently in retirement brownout)
  // --- PAID (cheapest to most expensive) ---
  'deepseek',    // ~$0.14/M tokens
  'mistral',     // cheap
  'grok',        // cheap-ish
  'openai',      // gpt-4o, expensive
  'claude',      // claude-sonnet-4, most expensive
] as const;

const MODEL_IDS = {
  glm: 'glm-4.7-flash',
  groq: 'llama-3.3-70b-versatile',
  sambanova: 'Meta-Llama-3.3-70B-Instruct',
  github: 'meta/Llama-3.3-70B-Instruct',
  nvidia: 'meta/llama-3.3-70b-instruct',
  cloudflare: '@cf/meta/llama-3.3-70b-instruct-fp8-fast',
  openrouter: 'meta-llama/llama-3.3-70b-instruct:free',
  openai: 'gpt-4o',
  claude: 'claude-sonnet-4-20250514',
  gemini: 'gemini-flash-latest',
  mistral: 'mistral-medium',
  grok: 'grok-4.20-0309-non-reasoning',
  deepseek: 'deepseek-chat',
} as const;

const BASE_URLS = {
  glm: 'https://api.z.ai/api/paas/v4',
  groq: 'https://api.groq.com/openai/v1',
  sambanova: 'https://api.sambanova.ai/v1',
  github: 'https://models.github.ai/inference',
  nvidia: 'https://integrate.api.nvidia.com/v1',
  openrouter: 'https://openrouter.ai/api/v1',
} as const;

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
    const preferred = options.preferredProvider;

    const ordered = preferred
      ? [preferred, ...PROVIDERS.filter((p) => p !== preferred)]
      : PROVIDERS;

    for (const provider of ordered) {
      if (!this.isProviderConfigured(provider)) {
        logger.debug(`[LLMRouter] Skipping unconfigured provider: ${provider}`);
        continue;
      }
      if (providerCircuitBreaker.isOpen(provider)) {
        logger.debug(`[LLMRouter] Skipping circuit-broken provider: ${provider}`);
        continue;
      }
      try {
        const text = await this.callProvider(provider, prompt);
        if (!text || !text.trim()) {
          throw new Error(`${provider} returned empty response`);
        }
        providerCircuitBreaker.recordSuccess(provider);
        return {
          text,
          provider,
          processingTime: performance.now() - startTime,
        };
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        providerCircuitBreaker.recordFailure(provider, errMsg);
        logger.warn(`[LLMRouter] Provider ${provider} failed`, { error: errMsg });
      }
    }

    throw new Error('[LLMRouter] All providers failed');
  }

  private async callProvider(provider: string, prompt: string): Promise<string> {
    switch (provider) {
      case 'glm':
        return this.callGLM(prompt);
      case 'groq':
        return this.callGroq(prompt);
      case 'sambanova':
        return this.callSambanova(prompt);
      case 'github':
        return this.callGithub(prompt);
      case 'nvidia':
        return this.callNvidia(prompt);
      case 'cloudflare':
        return this.callCloudflare(prompt);
      case 'openrouter':
        return this.callOpenrouter(prompt);
      case 'openai':
        return this.callOpenAI(prompt);
      case 'claude':
        return this.callClaude(prompt);
      case 'gemini':
        return this.callGemini(prompt);
      case 'mistral':
        return this.callMistral(prompt);
      case 'grok':
        return this.callGrok(prompt);
      case 'deepseek':
        return this.callDeepseek(prompt);
      default:
        throw new Error(`Unknown provider: ${provider}`);
    }
  }

  private async callGLM(prompt: string): Promise<string> {
    const apiKey = process.env.GLM_API_KEY;
    if (!apiKey) throw new Error('GLM_API_KEY not set');

    const glm = new OpenAI({ apiKey, baseURL: BASE_URLS.glm, timeout: 30_000, maxRetries: 0 });
    const response = await glm.chat.completions.create({
      model: MODEL_IDS.glm,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 512,
      thinking: { type: 'disabled' },
    } as any);

    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error('GLM returned empty content');
    return content;
  }

  private async callOpenAI(prompt: string): Promise<string> {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) throw new Error('OPENAI_API_KEY not set');

    const openai = new OpenAI({ apiKey, timeout: 30_000, maxRetries: 0 });
    const response = await openai.chat.completions.create({
      model: MODEL_IDS.openai,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 512,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error('OpenAI returned empty content');
    return content;
  }

  private async callGroq(prompt: string): Promise<string> {
    const apiKey = process.env.GROQ_API_KEY;
    if (!apiKey) throw new Error('GROQ_API_KEY not set');

    const groq = new OpenAI({ apiKey, baseURL: BASE_URLS.groq, timeout: 30_000, maxRetries: 0 });
    const response = await groq.chat.completions.create({
      model: MODEL_IDS.groq,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 512,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error('Groq returned empty content');
    return content;
  }

  private async callSambanova(prompt: string): Promise<string> {
    const apiKey = process.env.SAMBANOVA_API_KEY;
    if (!apiKey) throw new Error('SAMBANOVA_API_KEY not set');

    const client = new OpenAI({ apiKey, baseURL: BASE_URLS.sambanova, timeout: 30_000, maxRetries: 0 });
    const response = await client.chat.completions.create({
      model: MODEL_IDS.sambanova,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 512,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error('SambaNova returned empty content');
    return content;
  }

  private async callGithub(prompt: string): Promise<string> {
    const apiKey = process.env.GITHUB_TOKEN;
    if (!apiKey) throw new Error('GITHUB_TOKEN not set');

    const client = new OpenAI({ apiKey, baseURL: BASE_URLS.github, timeout: 30_000, maxRetries: 0 });
    const response = await client.chat.completions.create({
      model: MODEL_IDS.github,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 512,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error('GitHub Models returned empty content');
    return content;
  }

  private async callNvidia(prompt: string): Promise<string> {
    const apiKey = process.env.NVIDIA_API_KEY;
    if (!apiKey) throw new Error('NVIDIA_API_KEY not set');

    const client = new OpenAI({ apiKey, baseURL: BASE_URLS.nvidia, timeout: 30_000, maxRetries: 0 });
    const response = await client.chat.completions.create({
      model: MODEL_IDS.nvidia,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 512,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error('NVIDIA NIM returned empty content');
    return content;
  }

  private async callCloudflare(prompt: string): Promise<string> {
    const apiToken = process.env.CLOUDFLARE_API_TOKEN;
    const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;
    if (!apiToken) throw new Error('CLOUDFLARE_API_TOKEN not set');
    if (!accountId) throw new Error('CLOUDFLARE_ACCOUNT_ID not set');

    const baseURL = `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/v1`;
    const client = new OpenAI({ apiKey: apiToken, baseURL, timeout: 30_000, maxRetries: 0 });
    const response = await client.chat.completions.create({
      model: MODEL_IDS.cloudflare,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 512,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error('Cloudflare Workers AI returned empty content');
    return content;
  }

  private async callOpenrouter(prompt: string): Promise<string> {
    const apiKey = process.env.OPENROUTER_API_KEY;
    if (!apiKey) throw new Error('OPENROUTER_API_KEY not set');

    const client = new OpenAI({ apiKey, baseURL: BASE_URLS.openrouter, timeout: 30_000, maxRetries: 0 });
    const response = await client.chat.completions.create({
      model: MODEL_IDS.openrouter,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 512,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error('OpenRouter returned empty content');
    return content;
  }

  private async callClaude(prompt: string): Promise<string> {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) throw new Error('ANTHROPIC_API_KEY not set');

    const anthropic = new Anthropic({ apiKey, timeout: 30_000, maxRetries: 0 });
    const response = await anthropic.messages.create({
      model: MODEL_IDS.claude,
      max_tokens: 512,
      messages: [{ role: 'user', content: prompt }],
    });

    const block = response.content[0];
    return block && 'text' in block ? block.text : '';
  }

  private async callGemini(prompt: string): Promise<string> {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) throw new Error('GEMINI_API_KEY not set');

    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ model: MODEL_IDS.gemini });
    const result = await model.generateContent(prompt);
    return result.response.text();
  }

  private async callMistral(prompt: string): Promise<string> {
    const apiKey = process.env.MISTRAL_API_KEY;
    if (!apiKey) throw new Error('MISTRAL_API_KEY not set');

    const response = await axios.post(
      'https://api.mistral.ai/v1/chat/completions',
      {
        model: MODEL_IDS.mistral,
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

  private async callGrok(prompt: string): Promise<string> {
    const apiKey = process.env.GROK_API_KEY;
    if (!apiKey) throw new Error('GROK_API_KEY not set');

    const response = await axios.post(
      'https://api.x.ai/v1/chat/completions',
      {
        model: MODEL_IDS.grok,
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

  private async callDeepseek(prompt: string): Promise<string> {
    const apiKey = process.env.DEEPSEEK_API_KEY;
    if (!apiKey) throw new Error('DEEPSEEK_API_KEY not set');

    const response = await axios.post(
      'https://api.deepseek.com/v1/chat/completions',
      {
        model: MODEL_IDS.deepseek,
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
    switch (provider) {
      case 'glm':        return !!process.env.GLM_API_KEY;
      case 'openai':     return !!process.env.OPENAI_API_KEY;
      case 'claude':     return !!process.env.ANTHROPIC_API_KEY;
      case 'gemini':     return !!process.env.GEMINI_API_KEY;
      case 'mistral':    return !!process.env.MISTRAL_API_KEY;
      case 'deepseek':   return !!process.env.DEEPSEEK_API_KEY;
      case 'groq':       return !!process.env.GROQ_API_KEY;
      case 'grok':       return !!process.env.GROK_API_KEY;
      case 'sambanova':  return !!process.env.SAMBANOVA_API_KEY;
      case 'github':     return !!process.env.GITHUB_TOKEN;
      case 'nvidia':     return !!process.env.NVIDIA_API_KEY;
      case 'cloudflare': return !!process.env.CLOUDFLARE_API_TOKEN && !!process.env.CLOUDFLARE_ACCOUNT_ID;
      case 'openrouter': return !!process.env.OPENROUTER_API_KEY;
      default:           return false;
    }
  }
}
