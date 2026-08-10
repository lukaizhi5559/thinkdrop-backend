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
import {
  PROVIDER_CONFIG,
  HEAVY_CHAIN,
  LIGHT_CHAIN,
  PAID_CHAIN,
  detectTaskType,
  getProviderModels,
  getProviderBaseURL,
  isProviderConfiguredStatic,
  TaskType,
} from './providerConfig';

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
      ? 'light' : 'heavy';

    // Build ordered chain: preferred first, then fallback chain with round-robin
    const baseChain = taskType === 'light' ? [...LIGHT_CHAIN, ...PAID_CHAIN] : [...HEAVY_CHAIN, ...PAID_CHAIN];
    const rotatedChain = providerCircuitBreaker.getRotatedChain(baseChain);
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

      // Intra-provider model fallback
      const models = getProviderModels(provider, taskType);
      for (const model of models) {
        try {
          const text = await this.callProvider(provider, prompt, model.id);
          if (!text || !text.trim()) {
            throw new Error(`${provider}/${model.id} returned empty response`);
          }
          const estTokens = Math.ceil((prompt.length + text.length) / 4);
          providerCircuitBreaker.recordSuccess(provider, estTokens);
          return {
            text,
            provider,
            processingTime: performance.now() - startTime,
          };
        } catch (err) {
          const errMsg = err instanceof Error ? err.message : String(err);
          providerCircuitBreaker.recordFailure(provider, errMsg);
          logger.warn(`[LLMRouter] Provider ${provider} model ${model.id} failed`, { error: errMsg });
        }
      }
    }

    throw new Error('[LLMRouter] All providers failed');
  }

  private async callProvider(provider: string, prompt: string, modelId: string): Promise<string> {
    switch (provider) {
      case 'glm':
        return this.callGLM(prompt, modelId);
      case 'groq':
        return this.callGroq(prompt, modelId);
      case 'sambanova':
        return this.callSambanova(prompt, modelId);
      case 'nvidia':
        return this.callNvidia(prompt, modelId);
      case 'cloudflare':
        return this.callCloudflare(prompt, modelId);
      case 'openai':
        return this.callOpenAI(prompt, modelId);
      case 'claude':
        return this.callClaude(prompt, modelId);
      case 'gemini-free':
      case 'gemini-paid':
        return this.callGemini(prompt, modelId, provider);
      case 'mistral':
        return this.callMistral(prompt, modelId);
      case 'grok':
        return this.callGrok(prompt, modelId);
      case 'deepseek':
        return this.callDeepseek(prompt, modelId);
      default:
        throw new Error(`Unknown provider: ${provider}`);
    }
  }

  private async callGLM(prompt: string, modelId: string): Promise<string> {
    const apiKey = process.env.GLM_API_KEY;
    if (!apiKey) throw new Error('GLM_API_KEY not set');

    const glm = new OpenAI({ apiKey, baseURL: getProviderBaseURL('glm'), timeout: 30_000, maxRetries: 0 });
    const response = await glm.chat.completions.create({
      model: modelId,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 512,
      thinking: { type: 'disabled' },
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

    const groq = new OpenAI({ apiKey, baseURL: getProviderBaseURL('groq'), timeout: 30_000, maxRetries: 0 });
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

    const client = new OpenAI({ apiKey, baseURL: getProviderBaseURL('sambanova'), timeout: 30_000, maxRetries: 0 });
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

    const client = new OpenAI({ apiKey, baseURL: getProviderBaseURL('nvidia'), timeout: 30_000, maxRetries: 0 });
    const response = await client.chat.completions.create({
      model: modelId,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 512,
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
    const client = new OpenAI({ apiKey: apiToken, baseURL, timeout: 30_000, maxRetries: 0 });
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
