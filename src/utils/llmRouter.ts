/**
 * Base LLM router — handles non-streaming prompt dispatch with provider fallback chain.
 * Used by LLMElementMatcher for element matching calls.
 */

import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenerativeAI } from '@google/generative-ai';
import axios from 'axios';
import { logger } from './logger';

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

    const providers = ['openai', 'claude', 'gemini', 'mistral', 'deepseek'];
    const ordered = preferred
      ? [preferred, ...providers.filter((p) => p !== preferred)]
      : providers;

    for (const provider of ordered) {
      try {
        const text = await this.callProvider(provider, prompt);
        return {
          text,
          provider,
          processingTime: performance.now() - startTime,
        };
      } catch (err) {
        logger.warn(`[LLMRouter] Provider ${provider} failed`, {
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }

    throw new Error('[LLMRouter] All providers failed');
  }

  private async callProvider(provider: string, prompt: string): Promise<string> {
    switch (provider) {
      case 'openai':
        return this.callOpenAI(prompt);
      case 'claude':
        return this.callClaude(prompt);
      case 'gemini':
        return this.callGemini(prompt);
      case 'mistral':
        return this.callMistral(prompt);
      case 'deepseek':
        return this.callDeepseek(prompt);
      default:
        throw new Error(`Unknown provider: ${provider}`);
    }
  }

  private async callOpenAI(prompt: string): Promise<string> {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) throw new Error('OPENAI_API_KEY not set');

    const openai = new OpenAI({ apiKey });
    const response = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 512,
    });

    return response.choices[0]?.message?.content || '';
  }

  private async callClaude(prompt: string): Promise<string> {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) throw new Error('ANTHROPIC_API_KEY not set');

    const anthropic = new Anthropic({ apiKey });
    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
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
    const model = genAI.getGenerativeModel({ model: 'gemini-pro' });
    const result = await model.generateContent(prompt);
    return result.response.text();
  }

  private async callMistral(prompt: string): Promise<string> {
    const apiKey = process.env.MISTRAL_API_KEY;
    if (!apiKey) throw new Error('MISTRAL_API_KEY not set');

    const response = await axios.post(
      'https://api.mistral.ai/v1/chat/completions',
      {
        model: 'mistral-medium',
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
        model: 'deepseek-chat',
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
}
