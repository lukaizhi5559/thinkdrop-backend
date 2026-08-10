/**
 * Provider configuration — shared between llmRouter and llmStreamingRouter.
 *
 * Each provider has a HEAVY and LIGHT model chain for intra-provider fallback.
 * The router picks the chain based on task type, then iterates models within
 * each provider before moving to the next provider.
 *
 * Model intelligence scores from Artificial Analysis Intelligence Index v4.1.1
 * (https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index)
 */

export type TaskType = 'heavy' | 'light';

/**
 * Model category — distinguishes chat models from specialized models.
 * The discovery agent categorizes all models it finds; chat models go into
 * the heavy/light routing chains, while specialized models are cataloged
 * separately for use by vision, embedding, and other services.
 */
export type ModelCategory = 'chat' | 'vision' | 'embedding' | 'image-gen' | 'audio' | 'rerank' | 'other';

export interface ProviderModel {
  id: string;
  intelligence?: number;
  contextWindow?: number;
  /** Estimated output speed in tokens/sec (from benchmarks or runtime measurement) */
  speed?: number;
  /** Model category — chat models are routed; others are cataloged for specialized services */
  category?: ModelCategory;
}

export interface ProviderConfig {
  baseURL: string;
  envKey: string;
  catalogEndpoint?: string;
  heavy: ProviderModel[];
  light: ProviderModel[];
  /** Specialized (non-chat) models — vision, embedding, image-gen, etc. */
  special?: ProviderModel[];
}

export const PROVIDER_CONFIG: Record<string, ProviderConfig> = {
  // ─── FREE providers ──────────────────────────────────────────────────────

  nvidia: {
    baseURL: 'https://integrate.api.nvidia.com/v1',
    envKey: 'NVIDIA_API_KEY',
    catalogEndpoint: 'https://integrate.api.nvidia.com/v1/models',
    heavy: [
      { id: 'z-ai/glm-5.2', intelligence: 53, contextWindow: 1_000_000, speed: 17 },
      { id: 'nvidia/nemotron-3-ultra-550b-a55b', intelligence: 40, speed: 25 },
      { id: 'deepseek-ai/deepseek-v4-flash-0731', intelligence: 30, speed: 40 },
      { id: 'openai/gpt-oss-120b', intelligence: 24, contextWindow: 131_000, speed: 40 },
    ],
    light: [
      { id: 'meta/llama-3.1-8b-instruct', intelligence: 8, contextWindow: 128_000, speed: 200 },
      { id: 'openai/gpt-oss-20b', intelligence: 15, contextWindow: 131_000, speed: 150 },
      { id: 'nvidia/llama-3.1-nemotron-nano-8b-v1', intelligence: 7, speed: 200 },
    ],
    special: [
      { id: 'nvidia/nv-embedqa-e5-v5', category: 'embedding' },
      { id: 'nvidia/neva-22b', category: 'vision' },
    ],
  },

  groq: {
    baseURL: 'https://api.groq.com/openai/v1',
    envKey: 'GROQ_API_KEY',
    catalogEndpoint: 'https://api.groq.com/openai/v1/models',
    heavy: [
      { id: 'openai/gpt-oss-120b', intelligence: 24, contextWindow: 131_000, speed: 500 },
      { id: 'qwen/qwen3-32b', intelligence: 20, contextWindow: 131_000, speed: 400 },
      { id: 'llama-3.3-70b-versatile', intelligence: 9, contextWindow: 128_000, speed: 350 },
    ],
    light: [
      { id: 'llama-3.1-8b-instant', intelligence: 8, contextWindow: 128_000, speed: 700 },
      { id: 'openai/gpt-oss-20b', intelligence: 15, contextWindow: 128_000, speed: 1000 },
      { id: 'meta-llama/llama-4-scout-17b-16e-instruct', intelligence: 15, contextWindow: 128_000, speed: 600 },
    ],
  },

  glm: {
    baseURL: 'https://api.z.ai/api/paas/v4',
    envKey: 'GLM_API_KEY',
    heavy: [{ id: 'glm-4.7-flash', intelligence: 23, contextWindow: 200_000, speed: 97 }],
    light: [{ id: 'glm-4.7-flash', intelligence: 23, contextWindow: 200_000, speed: 97 }],
    special: [
      { id: 'glm-5v-turbo', category: 'vision', intelligence: 30 },
    ],
  },

  sambanova: {
    baseURL: 'https://api.sambanova.ai/v1',
    envKey: 'SAMBANOVA_API_KEY',
    catalogEndpoint: 'https://api.sambanova.ai/v1/models',
    heavy: [
      { id: 'gpt-oss-120b', intelligence: 24, contextWindow: 128_000, speed: 100 },
      { id: 'DeepSeek-V3.1', intelligence: 25, contextWindow: 128_000, speed: 80 },
      { id: 'MiniMax-M2.7', intelligence: 25, contextWindow: 192_000, speed: 80 },
    ],
    light: [
      { id: 'gpt-oss-120b', intelligence: 24, contextWindow: 128_000, speed: 100 },
    ],
  },

  'gemini-free': {
    baseURL: 'https://generativelanguage.googleapis.com/v1beta/openai',
    envKey: 'GEMINI_API_KEY_FREE',
    catalogEndpoint: 'https://generativelanguage.googleapis.com/v1beta/models',
    heavy: [
      { id: 'gemini-3.5-flash-lite', intelligence: 37, contextWindow: 1_000_000, speed: 397 },
      { id: 'gemini-3.1-flash-lite', intelligence: 30, contextWindow: 1_000_000, speed: 350 },
    ],
    light: [
      { id: 'gemini-3.5-flash-lite', intelligence: 37, contextWindow: 1_000_000, speed: 397 },
      { id: 'gemini-3.1-flash-lite', intelligence: 30, contextWindow: 1_000_000, speed: 350 },
    ],
  },

  cloudflare: {
    baseURL: '', // constructed dynamically from account ID
    envKey: 'CLOUDFLARE_API_TOKEN',
    heavy: [
      { id: '@cf/zai-org/glm-4.7-flash', intelligence: 23, speed: 50 },
      { id: '@cf/meta/llama-3.3-70b-instruct-fp8-fast', intelligence: 9, speed: 40 },
    ],
    light: [
      { id: '@cf/zai-org/glm-4.7-flash', intelligence: 23, speed: 50 },
      { id: '@cf/meta/llama-3.1-8b-instruct-fp8-fast', intelligence: 8, speed: 60 },
      { id: '@cf/meta/llama-3.2-3b-instruct', intelligence: 5, speed: 80 },
    ],
  },

  // ─── PAID providers ──────────────────────────────────────────────────────

  'gemini-paid': {
    baseURL: 'https://generativelanguage.googleapis.com/v1beta/openai',
    envKey: 'GEMINI_API_KEY',
    heavy: [{ id: 'gemini-3.6-flash', intelligence: 52, contextWindow: 1_000_000 }],
    light: [{ id: 'gemini-3.5-flash-lite', intelligence: 37, contextWindow: 1_000_000 }],
  },

  deepseek: {
    baseURL: 'https://api.deepseek.com/v1',
    envKey: 'DEEPSEEK_API_KEY',
    heavy: [{ id: 'deepseek-chat', intelligence: 30 }],
    light: [{ id: 'deepseek-chat', intelligence: 30 }],
  },

  mistral: {
    baseURL: 'https://api.mistral.ai/v1',
    envKey: 'MISTRAL_API_KEY',
    heavy: [{ id: 'mistral-medium', intelligence: 20 }],
    light: [{ id: 'mistral-medium', intelligence: 20 }],
  },

  grok: {
    baseURL: 'https://api.x.ai/v1',
    envKey: 'GROK_API_KEY',
    heavy: [{ id: 'grok-4.20-0309-non-reasoning', intelligence: 25 }],
    light: [{ id: 'grok-4.20-0309-non-reasoning', intelligence: 25 }],
  },

  openai: {
    baseURL: '',
    envKey: 'OPENAI_API_KEY',
    heavy: [{ id: 'gpt-4o', intelligence: 55 }],
    light: [{ id: 'gpt-4o', intelligence: 55 }],
  },

  claude: {
    baseURL: '',
    envKey: 'ANTHROPIC_API_KEY',
    heavy: [{ id: 'claude-sonnet-4-20250514', intelligence: 60 }],
    light: [{ id: 'claude-sonnet-4-20250514', intelligence: 60 }],
  },
};

// ─── Fallback chains ────────────────────────────────────────────────────────

/**
 * HEAVY chain — for planning, synthesis, complex skill steps (>5K chars).
 * Ordered by speed-to-intelligence ratio: fast + decent intelligence first,
 * smartest-slowest as fallback. A 400-token response at 500 t/s takes <1s,
 * while the same at 17 t/s takes 24s — the intelligence gap (53 vs 24-37)
 * rarely justifies 24× slower response for most planning tasks.
 */
export const HEAVY_CHAIN = [
  'groq',          // gpt-oss-120b (intel 24, 500 t/s) — fast, 1K RPD per model
  'gemini-free',   // flash-lite (intel 37, 397 t/s) — fast + smart, ~1,500 RPD
  'glm',           // GLM-4.7-Flash (intel 23, 97 t/s) — reliable, no published limits
  'nvidia',        // GLM-5.2 (intel 53, 17 t/s) — smartest but slowest, fallback
  'sambanova',     // gpt-oss-120b (intel 24, ~100 t/s) — 20 RPD, last resort
  'cloudflare',    // GLM-4.7-Flash (intel 23, 50 t/s) — slow, expensive in neurons
] as const;

/**
 * LIGHT chain — for heartbeats, simple skill steps (<5K chars), classification.
 * Ordered by speed and rate limit generosity for high-volume simple tasks.
 */
export const LIGHT_CHAIN = [
  'groq',          // llama-3.1-8b-instant: 14,400 RPD, 500K TPD, 840 t/s
  'nvidia',        // llama-3.1-8b-instruct: fast, 40 RPM shared
  'glm',           // glm-4.7-flash: completely free, 97 t/s
  'cloudflare',    // GLM-4.7-Flash: low neuron cost
  'gemini-free',   // flash-lite: 397 t/s, ~1,500 RPD
  'sambanova',     // gpt-oss-120b: 20 RPD — last resort for light
] as const;

/**
 * PAID chain — fallback when all free providers fail.
 * Ordered cheapest to most expensive.
 */
export const PAID_CHAIN = [
  'gemini-paid',   // gemini-3.6-flash (52), $1.50/$7.50 per M tokens
  'deepseek',      // deepseek-chat, ~$0.14/M tokens
  'mistral',       // mistral-medium, cheap
  'grok',          // grok-4.20, cheap-ish
  'openai',        // gpt-4o, expensive
  'claude',        // claude-sonnet-4, most expensive
] as const;

// ─── Task type detection ────────────────────────────────────────────────────

/**
 * Detect task type from request metadata.
 * Uses clientId prefix, metadata.source, prompt length, and options.taskType.
 */
export function detectTaskType(
  clientId: string | undefined,
  source: string | undefined,
  promptLength: number,
  taskTypeHint: string | undefined
): TaskType {
  // Heartbeats — always light
  if (clientId?.startsWith('hb_')) return 'light';
  // Explicit task type hints
  if (taskTypeHint === 'heartbeat' || taskTypeHint === 'classification') return 'light';
  if (taskTypeHint === 'planning' || taskTypeHint === 'synthesis') return 'heavy';
  // Skill steps — light if short prompt, heavy if long
  if (clientId?.startsWith('skill_')) return promptLength < 5000 ? 'light' : 'heavy';
  // Stategraph — heavy (planning, synthesis)
  if (clientId?.startsWith('stategraph_')) return 'heavy';
  if (source === 'stategraph_module') return 'heavy';
  // Default: heavy (safer for quality)
  return 'heavy';
}

/**
 * Get the fallback chain for a task type, with paid chain appended.
 */
export function getFallbackChain(taskType: TaskType): readonly string[] {
  const freeChain = taskType === 'light' ? LIGHT_CHAIN : HEAVY_CHAIN;
  return [...freeChain, ...PAID_CHAIN];
}

/**
 * Get models for a provider and task type.
 * Returns only the first model (for backward compatibility with single-model callers).
 * Use getProviderModels() for intra-provider fallback.
 */
export function getProviderModelId(provider: string, taskType: TaskType): string | undefined {
  const config = PROVIDER_CONFIG[provider];
  if (!config) return undefined;
  const models = taskType === 'light' ? config.light : config.heavy;
  return models[0]?.id;
}

/**
 * Get all models for a provider and task type (for intra-provider fallback).
 */
export function getProviderModels(provider: string, taskType: TaskType): ProviderModel[] {
  const config = PROVIDER_CONFIG[provider];
  if (!config) return [];
  return taskType === 'light' ? config.light : config.heavy;
}

/**
 * Get the base URL for a provider.
 * Cloudflare is special — constructed from account ID.
 */
export function getProviderBaseURL(provider: string): string {
  const config = PROVIDER_CONFIG[provider];
  if (!config) return '';
  if (provider === 'cloudflare') {
    const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;
    return `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/v1`;
  }
  return config.baseURL;
}

/**
 * Check if a provider is configured (has the required env var set).
 */
export function isProviderConfiguredStatic(provider: string): boolean {
  const config = PROVIDER_CONFIG[provider];
  if (!config) return false;
  if (provider === 'cloudflare') {
    return !!process.env.CLOUDFLARE_API_TOKEN && !!process.env.CLOUDFLARE_ACCOUNT_ID;
  }
  return !!process.env[config.envKey];
}

/**
 * Get the env key name for a provider (used for error messages).
 */
export function getProviderEnvKey(provider: string): string {
  return PROVIDER_CONFIG[provider]?.envKey || 'UNKNOWN';
}
