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

export type TaskType = 'complex' | 'super-heavy' | 'heavy' | 'light';

/**
 * Check if a model ID qualifies as "super-heavy" (70B+ effective parameters).
 * Filters out small models that sneaked into heavy arrays (nano MoE, 8B, 20B, etc.)
 */
export function isSuperHeavyModel(modelId: string): boolean {
  const lower = modelId.toLowerCase();
  // Exclude small / distilled / MoE-with-small-active models
  const smallPatterns = ['nano', '8b', '20b', 'mini', 'instant', 'scout', 'flash-lite', 'gpt-oss-20b'];
  for (const pat of smallPatterns) {
    if (lower.includes(pat)) return false;
  }
  return true;
}

/**
 * Model category — distinguishes chat models from specialized models.
 * The discovery agent categorizes all models it finds; chat models go into
 * the heavy/light routing chains, while specialized models are cataloged
 * separately for use by vision, embedding, and other services.
 */
export type ModelCategory = 'chat' | 'vision' | 'embedding' | 'image-gen' | 'audio' | 'rerank' | 'other';

/**
 * API protocol type — determines which handler method to use.
 * Most new LLM providers are 'openai-compatible' and can be added dynamically
 * without writing a new handler.
 */
export type ProviderAPIType = 'openai-compatible' | 'anthropic' | 'google' | 'mistral';

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
  apiType: ProviderAPIType;
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
    apiType: 'openai-compatible',
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
    apiType: 'openai-compatible',
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
    apiType: 'openai-compatible',
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
    apiType: 'openai-compatible',
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
    apiType: 'google',
    heavy: [
      { id: 'gemini-3.6-flash', intelligence: 52, contextWindow: 1_000_000, speed: 300 },
      { id: 'gemini-3.5-flash', intelligence: 50, contextWindow: 1_000_000, speed: 300 },
      { id: 'gemini-3.5-flash-lite', intelligence: 37, contextWindow: 1_000_000, speed: 397 },
    ],
    light: [
      { id: 'gemini-3.5-flash-lite', intelligence: 37, contextWindow: 1_000_000, speed: 397 },
      { id: 'gemini-3.1-flash-lite', intelligence: 30, contextWindow: 1_000_000, speed: 350 },
    ],
  },

  cloudflare: {
    baseURL: '', // constructed dynamically from account ID
    envKey: 'CLOUDFLARE_API_TOKEN',
    apiType: 'openai-compatible',
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

  // ─── PAID providers (real cheap) ─────────────────────────────────────────

  cerebras: {
    baseURL: 'https://api.cerebras.ai/v1',
    envKey: 'CEREBRAS_API_KEY',
    apiType: 'openai-compatible',
    heavy: [
      { id: 'gpt-oss-120b', intelligence: 24, contextWindow: 131_000, speed: 1872 },
    ],
    light: [
      { id: 'gpt-oss-120b', intelligence: 24, contextWindow: 131_000, speed: 1872 },
    ],
  },

  deepseek: {
    baseURL: 'https://api.deepseek.com/v1',
    envKey: 'DEEPSEEK_API_KEY',
    apiType: 'openai-compatible',
    heavy: [
      { id: 'deepseek-v4-flash', intelligence: 52, contextWindow: 1_000_000, speed: 132 },
      { id: 'deepseek-chat', intelligence: 30, contextWindow: 128_000, speed: 59 },
    ],
    light: [
      { id: 'deepseek-v4-flash', intelligence: 52, contextWindow: 1_000_000, speed: 132 },
    ],
  },

  'gemini-paid': {
    baseURL: 'https://generativelanguage.googleapis.com/v1beta/openai',
    envKey: 'GEMINI_API_KEY',
    apiType: 'google',
    heavy: [
      { id: 'gemini-3.6-flash', intelligence: 52, contextWindow: 1_000_000, speed: 300 },
      { id: 'gemini-3.5-flash-lite', intelligence: 37, contextWindow: 1_000_000, speed: 397 },
    ],
    light: [
      { id: 'gemini-3.5-flash-lite', intelligence: 37, contextWindow: 1_000_000, speed: 397 },
    ],
  },

  claude: {
    baseURL: '',
    envKey: 'ANTHROPIC_API_KEY',
    apiType: 'anthropic',
    heavy: [
      { id: 'claude-haiku-4-5', intelligence: 60, contextWindow: 200_000, speed: 125 },
      { id: 'claude-sonnet-4-20250514', intelligence: 60, contextWindow: 200_000, speed: 80 },
    ],
    light: [
      { id: 'claude-haiku-4-5', intelligence: 60, contextWindow: 200_000, speed: 125 },
    ],
  },

  // ─── PAID providers (silenced — kept for manual selection only) ───────────

  mistral: {
    baseURL: 'https://api.mistral.ai/v1',
    envKey: 'MISTRAL_API_KEY',
    apiType: 'mistral',
    heavy: [{ id: 'mistral-medium', intelligence: 20 }],
    light: [{ id: 'mistral-medium', intelligence: 20 }],
  },

  grok: {
    baseURL: 'https://api.x.ai/v1',
    envKey: 'GROK_API_KEY',
    apiType: 'openai-compatible',
    heavy: [{ id: 'grok-4.20-0309-non-reasoning', intelligence: 25 }],
    light: [{ id: 'grok-4.20-0309-non-reasoning', intelligence: 25 }],
  },

  openai: {
    baseURL: '',
    envKey: 'OPENAI_API_KEY',
    apiType: 'openai-compatible',
    heavy: [{ id: 'gpt-4o', intelligence: 55 }],
    light: [{ id: 'gpt-4o', intelligence: 55 }],
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
 * SUPER-HEAVY chain — for complex reasoning tasks (gatherPlanContext, multi-round
 * clarification generation). Uses the same providers as HEAVY but filters out
 * small models (nano MoE, 8B, 20B) that can't handle complex JSON generation.
 * Groq first for speed (llama-3.3-70b after gpt-oss-120b rate-limits),
 * nvidia second for GLM-5.2 (intel 53) as a smart fallback.
 */
export const SUPER_HEAVY_CHAIN = [
  'groq',          // llama-3.3-70b-versatile (70B) or gpt-oss-120b (120B)
  'nvidia',        // glm-5.2 (intel 53) or nemotron-3-ultra-550b (550B)
  'sambanova',     // gpt-oss-120b (120B)
  'gemini-free',   // flash-lite (intel 37)
  'glm',           // GLM-4.7-Flash (intel 23)
  'cloudflare',    // GLM-4.7-Flash
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
 * COMPLEX chain — for command_automate tasks that need high intelligence + fast speed.
 * Paid-only — skips free providers entirely. Ordered by speed-to-cost ratio:
 * Cerebras (fastest) → DeepSeek (smartest cheap) → Gemini (cheap + 1M context) → Claude (frontier).
 */
export const COMPLEX_CHAIN = [
  'gemini-free',   // 3.6 Flash (intel 52, 1M context) — FREE, 1500 RPD, 15 RPM
  'cerebras',      // GPT-OSS-120B (intel 24, 1872 t/s) — paid, $0.25/$0.69
  'deepseek',      // V4 Flash (intel 52, 132 t/s) — paid, $0.14/$0.28
  'claude',        // Haiku 4.5 (intel 60, 125 t/s) — paid, $1/$5
] as const;

/**
 * PAID chain — fallback when all free providers fail.
 * Ordered cheapest to most expensive. Silenced providers (mistral, grok, openai)
 * are kept in PROVIDER_CONFIG for manual selection but not in the automatic chain.
 */
export const PAID_CHAIN = [
  'cerebras',      // GPT-OSS-120B — fastest paid
  'deepseek',      // V4 Flash — smartest cheap
  'claude',        // Haiku 4.5 — frontier quality
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
  if (taskTypeHint === 'complex' || taskTypeHint === 'command_automate') return 'complex';
  if (taskTypeHint === 'super-heavy') return 'super-heavy';
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
  if (taskType === 'complex') return [...COMPLEX_CHAIN]; // paid-only, no free providers
  const freeChain = taskType === 'light'
    ? LIGHT_CHAIN
    : taskType === 'super-heavy'
      ? SUPER_HEAVY_CHAIN
      : HEAVY_CHAIN;
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
  if (taskType === 'light') return config.light[0]?.id;
  // complex and super-heavy both use heavy models (complex = paid providers only)
  const heavyModels = taskType === 'super-heavy'
    ? config.heavy.filter(m => isSuperHeavyModel(m.id))
    : config.heavy;
  return heavyModels[0]?.id;
}

/**
 * Get all models for a provider and task type (for intra-provider fallback).
 */
export function getProviderModels(provider: string, taskType: TaskType): ProviderModel[] {
  const config = PROVIDER_CONFIG[provider];
  if (!config) return [];
  if (taskType === 'light') return config.light;
  // complex and super-heavy both use heavy models (complex = paid providers only)
  return taskType === 'super-heavy'
    ? config.heavy.filter(m => isSuperHeavyModel(m.id))
    : config.heavy;
}

/**
 * Get the base URL for a provider.
 * Cloudflare is special — constructed from account ID.
 * Falls back to catalogManager for dynamically added providers.
 */
export function getProviderBaseURL(provider: string): string {
  const config = PROVIDER_CONFIG[provider];
  if (config) {
    if (provider === 'cloudflare') {
      const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;
      return `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/v1`;
    }
    return config.baseURL;
  }
  // Check catalog for dynamically added providers (lazy import to avoid circular dep)
  try {
    const { catalogManager } = require('./catalogManager');
    const catProvider = catalogManager.getProvider(provider);
    if (catProvider) return catProvider.baseURL;
  } catch { /* ignore */ }
  return '';
}

/**
 * Get the API type for a provider.
 * Falls back to catalogManager for dynamically added providers.
 */
export function getProviderAPIType(provider: string): ProviderAPIType {
  const config = PROVIDER_CONFIG[provider];
  if (config) return config.apiType;
  try {
    const { catalogManager } = require('./catalogManager');
    const catProvider = catalogManager.getProvider(provider);
    if (catProvider?.apiType) return catProvider.apiType;
  } catch { /* ignore */ }
  return 'openai-compatible'; // default — most providers are OpenAI-compatible
}

/**
 * Get the env key for a provider.
 * Falls back to catalogManager for dynamically added providers.
 */
export function getProviderEnvKeyDynamic(provider: string): string {
  const config = PROVIDER_CONFIG[provider];
  if (config) return config.envKey;
  try {
    const { catalogManager } = require('./catalogManager');
    const catProvider = catalogManager.getProvider(provider);
    if (catProvider) return catProvider.envKey;
  } catch { /* ignore */ }
  return '';
}

/**
 * Check if a provider is configured (has the required env var set).
 * Falls back to catalogManager for dynamically added providers.
 */
export function isProviderConfiguredStatic(provider: string): boolean {
  const config = PROVIDER_CONFIG[provider];
  if (config) {
    if (provider === 'cloudflare') {
      return !!process.env.CLOUDFLARE_API_TOKEN && !!process.env.CLOUDFLARE_ACCOUNT_ID;
    }
    return !!process.env[config.envKey];
  }
  // Check catalog for dynamically added providers
  try {
    const { catalogManager } = require('./catalogManager');
    const catProvider = catalogManager.getProvider(provider);
    if (catProvider && catProvider.status !== 'dead') {
      return !!process.env[catProvider.envKey];
    }
  } catch { /* ignore */ }
  return false;
}

/**
 * Get the env key name for a provider (used for error messages).
 */
export function getProviderEnvKey(provider: string): string {
  return PROVIDER_CONFIG[provider]?.envKey || 'UNKNOWN';
}

/**
 * Get special (non-chat) models for a provider, filtered by category.
 * Used by vision, embedding, and other specialized services to discover
 * available models from the catalog.
 */
export function getSpecialModels(provider: string, category?: ModelCategory): ProviderModel[] {
  const config = PROVIDER_CONFIG[provider];
  if (!config?.special) return [];
  if (!category) return config.special;
  return config.special.filter(m => m.category === category);
}

/**
 * Get all special models across all configured providers, filtered by category.
 * Returns array of { provider, model } pairs.
 */
export function getAllSpecialModels(category?: ModelCategory): Array<{ provider: string; model: ProviderModel }> {
  const results: Array<{ provider: string; model: ProviderModel }> = [];
  for (const [name, config] of Object.entries(PROVIDER_CONFIG)) {
    if (!config.special) continue;
    for (const model of config.special) {
      if (category && model.category !== category) continue;
      results.push({ provider: name, model });
    }
  }
  return results;
}

/**
 * Classify a model ID into a category based on its name.
 * Used by the discovery agent when probing new models from provider catalogs.
 */
export function classifyModelCategory(modelId: string): ModelCategory {
  const lower = modelId.toLowerCase();
  // Obvious non-chat patterns (stable model families, unlikely to change).
  // This is a FAST-PATH FALLBACK only — the primary classifier is the probe
  // (probeModel sends "Say OK" and checks if the model responds to chat completion).
  // Anything not matched here defaults to 'chat' and gets probed at discovery time.
  if (/diffusion|sdxl|flux|dall|stable-diffusion|imagen|gpt-image/i.test(lower)) return 'image-gen';
  if (/whisper|tts|parakeet|canary|piper|bark/i.test(lower)) return 'audio';
  if (/embed|bge|gte|jina|nomic/i.test(lower)) return 'embedding';
  if (/rerank|re-rank|colbert/i.test(lower)) return 'rerank';
  if (/neva|llava|pixtral|\bvlm\b/i.test(lower)) return 'vision';
  // Default: assume chat — the probe will filter out non-chat models.
  // Multimodal chat models (gpt-4o, claude-3, gemini-3.x) correctly default here.
  // Safety guards, tiny models, etc. get probed and filtered by the probe result.
  return 'chat';
}
