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

export type TaskType = 'complex' | 'super-heavy' | 'heavy' | 'light' | 'conversational';

/**
 * Sanitize prompt text to remove malformed UTF-8 that causes 400 errors on
 * all providers. Malformed hex escapes and lone Unicode surrogates in prompt
 * content (e.g. from screen capture / OCR) cause every provider to reject the
 * request with "unexpected end of hex escape" or "no low surrogate" errors.
 *
 * This strips the problematic sequences before sending, so a single bad prompt
 * doesn't take down the entire fallback chain.
 */
export function sanitizePrompt(text: string): string {
  if (!text) return text;
  return text
    // Remove malformed hex escape sequences (\x not followed by valid hex pair)
    .replace(/\\x([^0-9a-fA-F]{2}|$)/g, '')
    // Remove lone high surrogates (not followed by a low surrogate)
    .replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])/g, '')
    // Remove lone low surrogates (not preceded by a high surrogate)
    .replace(/(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g, '');
}

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
  /** Does this model do reasoning by default? (from preflight profiler) */
  isReasoning?: boolean;
  /** Can reasoning/thinking be disabled via API params? (from preflight profiler) */
  canDisableThinking?: boolean;
  /**
   * How this model exposes reasoning content (from preflight profiler):
   *  - 'separate' : model emits reasoning in a dedicated response field
   *                 (e.g. `delta.reasoning_content`); the router reads it
   *                 separately and forwards it as `chunk.reasoning`.
   *  - 'disabled' : reasoning is actively turned off via provider params
   *                 (see getDisableThinkingParams) for latency/output cleanliness.
   *  - 'none'     : model does not reason, or reasoning is not exposed.
   *  - undefined  : not yet profiled. The router applies an optimistic
   *                 separate-params attempt (where a registry entry exists)
   *                 plus the inline-think safety net.
   */
  reasoningMode?: 'none' | 'separate' | 'disabled';
  /** Does this model support stream:true? (from preflight profiler) */
  supportsStreaming?: boolean;
  /** When was this model last benchmarked by the profiler? (ISO timestamp) */
  benchmarkedAt?: string;
  /** Raw benchmark speed (t/s) from preflight — distinct from runtime EMA speed */
  benchmarkSpeed?: number;
}

export interface ProviderConfig {
  baseURL: string;
  envKey: string;
  catalogEndpoint?: string;
  apiType: ProviderAPIType;
  heavy: ProviderModel[];
  light: ProviderModel[];
  /** Conversational models — optimized for real-time chat (fast TTFT, streaming, short responses) */
  conversational?: ProviderModel[];
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
      // Verified live on /v1/chat/completions Sep 2026 — prior seeds were EOL
      // (openai/gpt-oss-120b 410) or hung >20s (deepseek-v4-flash-0731).
      { id: 'nvidia/nemotron-3-super-120b-a12b', intelligence: 45, speed: 30 },
      { id: 'nvidia/nemotron-3-ultra-550b-a55b', intelligence: 40, speed: 25 },
    ],
    light: [
      { id: 'openai/gpt-oss-20b', intelligence: 15, contextWindow: 131_000, speed: 150 },
      { id: 'nvidia/nemotron-3.5-lightning-30b-a3b', intelligence: 20, contextWindow: 128_000, speed: 100 },
      { id: 'meta/llama-3.2-11b-vision-instruct', intelligence: 10, contextWindow: 128_000, speed: 100 },
    ],
    special: [
      { id: 'nvidia/llama-3.2-nv-embedqa-1b-v1', category: 'embedding' },
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
    conversational: [
      { id: 'openai/gpt-oss-20b', intelligence: 15, contextWindow: 128_000, speed: 1000 },
      { id: 'llama-3.1-8b-instant', intelligence: 8, contextWindow: 128_000, speed: 840 },
    ],
  },

  glm: {
    baseURL: 'https://api.z.ai/api/paas/v4',
    envKey: 'GLM_API_KEY',
    apiType: 'openai-compatible',
    heavy: [{ id: 'glm-4.7-flash', intelligence: 23, contextWindow: 200_000, speed: 97 }],
    light: [{ id: 'glm-4.7-flash', intelligence: 23, contextWindow: 200_000, speed: 97 }],
    conversational: [{ id: 'glm-4.7-flash', intelligence: 23, contextWindow: 200_000, speed: 97 }],
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
    conversational: [
      { id: 'gemini-3.5-flash-lite', intelligence: 37, contextWindow: 1_000_000, speed: 397 },
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

  // ─── FREE aggregator (OpenRouter free variants — slow but smart) ──────────

  openrouter: {
    baseURL: 'https://openrouter.ai/api/v1',
    envKey: 'OPENROUTER_API_KEY',
    apiType: 'openai-compatible',
    heavy: [
      { id: 'z-ai/glm-5.2:free', intelligence: 86, contextWindow: 256_000, speed: 50 },
    ],
    light: [
      { id: 'z-ai/glm-5.2:free', intelligence: 86, contextWindow: 256_000, speed: 50 },
    ],
  },

  // ─── FREE-PREMIUM providers (free tier, paid-quality) ─────────────────────

  cerebras: {
    baseURL: 'https://api.cerebras.ai/v1',
    envKey: 'CEREBRAS_API_KEY',
    apiType: 'openai-compatible',
    // Free tier: 1M TPD, 30 RPM, no credit card. Ultra-fast WSE-3 hardware.
    // gemma-4-31b removed — 404s on this account (catalog /v1/models exposes only
    // gpt-oss-120b + zai-glm-4.7; "degraded after 53 failures" in logs).
    // zai-glm-4.7 added — GLM 4.7, 355B, ~1000 t/s, smart + fast fallback.
    heavy: [
      { id: 'gpt-oss-120b', intelligence: 24, contextWindow: 131_000, speed: 3000 },
      { id: 'zai-glm-4.7', intelligence: 53, contextWindow: 131_000, speed: 1000 },
    ],
    light: [
      { id: 'gpt-oss-120b', intelligence: 24, contextWindow: 131_000, speed: 3000 },
      { id: 'zai-glm-4.7', intelligence: 53, contextWindow: 131_000, speed: 1000 },
    ],
    conversational: [
      { id: 'gpt-oss-120b', intelligence: 24, contextWindow: 131_000, speed: 3000 },
    ],
  },

  cohere: {
    baseURL: 'https://api.cohere.ai/compatibility/v1',
    envKey: 'COHERE_API_KEY',
    catalogEndpoint: 'https://api.cohere.ai/compatibility/v1/models',
    apiType: 'openai-compatible',
    // Free trial key: 1K calls/month, 20 RPM, no credit card.
    // Uses Cohere's OpenAI-compatible endpoint (not native V2 API).
    heavy: [
      { id: 'command-a-plus-05-2026', intelligence: 50, contextWindow: 436_000, speed: 80 },
      { id: 'command-a-03-2025', intelligence: 40, contextWindow: 288_000, speed: 100 },
    ],
    light: [
      { id: 'command-r7b', intelligence: 20, contextWindow: 128_000, speed: 150 },
    ],
    special: [
      { id: 'embed-4', category: 'embedding' },
      { id: 'rerank-3-5', category: 'rerank' },
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

  // ─── FREE-PREMIUM providers (free mode, paid-quality) ─────────────────────

  mistral: {
    baseURL: 'https://api.mistral.ai/v1',
    envKey: 'MISTRAL_API_KEY',
    apiType: 'mistral',
    // Free mode: 1B tokens/month, ~1-2 RPS. Discovery agent auto-found these
    // free-tier models (see backend.log). Low RPM — circuit breaker handles rate limits.
    heavy: [
      { id: 'open-mixtral-8x7b', intelligence: 58, contextWindow: 128_000, speed: 300 },
      { id: 'mistral-medium', intelligence: 20 },
    ],
    light: [
      { id: 'mistral-small-latest', intelligence: 28, speed: 460 },
      { id: 'mistral-tiny', intelligence: 15, speed: 800 },
      { id: 'open-mistral-7b', intelligence: 28, speed: 420 },
    ],
    conversational: [
      { id: 'mistral-small-latest', intelligence: 28, speed: 460 },
    ],
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
  'nvidia',        // deepseek-v4-flash (intel 30) or nemotron-ultra (intel 40) — fallback
  'sambanova',     // gpt-oss-120b (intel 24, ~100 t/s) — 20 RPD, last resort
  'openrouter',    // GLM-5.2:free (intel 86, ~50 t/s) — slow but smart, 50 RPD, last resort
  // cloudflare removed — can't disable reasoning, 20s for simple tasks
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
  'gemini-free',   // 3.6-flash (intel 52) or 3.5-flash (intel 50) — fast + smart
  'sambanova',     // gpt-oss-120b (120B)
  'nvidia',        // nemotron-ultra-550b (intel 40) or deepseek-v4-flash (intel 30)
  'glm',           // GLM-4.7-Flash (intel 23)
  'openrouter',    // GLM-5.2:free (intel 86, 256K context) — slow but smart, last resort
  // cloudflare removed — can't disable reasoning, 20s for simple tasks
] as const;

/**
 * LIGHT chain — for heartbeats, simple skill steps (<5K chars), classification.
 * Ordered by speed and rate limit generosity for high-volume simple tasks.
 */
export const LIGHT_CHAIN = [
  'groq',          // llama-3.1-8b-instant: 14,400 RPD, 500K TPD, 840 t/s
  'nvidia',        // llama-3.1-8b-instruct: fast, 40 RPM shared
  'glm',           // glm-4.7-flash: completely free, 97 t/s, thinking disabled
  'gemini-free',   // flash-lite: 397 t/s, ~1,500 RPD
  'sambanova',     // gpt-oss-120b: 20 RPD — last resort for light
  'openrouter',    // GLM-5.2:free — slow but smart, 50 RPD, last resort
  // cloudflare removed — can't disable reasoning, ignores max_tokens for GLM
] as const;

/**
 * CONVERSATIONAL chain — for real-time chat (comms-graph). Optimized for fast TTFT,
 * streaming support, and good-enough quality. Short timeouts (3s watchdog, 8s max).
 * Ordered by TTFT + RPD generosity: Groq (highest RPD) → Cerebras (ultra-fast) →
 * Gemini (smartest) → GLM (reliable) → Mistral (good quality, low RPM).
 */
export const CONVERSATIONAL_CHAIN = [
  'groq',          // gpt-oss-20b (intel 15, 1000 t/s) — fastest TTFT, 1K RPD
  'cerebras',      // gpt-oss-120b (intel 24, 3000 t/s) — ultra-fast, 30 RPM cap
  'gemini-free',   // flash-lite (intel 37, 397 t/s) — best quality in chain
  'glm',           // glm-4.7-flash (intel 23, 97 t/s) — reliable, no limits
  'mistral',       // listed but excluded for conversational — see TASK_TYPE_PROVIDER_EXCLUSIONS
] as const;

/**
 * FREE-PREMIUM chain — free providers with paid-quality models. Tried after regular
 * free chains but before PAID_CHAIN. These providers have free tiers/modes but offer
 * intelligence comparable to paid providers. Ordered by speed-to-intelligence ratio.
 */
export const FREE_PREMIUM_CHAIN = [
  'cerebras',      // gpt-oss-120b (intel 24, 3000 t/s) — 1M TPD free, no credit card
  'mistral',       // open-mixtral-8x7b (intel 58, 300 t/s) — 1B tokens/month free mode
  'cohere',        // command-a-plus 218B (intel ~50+, 436K context) — 1K calls/month free trial
] as const;

/**
 * COMPLEX chain — for command_automate tasks that need high intelligence + fast speed.
 * Free-premium providers first (Gemini, Cerebras, Mistral, Cohere), then paid providers
 * (DeepSeek, Claude). Free providers from HEAVY_CHAIN are appended as a safety net
 * via getFallbackChain() so complex tasks don't hard-fail.
 */
export const COMPLEX_CHAIN = [
  'gemini-free',   // 3.6 Flash (intel 52, 1M context) — FREE, 1500 RPD, 15 RPM
  'cerebras',      // GPT-OSS-120B (intel 24, 3000 t/s) — FREE tier, 1M TPD
  'mistral',       // open-mixtral-8x7b (intel 58, 300 t/s) — FREE mode, 1B tokens/month
  'cohere',        // command-a-plus 218B (intel ~50+, 436K context) — FREE trial, 1K calls/month
  'deepseek',      // V4 Flash (intel 52, 132 t/s) — paid, $0.14/$0.28
  'claude',        // Haiku 4.5 (intel 60, 125 t/s) — paid, $1/$5
] as const;

/**
 * PAID chain — fallback when all free and free-premium providers fail.
 * Only actual paid providers. Cerebras removed (has free tier — now in FREE_PREMIUM_CHAIN).
 * Silenced providers (grok, openai) are kept in PROVIDER_CONFIG for manual selection
 * but not in the automatic chain.
 */
export const PAID_CHAIN = [
  'deepseek',      // V4 Flash — smartest cheap
  'claude',        // Haiku 4.5 — frontier quality
] as const;

// ─── Per-task-type provider exclusions ──────────────────────────────────────

/**
 * Providers that must never serve a given task type, regardless of chain
 * membership (including safety-net tails that would otherwise re-add them).
 *
 * mistral → conversational: mistral's safety layer over-triggers under
 * persona-style system prompts ("lovingly redirect anything not in their best
 * interest"), returning the 36-char canned refusal "I'm afraid I can't assist
 * with that." for innocuous chat. Its remaining active catalog models are
 * voxtral audio models and ministral-3b/8b — all wrong for persona chat.
 * Mistral stays in FREE_PREMIUM/COMPLEX for non-persona tasks.
 */
export const TASK_TYPE_PROVIDER_EXCLUSIONS: Partial<Record<TaskType, readonly string[]>> = {
  conversational: ['mistral'],
};

export function isProviderExcludedForTaskType(provider: string, taskType: TaskType): boolean {
  return TASK_TYPE_PROVIDER_EXCLUSIONS[taskType]?.includes(provider) ?? false;
}

// ─── Reasoning separation ────────────────────────────────────────────────────

/**
 * Per-provider configuration for models that expose reasoning in a separate
 * response field rather than inline in `delta.content`.
 *
 * To add a new provider: add an entry here. The streaming router will then
 * send `separateParams` on profiled-separate models (and optimistically on
 * unprofiled models from this provider), and read reasoning from
 * `delta[responseField]`.
 */
export interface ReasoningProviderConfig {
  /** Request params that move reasoning into a separate response field. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  separateParams?: Record<string, any>;
  /** Response delta field that carries reasoning (default 'reasoning_content'). */
  responseField?: string;
}

export const REASONING_PROVIDER_CONFIG: Record<string, ReasoningProviderConfig> = {
  // Groq defaults to 'raw' reasoning (inline in delta.content) when no
  // reasoning_format is supplied. 'parsed' moves it to delta.reasoning_content.
  groq: { separateParams: { reasoning_format: 'parsed' }, responseField: 'reasoning_content' },
  // Cerebras hosts gpt-oss models behind an OpenAI-compatible API; the same
  // reasoning_format param is accepted (profiler confirms at runtime; 400-retry
  // protects the request path if it is rejected).
  cerebras: { separateParams: { reasoning_format: 'parsed' }, responseField: 'reasoning_content' },
  // DeepSeek returns reasoning in delta.reasoning_content by default — no
  // request param needed, just read the field.
  deepseek: { responseField: 'reasoning_content' },
  // OpenRouter normalises provider reasoning into delta.reasoning.
  openrouter: { responseField: 'reasoning' },
};

/**
 * Resolve the reasoning request params and the response field to read, based
 * on the catalog profile (preferred) or the provider registry (optimistic
 * fallback for unprofiled models). Shared by the streaming and non-streaming
 * routers so behaviour stays consistent.
 *
 * Returns:
 *  - params  : request params to spread into the chat.completions call
 *  - responseField : delta field to read reasoning from, or null if reasoning
 *                    is disabled / not exposed / not expected
 */
export function resolveReasoningParams(
  provider: string,
  _modelId: string,
  catalogModel?: { reasoningMode?: 'none' | 'separate' | 'disabled'; canDisableThinking?: boolean; isReasoning?: boolean }
): { params: Record<string, any>; responseField: string | null } {
  const registry = REASONING_PROVIDER_CONFIG[provider];

  // Profiled models — trust the catalog.
  if (catalogModel?.reasoningMode === 'separate') {
    return {
      params: { ...(registry?.separateParams ?? {}) },
      responseField: registry?.responseField ?? 'reasoning_content',
    };
  }
  if (catalogModel?.reasoningMode === 'disabled' || catalogModel?.canDisableThinking) {
    // Reasoning is actively turned off — use disable params, no field to read.
    return { params: getDisableThinkingParams(provider, _modelId, catalogModel), responseField: null };
  }
  if (catalogModel?.reasoningMode === 'none') {
    return { params: {}, responseField: null };
  }
  if (catalogModel?.isReasoning === false) {
    return { params: {}, responseField: null };
  }

  // Unprofiled model — optimistic fallback.
  if (registry?.separateParams) {
    // Provider accepts a separate-reasoning param (groq/cerebras). Send it
    // optimistically; it is harmless on non-reasoning models. Read the field.
    return { params: { ...registry.separateParams }, responseField: registry.responseField ?? 'reasoning_content' };
  }
  // No registry entry — fall back to existing disable-thinking logic (glm,
  // nvidia, cloudflare) so current behaviour is preserved for known providers.
  return { params: getDisableThinkingParams(provider, _modelId, catalogModel), responseField: null };
}

/**
 * Returns provider-specific params to disable thinking/reasoning mode.
 * - GLM (z.ai) / Cloudflare: `thinking: { type: 'disabled' }`
 * - NVIDIA reasoning models: `chat_template_kwargs: { enable_thinking: false }`
 * - Others: no params (no reasoning or can't disable)
 *
 * Checks catalog profile data (canDisableThinking) first, falls back to
 * hardcoded set + pattern matching for models not yet profiled. Shared by
 * both routers (streaming and non-streaming) and by resolveReasoningParams.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function getDisableThinkingParams(provider: string, modelId: string, catalogModel?: { canDisableThinking?: boolean; isReasoning?: boolean }): Record<string, any> {
  // Check catalog profile data first (from preflight profiler)
  if (catalogModel?.canDisableThinking) {
    if (provider === 'glm' || provider === 'cloudflare') {
      return { thinking: { type: 'disabled' } };
    }
    return { chat_template_kwargs: { enable_thinking: false } };
  }
  // If catalog says it's NOT reasoning, skip
  if (catalogModel?.isReasoning === false) {
    return {};
  }
  // Fallback to hardcoded logic for models not yet profiled
  if (provider === 'glm') {
    return { thinking: { type: 'disabled' } };
  }
  if (provider === 'cloudflare') {
    return { thinking: { type: 'disabled' } };
  }
  if (provider === 'nvidia' && isNvidiaReasoningModel(modelId)) {
    return { chat_template_kwargs: { enable_thinking: false } };
  }
  return {};
}

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
  'nvidia/llama-3.3-nemotron-super-49b-v1',
  'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'nvidia/nemotron-3.5-lightning-30b-a3b',
  'nvidia/nemotron-3-nano-30b-a3b',
]);

/**
 * Pattern-based reasoning model detection for NVIDIA.
 * Catches future nemotron models with "super", "lightning", or "reasoning"
 * in the name that aren't in the explicit set above.
 */
export function isNvidiaReasoningModel(modelId: string): boolean {
  if (NVIDIA_REASONING_MODELS.has(modelId)) return true;
  const lower = modelId.toLowerCase();
  if (lower.includes('nemotron') && (lower.includes('super') || lower.includes('lightning') || lower.includes('reasoning'))) {
    return true;
  }
  return false;
}

// ─── Inline-think safety net ────────────────────────────────────────────────

/**
 * Stream-aware stripper for inline reasoning blocks that leak into
 * `delta.content` despite separate-reasoning params. Handles tags split
 * across chunk boundaries by buffering a small carry.
 *
 * Recognised wrappers (case-insensitive, matched as full literal tags):
 *    <think> ...      (groq gpt-oss raw format - bare tags, no angle brackets)
 *   <thinking>...</thinking>
 *   <reasoning>...</reasoning>
 *
 * Conservative: only buffers when the trailing text could be a prefix of a
 * known opening/closing tag. Any other text is flushed immediately so
 * legitimate HTML/code in answers is preserved.
 */
export class ThinkStripper {
  private inThink = false;
  private carry = '';
  private stripped = false;

  // Full opening tag strings (lowercase). Matched as direct substrings.
  private static OPEN_TAGS = ['<think>', '<thinking>', '<reasoning>'];
  private static CLOSE_TAGS = ['</think>', '</thinking>', '</reasoning>'];

  /**
   * Feed one chunk of content. Returns the safe-to-emit text for this chunk.
   * May return '' (all content was reasoning or a partial tag was buffered).
   */
  feed(chunk: string): string {
    if (!chunk) return '';
    let work = this.carry + chunk;
    this.carry = '';
    let out = '';

    while (work.length > 0) {
      if (this.inThink) {
        // Look for any closing tag.
        const closeIdx = this.findTag(work, ThinkStripper.CLOSE_TAGS);
        if (closeIdx.found) {
          // Drop everything up to and including the closing tag.
          this.inThink = false;
          this.stripped = true;
          work = work.slice(closeIdx.end);
          continue;
        }
        // No close yet - but the tail could be a partial closing tag.
        const safeLen = this.safeEmitLen(work, ThinkStripper.CLOSE_TAGS);
        if (safeLen < work.length) {
          // Buffer the partial closing tag; emit nothing more from this chunk.
          this.carry = work.slice(safeLen);
          break;
        }
        // Entire remaining chunk is inside a think block - discard it.
        break;
      } else {
        // Look for any opening tag.
        const openIdx = this.findTag(work, ThinkStripper.OPEN_TAGS);
        if (openIdx.found) {
          // Emit any text before the opening tag.
          if (openIdx.start > 0) out += work.slice(0, openIdx.start);
          this.inThink = true;
          this.stripped = true;
          work = work.slice(openIdx.end);
          continue;
        }
        // No full opening tag - but the tail could be a partial one.
        const safeLen = this.safeEmitLen(work, ThinkStripper.OPEN_TAGS);
        out += work.slice(0, safeLen);
        if (safeLen < work.length) {
          this.carry = work.slice(safeLen);
        }
        break;
      }
    }
    return out;
  }

  /**
   * Flush at end of stream. Returns any buffered carry that turned out not to
   * be a tag (emitted as normal text). If we're still inside a think block,
   * the carry is discarded (unclosed reasoning - treat as leaked reasoning).
   */
  flush(): string {
    const leftover = this.carry;
    this.carry = '';
    if (this.inThink) {
      // Unclosed think block - discard the leftover; it was reasoning.
      return '';
    }
    return leftover;
  }

  /** Did we ever strip anything? Used to log a profiling hint. */
  didStrip(): boolean {
    return this.stripped;
  }

  // Find the earliest tag in `work` from the given list. Returns {found, start, end}
  // where start/end are character indices of the full tag.
  private findTag(work: string, tags: string[]): { found: boolean; start: number; end: number } {
    const lower = work.toLowerCase();
    let best: { start: number; end: number } | null = null;
    for (const tag of tags) {
      const idx = lower.indexOf(tag);
      if (idx === -1) continue;
      if (!best || idx < best.start) best = { start: idx, end: idx + tag.length };
    }
    return best ? { found: true, start: best.start, end: best.end } : { found: false, start: -1, end: -1 };
  }

  /**
   * Returns the length of the safe-to-emit prefix of `work`. The tail is held
   * back only if it could be a prefix of one of `candidates` (a partial tag).
   * We check the longest candidate length so a partial match is never emitted.
   */
  private safeEmitLen(work: string, candidates: string[]): number {
    const lower = work.toLowerCase();
    let holdFrom = work.length;
    for (const cand of candidates) {
      // Check progressively shorter tails of `work` against the candidate prefix.
      const maxOverlap = Math.min(cand.length - 1, work.length);
      for (let len = maxOverlap; len > 0; len--) {
        if (lower.endsWith(cand.slice(0, len))) {
          if (work.length - len < holdFrom) holdFrom = work.length - len;
          break;
        }
      }
    }
    return holdFrom;
  }
}

export function stripInlineThinking(text: string): string {
  if (!text) return text;
  const stripper = new ThinkStripper();
  return stripper.feed(text) + stripper.flush();
}



// ─── Canned-refusal safety net ──────────────────────────────────────────────

/**
 * Some providers (notably Mistral's free-tier models) respond to innocuous
 * prompts with a short canned refusal when their safety layer over-triggers —
 * e.g. the 36-char "I'm afraid I can't assist with that." These responses are
 * technically "successful" (non-empty, finish_reason=stop) so they slip past
 * empty-response checks and get surfaced to the user.
 *
 * The helpers below let the routers (a) hold back streamed text until a
 * canned refusal can be ruled out, and (b) treat a detected refusal as a
 * provider failure so the fallback chain tries the next provider.
 */

/**
 * Normalized (lowercase, straight apostrophes, collapsed whitespace) refusal
 * openers. Deliberately refusal-SPECIFIC to avoid false positives on normal
 * answers — e.g. "i can't help with" is listed but bare "i can't help" is not
 * ("I can't help but notice…" is a legitimate sentence opener).
 */
const REFUSAL_OPENERS: readonly string[] = [
  "i'm afraid i can't",
  "i'm afraid i cannot",
  "i am afraid i can't",
  "i am afraid i cannot",
  "i can't assist",
  "i cannot assist",
  "i can't help with",
  "i cannot help with",
  "i can't help you with",
  "i cannot help you with",
  "i'm sorry, but i can't",
  "i'm sorry but i can't",
  "i'm sorry, but i cannot",
  "i'm sorry but i cannot",
  "i'm sorry, i can't assist",
  "sorry, but i can't",
  "i'm not able to assist",
  "i'm unable to",
  "i am unable to",
  "i must decline",
  "i have to decline",
  "i cannot fulfill",
  "i can't fulfill",
  "i cannot comply",
  "i can't comply",
  "i can't provide that",
  "i cannot provide that",
  "as an ai, i can't",
  "as an ai, i cannot",
  "as an ai language model",
  "unfortunately, i can't assist",
  "unfortunately, i cannot assist",
  "unfortunately, i'm unable",
  "i apologize, but i can't",
  "i apologize, but i cannot",
];

/**
 * Canned refusals are short (Mistral's is 36 chars; most are <150). Responses
 * longer than this that merely START with a refusal opener are treated as
 * legitimate refusal-led answers (e.g. "I can't help with that, but here's
 * what I can do: …") and stream normally.
 */
export const REFUSAL_MAX_LEN = 300;

/** Normalize for refusal matching: lowercase, straight quotes, single spaces. */
function normalizeForRefusal(text: string): string {
  return text
    .trimStart()
    .toLowerCase()
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/\s+/g, ' ');
}

/**
 * Final determination: is this complete response a canned refusal?
 * Requires a refusal opener AND short overall length.
 */
export function isCannedRefusal(text: string): boolean {
  const n = normalizeForRefusal(text);
  if (!n || n.length > REFUSAL_MAX_LEN) return false;
  return REFUSAL_OPENERS.some(o => n.startsWith(o));
}

/**
 * Streaming-time check: could the text accumulated so far still turn out to
 * be a canned refusal? True while the buffer is a prefix of an opener (or an
 * opener is a prefix of it). The streaming router holds text back while this
 * returns true so a refusal never reaches the client.
 */
export function couldBeRefusalPrefix(text: string): boolean {
  const n = normalizeForRefusal(text);
  if (!n) return true; // nothing yet — could still become anything
  return REFUSAL_OPENERS.some(o => o.startsWith(n) || n.startsWith(o));
}

/**
 * Thrown when a provider returns a canned refusal. Distinct error type so
 * routers can stash it for last-refusal grace: if every provider refuses,
 * the last refusal text is returned to the user instead of a generic error.
 */
export class RefusalError extends Error {
  readonly refusalText: string;
  constructor(message: string, refusalText: string) {
    super(message);
    this.name = 'RefusalError';
    this.refusalText = refusalText;
  }
}



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
  if (taskTypeHint === 'conversational' || taskTypeHint === 'chat') return 'conversational';
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
  // Complex tasks: complex chain first, then free HEAVY_CHAIN as safety net
  if (taskType === 'complex') return [...COMPLEX_CHAIN, ...HEAVY_CHAIN].filter(p => !isProviderExcludedForTaskType(p, taskType));
  // Conversational tasks: conversational chain, then free-premium, then paid
  if (taskType === 'conversational') return [...CONVERSATIONAL_CHAIN, ...FREE_PREMIUM_CHAIN, ...PAID_CHAIN].filter(p => !isProviderExcludedForTaskType(p, taskType));
  const freeChain = taskType === 'light'
    ? LIGHT_CHAIN
    : taskType === 'super-heavy'
      ? SUPER_HEAVY_CHAIN
      : HEAVY_CHAIN;
  return [...freeChain, ...FREE_PREMIUM_CHAIN, ...PAID_CHAIN].filter(p => !isProviderExcludedForTaskType(p, taskType));
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
  if (taskType === 'conversational') return (config.conversational ?? config.light)[0]?.id;
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
  if (taskType === 'conversational') return config.conversational ?? config.light;
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
  if (/whisper|tts|parakeet|canary|piper|bark|voxtral/i.test(lower)) return 'audio';
  if (/embed|bge|gte|jina|nomic/i.test(lower)) return 'embedding';
  if (/rerank|re-rank|colbert/i.test(lower)) return 'rerank';
  if (/neva|llava|pixtral|\bvlm\b/i.test(lower)) return 'vision';
  // Physics / scientific models (Ising, molecular dynamics, etc.) — not chat
  if (/ising|calibration|physics|crystal|lattice|molecule|protein|genomic/i.test(lower)) return 'other';
  // Default: assume chat — the probe will filter out non-chat models.
  // Multimodal chat models (gpt-4o, claude-3, gemini-3.x) correctly default here.
  // Safety guards, tiny models, etc. get probed and filtered by the probe result.
  return 'chat';
}
