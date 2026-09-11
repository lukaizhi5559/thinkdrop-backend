/**
 * DiscoveryAgent — self-healing catalog maintenance.
 *
 * When a model fails (5× consecutive failures, or 404/403), the catalogManager
 * triggers this agent. It:
 *   1. Fetches the provider's /models endpoint to get the live model list
 *   2. Marks removed models as dead
 *   3. Probes new models with a minimal "Say OK" request
 *   4. Classifies new models (heavy/light, intelligence) using a DIFFERENT provider
 *   5. Re-probes dead models (maybe they came back)
 *   6. Saves the updated catalog
 *
 * Three scheduling tiers:
 *   - On-failure: triggered immediately when a model fails (5× or 404/403)
 *   - Daily: re-probes all existing providers for model list changes
 *   - Weekly: deep check — probes PAID providers for free-tier models,
 *     re-evaluates 403'd models (maybe moved back to free), and checks
 *     known provider catalog endpoints for entirely new free providers
 *
 * Providers maintain each other: if NVIDIA's GLM-5.2 disappears, the agent
 * uses Groq or GLM to research what replaced it, probes NVIDIA's API for
 * new models, and updates the catalog — all without human intervention.
 */

import { catalogManager, CatalogProviderEntry } from './catalogManager';
import { logger } from './logger';
import { LLMRouter } from './llmRouter';
import { classifyModelCategory, ModelCategory, REASONING_PROVIDER_CONFIG } from './providerConfig';

const PROBE_TIMEOUT_MS = 15_000;
const DAILY_REFRESH_INTERVAL_MS = 24 * 60 * 60 * 1000; // 24 hours
const WEEKLY_REFRESH_INTERVAL_MS = 7 * 24 * 60 * 60 * 1000; // 7 days
const DISCOVERY_COOLDOWN_MS = 5 * 60 * 1000; // 5 min between discovery runs for same provider

/**
 * Known paid providers that MAY have free tiers or free models.
 * The weekly deep check probes these to see if any models work without
 * billing enabled, or if the provider has introduced a free tier.
 *
 * Format: provider name → { baseURL, envKey, catalogEndpoint }
 * The envKey is checked — if not set, we skip (user hasn't configured it).
 * If set, we probe the /models endpoint and then try minimal chat completions
 * to see if any models work without payment.
 */
const PAID_PROVIDERS_TO_PROBE: Array<{
  name: string;
  baseURL: string;
  envKey: string;
  catalogEndpoint: string;
}> = [
  {
    name: 'mistral',
    baseURL: 'https://api.mistral.ai/v1',
    envKey: 'MISTRAL_API_KEY',
    catalogEndpoint: 'https://api.mistral.ai/v1/models',
  },
  {
    name: 'deepseek',
    baseURL: 'https://api.deepseek.com/v1',
    envKey: 'DEEPSEEK_API_KEY',
    catalogEndpoint: 'https://api.deepseek.com/v1/models',
  },
  {
    name: 'grok',
    baseURL: 'https://api.x.ai/v1',
    envKey: 'GROK_API_KEY',
    catalogEndpoint: 'https://api.x.ai/v1/models',
  },
  {
    name: 'openai',
    baseURL: 'https://api.openai.com/v1',
    envKey: 'OPENAI_API_KEY',
    catalogEndpoint: 'https://api.openai.com/v1/models',
  },
  {
    name: 'claude',
    baseURL: 'https://api.anthropic.com/v1',
    envKey: 'ANTHROPIC_API_KEY',
    // Anthropic doesn't have a /models endpoint, but we can probe known models
    catalogEndpoint: '',
  },
  {
    name: 'cohere',
    baseURL: 'https://api.cohere.ai/compatibility/v1',
    envKey: 'COHERE_API_KEY',
    catalogEndpoint: 'https://api.cohere.ai/compatibility/v1/models',
  },
];

/**
 * Known free models on paid providers that we should probe during weekly checks.
 * These are models that have been free at some point or are rumored to have
 * free tiers. If the probe succeeds (200), we add them to the catalog as free.
 */
const FREE_TIER_CANDIDATES: Record<string, string[]> = {
  mistral: ['mistral-tiny', 'mistral-small-latest', 'open-mistral-7b', 'open-mixtral-8x7b'],
  deepseek: ['deepseek-chat', 'deepseek-coder'],
  grok: ['grok-2-mini', 'grok-beta'],
  openai: ['gpt-4o-mini', 'gpt-3.5-turbo'],
  // Anthropic doesn't currently have free models, but probe in case they add one
  claude: ['claude-3-5-haiku-20241022'],
};

class DiscoveryAgent {
  private running = false;
  private lastDiscovery: Map<string, number> = new Map();
  private dailyTimer: NodeJS.Timeout | null = null;
  private weeklyTimer: NodeJS.Timeout | null = null;
  private llmRouter: LLMRouter | null = null;

  /**
   * Set the LLM router instance (for classification calls).
   * Injected by index.ts to avoid circular dependency.
   */
  setLLMRouter(router: LLMRouter): void {
    this.llmRouter = router;
  }

  /**
   * Start the daily and weekly refresh timers.
   * - Daily: re-probes all existing providers for model list changes
   * - Weekly: deep check — probes paid providers for free-tier models,
   *   re-evaluates 403'd models, checks for new free providers
   */
  start(): void {
    if (this.dailyTimer) return;
    // Daily: first run 60s after startup, then every 24h
    setTimeout(() => this.refreshAllProviders(), 60_000);
    this.dailyTimer = setInterval(() => this.refreshAllProviders(), DAILY_REFRESH_INTERVAL_MS);

    // Weekly: first run 5 min after startup, then every 7 days
    setTimeout(() => this.weeklyDeepCheck(), 5 * 60_000);
    this.weeklyTimer = setInterval(() => this.weeklyDeepCheck(), WEEKLY_REFRESH_INTERVAL_MS);

    logger.info('[DiscoveryAgent] Started — daily + weekly refresh scheduled');
  }

  /**
   * Stop all refresh timers.
   */
  stop(): void {
    if (this.dailyTimer) {
      clearInterval(this.dailyTimer);
      this.dailyTimer = null;
    }
    if (this.weeklyTimer) {
      clearInterval(this.weeklyTimer);
      this.weeklyTimer = null;
    }
  }

  /**
   * Startup verification — probe all active models in the background to verify
   * their categories. Runs immediately on startup, doesn't block the server.
   * Uses lastVerifiedAt to skip models probed within the last 24 hours (cached).
   *
   * Reclassification logic:
   *   - Probe succeeds + was non-chat → reclassify as 'chat' (it can do text chat)
   *   - Probe fails + was 'chat' + regex says non-chat → reclassify as non-chat
   *   - Probe fails + regex says chat → leave it (might be temporarily down)
   */
  async verifyAllModelsOnStartup(): Promise<void> {
    const providers = catalogManager.getAllProviders();
    let probed = 0, reclassified = 0, skipped = 0;
    const STALE_THRESHOLD_MS = 24 * 60 * 60 * 1000; // 24 hours
    const now = Date.now();

    logger.info(`[Discovery] Starting startup verification probe for ${providers.length} providers`);

    for (const provider of providers) {
      if (!provider.catalogEndpoint) continue;
      if (this.running) {
        logger.debug(`[Discovery] Startup probe waiting for discovery to finish for ${provider.name}`);
      }

      for (const model of provider.models) {
        if (model.status !== 'active') continue;

        // Skip models probed recently (cached via lastVerifiedAt) — UNLESS the
        // stored category doesn't match the current regex classification, which
        // means the regex was updated and we need to re-verify.
        const regexCategory = classifyModelCategory(model.id);
        const categoryMismatch = model.category && model.category !== regexCategory;
        if (model.lastVerifiedAt && !categoryMismatch) {
          const age = now - new Date(model.lastVerifiedAt).getTime();
          if (age < STALE_THRESHOLD_MS) { skipped++; continue; }
        }

        // Probe with "Say OK"
        const probeResult = await this.probeModel(provider, model.id);
        probed++;

        // Check for obvious non-chat name patterns that the simplified regex misses.
        // These models may respond to chat completion but aren't general chat models
        // (safety guards return JSON classifications, code models only do code, etc.)
        const lowerId = model.id.toLowerCase();
        const isObviousNonChat = /guard|safety|nemoguard|prompt.?guard|content.?safety|topic.?control|moderation|classifier|filter|riva-translate|deplot|nvclip|cosmos-reason|ai-synthetic-video|nemoretriever-parse|orca-stt|whisper|tts|parakeet|canary|piper|bark|orpheus|embed|bge|gte|jina|nomic|rerank|colbert|diffusion|sdxl|flux|dall|imagen|gpt-image|ising|calibration|physics/.test(lowerId);

        if (isObviousNonChat && model.category === 'chat') {
          // Model is classified as chat but name indicates non-chat (e.g., safety guard
          // that was wrongly reclassified by a previous probe). Fix it regardless of probe result.
          model.category = 'other';
          reclassified++;
          logger.info(`[Discovery] Reclassified ${provider.name}/${model.id}: chat → other (name indicates non-chat)`);
        } else if (probeResult.alive && model.category !== 'chat' && !isObviousNonChat) {
          // Model responds to chat and isn't an obvious non-chat pattern → reclassify as chat
          const oldCategory = model.category;
          model.category = 'chat';
          reclassified++;
          logger.info(`[Discovery] Reclassified ${provider.name}/${model.id}: ${oldCategory} → chat (probe succeeded)`);
        } else if (probeResult.alive && model.category !== 'chat' && isObviousNonChat) {
          // Probe succeeded but name indicates non-chat (e.g., safety guard returns JSON)
          // Keep the non-chat category — the model responds but isn't useful for chat routing
          logger.debug(`[Discovery] ${provider.name}/${model.id} probe succeeded but name indicates non-chat — keeping category ${model.category}`);
        } else if (!probeResult.alive && model.category === 'chat') {
          // Model was classified as chat but probe failed → check if it's actually non-chat
          if (regexCategory !== 'chat') {
            logger.info(`[Discovery] Reclassified ${provider.name}/${model.id}: chat → ${regexCategory} (probe failed, regex confirms non-chat)`);
            model.category = regexCategory;
            reclassified++;
          }
          // If regex also says chat, leave it — might be temporarily down, not misclassified
        }

        // Update lastVerifiedAt
        model.lastVerifiedAt = new Date().toISOString();

        // Small delay between probes to respect rate limits (e.g., NVIDIA: 40 RPM)
        await new Promise(r => setTimeout(r, 300));
      }
      // Save after each provider so progress isn't lost if killed mid-probe
      await catalogManager.save();
    }

    logger.info(`[Discovery] Startup verification complete: ${probed} probed, ${reclassified} reclassified, ${skipped} skipped (cached)`);
  }

  /**
   * Triggered by catalogManager when a model fails.
   * Cooldown: don't re-discover the same provider within 5 minutes.
   */
  async onModelFailure(provider: string, _modelId: string, _statusCode: number): Promise<void> {
    if (this.running) return;
    const now = Date.now();
    const last = this.lastDiscovery.get(provider) ?? 0;
    if (now - last < DISCOVERY_COOLDOWN_MS) return;
    this.lastDiscovery.set(provider, now);
    await this.discoverProvider(provider);
  }

  /**
   * Discover/probe a single provider's catalog.
   */
  async discoverProvider(providerName: string): Promise<void> {
    if (this.running) {
      logger.debug(`[Discovery] Skipping ${providerName} — discovery already running`);
      return;
    }

    const provider = catalogManager.getProvider(providerName);
    if (!provider) {
      logger.warn(`[Discovery] Unknown provider: ${providerName}`);
      return;
    }
    if (!provider.catalogEndpoint) {
      logger.debug(`[Discovery] ${providerName} has no catalog endpoint — skipping`);
      return;
    }

    this.running = true;
    try {
      logger.info(`[Discovery] Discovering ${providerName}...`);

      // Step 1: Fetch the provider's live model list
      const remoteModelIds = await this.fetchProviderModels(provider);
      if (remoteModelIds.length === 0) {
        logger.warn(`[Discovery] ${providerName} returned 0 models — marking provider as dead`);
        catalogManager.markProviderDead(providerName, 'Catalog endpoint returned 0 models');
        await catalogManager.save();
        return;
      }

      // Step 2: Find models that disappeared (deprecated/removed)
      const localModelIds = provider.models.map(m => m.id);
      const removed = localModelIds.filter(id => !remoteModelIds.includes(id));
      const newModels = remoteModelIds.filter(id => !localModelIds.includes(id));

      // Mark removed models as dead
      for (const id of removed) {
        const entry = provider.models.find(m => m.id === id);
        if (entry && entry.status !== 'dead') {
          catalogManager.disableModel(providerName, id, 'Removed from provider catalog');
          logger.info(`[Discovery] ${providerName}/${id} removed from catalog — marked dead`);
        }
      }

      // Step 3: Probe ALL new models with "Say OK" — the probe is the primary
      // chat/non-chat detector. If a model responds to chat completion, it's
      // chat-capable regardless of its name. If it doesn't, use the regex fallback
      // to determine its non-chat category.
      for (const modelId of newModels) {
        // Use the preflight profiler instead of the simple "Say OK" probe.
        // This benchmarks actual speed, detects reasoning capability, and tests streaming.
        const profile = await this.profileModel(provider, modelId);

        if (profile.alive) {
          // Model responds to chat completion → it's chat-capable.
          // Use LLM classifier for intelligence/category (still a guess from name),
          // but use the ACTUAL benchmark speed from the profiler.
          const classification = await this.classifyModel(providerName, modelId);
          const now = new Date().toISOString();
          // Use benchmarked speed if available, else fall back to classifier estimate
          const modelSpeed = profile.speed || classification.speed || 0;
          // If profiler detected it's a light model (fast), override classification
          const modelTaskType = modelSpeed > 150 ? 'light' : classification.taskType;
          catalogManager.addModel(providerName, {
            id: modelId,
            taskType: modelTaskType,
            intelligence: classification.intelligence,
            speed: modelSpeed,
            contextWindow: profile.contextWindow,
            category: classification.category,
            status: 'active',
            isReasoning: profile.isReasoning,
            canDisableThinking: profile.canDisableThinking,
            supportsStreaming: profile.supportsStreaming,
            reasoningMode: profile.reasoningMode,
            benchmarkedAt: now,
            benchmarkSpeed: profile.speed,
            discoveredAt: now,
            lastVerifiedAt: now,
            consecutiveFailures: 0,
            totalCalls: 0,
            totalSuccesses: 0,
          });
          logger.info(`[Discovery] NEW model: ${providerName}/${modelId} (category: ${classification.category}, ${modelTaskType}, intel ~${classification.intelligence}, speed ~${modelSpeed} t/s, reasoning=${profile.isReasoning ?? '?'}, canDisable=${profile.canDisableThinking ?? '?'}, reasoningMode=${profile.reasoningMode ?? '?'}, streaming=${profile.supportsStreaming ?? '?'})`);
        } else {
          // Probe failed — model doesn't respond to chat completion.
          // Use regex fallback to determine non-chat category.
          const category = classifyModelCategory(modelId);
          const now = new Date().toISOString();
          if (category === 'chat') {
            // Regex says chat but probe failed — mark as dead (might come back later)
            catalogManager.addModel(providerName, {
              id: modelId,
              taskType: 'heavy',
              category: 'chat',
              status: 'dead',
              discoveredAt: now,
              lastVerifiedAt: now,
              consecutiveFailures: 1,
              totalCalls: 0,
              totalSuccesses: 0,
            });
            logger.debug(`[Discovery] ${providerName}/${modelId} probe failed — marked as dead (regex says chat, might be temporarily down)`);
          } else {
            // Non-chat model (vision, embedding, image-gen, audio, rerank, other) —
            // catalog it as a special model. The specialized service will probe it
            // with the right input format when it actually uses it.
            catalogManager.addModel(providerName, {
              id: modelId,
              taskType: 'heavy',
              category,
              status: 'active',
              discoveredAt: now,
              lastVerifiedAt: now,
              consecutiveFailures: 0,
              totalCalls: 0,
              totalSuccesses: 0,
            });
            logger.info(`[Discovery] NEW special model: ${providerName}/${modelId} (category: ${category})`);
          }
        }

        // Small delay between probes to respect rate limits (e.g., NVIDIA: 40 RPM)
        await new Promise(r => setTimeout(r, 500));
      }

      // Step 5: Re-probe existing dead/disabled models (maybe they came back)
      for (const model of provider.models.filter(m => m.status === 'dead' || m.status === 'disabled')) {
        const probeResult = await this.probeModel(provider, model.id);
        if (probeResult.alive) {
          catalogManager.reactivateModel(providerName, model.id);
          logger.info(`[Discovery] ${providerName}/${model.id} came back online — reactivating`);
        }
      }

      // Step 6: Save updated catalog
      await catalogManager.save();
      const activeCount = provider.models.filter(m => m.status === 'active').length;
      logger.info(`[Discovery] ${providerName} complete: ${newModels.length} new, ${removed.length} removed, ${activeCount} active`);

    } catch (err) {
      logger.error(`[Discovery] Failed to discover ${providerName}: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      this.running = false;
    }
  }

  /**
   * Fetch the provider's /models endpoint to get the live model list.
   * Returns an array of model IDs.
   */
  private async fetchProviderModels(provider: CatalogProviderEntry): Promise<string[]> {
    if (!provider.catalogEndpoint) return [];
    const apiKey = process.env[provider.envKey];
    if (!apiKey) return [];

    try {
      // Gemini native API uses ?key= query param, not Bearer header
      const isGoogle = provider.apiType === 'google';
      const url = isGoogle
        ? `${provider.catalogEndpoint}?key=${apiKey}`
        : provider.catalogEndpoint;
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (!isGoogle) {
        headers['Authorization'] = `Bearer ${apiKey}`;
      }

      const response = await fetch(url, {
        headers,
        signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
      });

      if (!response.ok) {
        logger.warn(`[Discovery] ${provider.name} catalog endpoint returned ${response.status}`);
        return [];
      }

      const data: any = await response.json();
      // Handle different response formats:
      // OpenAI-compatible: { data: [{ id: "model-name" }] }
      // NVIDIA: { data: [{ id: "model-name" }] }
      // Gemini: { models: [{ name: "models/gemini-3.5-flash-lite" }] }
      if (Array.isArray(data.data)) {
        return data.data.map((m: any) => m.id).filter(Boolean);
      }
      if (Array.isArray(data.models)) {
        return data.models.map((m: any) => {
          // Gemini format: "models/gemini-3.5-flash-lite" → "gemini-3.5-flash-lite"
          const name = m.name || m.id;
          return name?.replace(/^models\//, '');
        }).filter(Boolean);
      }
      return [];
    } catch (err) {
      logger.warn(`[Discovery] Failed to fetch ${provider.name} models: ${err instanceof Error ? err.message : String(err)}`);
      return [];
    }
  }

  /**
   * Probe a model with a minimal chat completion to check if it's alive.
   */
  private async probeModel(
    provider: CatalogProviderEntry,
    modelId: string
  ): Promise<{ alive: boolean; contextWindow?: number }> {
    const apiKey = process.env[provider.envKey];
    if (!apiKey) return { alive: false };

    // Cloudflare needs account-id-based URL
    let baseURL = provider.baseURL;
    if (provider.name === 'cloudflare') {
      const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;
      if (!accountId) return { alive: false };
      baseURL = `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/v1`;
    }

    try {
      const response = await fetch(`${baseURL}/chat/completions`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: modelId,
          messages: [{ role: 'user', content: 'Say OK' }],
          max_tokens: 8,
          stream: false,
        }),
        signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
      });

      if (response.ok) return { alive: true };
      // 429 = rate-limited but alive
      if (response.status === 429) return { alive: true };
      // 404/403 = model removed or moved to paid
      if (response.status === 404 || response.status === 403) return { alive: false };
      logger.debug(`[Discovery] Probe ${provider.name}/${modelId} returned ${response.status}`);
      return { alive: false };
    } catch {
      return { alive: false };
    }
  }

  /**
   * Preflight model profiler — benchmarks a model for speed, reasoning capability,
   * and streaming support. Replaces the simple "Say OK" probe with comprehensive
   * profiling that detects:
   *   1. Actual speed (tokens/sec) from a standardized benchmark prompt
   *   2. Whether the model does reasoning by default (and if thinking can be disabled)
   *   3. Whether the model supports streaming
   *
   * This data is stored in the catalog and used by the router to:
   *   - Disable thinking for reasoning models (3-19x speedup)
   *   - Skip non-streaming models for streaming requests
   *   - Penalize un-disableable reasoning models in scoring (effectively drops them)
   */
  async profileModel(
    provider: CatalogProviderEntry,
    modelId: string
  ): Promise<{
    alive: boolean;
    speed?: number;
    isReasoning?: boolean;
    canDisableThinking?: boolean;
    supportsStreaming?: boolean;
    contextWindow?: number;
    reasoningMode?: 'none' | 'separate' | 'disabled';
  }> {
    const apiKey = process.env[provider.envKey];
    if (!apiKey) return { alive: false };

    // Cloudflare needs account-id-based URL
    let baseURL = provider.baseURL;
    if (provider.name === 'cloudflare') {
      const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;
      if (!accountId) return { alive: false };
      baseURL = `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/v1`;
    }

    const isGoogle = provider.apiType === 'google';
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (!isGoogle) {
      headers['Authorization'] = `Bearer ${apiKey}`;
    }

    // Use a prompt that triggers reasoning in reasoning models but doesn't require
    // a huge response. Math/logic problems trigger reasoning; the answer is short.
    const BENCHMARK_PROMPT = 'A farmer has 100 meters of fence to enclose a rectangular field. What dimensions maximize the area? Answer briefly.';
    const BENCHMARK_MAX_TOKENS = 200;
    const PROFILE_TIMEOUT_MS = 45_000; // 45s — reasoning models need more time

    // ─── Test 1: Speed benchmark (non-streaming) ────────────────────────────
    let speedNormal: number | undefined;
    let tokensNormal: number | undefined;
    let timeNormal: number | undefined;
    let alive = false;
    let normalHasReasoning = false; // does normal response include reasoning_content?
    let normalHasInlineThink = false; // does normal content embed inline Mattis blocks?

    try {
      const url = isGoogle
        ? `${baseURL}/chat/completions`
        : `${baseURL}/chat/completions`;
      const start = Date.now();
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model: modelId,
          messages: [{ role: 'user', content: BENCHMARK_PROMPT }],
          max_tokens: BENCHMARK_MAX_TOKENS,
          stream: false,
        }),
        signal: AbortSignal.timeout(PROFILE_TIMEOUT_MS),
      });

      if (response.ok || response.status === 429) {
        alive = true;
        if (response.ok) {
          const data: any = await response.json();
          const elapsed = (Date.now() - start) / 1000;
          const completionTokens = data?.usage?.completion_tokens || 0;
          if (completionTokens > 0 && elapsed > 0) {
            speedNormal = Math.round(completionTokens / elapsed);
            tokensNormal = completionTokens;
            timeNormal = elapsed;
          }
          // Check if normal response includes reasoning_content (NVIDIA/Cloudflare)
          normalHasReasoning = !!data?.choices?.[0]?.message?.reasoning_content;
          // Check if normal content embeds inline Mattis blocks (groq gpt-oss
          // raw format when no reasoning_format is supplied). Cheap signal that
          // this is a reasoning model whose reasoning leaks into content.
          const normalContent: string = data?.choices?.[0]?.message?.content ?? '';
          normalHasInlineThink = / Mattis/i.test(normalContent) || /<thinking>/i.test(normalContent) || /<reasoning>/i.test(normalContent);
        }
      } else if (response.status === 404 || response.status === 403) {
        return { alive: false };
      }
    } catch {
      // Timeout or network error — model is too slow or unreachable
      return { alive: false };
    }

    if (!alive) return { alive: false };

    // ─── Test 2: Reasoning detection + can disable? ─────────────────────────
    // Send the same prompt with thinking-disable params, compare token count/time.
    // If tokens drop >30% or speed increases >30%, it's a reasoning model AND
    // we can disable thinking.
    let isReasoning: boolean | undefined;
    let canDisableThinking: boolean | undefined;

    // Determine which thinking-disable param to try based on provider
    let thinkingDisableParam: Record<string, any> | null = null;
    if (provider.name === 'glm' || provider.name === 'cloudflare') {
      thinkingDisableParam = { thinking: { type: 'disabled' } };
    } else if (provider.name === 'nvidia') {
      thinkingDisableParam = { chat_template_kwargs: { enable_thinking: false } };
    }

    if (thinkingDisableParam && speedNormal !== undefined) {
      try {
        const startDisabled = Date.now();
        const responseDisabled = await fetch(`${baseURL}/chat/completions`, {
          method: 'POST',
          headers,
          body: JSON.stringify({
            model: modelId,
            messages: [{ role: 'user', content: BENCHMARK_PROMPT }],
            max_tokens: BENCHMARK_MAX_TOKENS,
            stream: false,
            ...thinkingDisableParam,
          }),
          signal: AbortSignal.timeout(PROFILE_TIMEOUT_MS),
        });

        if (responseDisabled.ok) {
          const dataDisabled: any = await responseDisabled.json();
          const elapsedDisabled = (Date.now() - startDisabled) / 1000;
          const tokensDisabled = dataDisabled?.usage?.completion_tokens || 0;
          const speedDisabled = tokensDisabled > 0 && elapsedDisabled > 0
            ? Math.round(tokensDisabled / elapsedDisabled)
            : 0;

          // Check for reasoning_content field (some providers return it)
          const hasReasoningField = !!dataDisabled?.choices?.[0]?.message?.reasoning_content;

          logger.debug(
            `[Discovery] Reasoning test for ${provider.name}/${modelId}: ` +
            `normal=${tokensNormal}tok/${timeNormal?.toFixed(1)}s (${speedNormal}t/s), ` +
            `disabled=${tokensDisabled}tok/${elapsedDisabled.toFixed(1)}s (${speedDisabled}t/s), ` +
            `reasoning_field=${hasReasoningField}`
          );

          // If tokens dropped >30% or speed increased >30% with thinking disabled,
          // it's a reasoning model AND we can disable thinking.
          // Also check reasoning_content field — if present in normal but not in disabled,
          // that confirms reasoning + can disable.
          //
          // IMPORTANT: To avoid false positives from API variance, only flag as
          // reasoning if the normal speed is slow (< 50 t/s) — fast models don't
          // have reasoning to disable, and variance between calls can easily exceed 30%.
          //
          // Non-nemotron NVIDIA models (e.g. ising-calibration) require a higher
          // threshold (50% token drop) to reduce false positives from API variance
          // on non-chat models that happen to show token count changes.
          if (tokensNormal !== undefined && tokensDisabled > 0) {
            const tokenDrop = (tokensNormal - tokensDisabled) / tokensNormal;
            const speedIncrease = speedDisabled > 0 ? (speedDisabled - speedNormal) / speedNormal : 0;
            const isSignificantlyFaster = speedIncrease > 0.5 && speedNormal < 50; // 50% faster AND was slow
            const isNvidiaNonNemotron = provider.name === 'nvidia' && !modelId.toLowerCase().includes('nemotron');
            const tokenDropThreshold = isNvidiaNonNemotron ? 0.5 : 0.3;
            if (tokenDrop > tokenDropThreshold || isSignificantlyFaster || (normalHasReasoning && !hasReasoningField)) {
              isReasoning = true;
              canDisableThinking = true;
              // Use the faster (thinking-disabled) speed as the benchmark
              if (speedDisabled > speedNormal) {
                speedNormal = speedDisabled;
                tokensNormal = tokensDisabled;
                timeNormal = elapsedDisabled;
              }
            } else {
              // No significant change — either not reasoning, or can't disable
              // If normal response had reasoning_content, it IS reasoning but can't disable
              isReasoning = normalHasReasoning || hasReasoningField;
              canDisableThinking = false;
            }
          } else if (normalHasReasoning) {
            // No token comparison possible but normal response had reasoning_content
            isReasoning = true;
            canDisableThinking = hasReasoningField ? false : undefined;
          }
        } else {
          logger.debug(`[Discovery] Reasoning test for ${provider.name}/${modelId}: thinking-disabled response returned ${responseDisabled.status}`);
        }
      } catch (err) {
        // Thinking-disable request failed — inconclusive, leave undefined
        logger.debug(`[Discovery] Reasoning test for ${provider.name}/${modelId} failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    }

    // ─── Test 2b: Separate-reasoning probe ──────────────────────────────────
    // If this provider has a registry entry for separate-reasoning params
    // (groq/cerebras/deepseek/openrouter), probe whether the model actually
    // emits reasoning in a dedicated field when those params are sent. This
    // populates `reasoningMode` so the router can read reasoning separately
    // instead of relying on the inline-think safety net.
    let reasoningMode: 'none' | 'separate' | 'disabled' | undefined;
    const registry = REASONING_PROVIDER_CONFIG[provider.name];
    if (registry?.separateParams && !canDisableThinking) {
      try {
        const responseSeparate = await fetch(`${baseURL}/chat/completions`, {
          method: 'POST',
          headers,
          body: JSON.stringify({
            model: modelId,
            messages: [{ role: 'user', content: BENCHMARK_PROMPT }],
            max_tokens: BENCHMARK_MAX_TOKENS,
            stream: false,
            ...registry.separateParams,
          }),
          signal: AbortSignal.timeout(PROFILE_TIMEOUT_MS),
        });
        if (responseSeparate.ok) {
          const dataSeparate: any = await responseSeparate.json();
          const field = registry.responseField ?? 'reasoning_content';
          const separateReasoning = dataSeparate?.choices?.[0]?.message?.[field];
          const separateContent: string = dataSeparate?.choices?.[0]?.message?.content ?? '';
          const separateHasInlineThink = / Mattis/i.test(separateContent) || /<thinking>/i.test(separateContent) || /<reasoning>/i.test(separateContent);
          if (separateReasoning) {
            // Reasoning came back in a dedicated field — separate mode works.
            reasoningMode = 'separate';
            isReasoning = true;
            logger.debug(`[Discovery] Separate-reasoning test for ${provider.name}/${modelId}: reasoning in '${field}' (len ${String(separateReasoning).length})`);
          } else if (!separateHasInlineThink && !normalHasInlineThink) {
            // No reasoning anywhere — model doesn't reason.
            reasoningMode = 'none';
          }
          // If inline Mattis still present even with separate params, leave
          // reasoningMode undefined — the router safety net will handle it.
        } else if (responseSeparate.status === 400) {
          // Provider rejected the separate-params — leave reasoningMode
          // undefined; the router will fall back to disable-thinking logic
          // and the inline-think safety net.
          logger.debug(`[Discovery] Separate-reasoning test for ${provider.name}/${modelId}: params rejected (400) — leaving reasoningMode undefined`);
        }
      } catch (err) {
        logger.debug(`[Discovery] Separate-reasoning test for ${provider.name}/${modelId} failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    }

    // ─── Test 3: Streaming support ──────────────────────────────────────────
    let supportsStreaming: boolean | undefined;
    try {
      const streamResponse = await fetch(`${baseURL}/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model: modelId,
          messages: [{ role: 'user', content: 'Say OK' }],
          max_tokens: 10,
          stream: true,
        }),
        signal: AbortSignal.timeout(10_000),
      });
      // Check if response is SSE (text/event-stream) or has streaming chunks
      const contentType = streamResponse.headers.get('content-type') || '';
      supportsStreaming = streamResponse.ok && (
        contentType.includes('text/event-stream') ||
        contentType.includes('application/x-ndjson')
      );
      // Consume the stream to free the connection
      if (supportsStreaming) {
        await streamResponse.text().catch(() => {});
      }
    } catch {
      supportsStreaming = false;
    }

    // ─── Derive reasoningMode ────────────────────────────────────────────────
    // Priority: disabled (explicitly turned off) > separate (dedicated field)
    // > none (no reasoning) > undefined (unprofiled / inconclusive).
    if (canDisableThinking) {
      reasoningMode = 'disabled';
    } else if (reasoningMode === 'separate') {
      // Already set by Test 2b — keep it.
    } else if (isReasoning === false) {
      reasoningMode = 'none';
    } else if (normalHasInlineThink) {
      // Inline think detected in Test 1 but we couldn't separate it — leave
      // undefined so the router applies the inline-think safety net.
      isReasoning = true;
    }

    const profile = {
      alive,
      speed: speedNormal,
      isReasoning,
      canDisableThinking,
      supportsStreaming,
      reasoningMode,
    };

    logger.info(
      `[Discovery] Profiled ${provider.name}/${modelId}: ` +
      `${speedNormal || '?'} t/s, ` +
      `reasoning=${isReasoning ?? '?'}, ` +
      `canDisable=${canDisableThinking ?? '?'}, ` +
      `reasoningMode=${reasoningMode ?? '?'}, ` +
      `streaming=${supportsStreaming ?? '?'}`
    );

    return profile;
  }

  /**
   * Use a DIFFERENT provider to classify a new model.
   * This is the "providers maintain providers" pattern.
   * Returns taskType, intelligence, and estimated speed (tokens/sec).
   */
  private async classifyModel(
    sourceProvider: string,
    modelId: string
  ): Promise<{ taskType: 'heavy' | 'light'; intelligence?: number; speed?: number; category: ModelCategory }> {
    if (!this.llmRouter) return { taskType: 'heavy', category: classifyModelCategory(modelId) };

    // Provider speed baselines (tokens/sec) — used as fallback if LLM can't estimate
    const providerSpeedBaseline: Record<string, number> = {
      groq: 500, 'gemini-free': 397, glm: 97, sambanova: 100,
      nvidia: 30, cloudflare: 50, openai: 80, claude: 70,
      mistral: 60, deepseek: 50, grok: 40, 'gemini-paid': 300,
    };

    const prompt = `You are a model catalog classifier. A new LLM model was discovered on ${sourceProvider}.
Model ID: "${modelId}"

Based on the model ID, classify it:
1. Category: "chat" (text generation / instruction following), "vision" (image understanding only, cannot do text chat), "embedding" (text embeddings), "image-gen" (image generation), "audio" (speech/TTS), "rerank" (re-ranking), or "other" (safety guard, content filter, classifier, moderation, etc.)
2. Is this a large/heavy model (70B+ params, reasoning model) or a small/light model (8B, fast)?
3. Estimate its intelligence score (0-100, where 100 = best, 9 = Llama 3.3 70B, 53 = GLM-5.2, 37 = Gemini Flash-Lite)
4. Estimate its output speed in tokens/sec. Consider:
   - Provider: ${sourceProvider} (baseline ~${providerSpeedBaseline[sourceProvider] || 50} t/s)
   - Model size: 8B models are fast (~500+ t/s on Groq), 70B are slower (~100 t/s)
   - "flash" / "lite" / "mini" / "tiny" variants are faster
   - "ultra" / "max" / "pro" variants are slower but smarter

Respond in JSON only:
{"category": "chat" | "vision" | "embedding" | "image-gen" | "audio" | "rerank" | "other", "taskType": "heavy" | "light", "intelligence": <number>, "speed": <number>}`;

    try {
      const result = await this.llmRouter.processPrompt(prompt, {
        preferredProvider: 'auto',
        taskType: 'classification',
      });
      const cleaned = result.text.replace(/```json|```/g, '').trim();
      const parsed = JSON.parse(cleaned);
      const validCategories: ModelCategory[] = ['chat', 'vision', 'embedding', 'image-gen', 'audio', 'rerank', 'other'];
      const category: ModelCategory = validCategories.includes(parsed.category) ? parsed.category : classifyModelCategory(modelId);
      return {
        category,
        taskType: parsed.taskType === 'light' ? 'light' : 'heavy',
        intelligence: typeof parsed.intelligence === 'number' ? parsed.intelligence : undefined,
        speed: typeof parsed.speed === 'number' ? parsed.speed : providerSpeedBaseline[sourceProvider],
      };
    } catch {
      // If classification fails, use regex fallback for category + heavy default
      return { taskType: 'heavy', speed: providerSpeedBaseline[sourceProvider], category: classifyModelCategory(modelId) };
    }
  }

  /**
   * Scheduled daily refresh of all existing providers.
   * Re-probes model lists, marks removed models, discovers new ones.
   */
  async refreshAllProviders(): Promise<void> {
    const providers = catalogManager.getAllProviders();
    logger.info(`[Discovery] Starting daily refresh of ${providers.length} providers`);
    for (const provider of providers) {
      if (!provider.catalogEndpoint) continue;
      await this.discoverProvider(provider.name);
      // 2s between providers to avoid hammering
      await new Promise(r => setTimeout(r, 2000));
    }
    logger.info('[Discovery] Daily refresh complete');
  }

  // ─── Weekly deep check ─────────────────────────────────────────────────────

  /**
   * Weekly deep check — goes beyond the daily refresh:
   *   1. Probes PAID providers for free-tier models (they may have introduced free tiers)
   *   2. Re-evaluates models that were 403'd (moved to paid) — maybe they moved back to free
   *   3. Re-probes all dead providers (maybe they came back online)
   *
   * This is the "providers maintain providers" deep scan that catches changes
   * the daily refresh misses (which only checks providers already in the catalog
   * with known catalog endpoints).
   */
  async weeklyDeepCheck(): Promise<void> {
    if (this.running) {
      logger.debug('[Discovery] Weekly deep check skipped — discovery already running');
      return;
    }
    logger.info('[Discovery] Starting weekly deep check');

    // 1. Probe paid providers for free-tier models
    for (const paidProvider of PAID_PROVIDERS_TO_PROBE) {
      const apiKey = process.env[paidProvider.envKey];
      if (!apiKey) continue; // user hasn't configured this provider

      await this.probePaidProviderForFreeTier(paidProvider);
      await new Promise(r => setTimeout(r, 3000)); // 3s between paid providers
    }

    // 2. Re-evaluate 403'd models across all providers
    await this.reevaluateForbiddenModels();

    // 3. Re-probe all dead providers
    await this.reprobeDeadProviders();

    // 4. Re-profile active chat models that haven't been benchmarked in 7 days
    await this.reprofileStaleModels();

    // 5. Save catalog
    await catalogManager.save();
    logger.info('[Discovery] Weekly deep check complete');
  }

  /**
   * Re-profile active chat models that haven't been benchmarked in 7 days.
   * This catches speed changes (provider upgrades/downgrades), reasoning capability
   * changes (new thinking-disable support), and streaming support changes.
   */
  private async reprofileStaleModels(): Promise<void> {
    const STALE_THRESHOLD_MS = 7 * 24 * 60 * 60 * 1000; // 7 days
    const now = Date.now();
    let profiled = 0;

    for (const provider of catalogManager.getAllProviders()) {
      if (provider.status === 'dead') continue;

      // Only profile chat models (skip vision, embedding, etc.)
      const chatModels = provider.models.filter(
        m => m.status === 'active' && (!m.category || m.category === 'chat')
      );

      for (const model of chatModels) {
        // Skip models benchmarked recently
        if (model.benchmarkedAt) {
          const age = now - new Date(model.benchmarkedAt).getTime();
          if (age < STALE_THRESHOLD_MS) continue;
        }

        const profile = await this.profileModel(provider, model.id);
        if (profile.alive) {
          // Update model with fresh profile data
          catalogManager.updateModelProfile(provider.name, model.id, {
            speed: profile.speed,
            isReasoning: profile.isReasoning,
            canDisableThinking: profile.canDisableThinking,
            supportsStreaming: profile.supportsStreaming,
            reasoningMode: profile.reasoningMode,
            benchmarkedAt: new Date().toISOString(),
            benchmarkSpeed: profile.speed,
          });
          profiled++;
        }
        // Small delay between profiles to respect rate limits
        await new Promise(r => setTimeout(r, 500));
      }
    }

    if (profiled > 0) {
      logger.info(`[Discovery] Re-profiled ${profiled} stale models`);
    }
  }

  /**
   * Probe a paid provider to see if it has any free-tier models.
   *
   * Strategy:
   *   a) Fetch the /models endpoint (if available) to get all model IDs
   *   b) For each known free-tier candidate model, send a minimal chat completion
   *   c) If the response is 200 (not 402/403/payment required), the model is free
   *   d) Add surviving models to the catalog as free-tier entries
   *
   * This catches cases like:
   *   - Mistral introducing a free tier for mistral-tiny
   *   - DeepSeek offering free API access for deepseek-chat
   *   - Grok adding a free model
   *   - OpenAI adding gpt-4o-mini to a free tier
   */
  private async probePaidProviderForFreeTier(paidProvider: {
    name: string;
    baseURL: string;
    envKey: string;
    catalogEndpoint: string;
  }): Promise<void> {
    const apiKey = process.env[paidProvider.envKey];
    if (!apiKey) return;

    logger.info(`[Discovery] Probing paid provider ${paidProvider.name} for free-tier models`);

    // Gather candidate model IDs to probe
    const candidates = new Set<string>(FREE_TIER_CANDIDATES[paidProvider.name] || []);

    // Also fetch the /models endpoint for additional candidates
    if (paidProvider.catalogEndpoint) {
      try {
        const response = await fetch(paidProvider.catalogEndpoint, {
          headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
          signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
        });
        if (response.ok) {
          const data: any = await response.json();
          const modelIds: string[] = Array.isArray(data?.data)
            ? data.data.map((m: any) => m.id).filter(Boolean)
            : Array.isArray(data?.models)
              ? data.models.map((m: any) => (m.name || m.id)?.replace(/^models\//, '')).filter(Boolean)
              : [];
          // Add small/cheap models that might be free (filter by name patterns)
          for (const id of modelIds) {
            const lower = id.toLowerCase();
            if (/mini|tiny|small|lite|flash|haiku|nano|free|8b|7b|3b|1b/i.test(lower)) {
              candidates.add(id);
            }
          }
        }
      } catch (err) {
        logger.debug(`[Discovery] Could not fetch ${paidProvider.name} models list: ${err instanceof Error ? err.message : String(err)}`);
      }
    }

    if (candidates.size === 0) {
      logger.debug(`[Discovery] No free-tier candidates for ${paidProvider.name}`);
      return;
    }

    // Probe each candidate with a minimal chat completion — probe ALL candidates,
    // don't pre-filter by regex. The probe determines if the model is chat-capable.
    let freeModelsFound = 0;
    for (const modelId of candidates) {
      const isFree = await this.probeModelFree(paidProvider.baseURL, apiKey, modelId);
      if (isFree) {
        // Check if we already have this model in the catalog
        const existing = catalogManager.getProvider(paidProvider.name);
        const alreadyExists = existing?.models.some(m => m.id === modelId);

        if (!alreadyExists) {
          // Classify and add as a free model — classifyModel now returns category too
          const classification = await this.classifyModel(paidProvider.name, modelId);
          const now = new Date().toISOString();
          catalogManager.addModel(paidProvider.name, {
            id: modelId,
            taskType: classification.taskType,
            intelligence: classification.intelligence,
            speed: classification.speed,
            category: classification.category,
            status: 'active',
            discoveredAt: now,
            lastVerifiedAt: now,
            consecutiveFailures: 0,
            totalCalls: 0,
            totalSuccesses: 0,
          });
          logger.info(`[Discovery] FREE-TIER model found on paid provider: ${paidProvider.name}/${modelId} (category: ${classification.category}, ${classification.taskType}, intel ~${classification.intelligence}, speed ~${classification.speed} t/s)`);
          freeModelsFound++;
        } else {
          logger.debug(`[Discovery] ${paidProvider.name}/${modelId} is free but already in catalog`);
        }
      }
    }

    if (freeModelsFound > 0) {
      logger.info(`[Discovery] ${paidProvider.name}: found ${freeModelsFound} free-tier model(s)`);
    } else {
      logger.debug(`[Discovery] ${paidProvider.name}: no free-tier models found`);
    }
  }

  /**
   * Probe a single model to check if it's accessible without payment.
   * Returns true if the model responds with 200 or 429 (rate-limited but free).
   * Returns false if it returns 402 (payment required) or 403 (forbidden).
   */
  private async probeModelFree(baseURL: string, apiKey: string, modelId: string): Promise<boolean> {
    try {
      const response = await fetch(`${baseURL}/chat/completions`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: modelId,
          messages: [{ role: 'user', content: 'Say OK' }],
          max_tokens: 8,
          stream: false,
        }),
        signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
      });

      // 200 = works, 429 = rate-limited but free
      if (response.ok || response.status === 429) return true;
      // 402 = payment required, 403 = forbidden (paid only)
      if (response.status === 402 || response.status === 403) return false;
      // Other errors (500, 503, etc.) — inconclusive, assume not free
      logger.debug(`[Discovery] Free-tier probe ${modelId} returned ${response.status}`);
      return false;
    } catch {
      return false;
    }
  }

  /**
   * Re-evaluate models that were disabled with a 403 (moved to paid).
   * Providers sometimes move models back to free, or introduce free tiers
   * for previously paid-only models. This re-probes them.
   */
  private async reevaluateForbiddenModels(): Promise<void> {
    const providers = catalogManager.getAllProviders();
    let reactivated = 0;

    for (const provider of providers) {
      const forbiddenModels = provider.models.filter(
        m => (m.status === 'dead' || m.status === 'disabled') &&
             /HTTP 40[23]|moved to paid|payment required|403|402/i.test(m.disabledReason || '')
      );

      for (const model of forbiddenModels) {
        const apiKey = process.env[provider.envKey];
        if (!apiKey) continue;

        let baseURL = provider.baseURL;
        if (provider.name === 'cloudflare') {
          const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;
          if (!accountId) continue;
          baseURL = `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/v1`;
        }

        const isFree = await this.probeModelFree(baseURL, apiKey, model.id);
        if (isFree) {
          catalogManager.reactivateModel(provider.name, model.id);
          logger.info(`[Discovery] ${provider.name}/${model.id} re-evaluated: now FREE — reactivating`);
          reactivated++;
        }
      }
    }

    if (reactivated > 0) {
      logger.info(`[Discovery] Re-evaluated forbidden models: ${reactivated} reactivated`);
    }
  }

  /**
   * Re-probe all providers marked as 'dead' — maybe they came back online.
   */
  private async reprobeDeadProviders(): Promise<void> {
    const providers = catalogManager.getAllProviders();
    let revived = 0;

    for (const provider of providers) {
      if (provider.status !== 'dead') continue;
      if (!provider.catalogEndpoint) continue;

      const apiKey = process.env[provider.envKey];
      if (!apiKey) continue;

      // Gemini native API uses ?key= query param, not Bearer header
      const isGoogle = provider.apiType === 'google';
      const url = isGoogle
        ? `${provider.catalogEndpoint}?key=${apiKey}`
        : provider.catalogEndpoint;
      const headers: Record<string, string> = {};
      if (!isGoogle) {
        headers['Authorization'] = `Bearer ${apiKey}`;
      }

      // Try fetching the catalog endpoint
      try {
        const response = await fetch(url, {
          headers,
          signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
        });

        if (response.ok) {
          // Provider is back! Re-probe its models
          logger.info(`[Discovery] Dead provider ${provider.name} is back online — re-probing models`);
          await this.discoverProvider(provider.name);
          revived++;
        }
      } catch {
        // Still dead
      }
    }

    if (revived > 0) {
      logger.info(`[Discovery] Dead provider revival: ${revived} provider(s) back online`);
    }
  }

  // ─── Natural language instruction API ──────────────────────────────────────

  /**
   * Execute a natural language instruction from the user.
   *
   * Flow:
   *   1. Build a context prompt with current catalog state
   *   2. Send instruction + context to LLM → get JSON action plan
   *   3. Execute each action sequentially
   *   4. Send results back to LLM → get human-readable summary
   *   5. Return { summary, actions, results }
   */
  async executeInstruction(instruction: string): Promise<{
    summary: string;
    actions: ParsedAction[];
    results: ActionResult[];
  }> {
    if (!this.llmRouter) {
      throw new Error('LLM router not configured — cannot execute natural language instruction');
    }

    logger.info(`[Discovery] Executing natural language instruction: "${instruction}"`);

    // Step 1: Build catalog context for the LLM
    const catalogContext = this.buildCatalogContext();

    // Step 2: Ask the LLM to parse the instruction into actions
    const parsePrompt = this.buildParsePrompt(instruction, catalogContext);
    const parseResult = await this.llmRouter.processPrompt(parsePrompt, {
      preferredProvider: 'auto',
      taskType: 'classification',
    });

    let actions: ParsedAction[];
    try {
      const cleaned = parseResult.text.replace(/```json|```/g, '').trim();
      const parsed = JSON.parse(cleaned);
      actions = Array.isArray(parsed.actions) ? parsed.actions : [];
    } catch {
      throw new Error(`Failed to parse LLM action plan. Raw response: ${parseResult.text.substring(0, 500)}`);
    }

    if (actions.length === 0) {
      return {
        summary: 'I could not determine what action to take from your instruction. Try something like:\n' +
          '- "check if any of my providers have new fast models"\n' +
          '- "find free tier models on mistral"\n' +
          '- "check health of all providers"\n' +
          '- "look for new vision models"\n' +
          '- "re-probe dead models"',
        actions: [],
        results: [],
      };
    }

    // Step 3: Execute each action
    const results: ActionResult[] = [];
    for (const action of actions) {
      logger.info(`[Discovery] Executing action: ${action.action}`, { params: action.params });
      try {
        const result = await this.executeAction(action);
        results.push({ action: action.action, success: true, ...result });
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        results.push({ action: action.action, success: false, error: errMsg });
        logger.warn(`[Discovery] Action ${action.action} failed: ${errMsg}`);
      }
    }

    // Step 4: Generate human-readable summary
    const summaryPrompt = this.buildSummaryPrompt(instruction, actions, results);
    const summaryResult = await this.llmRouter.processPrompt(summaryPrompt, {
      preferredProvider: 'auto',
      taskType: 'classification',
    });

    logger.info('[Discovery] Natural language instruction complete', {
      actions: actions.length,
      successes: results.filter(r => r.success).length,
    });

    return {
      summary: summaryResult.text.trim(),
      actions,
      results,
    };
  }

  /**
   * Build a JSON summary of the current catalog state for the LLM context.
   */
  private buildCatalogContext(): string {
    const report = catalogManager.getHealthReport();
    const tpd = this.getTpdContext();

    // Compact summary — don't send full model details, just counts + statuses
    const providers = report.providers.map((p: any) => ({
      name: p.name,
      status: p.status,
      activeModels: p.activeModels,
      totalModels: p.totalModels,
      models: p.models.filter((m: any) => m.status === 'active').map((m: any) => ({
        id: m.id,
        taskType: m.taskType,
        intelligence: m.intelligence,
        successRate: m.successRate,
      })),
    }));

    return JSON.stringify({
      catalog: { summary: report.summary, providers },
      tpdUsage: tpd,
      specialModels: catalogManager.getSpecialModels().map(({ provider, model }) => ({
        provider, id: model.id, category: model.category,
      })),
    }, null, 2);
  }

  /**
   * Get TPD usage context (imported lazily to avoid circular dependency).
   */
  private getTpdContext(): Record<string, { used: number; limit: number | undefined; percent: number }> {
    try {
      // Dynamic require to avoid circular dependency
      const { providerCircuitBreaker } = require('./providerCircuitBreaker');
      return providerCircuitBreaker.getAllTpdUsage();
    } catch {
      return {};
    }
  }

  /**
   * Build the prompt that asks the LLM to parse the user's instruction into actions.
   */
  private buildParsePrompt(instruction: string, catalogContext: string): string {
    return `You are the ThinkDrop Discovery Agent controller. A user gave you a natural language instruction about their LLM provider catalog. Your job is to determine what discovery actions to take and respond with a JSON action plan.

USER INSTRUCTION: "${instruction}"

CURRENT CATALOG STATE:
${catalogContext}

AVAILABLE ACTIONS (choose one or more):
- discover_provider   params: { "provider": "<name>" }          — Probe a specific provider's /models endpoint for new/removed models
- discover_all        params: {}                                 — Probe ALL providers for model list changes
- weekly_check        params: {}                                 — Deep check: probe paid providers for free tiers, re-evaluate 403'd models, revive dead providers
- find_fast_models    params: { "min_speed": <number> }          — Discover all, then report models with speed >= min_speed tokens/sec
- find_free_tier      params: { "provider": "<name>" }           — Probe a specific paid provider (or all if no provider) for free-tier models
- check_health        params: {}                                 — Return current catalog health report (no discovery needed)
- reevaluate_dead     params: {}                                 — Re-probe all dead/disabled models to see if they came back
- find_special        params: { "category": "vision|embedding|image-gen|audio|rerank" } — Discover all, then report special models of the given category

RULES:
1. Respond with JSON ONLY — no markdown, no explanation outside JSON
2. Choose the MINIMUM set of actions that satisfy the user's request
3. If the user asks about "fast" models, use find_fast_models with min_speed (default 200 if not specified)
4. If the user asks about "free" models on a specific paid provider, use find_free_tier with that provider
5. If the user asks about "new" models, use discover_all (or discover_provider if they named one)
6. If the user asks about "health" or "status", use check_health
7. If the user asks about "vision" or "embedding" or "image" models, use find_special with the appropriate category
8. If the user asks to "re-check" or "re-probe" dead models, use reevaluate_dead
9. If the instruction is ambiguous, pick the most likely action and note the ambiguity in the "reason" field

RESPONSE FORMAT:
{
  "actions": [
    {
      "action": "discover_all",
      "params": {},
      "reason": "User asked to find new models across all providers"
    }
  ]
}`;
  }

  /**
   * Build the prompt that asks the LLM to summarize the action results.
   */
  private buildSummaryPrompt(instruction: string, actions: ParsedAction[], results: ActionResult[]): string {
    const resultsJson = JSON.stringify(results, null, 2);
    const actionsJson = JSON.stringify(actions.map(a => ({ action: a.action, params: a.params, reason: a.reason })), null, 2);

    return `You are the ThinkDrop Discovery Agent. The user asked: "${instruction}"

You executed these actions:
${actionsJson}

Here are the results:
${resultsJson}

Write a concise, human-readable summary of what you found and what you did. Include:
1. What actions you took
2. Key findings (new models, dead models, free-tier discoveries, health issues, etc.)
3. Any recommendations for the user

Be specific — mention model names, providers, and numbers. Don't be verbose. Use bullet points if helpful.`;
  }

  /**
   * Execute a single parsed action.
   */
  private async executeAction(action: ParsedAction): Promise<Omit<ActionResult, 'action' | 'success'>> {
    const params = action.params || {};

    switch (action.action) {
      case 'discover_provider': {
        const provider = params.provider;
        if (!provider) throw new Error('discover_provider requires "provider" param');
        const exists = catalogManager.getProvider(provider);
        if (!exists) throw new Error(`Unknown provider: ${provider}`);
        await this.discoverProvider(provider);
        const p = catalogManager.getProvider(provider)!;
        return {
          result: {
            provider,
            activeModels: p.models.filter(m => m.status === 'active').length,
            totalModels: p.models.length,
            models: p.models.filter(m => m.status === 'active').map(m => ({
              id: m.id, taskType: m.taskType, category: m.category, intelligence: m.intelligence,
            })),
          },
        };
      }

      case 'discover_all': {
        await this.refreshAllProviders();
        const report = catalogManager.getHealthReport();
        return {
          result: {
            summary: report.summary,
            providers: report.providers.map((p: any) => ({
              name: p.name, status: p.status, activeModels: p.activeModels, totalModels: p.totalModels,
            })),
          },
        };
      }

      case 'weekly_check': {
        await this.weeklyDeepCheck();
        const report = catalogManager.getHealthReport();
        return {
          result: {
            summary: report.summary,
            message: 'Weekly deep check complete — probed paid providers for free tiers, re-evaluated forbidden models, revived dead providers',
          },
        };
      }

      case 'find_fast_models': {
        const minSpeed = typeof params.min_speed === 'number' ? params.min_speed : 200;
        await this.refreshAllProviders();
        const fastModels: Array<{ provider: string; id: string; speed?: number; intelligence?: number }> = [];
        for (const p of catalogManager.getAllProviders()) {
          for (const m of p.models) {
            if (m.status !== 'active' || m.category !== 'chat' && m.category !== undefined) continue;
            if (m.category && m.category !== 'chat') continue;
            // Use speed from catalog, or estimate based on provider
            const speed = (m as any).speed || this.estimateSpeed(p.name);
            if (speed >= minSpeed) {
              fastModels.push({ provider: p.name, id: m.id, speed, intelligence: m.intelligence });
            }
          }
        }
        fastModels.sort((a, b) => (b.speed || 0) - (a.speed || 0));
        return { result: { minSpeed, count: fastModels.length, models: fastModels } };
      }

      case 'find_free_tier': {
        const provider = params.provider;
        if (provider) {
          // Probe a single paid provider
          const paidProvider = PAID_PROVIDERS_TO_PROBE.find(p => p.name === provider);
          if (!paidProvider) throw new Error(`Cannot probe ${provider} — not a known paid provider`);
          await this.probePaidProviderForFreeTier(paidProvider);
          const p = catalogManager.getProvider(provider);
          return {
            result: {
              provider,
              activeModels: p?.models.filter(m => m.status === 'active').length || 0,
              models: p?.models.filter(m => m.status === 'active').map(m => ({
                id: m.id, category: m.category, intelligence: m.intelligence,
              })),
            },
          };
        } else {
          // Probe all paid providers
          await this.weeklyDeepCheck();
          const report = catalogManager.getHealthReport();
          return { result: { summary: report.summary, message: 'Probed all paid providers for free-tier models' } };
        }
      }

      case 'check_health': {
        const report = catalogManager.getHealthReport();
        const tpd = this.getTpdContext();
        return { result: { health: report.summary, tpdUsage: tpd, providers: report.providers.map((p: any) => ({
          name: p.name, status: p.status, activeModels: p.activeModels, totalModels: p.totalModels,
        })) } };
      }

      case 'reevaluate_dead': {
        await this.reevaluateForbiddenModels();
        await this.reprobeDeadProviders();
        await catalogManager.save();
        const report = catalogManager.getHealthReport();
        return { result: { summary: report.summary, message: 'Re-evaluated all dead/disabled models' } };
      }

      case 'find_special': {
        const category = params.category as ModelCategory | undefined;
        await this.refreshAllProviders();
        const models = catalogManager.getSpecialModels(category);
        return {
          result: {
            category: category || 'all',
            count: models.length,
            models: models.map(({ provider, model }) => ({
              provider, id: model.id, category: model.category, intelligence: model.intelligence,
            })),
          },
        };
      }

      default:
        throw new Error(`Unknown action: ${action.action}`);
    }
  }

  /**
   * Estimate speed for a model if not explicitly set.
   */
  private estimateSpeed(provider: string): number {
    const estimates: Record<string, number> = {
      groq: 500,
      'gemini-free': 397,
      glm: 97,
      sambanova: 100,
      nvidia: 30,
      cloudflare: 50,
    };
    return estimates[provider] || 50;
  }
}

interface ParsedAction {
  action: string;
  params: Record<string, any>;
  reason?: string;
}

interface ActionResult {
  action: string;
  success: boolean;
  result?: any;
  error?: string;
}

export const discoveryAgent = new DiscoveryAgent();
