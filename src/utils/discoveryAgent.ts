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
import { classifyModelCategory, ModelCategory } from './providerConfig';

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

      // Step 3: Probe new models with minimal request
      for (const modelId of newModels) {
        // Categorize the model by its name — chat models get probed + classified,
        // non-chat models (vision, embedding, image-gen, audio) get cataloged as special
        const category = classifyModelCategory(modelId);

        if (category === 'chat') {
          // Chat model — probe with "Say OK" to verify it works
          const probeResult = await this.probeModel(provider, modelId);
          if (probeResult.alive) {
            // Step 4: Classify the new model using a DIFFERENT provider
            const classification = await this.classifyModel(providerName, modelId);
            const now = new Date().toISOString();
            catalogManager.addModel(providerName, {
              id: modelId,
              taskType: classification.taskType,
              intelligence: classification.intelligence,
              speed: classification.speed,
              contextWindow: probeResult.contextWindow,
              category: 'chat',
              status: 'active',
              discoveredAt: now,
              lastVerifiedAt: now,
              consecutiveFailures: 0,
              totalCalls: 0,
              totalSuccesses: 0,
            });
            logger.info(`[Discovery] NEW chat model: ${providerName}/${modelId} (${classification.taskType}, intel ~${classification.intelligence}, speed ~${classification.speed} t/s)`);
          }
        } else {
          // Non-chat model (vision, embedding, image-gen, audio, rerank) —
          // catalog it as a special model without probing with "Say OK"
          // (probing a vision model with text-only input would fail or be meaningless).
          // The specialized service (vision.ts, screen-intelligence, etc.) will
          // probe it with the right input format when it actually uses it.
          const now = new Date().toISOString();
          catalogManager.addModel(providerName, {
            id: modelId,
            taskType: 'heavy', // special models don't use taskType routing
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
      const response = await fetch(provider.catalogEndpoint, {
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
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
   * Use a DIFFERENT provider to classify a new model.
   * This is the "providers maintain providers" pattern.
   * Returns taskType, intelligence, and estimated speed (tokens/sec).
   */
  private async classifyModel(
    sourceProvider: string,
    modelId: string
  ): Promise<{ taskType: 'heavy' | 'light'; intelligence?: number; speed?: number }> {
    if (!this.llmRouter) return { taskType: 'heavy' };

    // Provider speed baselines (tokens/sec) — used as fallback if LLM can't estimate
    const providerSpeedBaseline: Record<string, number> = {
      groq: 500, 'gemini-free': 397, glm: 97, sambanova: 100,
      nvidia: 30, cloudflare: 50, openai: 80, claude: 70,
      mistral: 60, deepseek: 50, grok: 40, 'gemini-paid': 300,
    };

    const prompt = `You are a model catalog classifier. A new LLM model was discovered on ${sourceProvider}.
Model ID: "${modelId}"

Based on the model ID, classify it:
1. Is this a large/heavy model (70B+ params, reasoning model) or a small/light model (8B, fast)?
2. Estimate its intelligence score (0-100, where 100 = best, 9 = Llama 3.3 70B, 53 = GLM-5.2, 37 = Gemini Flash-Lite)
3. Estimate its output speed in tokens/sec. Consider:
   - Provider: ${sourceProvider} (baseline ~${providerSpeedBaseline[sourceProvider] || 50} t/s)
   - Model size: 8B models are fast (~500+ t/s on Groq), 70B are slower (~100 t/s)
   - "flash" / "lite" / "mini" / "tiny" variants are faster
   - "ultra" / "max" / "pro" variants are slower but smarter

Respond in JSON only:
{"taskType": "heavy" | "light", "intelligence": <number>, "speed": <number>}`;

    try {
      const result = await this.llmRouter.processPrompt(prompt, {
        preferredProvider: 'auto',
        taskType: 'classification',
      });
      const cleaned = result.text.replace(/```json|```/g, '').trim();
      const parsed = JSON.parse(cleaned);
      return {
        taskType: parsed.taskType === 'light' ? 'light' : 'heavy',
        intelligence: typeof parsed.intelligence === 'number' ? parsed.intelligence : undefined,
        speed: typeof parsed.speed === 'number' ? parsed.speed : providerSpeedBaseline[sourceProvider],
      };
    } catch {
      // If classification fails, default to heavy with provider baseline speed
      return { taskType: 'heavy', speed: providerSpeedBaseline[sourceProvider] };
    }
  }

  /**
   * Check if a model is a chat/text model (used by free-tier probe to skip
   * non-chat models that can't be probed with "Say OK").
   */
  private isChatModel(modelId: string): boolean {
    return classifyModelCategory(modelId) === 'chat';
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

    // 4. Save catalog
    await catalogManager.save();
    logger.info('[Discovery] Weekly deep check complete');
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

    // Probe each candidate with a minimal chat completion
    let freeModelsFound = 0;
    for (const modelId of candidates) {
      if (!this.isChatModel(modelId)) continue;

      const isFree = await this.probeModelFree(paidProvider.baseURL, apiKey, modelId);
      if (isFree) {
        // Check if we already have this model in the catalog
        const existing = catalogManager.getProvider(paidProvider.name);
        const alreadyExists = existing?.models.some(m => m.id === modelId);

        if (!alreadyExists) {
          // Classify and add as a free model
          const classification = await this.classifyModel(paidProvider.name, modelId);
          const now = new Date().toISOString();
          catalogManager.addModel(paidProvider.name, {
            id: modelId,
            taskType: classification.taskType,
            intelligence: classification.intelligence,
            speed: classification.speed,
            category: classifyModelCategory(modelId),
            status: 'active',
            discoveredAt: now,
            lastVerifiedAt: now,
            consecutiveFailures: 0,
            totalCalls: 0,
            totalSuccesses: 0,
          });
          logger.info(`[Discovery] FREE-TIER model found on paid provider: ${paidProvider.name}/${modelId} (${classification.taskType}, intel ~${classification.intelligence}, speed ~${classification.speed} t/s)`);
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

      // Try fetching the catalog endpoint
      try {
        const response = await fetch(provider.catalogEndpoint, {
          headers: { 'Authorization': `Bearer ${apiKey}` },
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
