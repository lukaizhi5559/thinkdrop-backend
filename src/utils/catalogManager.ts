/**
 * CatalogManager — singleton holding the live provider/model catalog.
 *
 * Tracks per-model health (success rate, consecutive failures, last error)
 * and per-provider status (active/degraded/dead). The router reads from this
 * to decide which models to try. The discovery agent updates it when models
 * fail or new models appear.
 *
 * On startup, loads from data/provider-catalog.json. If the file doesn't exist,
 * seeds from the hardcoded PROVIDER_CONFIG in providerConfig.ts.
 */

import fs from 'fs';
import path from 'path';
import { logger } from './logger';
import { PROVIDER_CONFIG, HEAVY_CHAIN, LIGHT_CHAIN, SUPER_HEAVY_CHAIN, COMPLEX_CHAIN, PAID_CHAIN, TaskType, ProviderModel, ModelCategory, isSuperHeavyModel } from './providerConfig';

export type ModelStatus = 'active' | 'degraded' | 'disabled' | 'dead';
export type ProviderStatus = 'active' | 'degraded' | 'dead';

export interface CatalogModelEntry extends ProviderModel {
  provider: string;
  taskType: TaskType;
  category?: ModelCategory;
  rpm?: number;
  rpd?: number;
  tpd?: number;
  status: ModelStatus;
  disabledReason?: string;
  discoveredAt: string;
  lastVerifiedAt: string;
  // Runtime health (not persisted)
  consecutiveFailures: number;
  lastError?: string;
  lastErrorTime?: string;
  lastSuccessTime?: string;
  lastResponseMs?: number;
  totalCalls: number;
  totalSuccesses: number;
}

export interface CatalogProviderEntry {
  name: string;
  baseURL: string;
  envKey: string;
  catalogEndpoint?: string;
  apiType?: string; // ProviderAPIType — stored as string for JSON serialization
  tier?: 'free' | 'paid';
  status: ProviderStatus;
  models: CatalogModelEntry[];
}

interface CatalogFile {
  version: number;
  lastUpdated: string;
  providers: CatalogProviderEntry[];
}

const CATALOG_PATH = path.join(process.cwd(), 'data', 'provider-catalog.json');
const DISCOVERY_TRIGGER_FAILURES = 5;

class CatalogManager {
  private providers: Map<string, CatalogProviderEntry> = new Map();
  private loaded = false;
  private discoveryCallback: ((provider: string, modelId: string, statusCode: number) => void) | null = null;

  /**
   * Set the callback that the discovery agent registers.
   * Called when a model fails enough to trigger discovery.
   */
  setDiscoveryCallback(cb: (provider: string, modelId: string, statusCode: number) => void): void {
    this.discoveryCallback = cb;
  }

  /**
   * Load catalog from JSON file, or seed from hardcoded config if file doesn't exist.
   */
  async load(): Promise<void> {
    try {
      if (fs.existsSync(CATALOG_PATH)) {
        const raw = fs.readFileSync(CATALOG_PATH, 'utf-8');
        const data: CatalogFile = JSON.parse(raw);
        this.providers.clear();
        for (const p of data.providers) {
          this.providers.set(p.name, p);
        }
        logger.info(`[CatalogManager] Loaded ${this.providers.size} providers from ${CATALOG_PATH}`);
      } else {
        await this.seedFromConfig();
        logger.info(`[CatalogManager] No catalog file found — seeded from providerConfig.ts`);
      }
      // Fix up models missing contextWindow data so the router can skip
      // models with insufficient context (e.g. nemotron-mini-4b = 4096).
      this.fixupMissingContextWindows();
      // Sync new/updated providers from PROVIDER_CONFIG into the loaded catalog
      this.syncWithConfig();
      // Note: model category verification is handled by discoveryAgent.verifyAllModelsOnStartup()
      // which probes each model with "Say OK" to functionally verify its category.
      this.loaded = true;
    } catch (err) {
      logger.error(`[CatalogManager] Failed to load catalog: ${err instanceof Error ? err.message : String(err)}`);
      await this.seedFromConfig();
      this.loaded = true;
    }
  }

  /**
   * Set contextWindow for models that are missing it, based on model name patterns.
   * This lets the router's context-window check skip models with small context
   * (e.g. nemotron-mini-4b-instruct = 4096 tokens) instead of failing with a 400.
   */
  private fixupMissingContextWindows(): void {
    let fixed = 0;
    for (const [, p] of this.providers) {
      for (const m of p.models) {
        if (m.contextWindow) continue; // already set
        const lower = m.id.toLowerCase();
        if (/mini-4b|nano-[0-9]b|-1b|-2b|-3b/.test(lower)) {
          m.contextWindow = 4096;  // tiny models typically have 4k context
          fixed++;
        } else if (/8b|7b|13b|14b|15b|20b|30b|32b|34b|49b|70b|120b|550b/.test(lower)) {
          m.contextWindow = 128000; // most modern models support 128k
          fixed++;
        }
      }
    }
    if (fixed > 0) {
      logger.info(`[CatalogManager] Set contextWindow for ${fixed} models missing it`);
      this.save().catch(() => {});
    }
  }

  /**
   * Seed the catalog from the hardcoded PROVIDER_CONFIG.
   */
  private async seedFromConfig(): Promise<void> {
    this.providers.clear();
    const now = new Date().toISOString();
    for (const [name, config] of Object.entries(PROVIDER_CONFIG)) {
      const models: CatalogModelEntry[] = [
        ...config.heavy.map(m => this.toCatalogEntry(m, name, 'heavy', now)),
        ...config.light.map(m => this.toCatalogEntry(m, name, 'light', now)),
      ];
      // Deduplicate models that appear in both heavy and light (e.g. glm-4.7-flash)
      const seen = new Set<string>();
      const deduped = models.filter(m => {
        const key = `${m.id}:${m.taskType}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
      this.providers.set(name, {
        name,
        baseURL: config.baseURL,
        envKey: config.envKey,
        catalogEndpoint: config.catalogEndpoint,
        apiType: config.apiType,
        status: 'active',
        models: deduped,
      });
    }
    await this.save();
  }

  /**
   * Sync providers from PROVIDER_CONFIG into the loaded catalog.
   * - Adds new providers that aren't in the catalog (e.g. newly configured paid providers)
   * - Updates models for existing providers if the config has new/changed models
   * - Preserves runtime health data (status, failures, etc.) for existing models
   */
  private syncWithConfig(): void {
    const now = new Date().toISOString();
    let addedProviders = 0;
    let updatedModels = 0;

    for (const [name, config] of Object.entries(PROVIDER_CONFIG)) {
      const existing = this.providers.get(name);

      if (!existing) {
        // New provider — seed it from config
        const models: CatalogModelEntry[] = [
          ...config.heavy.map(m => this.toCatalogEntry(m, name, 'heavy', now)),
          ...config.light.map(m => this.toCatalogEntry(m, name, 'light', now)),
        ];
        const seen = new Set<string>();
        const deduped = models.filter(m => {
          const key = `${m.id}:${m.taskType}`;
          if (seen.has(key)) return false;
          seen.add(key);
          return true;
        });
        this.providers.set(name, {
          name,
          baseURL: config.baseURL,
          envKey: config.envKey,
          catalogEndpoint: config.catalogEndpoint,
          apiType: config.apiType,
          status: 'active',
          models: deduped,
        });
        addedProviders++;
        continue;
      }

      // Existing provider — check for new/updated models
      const configModelIds = new Set([
        ...config.heavy.map(m => `${m.id}:heavy`),
        ...config.light.map(m => `${m.id}:light`),
      ]);
      const allConfigIds = new Set([...config.heavy.map(m => m.id), ...config.light.map(m => m.id)]);

      // Remove models that are no longer in the config (stale entries)
      // For providers with a catalogEndpoint, keep discovered models (discovery agent adds them)
      // For static providers (no catalogEndpoint), only keep models that match the config
      const beforeCount = existing.models.length;
      existing.models = existing.models.filter(m => {
        const key = `${m.id}:${m.taskType}`;
        if (configModelIds.has(key)) return true; // exact match with config
        // If provider has a catalogEndpoint, discovered models are allowed
        if (config.catalogEndpoint && !allConfigIds.has(m.id)) return true;
        // Stale config model (ID was in config but was removed/renamed) — remove it
        return false;
      });
      if (existing.models.length < beforeCount) {
        updatedModels += beforeCount - existing.models.length;
      }

      // Add models from config that aren't in the catalog
      for (const m of config.heavy) {
        const key = `${m.id}:heavy`;
        if (!existing.models.some(em => `${em.id}:${em.taskType}` === key)) {
          existing.models.push(this.toCatalogEntry(m, name, 'heavy', now));
          updatedModels++;
        }
      }
      for (const m of config.light) {
        const key = `${m.id}:light`;
        if (!existing.models.some(em => `${em.id}:${em.taskType}` === key)) {
          existing.models.push(this.toCatalogEntry(m, name, 'light', now));
          updatedModels++;
        }
      }

      // Update metadata (baseURL, apiType) for existing models from config
      for (const m of existing.models) {
        const configModels = m.taskType === 'light' ? config.light : config.heavy;
        const configModel = configModels.find(cm => cm.id === m.id);
        if (configModel) {
          if (configModel.intelligence !== undefined) m.intelligence = configModel.intelligence;
          if (configModel.contextWindow !== undefined) m.contextWindow = configModel.contextWindow;
          if (configModel.speed !== undefined) m.speed = configModel.speed;
          if (configModel.category !== undefined) m.category = configModel.category;
        }
      }
    }

    if (addedProviders > 0 || updatedModels > 0) {
      logger.info(`[CatalogManager] Synced config: ${addedProviders} new providers, ${updatedModels} new models`);
      this.save().catch(() => {});
    }
  }

  private toCatalogEntry(m: ProviderModel, provider: string, taskType: TaskType, now: string): CatalogModelEntry {
    return {
      ...m,
      provider,
      taskType,
      status: 'active',
      discoveredAt: now,
      lastVerifiedAt: now,
      consecutiveFailures: 0,
      totalCalls: 0,
      totalSuccesses: 0,
    };
  }

  /**
   * Save catalog to JSON file.
   */
  async save(): Promise<void> {
    try {
      const dir = path.dirname(CATALOG_PATH);
      if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
      const data: CatalogFile = {
        version: 1,
        lastUpdated: new Date().toISOString(),
        providers: Array.from(this.providers.values()).map(p => ({
          ...p,
          // Don't persist runtime health fields
          models: p.models.map(m => ({
            ...m,
            consecutiveFailures: 0,
            lastError: undefined,
            lastErrorTime: undefined,
            lastSuccessTime: undefined,
            totalCalls: 0,
            totalSuccesses: 0,
            lastResponseMs: 0,
          })),
        })),
      };
      fs.writeFileSync(CATALOG_PATH, JSON.stringify(data, null, 2));
      logger.debug(`[CatalogManager] Saved catalog to ${CATALOG_PATH}`);
    } catch (err) {
      logger.error(`[CatalogManager] Failed to save catalog: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  /**
   * Get models for routing — returns only 'active' models for the given task type.
   */
  getModels(provider: string, taskType: TaskType): CatalogModelEntry[] {
    const p = this.providers.get(provider);
    if (!p || p.status === 'dead') return [];
    // For super-heavy, use heavy models but filter out small ones (nano, 8b, 20b, etc.)
    if (taskType === 'super-heavy') {
      return p.models.filter(m =>
        m.taskType === 'heavy' &&
        m.status === 'active' &&
        (m.category === 'chat' || m.category === undefined) &&
        isSuperHeavyModel(m.id)
      );
    }
    // For complex, use heavy models from paid providers (no free providers)
    if (taskType === 'complex') {
      return p.models.filter(m =>
        m.taskType === 'heavy' &&
        m.status === 'active' &&
        (m.category === 'chat' || m.category === undefined)
      );
    }
    return p.models.filter(m =>
      m.taskType === taskType &&
      m.status === 'active' &&
      (m.category === 'chat' || m.category === undefined) // only chat models in routing
    );
  }

  /**
   * Get special (non-chat) models by category across all providers.
   * Used by vision, embedding, and other specialized services.
   * Returns only 'active' models.
   * If category is undefined, returns all non-chat special models.
   */
  getSpecialModels(category?: ModelCategory): Array<{ provider: string; model: CatalogModelEntry }> {
    const results: Array<{ provider: string; model: CatalogModelEntry }> = [];
    for (const [name, p] of this.providers) {
      if (p.status === 'dead') continue;
      for (const m of p.models) {
        // Skip chat models (they're handled by the routing chain)
        if (!m.category || m.category === 'chat') continue;
        if (m.status !== 'active') continue;
        // If category specified, filter; otherwise return all special
        if (category && m.category !== category) continue;
        results.push({ provider: name, model: m });
      }
    }
    return results;
  }

  // ─── Model scoring & ranking ───────────────────────────────────────────────

  /**
   * Free providers get a bonus, paid providers with free-tier models get partial bonus.
   */
  private isFreeProvider(providerName: string): boolean {
    return HEAVY_CHAIN.includes(providerName as any) || LIGHT_CHAIN.includes(providerName as any);
  }

  /**
   * Score a model on a 0-100 scale based on:
   *   - intelligence (20%): smarter is better
   *   - speed (50%): faster is better — tiered scoring, dominant factor for UX
   *   - free-tier bonus (15%): free providers get 100, paid get 0
   *   - TPD remaining (15%): more daily budget remaining is better
   *
   * Also includes a trust factor: new models with 0 calls get a small penalty
   * so proven models aren't immediately displaced.
   */
  scoreModel(model: CatalogModelEntry, providerName: string): { score: number; breakdown: Record<string, number> } {
    const intelligence = model.intelligence || 20;
    const speed = model.speed || 0;

    // Tiered speed scoring — harsh penalties for slow providers
    let speedScore: number;
    if (speed >= 200) speedScore = 100;       // Groq (500 t/s) — fast
    else if (speed >= 80) speedScore = 65;    // Sambanova (100 t/s) — OK
    else if (speed >= 30) speedScore = 35;    // NVIDIA (30 t/s) — marginal
    else if (speed >= 10) speedScore = 10;    // GLM (15 t/s) — slow
    else if (speed > 0) speedScore = 5;       // Very slow — minimal
    else speedScore = 15;                      // Unknown — give benefit of doubt, but below known-fast

    // Penalize models with slow last response regardless of theoretical speed
    if (model.lastResponseMs && model.lastResponseMs > 10000) {
      speedScore = Math.min(speedScore, 10);
    }

    const freeBonus = this.isFreeProvider(providerName) ? 100 : 0;

    // TPD remaining — lazy import to avoid circular dependency
    let tpdBonus = 100; // default for providers without TPD limits
    try {
      const { providerCircuitBreaker } = require('./providerCircuitBreaker');
      const usage = providerCircuitBreaker.getTpdUsage(providerName);
      if (usage.limit && usage.limit > 0) {
        tpdBonus = Math.round((1 - usage.percent / 100) * 100);
      }
    } catch { /* ignore */ }

    // Trust factor — models with more successful calls are trusted more.
    // New models (0 calls) start at 50% trust. After the first call, trust
    // ramps up from 50% toward 100% as totalCalls approaches 50.
    const totalCalls = model.totalCalls || 0;
    const trustFactor = totalCalls === 0 ? 0.5 : 0.5 + 0.5 * Math.min(totalCalls / 50, 1.0);

    const rawScore = (intelligence * 0.2) + (speedScore * 0.5) + (freeBonus * 0.15) + (tpdBonus * 0.15);
    const score = Math.round(rawScore * trustFactor);

    return {
      score,
      breakdown: {
        intelligence: Math.round(intelligence * 0.2),
        speed: Math.round(speedScore * 0.5),
        freeTier: Math.round(freeBonus * 0.15),
        tpdRemaining: Math.round(tpdBonus * 0.15),
        trustFactor: Math.round(trustFactor * 100),
        rawScore: Math.round(rawScore),
        finalScore: score,
      },
    };
  }

  /**
   * Get models for a provider + task type, sorted by score (highest first).
   */
  getRankedModels(provider: string, taskType: TaskType): Array<{ model: CatalogModelEntry; score: number; breakdown: Record<string, number> }> {
    const models = this.getModels(provider, taskType);
    return models
      .map(m => {
        const { score, breakdown } = this.scoreModel(m, provider);
        return { model: m, score, breakdown };
      })
      .sort((a, b) => b.score - a.score);
  }

  /**
   * Get the best model for a provider + task type (highest score).
   */
  getBestModel(provider: string, taskType: TaskType): { model: CatalogModelEntry; score: number } | null {
    const ranked = this.getRankedModels(provider, taskType);
    return ranked.length > 0 ? { model: ranked[0].model, score: ranked[0].score } : null;
  }

  /**
   * Get the full fallback chain for a task type, with providers ordered by
   * their best model's score. This replaces the static HEAVY_CHAIN/LIGHT_CHAIN
   * with a dynamic chain that adapts as models are discovered/disabled.
   *
   * Falls back to static chain if catalog isn't loaded.
   */
  getRankedFallbackChain(taskType: TaskType): string[] {
    if (!this.loaded) {
      // Fallback to static config
      if (taskType === 'complex') return [...COMPLEX_CHAIN]; // paid-only, no free providers
      const staticChain = taskType === 'light'
        ? [...LIGHT_CHAIN]
        : taskType === 'super-heavy'
          ? [...SUPER_HEAVY_CHAIN]
          : [...HEAVY_CHAIN];
      return [...staticChain, ...PAID_CHAIN];
    }

    // complex = paid-only chain, no free providers
    // Preserve COMPLEX_CHAIN order (Cerebras → DeepSeek → Gemini → Claude)
    // rather than re-sorting by score — the order is intentional (speed-first)
    if (taskType === 'complex') {
      const result: string[] = [];
      for (const name of COMPLEX_CHAIN) {
        const p = this.providers.get(name);
        if (!p || p.status === 'dead') continue;
        const best = this.getBestModel(name, taskType);
        if (best) result.push(name);
      }
      return result;
    }

    const staticFreeChain = taskType === 'light'
      ? [...LIGHT_CHAIN]
      : taskType === 'super-heavy'
        ? [...SUPER_HEAVY_CHAIN]
        : [...HEAVY_CHAIN];

    // Score each free provider by its best model
    const scoredProviders: Array<{ name: string; bestScore: number }> = [];
    for (const name of staticFreeChain) {
      const p = this.providers.get(name);
      if (!p || p.status === 'dead') continue;
      const best = this.getBestModel(name, taskType);
      if (best) {
        scoredProviders.push({ name, bestScore: best.score });
      }
    }

    // Sort by best model score descending
    scoredProviders.sort((a, b) => b.bestScore - a.bestScore);

    // Add any providers from the catalog that aren't in the static chain
    // (e.g., a paid provider that now has free-tier models)
    for (const [name, p] of this.providers) {
      if (scoredProviders.some(s => s.name === name)) continue;
      if (p.status === 'dead') continue;
      const best = this.getBestModel(name, taskType);
      if (best && best.score > 50) {
        scoredProviders.push({ name, bestScore: best.score });
      }
    }

    // Re-sort after adding new providers
    scoredProviders.sort((a, b) => b.bestScore - a.bestScore);

    // Append paid chain (sorted by best model score too)
    const paidScored: Array<{ name: string; bestScore: number }> = [];
    for (const name of PAID_CHAIN) {
      if (scoredProviders.some(s => s.name === name)) continue; // already in free chain
      const p = this.providers.get(name);
      if (!p || p.status === 'dead') continue;
      const best = this.getBestModel(name, taskType);
      if (best) {
        paidScored.push({ name, bestScore: best.score });
      }
    }
    paidScored.sort((a, b) => b.bestScore - a.bestScore);

    return [...scoredProviders.map(s => s.name), ...paidScored.map(s => s.name)];
  }

  /**
   * Get all active providers for a chain.
   */
  getActiveProviders(taskType: TaskType): string[] {
    const chain = taskType === 'light'
      ? LIGHT_CHAIN
      : taskType === 'complex'
        ? COMPLEX_CHAIN
        : taskType === 'super-heavy'
          ? SUPER_HEAVY_CHAIN
          : HEAVY_CHAIN;
    return chain.filter(name => {
      const p = this.providers.get(name);
      return p && p.status !== 'dead' && this.getModels(name, taskType).length > 0;
    });
  }

  /**
   * Get the full fallback chain (free + paid) for a task type.
   */
  getFallbackChain(taskType: TaskType): string[] {
    const freeActive = this.getActiveProviders(taskType);
    const paidActive = PAID_CHAIN.filter(name => {
      const p = this.providers.get(name);
      return p && p.status !== 'dead' && this.getModels(name, taskType).length > 0;
    });
    return [...freeActive, ...paidActive];
  }

  /**
   * Record a successful call — resets consecutive failures.
   */
  markSuccess(provider: string, modelId: string, measuredSpeed?: number, responseTimeMs?: number): void {
    const p = this.providers.get(provider);
    if (!p) return;
    const m = p.models.find(m => m.id === modelId);
    if (!m) return;
    m.consecutiveFailures = 0;
    m.lastSuccessTime = new Date().toISOString();
    m.totalCalls++;
    m.totalSuccesses++;
    m.lastVerifiedAt = new Date().toISOString();
    // Track last response time for slow-responder penalty in scoring
    if (responseTimeMs && responseTimeMs > 0) {
      m.lastResponseMs = responseTimeMs;
    }
    // Update speed with exponential moving average if we have a measurement
    if (measuredSpeed && measuredSpeed > 0) {
      const alpha = 0.3; // weight new measurement at 30%
      m.speed = Math.round((m.speed || 50) * (1 - alpha) + measuredSpeed * alpha);
    }
    // If provider was degraded, restore to active
    if (p.status === 'degraded') {
      p.status = 'active';
      logger.info(`[CatalogManager] Provider ${provider} restored to active`);
    }
  }

  /**
   * Record a failure — increments consecutive failures, may trigger discovery.
   */
  markFailure(provider: string, modelId: string, error: string, statusCode?: number): void {
    const p = this.providers.get(provider);
    if (!p) return;
    const m = p.models.find(m => m.id === modelId);
    if (!m) return;
    m.consecutiveFailures++;
    m.lastError = error;
    m.lastErrorTime = new Date().toISOString();
    m.totalCalls++;

    // 404 = model removed, 403 = moved to paid — mark as dead immediately
    if (statusCode === 404 || statusCode === 403) {
      m.status = 'dead';
      m.disabledReason = `HTTP ${statusCode}`;
      logger.warn(`[CatalogManager] Model ${provider}/${modelId} marked DEAD (HTTP ${statusCode})`);
      this.checkProviderStatus(provider);
      this.triggerDiscovery(provider, modelId, statusCode);
      return;
    }

    // 5+ consecutive failures — mark as degraded and trigger discovery
    if (m.consecutiveFailures >= DISCOVERY_TRIGGER_FAILURES) {
      m.status = 'degraded';
      m.disabledReason = `${m.consecutiveFailures} consecutive failures`;
      logger.warn(`[CatalogManager] Model ${provider}/${modelId} degraded after ${m.consecutiveFailures} failures`);
      this.triggerDiscovery(provider, modelId, statusCode || 0);
    }

    this.checkProviderStatus(provider);
  }

  /**
   * Check if all models in a provider are dead/disabled — mark provider as dead.
   */
  private checkProviderStatus(provider: string): void {
    const p = this.providers.get(provider);
    if (!p) return;
    const activeCount = p.models.filter(m => m.status === 'active').length;
    if (activeCount === 0) {
      p.status = 'dead';
      logger.error(`[CatalogManager] Provider ${provider} marked DEAD — no active models remaining`);
    } else if (activeCount < p.models.length / 2) {
      p.status = 'degraded';
      logger.warn(`[CatalogManager] Provider ${provider} DEGRADED — ${activeCount}/${p.models.length} models active`);
    }
  }

  /**
   * Trigger discovery agent if callback is registered.
   */
  private triggerDiscovery(provider: string, modelId: string, statusCode: number): void {
    if (this.discoveryCallback) {
      // Fire-and-forget — don't block the router
      setImmediate(() => {
        try {
          this.discoveryCallback!(provider, modelId, statusCode);
        } catch (err) {
          logger.error(`[CatalogManager] Discovery callback error: ${err instanceof Error ? err.message : String(err)}`);
        }
      });
    }
  }

  /**
   * Mark a provider as dead (used by discovery agent when catalog endpoint fails).
   */
  markProviderDead(provider: string, reason: string): void {
    const p = this.providers.get(provider);
    if (!p) return;
    p.status = 'dead';
    for (const m of p.models) {
      m.status = 'dead';
      m.disabledReason = reason;
    }
    logger.error(`[CatalogManager] Provider ${provider} marked DEAD: ${reason}`);
  }

  /**
   * Add a new model discovered by the discovery agent.
   * Logs an auto-promotion notice if the new model scores higher than the
   * current best model for the same provider + task type.
   */
  addModel(provider: string, entry: Omit<CatalogModelEntry, 'provider'>): void {
    const p = this.providers.get(provider);
    if (!p) return;
    // Check if model already exists
    const existing = p.models.find(m => m.id === entry.id && m.taskType === entry.taskType);
    if (existing) {
      // Update existing entry
      Object.assign(existing, entry);
      return;
    }

    // Check for auto-promotion before adding
    const newEntry = { ...entry, provider } as CatalogModelEntry;
    const newScore = this.scoreModel(newEntry, provider);
    const currentBest = this.getBestModel(provider, entry.taskType);
    if (currentBest && newScore.score > currentBest.score) {
      logger.info(
        `[Catalog] AUTO-PROMOTION: ${provider}/${entry.id} (score: ${newScore.score}) ` +
        `now ranks above ${provider}/${currentBest.model.id} (score: ${currentBest.score}) ` +
        `for ${entry.taskType} tasks`
      );
    }

    p.models.push(newEntry);
    logger.info(`[CatalogManager] Added model ${provider}/${entry.id} (${entry.taskType}, score: ${newScore.score})`);
  }

  /**
   * Reactivate a previously dead model.
   */
  reactivateModel(provider: string, modelId: string): void {
    const p = this.providers.get(provider);
    if (!p) return;
    const m = p.models.find(m => m.id === modelId);
    if (!m) return;
    m.status = 'active';
    m.disabledReason = undefined;
    m.consecutiveFailures = 0;
    m.lastVerifiedAt = new Date().toISOString();
    logger.info(`[CatalogManager] Reactivated model ${provider}/${modelId}`);
    this.checkProviderStatus(provider);
  }

  /**
   * Manually disable a model.
   */
  disableModel(provider: string, modelId: string, reason: string): void {
    const p = this.providers.get(provider);
    if (!p) return;
    const m = p.models.find(m => m.id === modelId);
    if (!m) return;
    m.status = 'disabled';
    m.disabledReason = reason;
    logger.info(`[CatalogManager] Manually disabled model ${provider}/${modelId}: ${reason}`);
    this.checkProviderStatus(provider);
  }

  /**
   * Add a new provider at runtime (for dynamically discovered providers).
   * Returns { success, error? }.
   */
  addProvider(name: string, config: {
    baseURL: string;
    envKey: string;
    catalogEndpoint?: string;
    apiType?: string;
    tier?: 'free' | 'paid';
  }): { success: boolean; error?: string } {
    if (this.providers.has(name)) {
      return { success: false, error: `Provider already exists: ${name}` };
    }
    this.providers.set(name, {
      name,
      baseURL: config.baseURL,
      envKey: config.envKey,
      catalogEndpoint: config.catalogEndpoint,
      apiType: config.apiType || 'openai-compatible',
      tier: config.tier || 'free',
      status: 'active',
      models: [],
    });
    logger.info(`[CatalogManager] Added provider: ${name} (${config.apiType || 'openai-compatible'}, ${config.tier || 'free'})`);
    this.save().catch(() => {});
    return { success: true };
  }

  /**
   * Remove a provider (marks as dead, preserves history).
   */
  removeProvider(name: string): { success: boolean; error?: string } {
    const p = this.providers.get(name);
    if (!p) {
      return { success: false, error: `Provider not found: ${name}` };
    }
    p.status = 'dead';
    // Mark all models as dead too
    for (const m of p.models) {
      m.status = 'dead';
    }
    logger.info(`[CatalogManager] Removed provider: ${name} (marked dead)`);
    this.save().catch(() => {});
    return { success: true };
  }

  /**
   * Get a provider entry.
   */
  getProvider(provider: string): CatalogProviderEntry | undefined {
    return this.providers.get(provider);
  }

  /**
   * Get all providers.
   */
  getAllProviders(): CatalogProviderEntry[] {
    return Array.from(this.providers.values());
  }

  /**
   * Get a health report for API/logging.
   */
  getHealthReport(): any {
    const providers = this.getAllProviders().map(p => ({
      name: p.name,
      status: p.status,
      baseURL: p.baseURL,
      envKey: p.envKey,
      apiType: p.apiType || 'openai-compatible',
      tier: p.tier || 'free',
      activeModels: p.models.filter(m => m.status === 'active').length,
      totalModels: p.models.length,
      models: p.models.map(m => ({
        id: m.id,
        taskType: m.taskType,
        status: m.status,
        intelligence: m.intelligence,
        consecutiveFailures: m.consecutiveFailures,
        totalCalls: m.totalCalls,
        totalSuccesses: m.totalSuccesses,
        successRate: m.totalCalls > 0 ? (m.totalSuccesses / m.totalCalls * 100).toFixed(1) + '%' : 'N/A',
        lastError: m.lastError,
        lastSuccessTime: m.lastSuccessTime,
      })),
    }));
    return {
      loaded: this.loaded,
      catalogPath: CATALOG_PATH,
      providers,
      summary: {
        totalProviders: providers.length,
        activeProviders: providers.filter(p => p.status === 'active').length,
        degradedProviders: providers.filter(p => p.status === 'degraded').length,
        deadProviders: providers.filter(p => p.status === 'dead').length,
        totalModels: providers.reduce((sum, p) => sum + p.totalModels, 0),
        activeModels: providers.reduce((sum, p) => sum + p.activeModels, 0),
      },
    };
  }

  /**
   * Check if the catalog has been loaded.
   */
  isLoaded(): boolean {
    return this.loaded;
  }
}

export const catalogManager = new CatalogManager();
