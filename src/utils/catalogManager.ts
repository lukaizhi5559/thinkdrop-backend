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
import { PROVIDER_CONFIG, HEAVY_CHAIN, LIGHT_CHAIN, PAID_CHAIN, TaskType, ProviderModel } from './providerConfig';

export type ModelStatus = 'active' | 'degraded' | 'disabled' | 'dead';
export type ProviderStatus = 'active' | 'degraded' | 'dead';

export interface CatalogModelEntry extends ProviderModel {
  provider: string;
  taskType: TaskType;
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
  totalCalls: number;
  totalSuccesses: number;
}

export interface CatalogProviderEntry {
  name: string;
  baseURL: string;
  envKey: string;
  catalogEndpoint?: string;
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
      this.loaded = true;
    } catch (err) {
      logger.error(`[CatalogManager] Failed to load catalog: ${err instanceof Error ? err.message : String(err)}`);
      await this.seedFromConfig();
      this.loaded = true;
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
        status: 'active',
        models: deduped,
      });
    }
    await this.save();
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
    return p.models.filter(m => m.taskType === taskType && m.status === 'active');
  }

  /**
   * Get all active providers for a chain.
   */
  getActiveProviders(taskType: TaskType): string[] {
    const chain = taskType === 'light' ? LIGHT_CHAIN : HEAVY_CHAIN;
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
  markSuccess(provider: string, modelId: string): void {
    const p = this.providers.get(provider);
    if (!p) return;
    const m = p.models.find(m => m.id === modelId);
    if (!m) return;
    m.consecutiveFailures = 0;
    m.lastSuccessTime = new Date().toISOString();
    m.totalCalls++;
    m.totalSuccesses++;
    m.lastVerifiedAt = new Date().toISOString();
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
    p.models.push({ ...entry, provider });
    logger.info(`[CatalogManager] Added model ${provider}/${entry.id} (${entry.taskType})`);
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
