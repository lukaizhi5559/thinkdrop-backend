/**
 * ProviderCircuitBreaker — tracks per-provider rate-limit (429) failures
 * and suppresses retries for a configurable TTL window.
 *
 * When a provider returns a 429 it is "opened" for COOLDOWN_MS.
 * During that window isOpen() returns true and the router skips it.
 * After the TTL expires the breaker resets ("half-open") and the provider
 * is tried again automatically.
 */

import { logger } from './logger';

interface BreakerState {
  openedAt: number;
  reason: string;
  consecutiveFailures: number;
}

const MAX_COOLDOWN_MS = 3_600_000; // 1 hour cap on exponential backoff

const RATE_LIMIT_COOLDOWN_MS = 60_000; // 60 s default cooldown for 429s
const PER_MINUTE_RATE_LIMIT_COOLDOWN_MS = 15_000; // 15 s for simple per-minute rate limits (SambaNova)
const QUOTA_EXCEEDED_COOLDOWN_MS = 300_000; // 5 min for hard quota failures
const TPD_COOLDOWN_MS = 3_600_000; // 1 hour for tokens-per-day limits (reset in hours, not 60s)
const BILLING_ERROR_COOLDOWN_MS = 3_600_000; // 1 hour for billing/auth failures (needs manual fix)

function isRateLimitError(message: string): boolean {
  return /429|rate.?limit|quota|exceeded.*quota|too.many.requests/i.test(message);
}

function isHardQuotaError(message: string): boolean {
  return /exceeded your current quota/i.test(message);
}

function isTpdLimitError(message: string): boolean {
  // Groq: "tokens per day (TPD): Limit 100000, Used 89552"
  // Other providers may use similar phrasing
  return /tokens?\s*per\s*day|TPD/i.test(message);
}

function isBillingOrAuthError(message: string): boolean {
  // Added "no credits remaining" (OpenAI) and "insufficient balance" (z.ai GLM)
  return /credit balance is too low|402|payment required|insufficient.*credit|no credits remaining|insufficient balance|upgrade or purchase/i.test(message);
}

/**
 * Detects simple per-minute rate limits (e.g. SambaNova's "429 Rate limit exceeded")
 * that don't mention TPD, quota, or billing. These reset quickly (~1 min) so a
 * shorter cooldown lets the provider handle multiple calls per task.
 */
function isPerMinuteRateLimit(message: string): boolean {
  // Must be a rate limit but NOT a TPD, hard quota, or billing error
  if (!isRateLimitError(message)) return false;
  if (isTpdLimitError(message)) return false;
  if (isHardQuotaError(message)) return false;
  if (isBillingOrAuthError(message)) return false;
  // SambaNova: "429 Rate limit exceeded\n" — short generic message, no retry time
  // If there's a "try again in Xm" with large minutes, it's not per-minute
  const minuteMatch = message.match(/try again in (\d+)m/i);
  if (minuteMatch && parseInt(minuteMatch[1], 10) >= 5) return false;
  return true;
}

/**
 * Parse the retry time from error messages like "Please try again in 7h14m44.16s"
 * Returns the cooldown in ms, or null if no retry time is found.
 */
function parseRetryTime(message: string): number | null {
  const match = message.match(/try again in (\d+)h(\d+)m(\d+)/i);
  if (match) {
    const hours = parseInt(match[1], 10);
    const minutes = parseInt(match[2], 10);
    const seconds = parseInt(match[3], 10);
    return (hours * 3600 + minutes * 60 + seconds) * 1000;
  }
  // Also handle "try again in 1m16s" or "try again in 76s"
  const shortMatch = message.match(/try again in (\d+)m(\d+)s/i);
  if (shortMatch) {
    return (parseInt(shortMatch[1], 10) * 60 + parseInt(shortMatch[2], 10)) * 1000;
  }
  const secMatch = message.match(/try again in (\d+)s/i);
  if (secMatch) {
    return parseInt(secMatch[1], 10) * 1000;
  }
  return null;
}

class ProviderCircuitBreaker {
  private states: Map<string, BreakerState> = new Map();
  private failureCounts: Map<string, number> = new Map();

  // Round-robin: rotate starting index per task type to distribute load
  private rrCounters: Map<string, number> = new Map();

  // TPD tracking: estimated tokens used today per provider (reset at midnight UTC)
  private tpdUsed: Map<string, number> = new Map();
  private tpdDay: string = new Date().toISOString().slice(0, 10); // YYYY-MM-DD

  // Known TPD limits per provider (from provider docs)
  private static TPD_LIMITS: Record<string, number> = {
    groq: 500_000,        // 500K TPD for llama-3.1-8b-instant
    sambanova: 200_000,   // 200K TPD shared
    cloudflare: 100_000,  // ~10K neurons/day ≈ 100K tokens est.
  };

  /**
   * Call when a provider succeeds. Resets the consecutive-failure counter so
   * the exponential backoff starts fresh on the next failure.
   * Also records estimated token usage for TPD tracking.
   */
  recordSuccess(provider: string, estimatedTokens: number = 0): void {
    this.failureCounts.delete(provider);
    this.addTpdUsage(provider, estimatedTokens);
  }

  /**
   * Call when a provider throws an error.
   * Opens the breaker if the error is a rate-limit / quota error.
   * Returns true if the breaker was opened.
   *
   * Uses exponential backoff: each consecutive failure doubles the cooldown
   * (capped at 1 hour). This prevents daily-dead providers (e.g. SambaNova
   * quota exhausted) from being retried every 15s and wasting 2-4s per call.
   */
  recordFailure(provider: string, errorMessage: string): boolean {
    const billing = isBillingOrAuthError(errorMessage);
    if (!isRateLimitError(errorMessage) && !billing) return false;

    // Determine base cooldown: billing > TPD > hard quota > parsed retry > rate limit
    let baseCooldown: number;
    let cooldownReason: string;

    if (billing) {
      baseCooldown = BILLING_ERROR_COOLDOWN_MS;
      cooldownReason = 'billing/auth error';
    } else if (isTpdLimitError(errorMessage)) {
      // TPD limits reset in hours — use 1 hour default, or parse exact retry time
      const parsed = parseRetryTime(errorMessage);
      baseCooldown = parsed ?? TPD_COOLDOWN_MS;
      cooldownReason = `TPD limit${parsed ? ` (parsed retry in ${Math.round(parsed / 1000)}s)` : ''}`;
    } else if (isHardQuotaError(errorMessage)) {
      baseCooldown = QUOTA_EXCEEDED_COOLDOWN_MS;
      cooldownReason = 'hard quota exceeded';
    } else if (isPerMinuteRateLimit(errorMessage)) {
      // Simple per-minute rate limits (SambaNova) — short cooldown so provider
      // can handle multiple calls per task (intent, route, plan, etc.)
      baseCooldown = PER_MINUTE_RATE_LIMIT_COOLDOWN_MS;
      cooldownReason = 'per-minute rate limit';
    } else {
      // For regular rate limits, try to parse retry time from the error
      const parsed = parseRetryTime(errorMessage);
      baseCooldown = parsed ?? RATE_LIMIT_COOLDOWN_MS;
      cooldownReason = `rate limited${parsed ? ` (parsed retry in ${Math.round(parsed / 1000)}s)` : ''}`;
    }

    // Exponential backoff: double the cooldown for each consecutive failure
    const consecutiveFailures = (this.failureCounts.get(provider) ?? 0) + 1;
    this.failureCounts.set(provider, consecutiveFailures);
    const cooldown = Math.min(baseCooldown * 2 ** (consecutiveFailures - 1), MAX_COOLDOWN_MS);

    this.states.set(provider, { openedAt: Date.now(), reason: errorMessage, consecutiveFailures });

    logger.warn(`[CircuitBreaker] Opened for provider "${provider}" — cooldown ${cooldown / 1000}s (${cooldownReason}, failure #${consecutiveFailures})`, {
      reason: errorMessage.substring(0, 120),
    });

    // Auto-reset after cooldown
    setTimeout(() => {
      const state = this.states.get(provider);
      if (state) {
        this.states.delete(provider);
        logger.info(`[CircuitBreaker] Reset for provider "${provider}" — available again`);
      }
    }, cooldown);

    return true;
  }

  /**
   * Returns true if the provider is currently circuit-broken (skip it).
   */
  isOpen(provider: string): boolean {
    return this.states.has(provider);
  }

  /**
   * Manually reset a provider (e.g. on app restart or explicit unlock).
   */
  reset(provider: string): void {
    this.states.delete(provider);
  }

  getOpenProviders(): string[] {
    return Array.from(this.states.keys());
  }

  // ─── Round-robin ─────────────────────────────────────────────────────────

  /**
   * Get a rotated copy of the fallback chain so load is distributed.
   * Each call increments the counter, so consecutive requests start at
   * different providers. This prevents always hammering the first provider.
   */
  getRotatedChain(chain: readonly string[]): readonly string[] {
    if (chain.length <= 1) return chain;
    const key = chain.join(',');
    const idx = (this.rrCounters.get(key) ?? 0) % chain.length;
    this.rrCounters.set(key, idx + 1);
    return [...chain.slice(idx), ...chain.slice(0, idx)];
  }

  // ─── TPD tracking ─────────────────────────────────────────────────────────

  /**
   * Add estimated token usage for a provider. Resets daily at midnight UTC.
   */
  private addTpdUsage(provider: string, tokens: number): void {
    if (tokens <= 0) return;
    this.checkDayRollover();
    const current = this.tpdUsed.get(provider) ?? 0;
    this.tpdUsed.set(provider, current + tokens);
  }

  /**
   * Check if the TPD day has rolled over (midnight UTC) and reset counters.
   */
  private checkDayRollover(): void {
    const today = new Date().toISOString().slice(0, 10);
    if (today !== this.tpdDay) {
      this.tpdUsed.clear();
      this.tpdDay = today;
      logger.info(`[CircuitBreaker] TPD counters reset for new day (${today})`);
    }
  }

  /**
   * Check if a provider is near its TPD limit and should be skipped.
   * Returns true if the provider has used >= 90% of its daily token budget.
   */
  isTpdExhausted(provider: string): boolean {
    this.checkDayRollover();
    const limit = ProviderCircuitBreaker.TPD_LIMITS[provider];
    if (!limit) return false; // no known limit
    const used = this.tpdUsed.get(provider) ?? 0;
    return used >= limit * 0.9;
  }

  /**
   * Get TPD usage info for logging/API.
   */
  getTpdUsage(provider: string): { used: number; limit: number | undefined; percent: number } {
    this.checkDayRollover();
    const used = this.tpdUsed.get(provider) ?? 0;
    const limit = ProviderCircuitBreaker.TPD_LIMITS[provider];
    return {
      used,
      limit,
      percent: limit ? Math.round((used / limit) * 100) : 0,
    };
  }

  /**
   * Get TPD usage for all tracked providers.
   */
  getAllTpdUsage(): Record<string, { used: number; limit: number | undefined; percent: number }> {
    const result: Record<string, { used: number; limit: number | undefined; percent: number }> = {};
    for (const provider of Object.keys(ProviderCircuitBreaker.TPD_LIMITS)) {
      result[provider] = this.getTpdUsage(provider);
    }
    return result;
  }
}

export const providerCircuitBreaker = new ProviderCircuitBreaker();
