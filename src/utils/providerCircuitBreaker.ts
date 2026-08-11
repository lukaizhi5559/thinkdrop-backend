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
const SERVER_OVERLOAD_COOLDOWN_MS = 5_000; // 5 s for transient server overload (GLM "temporarily overloaded")
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
 * Detects transient server-side overload errors (e.g. GLM's "429 The service
 * may be temporarily overloaded"). These are NOT client-side rate limits —
 * the provider's servers are just busy. Use a short 5s cooldown so we retry
 * quickly rather than blocking the provider for 15s+.
 */
function isServerOverloadError(message: string): boolean {
  return /overloaded|temporarily.*unavailable|service.*temporarily|internal server error|503/i.test(message);
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
 * Parse the retry time from error messages.
 * Unified regex — handles all known provider formats:
 *   "try again in 2.745s"     (Groq — decimal seconds)
 *   "try again in 7h14m44.16s" (OpenAI — h+m+s with decimals)
 *   "try again in 1m16s"      (compact m+s)
 *   "retry after 30 seconds"  (word form + different phrasing)
 *   "try again in 500ms"      (milliseconds)
 *   "available in 2 minutes"  (word form + different phrasing)
 * Returns the cooldown in ms (minimum 1s), or null if no retry time is found.
 */
function parseRetryTime(message: string): number | null {
  const m = message.match(/(?:try again in|retry (?:after|in)|available in)\s+(?:(\d+(?:\.\d+)?)\s*(?:h|hours?))?(?:(\d+(?:\.\d+)?)\s*(?:m(?!s)|minutes?))?(?:(\d+(?:\.\d+)?)\s*(?:s(?!ms)|seconds?))?(?:(\d+(?:\.\d+)?)\s*ms)?/i);
  if (!m || (!m[1] && !m[2] && !m[3] && !m[4])) return null;
  const h = parseFloat(m[1] || '0');
  const min = parseFloat(m[2] || '0');
  const s = parseFloat(m[3] || '0');
  const ms = parseFloat(m[4] || '0');
  const totalMs = Math.ceil((h * 3600 + min * 60 + s) * 1000 + ms);
  return Math.max(totalMs, 1000); // minimum 1s cooldown
}

/**
 * Parse the standard HTTP Retry-After header.
 * Can be either seconds (e.g. "10") or an HTTP-date (e.g. "Wed, 21 Oct 2025 07:28:00 GMT").
 * Returns the cooldown in ms (minimum 1s), or null if the header is missing/invalid.
 */
function parseRetryAfterHeader(headerValue: string | undefined): number | null {
  if (!headerValue) return null;
  // Numeric form: seconds until retry
  const seconds = parseInt(headerValue, 10);
  if (!isNaN(seconds) && seconds > 0) return Math.max(seconds * 1000, 1000);
  // HTTP-date form: absolute timestamp
  const date = Date.parse(headerValue);
  if (!isNaN(date)) {
    const delta = date - Date.now();
    return delta > 0 ? delta : 1000;
  }
  return null;
}

/**
 * Detects tokens-per-minute (TPM) limits (e.g. Groq's "tokens per minute (TPM)").
 * These are per-model limits that reset quickly (~1 min). Distinct from TPD.
 */
function isTpmLimitError(message: string): boolean {
  return /tokens?\s*per\s*minute|TPM/i.test(message);
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
  recordFailure(provider: string, errorMessage: string, headers?: Record<string, string>): boolean {
    const billing = isBillingOrAuthError(errorMessage);
    if (!isRateLimitError(errorMessage) && !billing) return false;

    // 3-tier retry time priority:
    //   1. Retry-After HTTP header (most reliable — standard, provider-set)
    //   2. Parsed retry time from error message (fallback for providers without header)
    //   3. Default cooldown per error type (last resort)
    const retryAfterMs = parseRetryAfterHeader(headers?.['retry-after'] || headers?.['Retry-After']);
    const parsedRetryMs = parseRetryTime(errorMessage);

    // Determine base cooldown: billing > TPD > hard quota > TPM > server overload > per-minute > rate limit
    let baseCooldown: number;
    let cooldownReason: string;

    if (billing) {
      baseCooldown = BILLING_ERROR_COOLDOWN_MS;
      cooldownReason = 'billing/auth error';
    } else if (isTpdLimitError(errorMessage)) {
      // TPD limits reset in hours — use parsed retry time, or 1 hour default
      baseCooldown = retryAfterMs ?? parsedRetryMs ?? TPD_COOLDOWN_MS;
      cooldownReason = `TPD limit${retryAfterMs ? ` (Retry-After: ${Math.round(retryAfterMs / 1000)}s)` : parsedRetryMs ? ` (parsed: ${Math.round(parsedRetryMs / 1000)}s)` : ''}`;
    } else if (isHardQuotaError(errorMessage)) {
      baseCooldown = retryAfterMs ?? QUOTA_EXCEEDED_COOLDOWN_MS;
      cooldownReason = `hard quota exceeded${retryAfterMs ? ` (Retry-After: ${Math.round(retryAfterMs / 1000)}s)` : ''}`;
    } else if (isTpmLimitError(errorMessage)) {
      // Tokens-per-minute limits (Groq) — reset quickly, use Retry-After or parsed time
      baseCooldown = retryAfterMs ?? parsedRetryMs ?? 5_000;
      cooldownReason = `TPM limit${retryAfterMs ? ` (Retry-After: ${Math.round(retryAfterMs / 1000)}s)` : parsedRetryMs ? ` (parsed: ${Math.round(parsedRetryMs / 1000)}s)` : ''}`;
    } else if (isServerOverloadError(errorMessage)) {
      // Transient server-side overload (e.g. GLM "temporarily overloaded") —
      // NOT a client rate limit. Short cooldown so we retry quickly.
      baseCooldown = retryAfterMs ?? SERVER_OVERLOAD_COOLDOWN_MS;
      cooldownReason = `server overload (transient)${retryAfterMs ? ` (Retry-After: ${Math.round(retryAfterMs / 1000)}s)` : ''}`;
    } else if (isPerMinuteRateLimit(errorMessage)) {
      // Simple per-minute rate limits (SambaNova) — short cooldown so provider
      // can handle multiple calls per task (intent, route, plan, etc.)
      baseCooldown = retryAfterMs ?? PER_MINUTE_RATE_LIMIT_COOLDOWN_MS;
      cooldownReason = `per-minute rate limit${retryAfterMs ? ` (Retry-After: ${Math.round(retryAfterMs / 1000)}s)` : ''}`;
    } else {
      // For regular rate limits, try Retry-After header, then parse from message
      baseCooldown = retryAfterMs ?? parsedRetryMs ?? RATE_LIMIT_COOLDOWN_MS;
      cooldownReason = `rate limited${retryAfterMs ? ` (Retry-After: ${Math.round(retryAfterMs / 1000)}s)` : parsedRetryMs ? ` (parsed: ${Math.round(parsedRetryMs / 1000)}s)` : ''}`;
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
   *
   * When scores are provided, uses weighted rotation: high-score providers
   * (score ≥ 35) get 3 rotation slots, medium (≥ 25) get 2, slow (< 25) get 1.
   * This means fast providers like Groq are tried first ~3x more often than
   * slow providers like GLM, while still distributing load across all providers.
   */
  getRotatedChain(chain: readonly string[], scores?: ReadonlyMap<string, number>): readonly string[] {
    if (chain.length <= 1) return chain;

    // Build weighted list: high-score providers appear multiple times
    let weighted: string[];
    if (scores && scores.size > 0) {
      weighted = [];
      for (const p of chain) {
        const score = scores.get(p) ?? 0;
        const slots = score >= 35 ? 3 : score >= 25 ? 2 : 1;
        for (let i = 0; i < slots; i++) weighted.push(p);
      }
    } else {
      weighted = [...chain];
    }

    const key = chain.join(',');
    const idx = (this.rrCounters.get(key) ?? 0) % weighted.length;
    this.rrCounters.set(key, idx + 1);

    // Return the full chain (no duplicates) but starting from the rotated provider
    const startProvider = weighted[idx];
    const startPos = chain.indexOf(startProvider);
    if (startPos < 0) return chain;
    return [...chain.slice(startPos), ...chain.slice(0, startPos)];
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
