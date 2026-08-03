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
}

const RATE_LIMIT_COOLDOWN_MS = 60_000; // 60 s default cooldown for 429s
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

  /**
   * Call when a provider throws an error.
   * Opens the breaker if the error is a rate-limit / quota error.
   * Returns true if the breaker was opened.
   */
  recordFailure(provider: string, errorMessage: string): boolean {
    const billing = isBillingOrAuthError(errorMessage);
    if (!isRateLimitError(errorMessage) && !billing) return false;

    // Determine cooldown: billing > TPD > hard quota > parsed retry > rate limit
    let cooldown: number;
    let cooldownReason: string;

    if (billing) {
      cooldown = BILLING_ERROR_COOLDOWN_MS;
      cooldownReason = 'billing/auth error';
    } else if (isTpdLimitError(errorMessage)) {
      // TPD limits reset in hours — use 1 hour default, or parse exact retry time
      const parsed = parseRetryTime(errorMessage);
      cooldown = parsed ?? TPD_COOLDOWN_MS;
      cooldownReason = `TPD limit${parsed ? ` (parsed retry in ${Math.round(parsed / 1000)}s)` : ''}`;
    } else if (isHardQuotaError(errorMessage)) {
      cooldown = QUOTA_EXCEEDED_COOLDOWN_MS;
      cooldownReason = 'hard quota exceeded';
    } else {
      // For regular rate limits, try to parse retry time from the error
      const parsed = parseRetryTime(errorMessage);
      cooldown = parsed ?? RATE_LIMIT_COOLDOWN_MS;
      cooldownReason = `rate limited${parsed ? ` (parsed retry in ${Math.round(parsed / 1000)}s)` : ''}`;
    }

    this.states.set(provider, { openedAt: Date.now(), reason: errorMessage });

    logger.warn(`[CircuitBreaker] Opened for provider "${provider}" — cooldown ${cooldown / 1000}s (${cooldownReason})`, {
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
}

export const providerCircuitBreaker = new ProviderCircuitBreaker();
