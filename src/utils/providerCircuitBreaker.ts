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

function isRateLimitError(message: string): boolean {
  return /429|rate.?limit|quota|exceeded.*quota|too.many.requests/i.test(message);
}

function isHardQuotaError(message: string): boolean {
  return /exceeded your current quota/i.test(message);
}

class ProviderCircuitBreaker {
  private states: Map<string, BreakerState> = new Map();

  /**
   * Call when a provider throws an error.
   * Opens the breaker if the error is a rate-limit / quota error.
   * Returns true if the breaker was opened.
   */
  recordFailure(provider: string, errorMessage: string): boolean {
    if (!isRateLimitError(errorMessage)) return false;

    const cooldown = isHardQuotaError(errorMessage)
      ? QUOTA_EXCEEDED_COOLDOWN_MS
      : RATE_LIMIT_COOLDOWN_MS;

    this.states.set(provider, { openedAt: Date.now(), reason: errorMessage });

    logger.warn(`[CircuitBreaker] Opened for provider "${provider}" — cooldown ${cooldown / 1000}s`, {
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
