/**
 * OmniParser Warm-up Service
 * Keeps Replicate model warm by sending periodic requests.
 * Prevents cold boots by ensuring model is called at least every 3 minutes.
 * Only active when OMNIPARSER_WARMUP_ENABLED=true and REPLICATE_API_TOKEN is set.
 */

import Replicate from 'replicate';
import { logger } from '../utils/logger';

const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;
const WARMUP_INTERVAL_MS = 3 * 60 * 1000; // 3 minutes
const WARMUP_ENABLED = process.env.OMNIPARSER_WARMUP_ENABLED === 'true';

// Replicate's own playground screenshot — guaranteed to work with their API
const WARMUP_TEST_IMAGE = 'https://replicate.delivery/pbxt/MWb5PhmtW9qcXtvG1G9DQMo2TmBtsVK3DS1dETfEl78YNLZL/replicate-website.png';

let warmupInterval: NodeJS.Timeout | null = null;
let lastWarmupTime: number = 0;
let warmupCount: number = 0;

export class OmniParserWarmupService {
  private replicateClient: Replicate | null = null;

  constructor() {
    if (REPLICATE_API_TOKEN && WARMUP_ENABLED) {
      this.replicateClient = new Replicate({ auth: REPLICATE_API_TOKEN });
      logger.info('🔥 [WARMUP] OmniParser warmup service initialized', {
        intervalMinutes: WARMUP_INTERVAL_MS / 60000,
        enabled: true,
      });
    } else {
      logger.info('🔥 [WARMUP] OmniParser warmup service disabled', {
        enabled: false,
        reason: !REPLICATE_API_TOKEN ? 'no_replicate_token' : 'OMNIPARSER_WARMUP_ENABLED not set to true',
      });
    }
  }

  /**
   * Start periodic warmup. Fires immediately then on interval.
   */
  start(): void {
    if (!this.replicateClient || !WARMUP_ENABLED) {
      logger.warn('🔥 [WARMUP] Cannot start — service not initialized or disabled');
      return;
    }

    this.warmup().catch((error) => {
      logger.error('🔥 [WARMUP] Initial warmup failed', { error: error.message });
    });

    warmupInterval = setInterval(() => {
      this.warmup().catch((error) => {
        logger.error('🔥 [WARMUP] Scheduled warmup failed', { error: error.message });
      });
    }, WARMUP_INTERVAL_MS);

    logger.info('🔥 [WARMUP] Warmup service started', {
      intervalMs: WARMUP_INTERVAL_MS,
      intervalMinutes: WARMUP_INTERVAL_MS / 60000,
    });
  }

  /**
   * Stop the warmup service and clear the interval.
   */
  stop(): void {
    if (warmupInterval) {
      clearInterval(warmupInterval);
      warmupInterval = null;
    }
    logger.info('🔥 [WARMUP] Warmup service stopped', { totalWarmups: warmupCount });
  }

  /**
   * Perform a single warmup request against Replicate.
   */
  private async warmup(): Promise<void> {
    if (!this.replicateClient) return;

    const startTime = Date.now();
    warmupCount++;

    try {
      logger.info('🔥 [WARMUP] Sending warmup request', {
        warmupNumber: warmupCount,
        timeSinceLastWarmupSeconds: lastWarmupTime ? (startTime - lastWarmupTime) / 1000 : 0,
      });

      await this.replicateClient.run(
        'microsoft/omniparser-v2:49cf3d41b8d3aca1360514e83be4c97131ce8f0d99abfc365526d8384caa88df',
        { input: { image: WARMUP_TEST_IMAGE, box_threshold: 0.05, iou_threshold: 0.1 } }
      );

      const latency = Date.now() - startTime;
      lastWarmupTime = Date.now();

      logger.info('✅ [WARMUP] Warmup successful', {
        warmupNumber: warmupCount,
        latencyMs: latency,
        latencySeconds: (latency / 1000).toFixed(2),
        isColdBoot: latency > 60000,
      });

      if (latency > 60000) {
        logger.warn('⚠️ [WARMUP] Cold boot detected — consider reducing WARMUP_INTERVAL_MS', {
          latencySeconds: (latency / 1000).toFixed(2),
        });
      }
    } catch (error: any) {
      logger.error('❌ [WARMUP] Warmup request failed', {
        warmupNumber: warmupCount,
        error: error.message,
      });
    }
  }

  /**
   * Returns true if the model was warmed up within the last 3 minutes.
   */
  isWarm(): boolean {
    if (!WARMUP_ENABLED || !this.replicateClient) return false;
    if (!lastWarmupTime) return false;
    const timeSinceWarmup = (Date.now() - lastWarmupTime) / 1000;
    return timeSinceWarmup < 180;
  }

  /**
   * Trigger an immediate warmup if the model is cold.
   */
  async ensureWarm(): Promise<{ wasWarm: boolean; latencyMs?: number }> {
    if (this.isWarm()) return { wasWarm: true };
    const startTime = Date.now();
    await this.warmup();
    return { wasWarm: false, latencyMs: Date.now() - startTime };
  }

  getStats() {
    return {
      enabled: WARMUP_ENABLED,
      warmupCount,
      lastWarmupTime,
      timeSinceLastWarmupSeconds: lastWarmupTime ? (Date.now() - lastWarmupTime) / 1000 : null,
      intervalMinutes: WARMUP_INTERVAL_MS / 60000,
    };
  }
}

export const omniParserWarmup = new OmniParserWarmupService();
