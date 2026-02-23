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
const WARM_TTL_MS = 4 * 60 * 1000; // 4 minutes — must be > interval to avoid false cold on tight timing
const WARMUP_ENABLED = process.env.OMNIPARSER_WARMUP_ENABLED === 'true';

// Replicate's own playground screenshot — guaranteed to work with their API
const WARMUP_TEST_IMAGE = 'https://replicate.delivery/pbxt/MWb5PhmtW9qcXtvG1G9DQMo2TmBtsVK3DS1dETfEl78YNLZL/replicate-website.png';

let warmupInterval: NodeJS.Timeout | null = null;
let lastWarmupTime: number = 0;
let warmupCount: number = 0;
let inFlightWarmup: Promise<void> | null = null; // deduplicates concurrent warmup calls

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
   * Deduplicates concurrent calls — if a warmup is already in flight, awaits it instead of firing another.
   */
  private async warmup(): Promise<void> {
    if (!this.replicateClient) return;

    if (inFlightWarmup) {
      logger.info('🔥 [WARMUP] Warmup already in flight, awaiting existing request');
      return inFlightWarmup;
    }

    const startTime = Date.now();
    warmupCount++;
    const thisWarmupNumber = warmupCount;

    inFlightWarmup = (async () => {
      try {
        logger.info('🔥 [WARMUP] Sending warmup request', {
          warmupNumber: thisWarmupNumber,
          timeSinceLastWarmupSeconds: lastWarmupTime ? (startTime - lastWarmupTime) / 1000 : 0,
        });

        await this.replicateClient!.run(
          'microsoft/omniparser-v2:49cf3d41b8d3aca1360514e83be4c97131ce8f0d99abfc365526d8384caa88df',
          { input: { image: WARMUP_TEST_IMAGE, box_threshold: 0.05, iou_threshold: 0.1 } }
        );

        const latency = Date.now() - startTime;
        lastWarmupTime = Date.now();

        logger.info('✅ [WARMUP] Warmup successful', {
          warmupNumber: thisWarmupNumber,
          latencyMs: latency,
          latencySeconds: (latency / 1000).toFixed(2),
          isColdBoot: latency > 60000,
        });

        if (latency > 60000 && thisWarmupNumber > 1) {
          logger.warn('⚠️ [WARMUP] Cold boot detected on scheduled warmup — interval may be too long', {
            latencySeconds: (latency / 1000).toFixed(2),
            warmupNumber: thisWarmupNumber,
            intervalMinutes: WARMUP_INTERVAL_MS / 60000,
          });
        }
      } catch (error: any) {
        logger.error('❌ [WARMUP] Warmup request failed', {
          warmupNumber: thisWarmupNumber,
          error: error.message,
        });
      } finally {
        inFlightWarmup = null;
      }
    })();

    return inFlightWarmup;
  }

  /**
   * Returns true if the model was warmed up within the last 3 minutes.
   */
  isWarm(): boolean {
    if (!WARMUP_ENABLED || !this.replicateClient) return false;
    if (inFlightWarmup) return true; // warmup in progress — treat as warm to avoid stacking more calls
    if (!lastWarmupTime) return false;
    const timeSinceWarmup = Date.now() - lastWarmupTime;
    return timeSinceWarmup < WARM_TTL_MS;
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
