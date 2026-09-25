/**
 * Provider verification API — operator-facing smoke test.
 *
 *   GET /api/providers/verify  — Probe one configured model per provider
 *                                (protocol-aware: Anthropic /v1/messages,
 *                                OpenAI-compatible /chat/completions) and
 *                                return a live health table.
 *
 * Use this to answer "is my paid provider actually working?" without
 * digging through logs or reasoning about catalog state.
 */

import { Router, Request, Response } from 'express';
import { discoveryAgent } from '../utils/discoveryAgent';
import { catalogManager } from '../utils/catalogManager';
import { providerCircuitBreaker } from '../utils/providerCircuitBreaker';
import { logger } from '../utils/logger';

const router = Router();

router.get('/verify', async (_req: Request, res: Response) => {
  try {
    const results = await discoveryAgent.verifyProviders();

    // Annotate with circuit-breaker + catalog state so the table explains
    // WHY a provider is unroutable even when its key works.
    const table = results.map(r => {
      const cat = catalogManager.getProvider(r.provider);
      return {
        ...r,
        catalogStatus: cat?.status ?? 'unknown',
        circuitOpen: providerCircuitBreaker.isOpen(r.provider),
        activeModels: cat?.models.filter(m => m.status === 'active').length ?? 0,
      };
    });

    const healthy = table.filter(r => r.alive && !r.circuitOpen).length;
    res.json({
      ok: true,
      data: {
        healthy,
        total: table.length,
        providers: table,
      },
    });
  } catch (err) {
    const errMsg = err instanceof Error ? err.message : String(err);
    logger.error(`[VerifyAPI] /api/providers/verify failed: ${errMsg}`);
    res.status(500).json({ ok: false, error: errMsg });
  }
});

export default router;
