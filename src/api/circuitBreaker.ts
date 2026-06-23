import { Router, Request, Response } from 'express';
import { logger } from '../utils/logger';
import { providerCircuitBreaker } from '../utils/providerCircuitBreaker';

const router = Router();

/**
 * GET /api/circuit-breaker/status
 * Returns the list of currently open (circuit-broken) providers.
 */
router.get('/status', (_req: Request, res: Response): void => {
  const openProviders = providerCircuitBreaker.getOpenProviders();
  res.json({ openProviders, count: openProviders.length });
});

/**
 * POST /api/circuit-breaker/reset
 * Resets one or all open circuit breakers.
 * Body: { provider?: string }  — omit provider to reset all.
 */
router.post('/reset', (req: Request, res: Response): void => {
  const { provider } = req.body as { provider?: string };

  if (provider) {
    providerCircuitBreaker.reset(provider);
    logger.info(`[CircuitBreaker-API] Manually reset provider "${provider}"`);
    res.json({ success: true, reset: [provider] });
    return;
  }

  const open = providerCircuitBreaker.getOpenProviders();
  if (open.length === 0) {
    res.json({ success: true, reset: [], message: 'No open circuit breakers' });
    return;
  }

  for (const p of open) {
    providerCircuitBreaker.reset(p);
  }
  logger.info(`[CircuitBreaker-API] Manually reset all providers: ${open.join(', ')}`);
  res.json({ success: true, reset: open });
});

export default router;
