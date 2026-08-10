/**
 * Catalog API — endpoints for viewing and managing the provider catalog.
 *
 *   GET  /api/catalog                    — Full catalog with health report
 *   GET  /api/catalog/health             — Catalog health summary
 *   GET  /api/catalog/providers          — List all providers
 *   GET  /api/catalog/providers/:name    — Single provider detail
 *   POST /api/catalog/discover/:name     — Trigger discovery for a provider
 *   POST /api/catalog/discover-all       — Trigger discovery for all providers
 *   POST /api/catalog/disable            — Disable a model { provider, modelId, reason }
 *   POST /api/catalog/reactivate         — Reactivate a model { provider, modelId }
 *   GET  /api/catalog/tpd                — TPD usage for all tracked providers
 */

import { Router } from 'express';
import { catalogManager } from '../utils/catalogManager';
import { discoveryAgent } from '../utils/discoveryAgent';
import { providerCircuitBreaker } from '../utils/providerCircuitBreaker';
import { logger } from '../utils/logger';

const router = Router();

// GET /api/catalog — full catalog with health report
router.get('/', (_req, res) => {
  res.json(catalogManager.getHealthReport());
});

// GET /api/catalog/health — catalog health summary
router.get('/health', (_req, res) => {
  const report = catalogManager.getHealthReport();
  res.json({
    loaded: report.loaded,
    summary: report.summary,
  });
});

// GET /api/catalog/providers — list all providers
router.get('/providers', (_req, res) => {
  const providers = catalogManager.getAllProviders().map(p => ({
    name: p.name,
    status: p.status,
    baseURL: p.baseURL,
    envKey: p.envKey,
    activeModels: p.models.filter(m => m.status === 'active').length,
    totalModels: p.models.length,
  }));
  res.json({ providers });
});

// GET /api/catalog/providers/:name — single provider detail
router.get('/providers/:name', (req, res) => {
  const provider = catalogManager.getProvider(req.params.name);
  if (!provider) {
    res.status(404).json({ error: `Provider not found: ${req.params.name}` });
    return;
  }
  res.json(provider);
});

// POST /api/catalog/discover/:name — trigger discovery for a provider
router.post('/discover/:name', async (req, res) => {
  const providerName = req.params.name;
  const provider = catalogManager.getProvider(providerName);
  if (!provider) {
    res.status(404).json({ error: `Provider not found: ${providerName}` });
    return;
  }
  if (!provider.catalogEndpoint) {
    res.status(400).json({ error: `Provider ${providerName} has no catalog endpoint` });
    return;
  }
  res.json({ message: `Discovery started for ${providerName}`, provider: providerName });
  // Run async — don't block the response
  discoveryAgent.discoverProvider(providerName).catch(err => {
    logger.error(`[CatalogAPI] Discovery failed for ${providerName}: ${err instanceof Error ? err.message : String(err)}`);
  });
});

// POST /api/catalog/discover-all — trigger daily-style discovery for all providers
router.post('/discover-all', async (_req, res) => {
  res.json({ message: 'Discovery started for all providers' });
  discoveryAgent.refreshAllProviders().catch(err => {
    logger.error(`[CatalogAPI] Discovery all failed: ${err instanceof Error ? err.message : String(err)}`);
  });
});

// POST /api/catalog/weekly-check — trigger weekly deep check (paid free-tier probe + re-eval)
router.post('/weekly-check', async (_req, res) => {
  res.json({ message: 'Weekly deep check started — probing paid providers for free tiers' });
  discoveryAgent.weeklyDeepCheck().catch(err => {
    logger.error(`[CatalogAPI] Weekly check failed: ${err instanceof Error ? err.message : String(err)}`);
  });
});

// POST /api/catalog/disable — disable a model
router.post('/disable', (req, res) => {
  const { provider, modelId, reason } = req.body;
  if (!provider || !modelId) {
    res.status(400).json({ error: 'provider and modelId are required' });
    return;
  }
  catalogManager.disableModel(provider, modelId, reason || 'Manually disabled via API');
  res.json({ message: `Model ${provider}/${modelId} disabled`, provider, modelId });
});

// POST /api/catalog/reactivate — reactivate a model
router.post('/reactivate', (req, res) => {
  const { provider, modelId } = req.body;
  if (!provider || !modelId) {
    res.status(400).json({ error: 'provider and modelId are required' });
    return;
  }
  catalogManager.reactivateModel(provider, modelId);
  res.json({ message: `Model ${provider}/${modelId} reactivated`, provider, modelId });
});

// GET /api/catalog/tpd — TPD usage for all tracked providers
router.get('/tpd', (_req, res) => {
  res.json(providerCircuitBreaker.getAllTpdUsage());
});

// GET /api/catalog/special — list all special (non-chat) models, optionally filtered by category
// Example: /api/catalog/special?category=vision
router.get('/special', (req, res) => {
  const category = req.query.category as string | undefined;
  const validCategories = ['vision', 'embedding', 'image-gen', 'audio', 'rerank', 'other'];
  if (category && !validCategories.includes(category)) {
    res.status(400).json({ error: `Invalid category. Valid: ${validCategories.join(', ')}` });
    return;
  }
  const models = catalogManager.getSpecialModels(category as any || undefined);
  res.json({
    category: category || 'all',
    count: models.length,
    models: models.map(({ provider, model }) => ({
      provider,
      id: model.id,
      category: model.category,
      intelligence: model.intelligence,
      contextWindow: model.contextWindow,
      status: model.status,
    })),
  });
});

export default router;
