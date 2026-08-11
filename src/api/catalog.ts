/**
 * Catalog API — endpoints for viewing and managing the provider catalog.
 *
 *   GET  /api/catalog                    — Full catalog with health report
 *   GET  /api/catalog/health             — Catalog health summary
 *   GET  /api/catalog/providers          — List all providers
 *   GET  /api/catalog/providers/:name    — Single provider detail
 *   POST /api/catalog/providers          — Add a new provider at runtime
 *   POST /api/catalog/providers/confirm  — Confirm candidates + trigger discovery
 *   DELETE /api/catalog/providers/:name  — Remove a provider (marks as dead)
 *   POST /api/catalog/discover/:name     — Trigger discovery for a provider
 *   POST /api/catalog/discover-all       — Trigger discovery for all providers
 *   POST /api/catalog/weekly-check       — Trigger weekly deep check (paid free-tier probe + re-eval)
 *   POST /api/catalog/ask                — Natural language instruction (e.g. "find new fast models")
 *   POST /api/catalog/disable            — Disable a model { provider, modelId, reason }
 *   POST /api/catalog/reactivate         — Reactivate a model { provider, modelId }
 *   GET  /api/catalog/tpd                — TPD usage for all tracked providers
 *   GET  /api/catalog/special            — List special (non-chat) models, optional ?category=vision
 *   GET  /api/catalog/rankings           — Model rankings with score breakdown
 *   POST /api/catalog/promote            — Manually promote a model { provider, modelId, taskType }
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

// POST /api/catalog/providers — add a new provider at runtime
// Body: { "name": "together-ai", "baseURL": "https://api.together.xyz/v1", "envKey": "TOGETHER_API_KEY", "catalogEndpoint": "...", "apiType": "openai-compatible", "tier": "free", "autoDiscover": true }
router.post('/providers', async (req, res) => {
  const { name, baseURL, envKey, catalogEndpoint, apiType, tier, autoDiscover } = req.body;
  if (!name || !baseURL || !envKey) {
    res.status(400).json({ error: 'name, baseURL, and envKey are required' });
    return;
  }

  const result = catalogManager.addProvider(name, {
    baseURL,
    envKey,
    catalogEndpoint,
    apiType: apiType || 'openai-compatible',
    tier: tier || 'free',
  });

  if (!result.success) {
    res.status(409).json({ error: result.error });
    return;
  }

  // Optionally trigger discovery to auto-discover models
  let discoveryResult = null;
  if (autoDiscover) {
    try {
      await discoveryAgent.discoverProvider(name);
      const p = catalogManager.getProvider(name)!;
      discoveryResult = {
        activeModels: p.models.filter(m => m.status === 'active').length,
        totalModels: p.models.length,
      };
    } catch (e) {
      discoveryResult = { error: e instanceof Error ? e.message : String(e) };
    }
  }

  logger.info(`[CatalogAPI] Provider added: ${name} (autoDiscover: ${!!autoDiscover})`);
  res.json({ success: true, provider: name, discovery: discoveryResult });
});

// POST /api/catalog/providers/confirm — confirm candidate providers from discovery
// Body: { "providers": ["together-ai", "fireworks"] }
// Each provider must have been previously added via POST /providers
router.post('/providers/confirm', async (req, res) => {
  const { providers } = req.body;
  if (!Array.isArray(providers)) {
    res.status(400).json({ error: 'providers must be an array of names' });
    return;
  }

  const results = [];
  for (const name of providers) {
    const provider = catalogManager.getProvider(name);
    if (!provider) {
      results.push({ provider: name, success: false, error: 'Provider not found — add via POST /providers first' });
      continue;
    }
    try {
      await discoveryAgent.discoverProvider(name);
      const p = catalogManager.getProvider(name)!;
      results.push({
        provider: name,
        success: true,
        activeModels: p.models.filter(m => m.status === 'active').length,
        totalModels: p.models.length,
      });
    } catch (e) {
      results.push({ provider: name, success: false, error: e instanceof Error ? e.message : String(e) });
    }
  }

  res.json({ results });
});

// DELETE /api/catalog/providers/:name — remove a provider (marks as dead)
router.delete('/providers/:name', (req, res) => {
  const result = catalogManager.removeProvider(req.params.name);
  if (!result.success) {
    res.status(404).json({ error: result.error });
    return;
  }
  logger.info(`[CatalogAPI] Provider removed: ${req.params.name}`);
  res.json({ success: true, provider: req.params.name });
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

// POST /api/catalog/ask — natural language instruction to the discovery agent
// Body: { "instruction": "check if any of my providers have new fast models" }
// Returns: { summary, actions, results } — LLM-generated summary + parsed action plan + per-action results
router.post('/ask', async (req, res) => {
  const { instruction } = req.body;
  if (!instruction || typeof instruction !== 'string') {
    res.status(400).json({
      error: 'instruction is required (string)',
      examples: [
        'check if any of my providers have new fast models',
        'find free tier models on mistral',
        'check health of all providers',
        'look for new vision models',
        're-probe dead models',
      ],
    });
    return;
  }

  try {
    const result = await discoveryAgent.executeInstruction(instruction);
    res.json(result);
  } catch (err) {
    const errMsg = err instanceof Error ? err.message : String(err);
    logger.error(`[CatalogAPI] /ask failed: ${errMsg}`);
    res.status(500).json({ error: errMsg, instruction });
  }
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

// GET /api/catalog/rankings — show model rankings with score breakdown
// Optional: ?taskType=heavy or ?taskType=light (default: both)
router.get('/rankings', (req, res) => {
  const taskType = req.query.taskType as string | undefined;
  const taskTypes: Array<'heavy' | 'light'> = taskType === 'heavy' || taskType === 'light'
    ? [taskType] : ['heavy', 'light'];

  const result: Record<string, any> = {};
  for (const tt of taskTypes) {
    const chain = catalogManager.getRankedFallbackChain(tt);
    result[tt] = chain.map(provider => {
      const ranked = catalogManager.getRankedModels(provider, tt);
      return {
        provider,
        models: ranked.map(({ model, score, breakdown }) => ({
          id: model.id,
          score,
          breakdown,
          intelligence: model.intelligence,
          speed: model.speed,
          category: model.category,
          status: model.status,
          totalCalls: model.totalCalls,
          lastResponseMs: model.lastResponseMs || 0,
          successRate: model.totalCalls > 0
            ? (model.totalSuccesses / model.totalCalls * 100).toFixed(1) + '%'
            : 'N/A',
        })),
      };
    });
  }
  res.json(result);
});

// POST /api/catalog/promote — manually promote a model to the top of its chain
// Body: { "provider": "groq", "modelId": "qwen/qwen3.6-27b", "taskType": "heavy" }
router.post('/promote', (req, res) => {
  const { provider, modelId, taskType } = req.body;
  if (!provider || !modelId || !taskType) {
    res.status(400).json({ error: 'provider, modelId, and taskType are required' });
    return;
  }
  if (taskType !== 'heavy' && taskType !== 'light') {
    res.status(400).json({ error: 'taskType must be "heavy" or "light"' });
    return;
  }

  const p = catalogManager.getProvider(provider);
  if (!p) {
    res.status(404).json({ error: `Provider not found: ${provider}` });
    return;
  }
  const model = p.models.find(m => m.id === modelId && m.taskType === taskType);
  if (!model) {
    res.status(404).json({ error: `Model not found: ${provider}/${modelId} (${taskType})` });
    return;
  }

  // Boost the model's intelligence and speed to force it to the top
  // This is a manual override — the scoring system will naturally maintain it
  // if the model performs well (totalCalls/successes will build trust)
  const currentBest = catalogManager.getBestModel(provider, taskType as any);
  if (currentBest) {
    const currentScore = currentBest.score;
    // Set intelligence high enough to beat current best
    model.intelligence = Math.max(model.intelligence || 50, 100);
    model.speed = Math.max(model.speed || 100, 1000);
    logger.info(`[CatalogAPI] Manually promoted ${provider}/${modelId} for ${taskType} tasks (was score ${currentScore})`);
  }

  res.json({
    message: `Model ${provider}/${modelId} promoted for ${taskType} tasks`,
    provider,
    modelId,
    taskType,
  });
});

export default router;
