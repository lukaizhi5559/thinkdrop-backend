import { Router, Request, Response } from 'express';
import { logger } from '../utils/logger';
import { omniParserService } from '../services/omniParserService';

const router = Router();

/**
 * POST /api/omniparser/parse
 * Parse screenshot with OmniParser to detect all UI elements.
 * Used by ui.findAndClick skill in command-service MCP.
 */
router.post('/parse', async (req: Request, res: Response): Promise<void> => {
  try {
    const { screenshot, context } = req.body;

    if (!screenshot?.base64) {
      res.status(400).json({ success: false, error: 'Missing required field: screenshot.base64' });
      return;
    }

    if (!omniParserService.isAvailable()) {
      res.status(503).json({ success: false, error: 'OmniParser service not available', message: 'Configure HUGGINGFACE_OMNIPARSER_ENDPOINT, MODAL_OMNIPARSER_ENDPOINT, or REPLICATE_API_TOKEN' });
      return;
    }

    logger.info('🔍 [OMNIPARSER-API] Parse request received', {
      screenshotSize: screenshot.base64.length,
      hasContext: !!context,
      screenDimensions: context?.screenWidth && context?.screenHeight ? `${context.screenWidth}x${context.screenHeight}` : 'unknown',
      hasWindowBounds: !!context?.windowBounds,
    });

    const startTime = Date.now();

    const omniContext = {
      url: context?.url || context?.activeUrl || 'unknown',
      screenWidth: context?.screenWidth || context?.screenshotWidth || 1440,
      screenHeight: context?.screenHeight || context?.screenshotHeight || 900,
      screenshotWidth: context?.screenshotWidth || context?.screenWidth || 1440,
      screenshotHeight: context?.screenshotHeight || context?.screenHeight || 900,
      windowBounds: context?.windowBounds,
    };

    const result = await omniParserService.detectElement(screenshot, 'fetch_all_elements', omniContext);
    const latencyMs = Date.now() - startTime;
    const elements = result.allElements || [];

    logger.info('✅ [OMNIPARSER-API] Parse successful', {
      totalElements: elements.length,
      interactiveElements: elements.filter((e) => e.interactivity).length,
      cacheHit: result.cacheHit,
      latencyMs,
    });

    res.status(200).json({
      success: true,
      elements,
      metadata: {
        totalElements: elements.length,
        interactiveElements: elements.filter((e) => e.interactivity).length,
        byType: {
          text: elements.filter((e) => e.type === 'text').length,
          icon: elements.filter((e) => e.type === 'icon').length,
        },
        cacheHit: result.cacheHit,
        method: result.method,
      },
      latencyMs,
    });
  } catch (error: any) {
    logger.error('❌ [OMNIPARSER-API] Parse failed', { error: error.message });
    res.status(500).json({ success: false, error: 'Failed to parse screenshot with OmniParser', message: error.message });
  }
});

/**
 * POST /api/omniparser/detect
 * Detect a specific element in a screenshot using OmniParser + LLM matching.
 * Used by ui.findAndClick skill in command-service MCP.
 */
router.post('/detect', async (req: Request, res: Response): Promise<void> => {
  try {
    const { screenshot, description, context } = req.body;

    if (!screenshot?.base64 || !description) {
      res.status(400).json({ success: false, error: 'Missing required fields: screenshot.base64 and description' });
      return;
    }

    if (!omniParserService.isAvailable()) {
      res.status(503).json({ success: false, error: 'OmniParser service not available', message: 'Configure HUGGINGFACE_OMNIPARSER_ENDPOINT, MODAL_OMNIPARSER_ENDPOINT, or REPLICATE_API_TOKEN' });
      return;
    }

    logger.info('🎯 [OMNIPARSER-API] Detect request received', { description, screenshotSize: screenshot.base64.length, hasContext: !!context });

    const startTime = Date.now();

    const omniContext = {
      url: context?.url || context?.activeUrl || 'unknown',
      screenWidth: context?.screenWidth || context?.screenshotWidth || 1440,
      screenHeight: context?.screenHeight || context?.screenshotHeight || 900,
      screenshotWidth: context?.screenshotWidth || context?.screenWidth || 1440,
      screenshotHeight: context?.screenshotHeight || context?.screenHeight || 900,
      windowBounds: context?.windowBounds,
      intentType: context?.intentType,
      activeApp: context?.activeApp,
      activeUrl: context?.activeUrl,
    };

    const result = await omniParserService.detectElement(screenshot, description, omniContext);
    const latencyMs = Date.now() - startTime;

    logger.info('✅ [OMNIPARSER-API] Detect successful', {
      description,
      coordinates: result.coordinates,
      confidence: result.confidence,
      selectedElement: result.selectedElement,
      method: result.method,
      cacheHit: result.cacheHit,
      latencyMs,
    });

    res.status(200).json({
      success: true,
      coordinates: result.coordinates,
      confidence: result.confidence,
      selectedElement: result.selectedElement,
      method: result.method,
      cacheHit: result.cacheHit,
      latencyMs,
    });
  } catch (error: any) {
    logger.error('❌ [OMNIPARSER-API] Detect failed', { error: error.message });
    res.status(500).json({ success: false, error: 'Failed to detect element with OmniParser', message: error.message });
  }
});

/**
 * GET /api/omniparser/health
 * Health check for OmniParser service — reports which providers are configured.
 */
router.get('/health', async (_req: Request, res: Response): Promise<void> => {
  try {
    const available = omniParserService.isAvailable();
    const { omniParserWarmup } = await import('../services/omniParserWarmup');
    const warmupStats = omniParserWarmup.getStats();

    const status = {
      service: 'omniparser',
      status: available ? 'healthy' : 'unavailable',
      providers: {
        huggingface: {
          available: !!process.env.HUGGINGFACE_OMNIPARSER_ENDPOINT,
          priority: 1,
          endpoint: process.env.HUGGINGFACE_OMNIPARSER_ENDPOINT ? 'configured' : 'not configured',
        },
        modal: {
          available: !!(process.env.MODAL_API_KEY && process.env.MODAL_OMNIPARSER_ENDPOINT),
          priority: 2,
          endpoint: process.env.MODAL_OMNIPARSER_ENDPOINT ? 'configured' : 'not configured',
        },
        replicate: {
          available: !!process.env.REPLICATE_API_TOKEN,
          priority: 3,
          fallback: true,
          warmup: warmupStats,
        },
      },
      features: { caching: !!process.env.REDIS_HOST, elementDetection: true, batchParsing: true },
      timestamp: new Date().toISOString(),
    };

    res.status(available ? 200 : 503).json(status);
  } catch (error: any) {
    logger.error('OmniParser health check failed', { error: error.message });
    res.status(503).json({ service: 'omniparser', status: 'unhealthy', error: error.message, timestamp: new Date().toISOString() });
  }
});

export default router;
