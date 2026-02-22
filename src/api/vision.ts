/**
 * Vision API routes — for screen-intelligence MCP service.
 * Provides screenshot analysis and coordinate resolution for UI automation.
 * 
 * Future: This will be the primary endpoint for the screen-intelligence MCP.
 * Currently provides a stub that returns structured responses for integration testing.
 */

import { Router, Request, Response } from 'express';
import { logger } from '../utils/logger';

const router = Router();

/**
 * POST /api/vision/analyze
 * Analyze a screenshot and return structured UI understanding.
 * 
 * Request body:
 * {
 *   "screenshot": { "base64": "...", "mimeType": "image/png" },
 *   "query": "What is on the screen?",
 *   "context": {
 *     "windowBounds": { "x": 0, "y": 0, "width": 1440, "height": 900 },
 *     "activeApp": "Chrome",
 *     "activeUrl": "https://example.com"
 *   }
 * }
 */
router.post('/analyze', async (req: Request, res: Response): Promise<void> => {
  try {
    const { screenshot, query, context } = req.body;

    if (!screenshot?.base64) {
      res.status(400).json({ success: false, error: 'Missing required field: screenshot.base64' });
      return;
    }

    logger.info('👁️ [VISION-API] Analyze request received', {
      hasQuery: !!query,
      hasContext: !!context,
      activeApp: context?.activeApp,
      hasWindowBounds: !!context?.windowBounds,
    });

    // TODO: Integrate with vision LLM (GPT-4o vision, Claude vision, Gemini vision)
    // For now return a structured stub so screen-intelligence MCP can integrate
    res.status(200).json({
      success: true,
      analysis: {
        description: 'Vision analysis not yet implemented — integrate LLM vision provider here',
        query: query || null,
        activeApp: context?.activeApp || null,
        activeUrl: context?.activeUrl || null,
        windowBounds: context?.windowBounds || null,
      },
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    logger.error('❌ [VISION-API] Analyze failed', { error: error.message });
    res.status(500).json({ success: false, error: 'Vision analysis failed', message: error.message });
  }
});

/**
 * POST /api/vision/find
 * Find a UI element by natural language description using vision LLM.
 * Returns desktop-space coordinates (windowBounds offset applied).
 * 
 * Request body:
 * {
 *   "screenshot": { "base64": "...", "mimeType": "image/png" },
 *   "description": "the submit button",
 *   "context": {
 *     "windowBounds": { "x": 100, "y": 100, "width": 1200, "height": 800 }
 *   }
 * }
 */
router.post('/find', async (req: Request, res: Response): Promise<void> => {
  try {
    const { screenshot, description, context } = req.body;

    if (!screenshot?.base64 || !description) {
      res.status(400).json({ success: false, error: 'Missing required fields: screenshot.base64 and description' });
      return;
    }

    logger.info('🔎 [VISION-API] Find request received', {
      description,
      hasWindowBounds: !!context?.windowBounds,
    });

    // TODO: Integrate with vision LLM for coordinate extraction
    // IMPORTANT: Must apply windowBounds offset to returned coordinates so they are
    // in desktop space, not window-relative space.
    res.status(501).json({
      success: false,
      error: 'Vision find not yet implemented',
      message: 'Use /api/omniparser/detect for element detection until vision LLM is integrated',
    });
  } catch (error: any) {
    logger.error('❌ [VISION-API] Find failed', { error: error.message });
    res.status(500).json({ success: false, error: 'Vision find failed', message: error.message });
  }
});

/**
 * GET /api/vision/health
 */
router.get('/health', (_req: Request, res: Response): void => {
  res.status(200).json({
    service: 'vision',
    status: 'stub',
    message: 'Vision API is stubbed — ready for screen-intelligence MCP integration',
    timestamp: new Date().toISOString(),
  });
});

export default router;
