/**
 * Vision API routes — screen verification and analysis for UI automation.
 *
 * Primary use-case: the StateGraph brain calls /verify after each automation
 * step so the agent can confirm success or replan on failure instead of flying
 * blind with false-positives.
 *
 * Provider fallback chain (vision-capable models only):
 *   1. OpenAI  — gpt-4o          (OPENAI_API_KEY)
 *   2. Claude  — claude-opus-4-5 (ANTHROPIC_API_KEY)
 *   3. Gemini  — gemini-1.5-pro  (GEMINI_API_KEY)
 */

import { Router, Request, Response } from 'express';
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { logger } from '../utils/logger';

const router = Router();

// ─── Vision provider helpers ─────────────────────────────────────────────────

interface VisionResult {
  text: string;
  provider: string;
  processingTime: number;
}

async function callOpenAIVision(base64: string, mimeType: string, prompt: string): Promise<string> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error('OPENAI_API_KEY not set');

  const openai = new OpenAI({ apiKey });
  const response = await openai.chat.completions.create({
    model: 'gpt-4o',
    max_tokens: 1024,
    temperature: 0.1,
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'image_url',
            image_url: { url: `data:${mimeType};base64,${base64}`, detail: 'high' },
          },
          { type: 'text', text: prompt },
        ],
      },
    ],
  });

  return response.choices[0]?.message?.content || '';
}

async function callClaudeVision(base64: string, mimeType: string, prompt: string): Promise<string> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error('ANTHROPIC_API_KEY not set');

  const anthropic = new Anthropic({ apiKey });
  const response = await anthropic.messages.create({
    model: 'claude-opus-4-5',
    max_tokens: 1024,
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'image',
            source: {
              type: 'base64',
              media_type: mimeType as 'image/png' | 'image/jpeg' | 'image/gif' | 'image/webp',
              data: base64,
            },
          },
          { type: 'text', text: prompt },
        ],
      },
    ],
  });

  const block = response.content[0];
  return block && 'text' in block ? block.text : '';
}

async function callGeminiVision(base64: string, mimeType: string, prompt: string): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY not set');

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({ model: 'gemini-1.5-pro' });
  const result = await model.generateContent([
    { inlineData: { data: base64, mimeType } },
    prompt,
  ]);

  return result.response.text();
}

async function callVisionWithFallback(base64: string, mimeType: string, prompt: string): Promise<VisionResult> {
  const startTime = performance.now();
  const providers: Array<{ name: string; fn: () => Promise<string> }> = [
    { name: 'openai', fn: () => callOpenAIVision(base64, mimeType, prompt) },
    { name: 'claude', fn: () => callClaudeVision(base64, mimeType, prompt) },
    { name: 'gemini', fn: () => callGeminiVision(base64, mimeType, prompt) },
  ];

  for (const { name, fn } of providers) {
    try {
      const text = await fn();
      logger.info(`✅ [VISION] Provider succeeded`, { provider: name });
      return { text, provider: name, processingTime: performance.now() - startTime };
    } catch (err) {
      logger.warn(`⚠️ [VISION] Provider failed, trying next`, {
        provider: name,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }

  throw new Error('[VISION] All vision providers failed — check API keys');
}

// ─── Parse structured JSON from LLM response ─────────────────────────────────

function extractJSON(raw: string): any {
  let text = raw.trim();
  if (text.startsWith('```')) {
    text = text.replace(/```json?\n?/g, '').replace(/```\n?$/g, '').trim();
  }
  return JSON.parse(text);
}

// ─── Routes ───────────────────────────────────────────────────────────────────

/**
 * POST /api/vision/verify
 *
 * Called by the StateGraph after each automation step to confirm success.
 * Returns a structured verdict so the brain can replan on failure instead
 * of continuing with a false-positive.
 *
 * Request body:
 * {
 *   "screenshot": { "base64": "...", "mimeType": "image/png" },
 *   "prompt": "Verify that Chris Akers' DM is open in Slack",
 *   "stepDescription": "Open Chris Akers' DM",
 *   "context": {
 *     "activeApp": "Slack",
 *     "activeUrl": null,
 *     "stepIndex": 4,
 *     "totalSteps": 6
 *   }
 * }
 *
 * Response:
 * {
 *   "success": true,
 *   "verified": true | false,
 *   "confidence": 0.0–1.0,
 *   "reasoning": "The DM panel shows 'Chris Akers' in the header...",
 *   "suggestion": "Proceed to next step" | "Retry: click the DM again",
 *   "provider": "openai",
 *   "processingTime": 1234
 * }
 */
router.post('/verify', async (req: Request, res: Response): Promise<void> => {
  try {
    const { screenshot, prompt, stepDescription, context } = req.body;

    if (!screenshot?.base64) {
      res.status(400).json({ success: false, error: 'Missing required field: screenshot.base64' });
      return;
    }
    if (!prompt) {
      res.status(400).json({ success: false, error: 'Missing required field: prompt' });
      return;
    }

    const mimeType = screenshot.mimeType || 'image/png';

    logger.info('🔍 [VISION-VERIFY] Verification request received', {
      stepDescription,
      activeApp: context?.activeApp,
      stepIndex: context?.stepIndex,
    });

    const systemPrompt = `You are a UI automation verification agent. Your job is to look at a screenshot and determine whether a specific automation step succeeded.

Step that was just executed: "${stepDescription || prompt}"

Verification instruction from the automation brain:
${prompt}

Context:
- Active app: ${context?.activeApp || 'unknown'}
- Step ${context?.stepIndex != null ? context.stepIndex + 1 : '?'} of ${context?.totalSteps || '?'}

Respond ONLY with a JSON object in this exact format:
{
  "verified": true or false,
  "confidence": <number 0.0 to 1.0>,
  "reasoning": "<one or two sentences describing exactly what you see that supports your verdict>",
  "suggestion": "<if verified=false, a specific corrective action; if verified=true, write 'Proceed to next step'>"
}

Be strict: only return verified=true if you can clearly see evidence the step succeeded.`;

    const result = await callVisionWithFallback(screenshot.base64, mimeType, systemPrompt);

    let parsed: any;
    try {
      parsed = extractJSON(result.text);
    } catch {
      logger.warn('⚠️ [VISION-VERIFY] Could not parse JSON from LLM, returning raw', { raw: result.text });
      res.status(200).json({
        success: true,
        verified: false,
        confidence: 0,
        reasoning: result.text,
        suggestion: 'LLM response was not structured JSON — treat as unverified',
        provider: result.provider,
        processingTime: result.processingTime,
        timestamp: new Date().toISOString(),
      });
      return;
    }

    logger.info('✅ [VISION-VERIFY] Verification complete', {
      verified: parsed.verified,
      confidence: parsed.confidence,
      provider: result.provider,
      stepDescription,
    });

    res.status(200).json({
      success: true,
      verified: !!parsed.verified,
      confidence: typeof parsed.confidence === 'number' ? parsed.confidence : 0,
      reasoning: parsed.reasoning || '',
      suggestion: parsed.suggestion || '',
      provider: result.provider,
      processingTime: Math.round(result.processingTime),
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    logger.error('❌ [VISION-VERIFY] All providers failed — returning degraded response', { error: error.message });
    res.status(200).json({
      success: true,
      verified: null,
      degraded: true,
      confidence: 0,
      reasoning: 'Vision service unavailable — all providers failed. Cannot verify step.',
      suggestion: 'Vision unavailable — skip verification and proceed.',
      provider: null,
      processingTime: 0,
      timestamp: new Date().toISOString(),
    });
  }
});

/**
 * POST /api/vision/analyze
 *
 * General-purpose screenshot analysis. The prompt/query comes from the caller
 * (StateGraph brain). Returns a free-form description plus structured fields.
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

    const mimeType = screenshot.mimeType || 'image/png';

    logger.info('👁️ [VISION-ANALYZE] Analyze request received', {
      hasQuery: !!query,
      activeApp: context?.activeApp,
    });

    const prompt = `You are a screen analysis assistant. Analyze the screenshot and answer the following query.

Query: ${query || 'Describe what is visible on the screen in detail.'}

Context:
- Active app: ${context?.activeApp || 'unknown'}
- Active URL: ${context?.activeUrl || 'none'}

Respond with a JSON object:
{
  "description": "<detailed description of what is on screen>",
  "answer": "<direct answer to the query>",
  "uiState": "<brief summary of the current UI state>",
  "relevantElements": ["<list of notable UI elements visible>"]
}`;

    const result = await callVisionWithFallback(screenshot.base64, mimeType, prompt);

    let parsed: any = null;
    try {
      parsed = extractJSON(result.text);
    } catch {
      parsed = null;
    }

    res.status(200).json({
      success: true,
      analysis: {
        description: parsed?.description || result.text,
        answer: parsed?.answer || result.text,
        uiState: parsed?.uiState || null,
        relevantElements: parsed?.relevantElements || [],
        query: query || null,
        activeApp: context?.activeApp || null,
        activeUrl: context?.activeUrl || null,
        windowBounds: context?.windowBounds || null,
      },
      provider: result.provider,
      processingTime: Math.round(result.processingTime),
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    logger.error('❌ [VISION-ANALYZE] Analyze failed', { error: error.message });
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

    const mimeType = screenshot.mimeType || 'image/png';
    const wb = context?.windowBounds;

    logger.info('🔎 [VISION-FIND] Find request received', {
      description,
      hasWindowBounds: !!wb,
    });

    const prompt = `You are a UI element locator. Find the element described below in the screenshot and return its center coordinates.

Element to find: "${description}"

The screenshot dimensions correspond to a window at:
- x offset: ${wb?.x ?? 0}, y offset: ${wb?.y ?? 0}
- width: ${wb?.width ?? 'unknown'}, height: ${wb?.height ?? 'unknown'}

Return ONLY a JSON object:
{
  "found": true or false,
  "x": <desktop x coordinate — window x offset + element x within window>,
  "y": <desktop y coordinate — window y offset + element y within window>,
  "confidence": <0.0 to 1.0>,
  "reasoning": "<brief explanation of where you found it>"
}

If not found, set found=false and omit x/y.`;

    const result = await callVisionWithFallback(screenshot.base64, mimeType, prompt);

    let parsed: any;
    try {
      parsed = extractJSON(result.text);
    } catch {
      res.status(200).json({
        success: false,
        found: false,
        reasoning: result.text,
        provider: result.provider,
        processingTime: Math.round(result.processingTime),
        timestamp: new Date().toISOString(),
      });
      return;
    }

    res.status(200).json({
      success: true,
      found: !!parsed.found,
      x: parsed.x ?? null,
      y: parsed.y ?? null,
      confidence: parsed.confidence ?? 0,
      reasoning: parsed.reasoning || '',
      provider: result.provider,
      processingTime: Math.round(result.processingTime),
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    logger.error('❌ [VISION-FIND] Find failed', { error: error.message });
    res.status(500).json({ success: false, error: 'Vision find failed', message: error.message });
  }
});

/**
 * GET /api/vision/health
 */
router.get('/health', (_req: Request, res: Response): void => {
  const providers = {
    openai: !!process.env.OPENAI_API_KEY,
    claude: !!process.env.ANTHROPIC_API_KEY,
    gemini: !!process.env.GEMINI_API_KEY,
  };
  const available = Object.values(providers).some(Boolean);

  res.status(200).json({
    service: 'vision',
    status: available ? 'ready' : 'no_providers',
    providers,
    endpoints: {
      verify: 'POST /api/vision/verify — step verification for automation (StateGraph)',
      analyze: 'POST /api/vision/analyze — general screenshot analysis',
      find: 'POST /api/vision/find — locate UI element by description',
    },
    timestamp: new Date().toISOString(),
  });
});

export default router;
