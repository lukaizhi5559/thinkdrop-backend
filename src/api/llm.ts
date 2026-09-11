/**
 * LLM HTTP API — simple text-generation endpoints for services that don't
 * use the WebSocket streaming protocol (e.g. comms-graph).
 *
 *   POST /api/llm         — Non-streaming LLM query (returns full text)
 *   POST /api/llm/stream  — Streaming LLM query (Server-Sent Events)
 *
 * Both endpoints route through the backend's resilient provider chain:
 * catalog ranking, circuit breakers, model fallback, discovery, and
 * free-premium/paid chain escalation.
 *
 * Auth: Optional Bearer token via STATEGRAPH_API_KEY env var.
 * If STATEGRAPH_API_KEY is set, requests must include:
 *   Authorization: Bearer <key>
 */

import { Router, Request, Response } from 'express';
import { LLMRouter } from '../utils/llmRouter';
import { llmStreamingRouter } from '../utils/llmStreamingRouter';
import { logger } from '../utils/logger';
import {
  StreamingMessage,
  StreamingMessageType,
  LLMStreamRequest,
} from '../types/streaming';

const router = Router();

// Singleton router instance (same pattern as streamingHandler)
const llmRouter = new LLMRouter();

// ─── Auth middleware ────────────────────────────────────────────────────────

function authMiddleware(req: Request, res: Response, next: () => void): void {
  const expectedKey = process.env.STATEGRAPH_API_KEY;
  if (!expectedKey) {
    // No key configured — open access
    next();
    return;
  }
  const authHeader = req.headers.authorization;
  const providedKey = authHeader?.startsWith('Bearer ')
    ? authHeader.slice(7)
    : undefined;
  if (providedKey !== expectedKey) {
    res.status(401).json({ ok: false, error: 'Unauthorized' });
    return;
  }
  next();
}

router.use(authMiddleware);

// ─── POST /api/llm — non-streaming LLM query ───────────────────────────────

interface LLMRequestBody {
  prompt: string;
  systemPrompt?: string;
  options?: {
    maxTokens?: number;
    temperature?: number;
    taskType?: string;
    preferredProvider?: string;
    responseFormat?: {
      type: 'json_schema' | 'json_object';
      json_schema?: { name: string; schema: object; strict?: boolean };
    };
  };
}

router.post('/', async (req: Request, res: Response) => {
  const { prompt, systemPrompt, options } = (req.body || {}) as LLMRequestBody;

  if (!prompt || typeof prompt !== 'string') {
    res.status(400).json({ ok: false, error: 'prompt is required (string)' });
    return;
  }

  // Prepend system prompt if provided
  const fullPrompt = systemPrompt
    ? `${systemPrompt}\n\n${prompt}`
    : prompt;

  const startTime = Date.now();

  try {
    const result = await llmRouter.processPrompt(fullPrompt, {
      taskType: options?.taskType,
      preferredProvider: options?.preferredProvider,
      responseFormat: options?.responseFormat,
    });

    res.json({
      ok: true,
      data: {
        text: result.text,
        provider: result.provider,
        processingTime: Date.now() - startTime,
      },
    });
  } catch (err) {
    const errMsg = err instanceof Error ? err.message : String(err);
    logger.error(`[LLMAPI] /api/llm failed: ${errMsg}`);
    res.status(500).json({ ok: false, error: errMsg });
  }
});

// ─── POST /api/llm/stream — streaming LLM query (SSE) ──────────────────────

router.post('/stream', async (req: Request, res: Response) => {
  const { prompt, systemPrompt, options } = (req.body || {}) as LLMRequestBody;

  if (!prompt || typeof prompt !== 'string') {
    res.status(400).json({ ok: false, error: 'prompt is required (string)' });
    return;
  }

  // Set SSE headers
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no'); // Disable nginx buffering

  const startTime = Date.now();

  // Build the stream request
  const streamRequest: LLMStreamRequest = {
    prompt,
    provider: options?.preferredProvider === 'auto' ? 'auto' : options?.preferredProvider,
    options: {
      temperature: options?.temperature,
      maxTokens: options?.maxTokens,
      taskType: options?.taskType,
    },
    context: systemPrompt
      ? { systemInstructions: systemPrompt }
      : undefined,
  };

  // Metadata for the streaming router
  const metadata = {
    clientId: `http_${Date.now()}`,
    source: 'http_api',
  };

  let lastProvider = 'unknown';

  try {
    const result = await llmStreamingRouter.processPromptWithStreaming(
      streamRequest,
      (msg: StreamingMessage) => {
        if (msg.type === StreamingMessageType.LLM_STREAM_CHUNK) {
          const chunk = msg.payload as any;
          if (chunk.reasoning) {
            lastProvider = chunk.provider || lastProvider;
            res.write(`data: ${JSON.stringify({ reasoning: chunk.reasoning, provider: chunk.provider })}\n\n`);
          }
          if (chunk.text) {
            lastProvider = chunk.provider || lastProvider;
            res.write(`data: ${JSON.stringify({ text: chunk.text, provider: chunk.provider })}\n\n`);
          }
        }
      },
      metadata as any
    );

    // Send final event
    res.write(`data: ${JSON.stringify({
      done: true,
      provider: result.provider || lastProvider,
      processingTime: Date.now() - startTime,
    })}\n\n`);
    res.end();
  } catch (err) {
    const errMsg = err instanceof Error ? err.message : String(err);
    logger.error(`[LLMAPI] /api/llm/stream failed: ${errMsg}`);
    res.write(`data: ${JSON.stringify({ error: errMsg })}\n\n`);
    res.end();
  }
});

export default router;
