/**
 * thinkdrop-backend — lightweight LLM streaming pipe + OmniParser service
 *
 * Endpoints:
 *   ws://localhost:4000/ws/stream       — LLM streaming WebSocket (stategraph black box)
 *   POST /api/omniparser/parse          — Parse screenshot, return all UI elements
 *   POST /api/omniparser/detect         — Detect specific element by description
 *   GET  /api/omniparser/health         — OmniParser provider status
 *   POST /api/vision/verify             — Step verification after automation action (StateGraph → replan on failure)
 *   POST /api/vision/analyze            — General screenshot analysis via vision LLM
 *   POST /api/vision/find               — Locate UI element by description, returns desktop coords
 *   GET  /api/vision/health             — Vision service status + provider availability
 *   GET  /health                        — Overall service health
 *   GET  /api/circuit-breaker/status   — List currently open (circuit-broken) providers
 *   POST /api/circuit-breaker/reset    — Reset one or all open circuit breakers
 */

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import http from 'http';
import WebSocket, { WebSocketServer } from 'ws';
import { v4 as uuidv4 } from 'uuid';
import { logger } from './utils/logger';
import { StreamingHandler } from './websocket/streamingHandler';
import { StreamingMessage } from './types/streaming';
import omniparserRoutes from './api/omniparser';
import visionRoutes from './api/vision';
import circuitBreakerRoutes from './api/circuitBreaker';
import catalogRoutes from './api/catalog';
import { catalogManager } from './utils/catalogManager';
import { discoveryAgent } from './utils/discoveryAgent';
import { LLMRouter } from './utils/llmRouter';

const PORT = parseInt(process.env.PORT || '4000', 10);

// ─── Express app ────────────────────────────────────────────────────────────

const app = express();

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// API routes
app.use('/api/omniparser', omniparserRoutes);
app.use('/api/vision', visionRoutes);
app.use('/api/circuit-breaker', circuitBreakerRoutes);
app.use('/api/catalog', catalogRoutes);

// Overall health check
app.get('/health', (_req, res) => {
  res.json({
    service: 'thinkdrop-backend',
    status: 'healthy',
    version: '1.0.0',
    uptime: process.uptime(),
    timestamp: new Date().toISOString(),
    endpoints: {
      websocket: 'ws://localhost:' + PORT + '/ws/stream',
      omniparser: 'http://localhost:' + PORT + '/api/omniparser',
      vision: 'http://localhost:' + PORT + '/api/vision',
      circuitBreaker: 'http://localhost:' + PORT + '/api/circuit-breaker',
      catalog: 'http://localhost:' + PORT + '/api/catalog',
    },
  });
});

// ─── HTTP + WebSocket server ─────────────────────────────────────────────────

const server = http.createServer(app);

const wss = new WebSocketServer({ server, path: '/ws/stream' });

wss.on('connection', (ws: WebSocket, req) => {
  const sessionId = uuidv4();
  const url = new URL(req.url || '/', `http://localhost:${PORT}`);
  const userId = url.searchParams.get('userId') || undefined;
  const clientId = url.searchParams.get('clientId') || undefined;

  logger.debug('🔌 [WS] Client connected', { sessionId, userId, clientId, remoteAddress: req.socket.remoteAddress });

  const handler = new StreamingHandler(ws, sessionId, userId, clientId);

  ws.on('message', async (data: WebSocket.RawData) => {
    try {
      const message = JSON.parse(data.toString()) as StreamingMessage;
      await handler.handleMessage(message);
    } catch (error) {
      logger.error('❌ [WS] Failed to parse message', { error: error instanceof Error ? error.message : String(error) });
      ws.send(JSON.stringify({
        id: 'parse_error',
        type: 'error',
        payload: { code: 'PARSE_ERROR', message: 'Invalid JSON message', recoverable: true },
        timestamp: Date.now(),
      }));
    }
  });

  ws.on('close', (code, reason) => {
    // Only log abnormal disconnects at info; normal close (1000/1005) at debug
    if (code !== 1000 && code !== 1005) {
      logger.info('🔌 [WS] Client disconnected (abnormal)', { sessionId, code, reason: reason.toString() });
    } else {
      logger.debug('🔌 [WS] Client disconnected', { sessionId, code });
    }
    handler.cleanup();
  });

  ws.on('error', (error) => {
    logger.error('❌ [WS] WebSocket error', { sessionId, error: error.message });
    handler.cleanup();
  });

  // Send connection acknowledgement
  ws.send(JSON.stringify({
    id: 'connection_ack',
    type: 'connection_status',
    payload: { connected: true, sessionId, message: 'Connected to thinkdrop-backend /ws/stream' },
    timestamp: Date.now(),
  }));
});

wss.on('error', (error) => {
  logger.error('❌ [WSS] WebSocket server error', { error: error.message });
});

// ─── OmniParser warmup ───────────────────────────────────────────────────────

async function startWarmup(): Promise<void> {
  try {
    const { omniParserWarmup } = await import('./services/omniParserWarmup');
    omniParserWarmup.start();
    logger.info('🔥 [STARTUP] OmniParser warmup service started');
  } catch (error) {
    logger.warn('⚠️ [STARTUP] OmniParser warmup failed to start', { error: error instanceof Error ? error.message : String(error) });
  }
}

// ─── Start server ────────────────────────────────────────────────────────────

server.on('error', (error: NodeJS.ErrnoException) => {
  if (error.code === 'EADDRINUSE') {
    logger.error(`❌ [SERVER] Port ${PORT} is already in use. Kill the existing process (kill -9 $(lsof -t -i:${PORT})) and retry.`);
    process.exit(1);
  } else {
    throw error;
  }
});

server.listen(PORT, async () => {
  logger.info('🚀 thinkdrop-backend started', {
    port: PORT,
    env: process.env.NODE_ENV || 'development',
    wsEndpoint: `ws://localhost:${PORT}/ws/stream`,
    httpEndpoint: `http://localhost:${PORT}`,
  });

  // Load provider catalog and start discovery agent
  try {
    await catalogManager.load();
    discoveryAgent.setLLMRouter(new LLMRouter());
    catalogManager.setDiscoveryCallback((provider, modelId, statusCode) => {
      discoveryAgent.onModelFailure(provider, modelId, statusCode);
    });
    discoveryAgent.start();
    logger.info('📋 [STARTUP] Provider catalog loaded + discovery agent started', {
      providers: catalogManager.getAllProviders().length,
    });
    // Background startup probe — verifies model categories by probing with "Say OK".
    // Non-blocking: server is ready immediately. Cached via lastVerifiedAt (24h).
    discoveryAgent.verifyAllModelsOnStartup().catch(err => {
      logger.warn('⚠️ [STARTUP] Model verification probe failed', {
        error: err instanceof Error ? err.message : String(err),
      });
    });
  } catch (err) {
    logger.warn('⚠️ [STARTUP] Catalog load failed — using hardcoded config', {
      error: err instanceof Error ? err.message : String(err),
    });
  }

  if (process.env.OMNIPARSER_WARMUP_ENABLED !== 'false') {
    await startWarmup();
  }
});

// ─── Graceful shutdown ───────────────────────────────────────────────────────

async function shutdown(signal: string): Promise<void> {
  logger.info(`🛑 [SHUTDOWN] Received ${signal}, shutting down gracefully`);

  try {
    discoveryAgent.stop();
  } catch { /* non-fatal */ }

  try {
    const { omniParserWarmup } = await import('./services/omniParserWarmup');
    omniParserWarmup.stop();
  } catch { /* non-fatal */ }

  wss.clients.forEach((ws) => ws.terminate());
  wss.close();

  server.close(() => {
    logger.info('✅ [SHUTDOWN] HTTP server closed');
    process.exit(0);
  });

  // Force exit after 10s if graceful shutdown hangs
  setTimeout(() => {
    logger.error('❌ [SHUTDOWN] Forced exit after timeout');
    process.exit(1);
  }, 10000);
}

process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));
process.on('uncaughtException', (error) => {
  logger.error('❌ [UNCAUGHT] Uncaught exception', { error: error.message, stack: error.stack });
  process.exit(1);
});
process.on('unhandledRejection', (reason) => {
  logger.error('❌ [UNHANDLED] Unhandled promise rejection', { reason: String(reason) });
});
