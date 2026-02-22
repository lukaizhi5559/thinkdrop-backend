# thinkdrop-backend

Lightweight LLM streaming pipe + OmniParser service for Thinkdrop.

## Architecture

```
stategraph-module  →  decides WHAT to ask and HOW to ask it  (brain)
thinkdrop-backend  →  routes the prompt to an LLM and streams back tokens  (pipe)
```

The backend is a **black box** to the stategraph. It only cares that `/ws/stream` exists. All prompt construction, persona injection, and context formatting is owned by the stategraph. This backend passes prompts straight through.

## Endpoints

| Endpoint | Description |
|---|---|
| `ws://localhost:4000/ws/stream` | LLM streaming WebSocket — stategraph primary connection |
| `POST /api/omniparser/parse` | Parse screenshot, return all UI elements |
| `POST /api/omniparser/detect` | Detect specific element by natural language description |
| `GET /api/omniparser/health` | OmniParser provider status + warmup stats |
| `POST /api/vision/analyze` | Vision analysis stub (for screen-intelligence MCP) |
| `POST /api/vision/find` | Vision element find stub |
| `GET /health` | Overall service health |

## Setup

```bash
cp .env.example .env
# Fill in at least one LLM provider key and one OmniParser provider
yarn install
yarn dev
```

## Environment Variables

### LLM Providers (configure at least one)
```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
MISTRAL_API_KEY=
GROK_API_KEY=
DEEPSEEK_API_KEY=
LAMBDA_AI=
```

### OmniParser (configure at least one, priority order)
```
# Priority 1: Hugging Face Gradio — Microsoft's official deployment, no cold starts
HUGGINGFACE_OMNIPARSER_ENDPOINT=

# Priority 2: Modal.com serverless GPU
MODAL_API_KEY=
MODAL_OMNIPARSER_ENDPOINT=

# Priority 3: Replicate — has cold starts, warmup service mitigates this
REPLICATE_API_TOKEN=
OMNIPARSER_WARMUP_ENABLED=true
```

### Optional
```
REDIS_HOST=        # Enables OmniParser result caching (3-day TTL)
REDIS_PORT=6379
PORT=4000
```

## WebSocket Protocol

Send JSON messages to `ws://localhost:4000/ws/stream`:

```json
{
  "type": "llm_request",
  "id": "req_123",
  "payload": {
    "prompt": "User request: 'create a folder...'",
    "context": {
      "systemInstructions": "You are an automation planner. Output ONLY valid JSON..."
    }
  }
}
```

The backend streams back:
- `llm_stream_start` — stream beginning
- `llm_stream_chunk` — token chunks
- `llm_stream_end` — full text + provider + timing
- `error` — if all providers fail

## OmniParser Warmup

When `REPLICATE_API_TOKEN` + `OMNIPARSER_WARMUP_ENABLED=true` are set, the warmup service:
- Fires immediately on startup
- Pings the Replicate model every **3 minutes** to prevent cold boots
- Logs cold boot detection (>60s latency) with a recommendation to reduce interval

## File Structure

```
src/
  index.ts                      — Server entry point, WS server, graceful shutdown
  api/
    omniparser.ts               — /api/omniparser routes
    vision.ts                   — /api/vision routes (stub for screen-intelligence MCP)
  services/
    omniParserService.ts        — OmniParser API calls + Redis caching
    omniParserWarmup.ts         — Keep-hot service for Replicate
    llmElementMatcher.ts        — LLM-based UI element matching
  websocket/
    streamingHandler.ts         — Pure pass-through streaming handler
  utils/
    llmStreamingRouter.ts       — Multi-provider streaming (OpenAI, Claude, Gemini, etc.)
    llmRouter.ts                — Non-streaming LLM calls (used by element matcher)
    logger.ts                   — Winston logger
  types/
    streaming.ts                — All WebSocket message types
```

## Scripts

```bash
yarn dev        # ts-node-dev with hot reload
yarn build      # tsc compile to dist/
yarn start      # run compiled dist/index.js
yarn typecheck  # type check without emitting
```
