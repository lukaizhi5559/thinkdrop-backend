# Free LLM Provider & Model Catalog

> **Purpose**: Complete reference of all free LLM providers, their models, rate limits,
> intelligence scores, and API catalog endpoints. Used by the backend router to make
> intelligent routing decisions and track usage against known limits.
>
> **Last updated**: August 2026
> **Verification**: All model IDs and rate limits verified from official API endpoints and docs

---

## Table of Contents

1. [NVIDIA NIM](#nvidia-nim)
2. [Groq](#groq)
3. [GLM (z.ai)](#glm-zai)
4. [SambaNova](#sambanova)
5. [Gemini Free Tier](#gemini-free-tier)
6. [Cloudflare Workers AI](#cloudflare-workers-ai)
7. [OpenRouter](#openrouter)
8. [Intelligence Index Reference](#intelligence-index-reference)
9. [Intra-Provider Fallback Chains](#intra-provider-fallback-chains)
10. [Dynamic Catalog Refresh](#dynamic-catalog-refresh)

---

## NVIDIA NIM

**The largest free model catalog — 101+ models, 40 RPM shared cap**

| Property | Value |
|---|---|
| **API Base URL** | `https://integrate.api.nvidia.com/v1` |
| **Catalog Endpoint** | `https://integrate.api.nvidia.com/v1/models` |
| **Auth** | `Authorization: Bearer $NVIDIA_API_KEY` |
| **Env Var** | `NVIDIA_API_KEY` |
| **API Key URL** | https://build.nvidia.com/ (sign in → Get API Key) |
| **Free Tier** | Yes, NVIDIA Developer Program (free) |
| **Rate Limit** | **40 RPM hard cap** (shared across ALL models, cannot be increased) |
| **Rate Limit Type** | Per-account, not per-model |
| **Practical Daily Limit** | ~43,200 RPD (at sustained 30 RPM) |
| **Token Limits** | No published TPD limit — limited by RPM only |
| **API Compatibility** | OpenAI-compatible (`/v1/chat/completions`) |
| **Streaming** | Yes (`stream: true`) |
| **Credit System** | Retired — now per-model rate limits (but 40 RPM is global) |

### Key Free Models (verified from `/v1/models` endpoint)

#### HEAVY models (for planning, synthesis, complex tasks)

| Model ID | Intelligence | Context | Speed | Notes |
|---|---|---|---|---|
| `z-ai/glm-5.2` | **53** | 1M | 141 t/s | **Best free model.** Near Claude Opus level. MoE, reasoning |
| `nvidia/nemotron-3-ultra-550b-a55b` | ~40+ | — | — | 550B params, largest free model |
| `nvidia/nemotron-3-super-120b-a12b` | ~30+ | — | — | 120B MoE |
| `deepseek-ai/deepseek-v4-flash-0731` | ~30+ | — | — | Fast DeepSeek variant |
| `moonshotai/kimi-k2.6` | ~35+ | — | — | Kimi K2.6 |
| `openai/gpt-oss-120b` | **24** | 131K | 175 t/s | Good balance, open weights |
| `minimaxai/minimax-m3` | ~25+ | — | — | MiniMax M3 |
| `nvidia/llama-3.3-nemotron-super-49b-v1.5` | ~15+ | — | — | NVIDIA-tuned Llama |
| `nvidia/llama-3.1-nemotron-70b-instruct` | ~12+ | — | — | NVIDIA-tuned Llama 70B |
| `meta/llama-3.3-70b-instruct` | **9** | 128K | 82 t/s | Current model in code — **worst choice** |
| `mistralai/mistral-large-2-instruct` | ~20+ | — | — | Mistral Large 2 |
| `google/gemma-4-31b-it` | ~22 | 262K | 34 t/s | Gemma 4 31B |
| `01-ai/yi-large` | ~15+ | — | — | Yi Large |
| `writer/palmyra-creative-122b` | ~15+ | — | — | Writer Palmyra |
| `stepfun-ai/step-3.7-flash` | ~15+ | — | — | StepFun |
| `thinkingmachines/inkling` | ~15+ | 256K | — | Mamba-hybrid MoE |

#### LIGHT models (for heartbeats, classification, simple tasks)

| Model ID | Intelligence | Context | Speed | Notes |
|---|---|---|---|---|
| `meta/llama-3.1-8b-instruct` | ~5-10 | 128K | fast | Fast, good for simple tasks |
| `openai/gpt-oss-20b` | ~15 | 131K | 238 t/s | Smaller GPT-OSS, very fast |
| `nvidia/llama-3.1-nemotron-nano-8b-v1` | ~5-10 | — | fast | NVIDIA-tuned 8B |
| `nvidia/nemotron-3-nano-30b-a3b` | ~10+ | — | — | 30B MoE, 3B active |
| `meta/llama-3.2-3b-instruct` | ~5 | — | fast | Very small, very fast |
| `meta/llama-3.2-1b-instruct` | ~3 | — | fastest | Smallest model |
| `ibm/granite-3.0-8b-instruct` | ~5 | — | fast | IBM Granite |
| `mistralai/mistral-7b-instruct-v0.3` | ~5 | — | fast | Mistral 7B |
| `nv-mistralai/mistral-nemo-12b-instruct` | ~8 | — | fast | Mistral NeMo 12B |
| `nvidia/mistral-nemo-minitron-8b-8k-instruct` | ~5 | 8K | fast | Compressed NeMo |
| `nvidia/nemotron-mini-4b-instruct` | ~3 | — | fastest | Very small |
| `zyphra/zamba2-7b-instruct` | ~5 | — | fast | Zamba2 7B |

#### Specialized models (not for general chat)

| Model ID | Type | Notes |
|---|---|---|
| `nvidia/llama-3.1-nemoguard-8b-content-safety` | Safety | Content safety guard |
| `nvidia/llama-3.1-nemoguard-8b-topic-control` | Safety | Topic control guard |
| `nvidia/nemotron-3.5-content-safety` | Safety | Content safety |
| `meta/llama-guard-4-12b` | Safety | Llama Guard 4 |
| `nvidia/nv-embed-v1` | Embedding | Embedding model |
| `nvidia/nv-embedcode-7b-v1` | Embedding | Code embedding |
| `baai/bge-m3` | Embedding | BGE M3 embedding |
| `snowflake/arctic-embed-l` | Embedding | Arctic embedding |
| `nvidia/nemotron-parse` | Vision | Document parsing |
| `nvidia/nemotron-3-embed-1b` | Embedding | Small embedding |

#### Retired/404 models (in catalog but not hosted)

| Model ID | Status |
|---|---|
| `meta/llama2-70b` | 404 — retired |
| `databricks/dbrx-instruct` | 404 — retired |
| `adept/fuyu-8b` | 404 — retired |
| `nvidia/llama-3.1-nemotron-ultra-253b-v1` | Timeout — too large |
| `qwen/qwen3.5-397b-a17b` | Timeout — too large |

### NVIDIA Notes
- **40 RPM is a global hard cap** — shared across ALL models. Intra-provider model fallback
  helps with model errors (timeouts, empty responses) but NOT with rate limits.
- For rate limits, skip to the next provider immediately.
- No published TPD limit — the only constraint is the 40 RPM.
- The `/v1/models` endpoint returns all 101 models but doesn't distinguish free vs paid.
  All models with a valid chat completions response (HTTP 200 or 429) are free to use.

---

## Groq

**Fastest inference — LPU hardware, per-model rate limits**

| Property | Value |
|---|---|
| **API Base URL** | `https://api.groq.com/openai/v1` |
| **Catalog Endpoint** | `https://api.groq.com/openai/v1/models` (requires auth) |
| **Docs URL** | https://console.groq.com/docs/models |
| **Rate Limits URL** | https://console.groq.com/settings/limits |
| **Auth** | `Authorization: Bearer $GROQ_API_KEY` |
| **Env Var** | `GROQ_API_KEY` |
| **API Key URL** | https://console.groq.com/keys |
| **Free Tier** | Yes, free forever |
| **Rate Limit Type** | **Per-model** (each model has its own RPM/RPD/TPD) |
| **API Compatibility** | OpenAI-compatible |
| **Streaming** | Yes |

### Free Models (verified from Groq docs, March 2026)

| Model ID | Intelligence | RPM | RPD | TPD | TPM | Speed (t/s) | Context | Status |
|---|---|---|---|---|---|---|---|---|
| `llama-3.1-8b-instant` | ~5-10 | 30 | **14,400** | **500K** | 6,000 | 560-840 | 128K | Production |
| `openai/gpt-oss-20b` | ~15 | 30 | 1,000 | — | 8,000 | 1,000 | 128K | Production |
| `meta-llama/llama-4-scout-17b-16e-instruct` | ~15+ | 30 | 1,000 | 500K | 30,000 | 594-750 | 128K | Preview |
| `qwen/qwen3-32b` | ~20+ | 60 | 1,000 | — | 6,000 | 400-662 | 131K | Preview |
| `openai/gpt-oss-120b` | **24** | 30 | 1,000 | — | 8,000 | 500 | 128K | Production |
| `llama-3.3-70b-versatile` | **9** | 30 | 1,000 | **100K** | 12,000 | 280-394 | 128K | Production |
| `moonshotai/kimi-k2-instruct-0905` | ~40+ | 30 | — | — | — | — | — | — |

### Groq Notes
- **Per-model rate limits** — each model has its own RPM/RPD/TPD. Intra-provider fallback
  is highly effective: when `gpt-oss-120b` hits 30 RPM, switch to `qwen/qwen3-32b` (60 RPM!).
- `llama-3.1-8b-instant` has the most generous limits: **14,400 RPD, 500K TPD** — 5× more
  than any other Groq model.
- `qwen/qwen3-32b` has **60 RPM** — double the standard 30 RPM.
- Groq LPU hardware is the fastest: 560-1,000 t/s vs typical 50-100 t/s.
- Deprecated models: `gemma2-9b-it`, `llama3-8b-8192`, `llama3-70b-8192`, Mixtral.

---

## GLM (z.ai)

**Completely free — no published limits**

| Property | Value |
|---|---|
| **API Base URL** | `https://api.z.ai/api/paas/v4` |
| **Catalog Endpoint** | Not publicly documented |
| **Docs URL** | https://docs.z.ai/ |
| **Auth** | `Authorization: Bearer $GLM_API_KEY` |
| **Env Var** | `GLM_API_KEY` |
| **API Key URL** | https://z.ai/ (sign up → API keys) |
| **Free Tier** | **Yes — completely free** (input, output, caching all free) |
| **Rate Limit** | No published RPM/RPD/TPD limits |
| **API Compatibility** | OpenAI-compatible |
| **Streaming** | Yes |
| **Thinking Mode** | Supported (can be disabled via `thinking: { type: 'disabled' }`) |

### Free Models

| Model ID | Intelligence | Context | Speed | Params | Notes |
|---|---|---|---|---|---|
| `glm-4.7-flash` | **23** | 200K | 97 t/s | 31.2B (3B active MoE) | **Best truly free model.** Reasoning, MIT license |

### GLM Notes
- **Completely free** — no billing, no credits, no published limits.
- `glm-4.7-flash` is a reasoning model (31.2B params, 3B active MoE).
- Intelligence 23 — 2.6× smarter than Llama 3.3 70B (9).
- **Disable thinking mode** for faster responses: `thinking: { type: 'disabled' }`
- 200K context window — handles large prompts.
- Also available on NVIDIA as `z-ai/glm-5.2` (intelligence 53, newer version).
- Also available on Cloudflare as `@cf/zai-org/glm-4.7-flash` (free) and `@cf/zai-org/glm-5.2` (paid).

---

## SambaNova

**Fast RDU inference — 20 RPD free tier is the bottleneck**

| Property | Value |
|---|---|
| **API Base URL** | `https://api.sambanova.ai/v1` |
| **Catalog Endpoint** | `https://api.sambanova.ai/v1/models` (requires auth) |
| **Docs URL** | https://docs.sambanova.ai/docs/en/models/sambacloud-models |
| **Rate Limits URL** | https://docs.sambanova.ai/docs/en/models/rate-limits |
| **Pricing URL** | https://cloud.sambanova.ai/plans/pricing |
| **Auth** | `Authorization: Bearer $SAMBANOVA_API_KEY` |
| **Env Var** | `SAMBANOVA_API_KEY` |
| **API Key URL** | https://cloud.sambanova.ai/apis |
| **Free Tier** | Yes — forever free, no credit card required |
| **Rate Limit Type** | Shared across all models |
| **API Compatibility** | OpenAI-compatible |
| **Streaming** | Yes |

### Free Tier Rate Limits

| Limit | Free Tier | Developer Tier (paid) |
|---|---|---|
| RPM | 20 | 60-240 (per model) |
| RPD | **20** | 12,000-48,000 (per model) |
| TPD | 200,000 | Unlimited (pay per token) |

### Free Models (verified from SambaNova docs)

#### Production Models

| Model ID | Intelligence | Context | RPM (Free) | RPD (Free) | TPD (Free) | Notes |
|---|---|---|---|---|---|---|
| `gpt-oss-120b` | **24** | 128K | 20 | 20 | 200K | **Best SambaNova model.** Reasoning, function calling |
| `DeepSeek-V3.1` | ~25+ | 128K | 20 | 20 | 200K | DeepSeek V3.1, reasoning |
| `MiniMax-M2.7` | ~25+ | 192K | 20 | 20 | 200K | MiniMax M2.7, largest context |
| `Meta-Llama-3.3-70B-Instruct` | **9** | 128K | 20 | 20 | 200K | Current model — **worst choice** |

#### Preview Models

| Model ID | Intelligence | Context | RPM (Free) | RPD (Free) | TPD (Free) | Notes |
|---|---|---|---|---|---|---|
| `DeepSeek-V3.2` | ~30+ | 164K | 20 | 20 | 200K | Newer DeepSeek, may be removed |
| `gemma-4-31B-it` | ~22 | 262K | 20 | 20 | 200K | Gemma 4, largest context |

### SambaNova Notes
- **20 RPD is the binding constraint** — one user task makes 16-30 LLM calls, so
  SambaNova is exhausted in a single task.
- All models share the same 20 RPM / 20 RPD / 200K TPD quota.
- Intra-provider fallback doesn't help for rate limits (shared quota) but helps for
  model-specific errors.
- RDU hardware is very fast for inference.
- Deprecated models: `DeepSeek-V3-0324`, `DeepSeek-R1-0528`, `Meta-Llama-3.1-8B-Instruct`,
  `Qwen3-235B-A22B-Instruct-2507`, `Qwen3-32B`, `DeepSeek-R1-Distill-Llama-70B`.

---

## Gemini Free Tier

**Free Flash-Lite models — requires separate GCP project WITHOUT billing**

| Property | Value |
|---|---|
| **API Base URL** | `https://generativelanguage.googleapis.com/v1beta/openai` |
| **Catalog URL** | https://ai.google.dev/gemini-api/docs/models |
| **Pricing URL** | https://ai.google.dev/gemini-api/docs/pricing |
| **Rate Limits URL** | https://ai.google.dev/gemini-api/docs/rate-limits |
| **Auth** | `Authorization: Bearer $GEMINI_API_KEY_FREE` |
| **Env Var (free)** | `GEMINI_API_KEY_FREE` (from GCP project WITHOUT billing) |
| **Env Var (paid)** | `GEMINI_API_KEY` (from GCP project WITH billing) |
| **API Key URL** | https://aistudio.google.com/app/apikey |
| **Free Tier** | Yes — but ONLY if GCP project has NO billing enabled |
| **Rate Limit Type** | Per-project (not per-key) |
| **API Compatibility** | OpenAI-compatible |
| **Streaming** | Yes |
| **Reset Time** | Midnight Pacific Time (08:00 UTC) |

### ⚠️ Critical: Billing Deletes Free Tier
If your GCP project has billing enabled, you are automatically on the Paid tier.
**All Gemini calls cost money** ($1.50/M input, $7.50/M output for Gemini 3.6 Flash).
To use the free tier, create a NEW GCP project WITHOUT billing and generate a key from there.

### Free Tier Rate Limits (approximate, varies by model and region)

| Limit | Free Tier | Tier 1 (paid) | Tier 2 (paid) |
|---|---|---|---|
| RPM | 5-15 | Higher | Higher |
| RPD | 1,000-1,500 | Higher | Higher |
| TPM | 250,000 | Higher | Higher |
| Spend limit | N/A | $10/10min | $200/10min |

### Free Models (Flash and Flash-Lite only — Pro removed April 2026)

| Model ID | Intelligence | RPM | RPD | Speed (t/s) | Context | Input Price | Output Price |
|---|---|---|---|---|---|---|---|
| `gemini-3.5-flash-lite` | **37** | ~15 | ~1,500 | 397 | 1M | Free | Free |
| `gemini-3.1-flash-lite` | ~30+ | ~15 | ~1,500 | fast | 1M | Free | Free |
| `gemini-2.5-flash-lite` | ~25+ | ~15 | ~1,000 | fast | 1M | Free | Free |
| `gemini-3-flash-preview` | ~35+ | ~10 | ~1,500 | fast | 1M | Free | Free |

### Paid Models (for fallback — costs money)

| Model ID | Intelligence | Speed (t/s) | Input $/M | Output $/M |
|---|---|---|---|---|
| `gemini-3.6-flash` | **52** | 238 | $1.50 | $7.50 |
| `gemini-3.5-flash` | **47** | 188 | $1.50 | $9.00 |
| `gemini-flash-latest` | 52 | 238 | $1.50 | $7.50 | (alias for 3.6 Flash) |

### Gemini Notes
- Free tier only covers Flash and Flash-Lite models (Pro removed April 2026).
- `gemini-3.5-flash-lite` (intelligence 37) is **4× smarter than Llama 3.3 70B** and
  at 397 t/s is one of the fastest models available.
- Rate limits are per-project, not per-key. Generating extra keys in the same project
  does NOT add quota.
- RPD resets at midnight Pacific Time (08:00 UTC).
- Data on free tier may be used to improve Google's products.

---

## Cloudflare Workers AI

**10K free neurons/day — GLM-4.7-Flash is the standout free model**

| Property | Value |
|---|---|
| **API Base URL** | `https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/ai/v1` |
| **Catalog URL** | https://developers.cloudflare.com/workers-ai/models/ |
| **Pricing URL** | https://developers.cloudflare.com/workers-ai/platform/pricing/ |
| **Auth** | `Authorization: Bearer $CLOUDFLARE_API_TOKEN` |
| **Env Var** | `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` |
| **API Key URL** | https://dash.cloudflare.com/profile/api-tokens |
| **Free Tier** | Yes — 10,000 Neurons/day free |
| **Rate Limit** | 300 RPM |
| **API Compatibility** | OpenAI-compatible |
| **Streaming** | Yes |

### Free Tier

| Plan | Free Neurons/day | Pricing above free |
|---|---|---|
| Workers Free | 10,000 | N/A — upgrade required |
| Workers Paid ($5/mo) | 10,000 | $0.011 / 1,000 Neurons |

### Free Models (available on Workers Free plan)

| Model ID | Neurons/M input | Neurons/M output | Est. output tokens (10K free) | Notes |
|---|---|---|---|---|
| `@cf/zai-org/glm-4.7-flash` | low | low | ~500+ | **Best free Cloudflare model.** GLM-4.7-Flash |
| `@cf/meta/llama-3.2-1b-instruct` | 2,457 | 18,252 | ~547 | Smallest, cheapest |
| `@cf/meta/llama-3.2-3b-instruct` | 4,625 | 30,475 | ~328 | Small |
| `@cf/meta/llama-3.1-8b-instruct-fp8-fast` | 4,119 | 34,868 | ~287 | Fast 8B |
| `@cf/meta/llama-3.1-8b-instruct-fp8` | 13,778 | 26,128 | ~383 | FP8 8B |
| `@cf/meta/llama-3.1-8b-instruct-awq` | — | — | — | AWQ 8B |
| `@cf/meta/llama-3.1-8b-instruct` | 25,608 | 75,147 | ~133 | Non-quantized 8B |
| `@cf/nvidia/nemotron-3-120b-a12b` | — | — | — | NVIDIA Nemotron 120B |
| `@cf/google/gemma-4-26b-a4b-it` | — | — | — | Gemma 4 26B |

### Paid-Only Models (require Workers Paid plan as of July 28, 2026)

| Model ID | Notes |
|---|---|
| `@cf/zai-org/glm-5.2` | GLM-5.2 — now paid only |
| `@cf/moonshotai/kimi-k2.6` | Kimi K2.6 — now paid only |
| `@cf/moonshotai/kimi-k2.7-code` | Kimi K2.7 Code — now paid only |
| `@cf/meta/llama-3.3-70b-instruct-fp8-fast` | 70B — expensive in neurons (204K/M output) |
| `@cf/meta/llama-3.1-70b-instruct` | 70B — deprecated May 2026 |

### Cloudflare Notes
- **10K neurons/day is very limiting** for large models. The 70B model costs 204,805 neurons
  per million output tokens — meaning 10K free neurons = ~48 output tokens. Unusable.
- **`@cf/zai-org/glm-4.7-flash` is the standout** — GLM-4.7-Flash on Cloudflare's free tier.
  Low neuron cost, intelligence 23.
- **`@cf/meta/llama-3.1-8b-instruct-fp8-fast`** is the best budget option — 34,868 neurons/M
  output = ~287 output tokens on free tier. Usable for short responses.
- Context window is limited to 24K tokens (not the full 128K).
- GLM-5.2, Kimi K2.6, and Kimi K2.7 Code moved to paid-only on July 28, 2026.

---

## OpenRouter

**Free models available — 50 RPD without credits**

| Property | Value |
|---|---|
| **API Base URL** | `https://openrouter.ai/api/v1` |
| **Catalog URL** | https://openrouter.ai/models |
| **Docs URL** | https://openrouter.ai/docs |
| **Auth** | `Authorization: Bearer $OPENROUTER_API_KEY` |
| **Env Var** | `OPENROUTER_API_KEY` (optional) |
| **API Key URL** | https://openrouter.ai/keys |
| **Free Tier** | Yes — free models available |
| **Rate Limit (no credits)** | 20 RPM, **50 RPD** |
| **Rate Limit ($10+ credits)** | 20 RPM, 1,000 RPD |
| **API Compatibility** | OpenAI-compatible |
| **Streaming** | Yes |

### Free Models (suffix `:free`)

| Model ID | Intelligence | Notes |
|---|---|---|
| `meta-llama/llama-3.3-70b-instruct:free` | 9 | Llama 3.3 70B |
| `google/gemini-2.5-flash-lite-preview:free` | ~25+ | Gemini Flash-Lite |
| `qwen/qwen-2.5-72b-instruct:free` | ~15+ | Qwen 2.5 72B |
| `deepseek/deepseek-r1:free` | ~30+ | DeepSeek R1 (reasoning) |

### OpenRouter Notes
- 50 RPD without credits is very restrictive (one user task = 16-30 calls).
- With $10+ credits, jumps to 1,000 RPD.
- Free models may have higher latency (lower priority on shared infrastructure).
- Good as a last-resort fallback.

---

## Intelligence Index Reference

**Artificial Analysis Intelligence Index v4.1.1** — 0-100 scale

### Methodology
Weighted average of 9 independent benchmarks:
- **Agents (34%)**: GDPval-AA v2 (20%), τ³-Banking (14%)
- **Coding (24%)**: Terminal-Bench v2.1 (16%), SciCode (8%)
- **Scientific Reasoning (24%)**: Humanity's Last Exam (12%), GPQA Diamond (6%)
- **General (18%)**: AA-Omniscience Accuracy (8%) + Non-Hallucination (4%), AA-LCR (6%), CritPt (6%)

### Score Reference

| Score | Level | Example Models |
|---|---|---|
| 60-65 | Frontier | Claude Opus 5 (63), GPT-5.6 Sol (61) |
| 50-59 | Near-frontier | **GLM-5.2 max (53)**, Gemini 3.6 Flash (52) |
| 40-49 | Strong | Gemini 3.5 Flash (47), GLM-5.2 (35-53) |
| 30-39 | Good | **Gemini 3.5 Flash-Lite (37)**, GLM-5.2 non-reasoning (35) |
| 20-29 | Above average | **GLM-4.7-Flash (23)**, gpt-oss-120b (24), Gemma 4 31B (22) |
| 10-19 | Average | Llama 3.3 70B Nemotron (12-15), gpt-oss-20b (15) |
| 1-9 | Below average | **Llama 3.3 70B (9)**, Gemma 4 E4B (9) |

### Source
- **Leaderboard**: https://artificialanalysis.ai/leaderboards/models
- **Methodology**: https://artificialanalysis.ai/methodology/intelligence-benchmarking
- **Index**: https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index

---

## Intra-Provider Fallback Chains

### HEAVY chain (planning, synthesis, complex skills >5K chars)

```
nvidia:     z-ai/glm-5.2 (53) → nemotron-3-ultra-550b (40+) → deepseek-v4-flash (30+) → gpt-oss-120b (24)
groq:       gpt-oss-120b (24) → qwen3-32b (20+) → llama-3.3-70b (9)
glm:        glm-4.7-flash (23)
sambanova:  gpt-oss-120b (24) → DeepSeek-V3.1 (25+) → MiniMax-M2.7 (25+)
gemini-free: gemini-3.5-flash-lite (37) → gemini-3.1-flash-lite (30+)
cloudflare: @cf/zai-org/glm-4.7-flash (23) → @cf/meta/llama-3.3-70b-instruct-fp8-fast (9)
```

### LIGHT chain (heartbeats, simple skill steps <5K chars, classification)

```
groq:       llama-3.1-8b-instant (5-10) → gpt-oss-20b (15) → llama-4-scout-17b (15+)
nvidia:     llama-3.1-8b-instruct (5-10) → gpt-oss-20b (15) → nemotron-nano-8b (5-10)
glm:        glm-4.7-flash (23)
cloudflare: @cf/zai-org/glm-4.7-flash (23) → llama-3.1-8b-fp8-fast → llama-3.2-3b
gemini-free: gemini-3.5-flash-lite (37) → gemini-3.1-flash-lite (30+)
sambanova:  gpt-oss-120b (24) → Meta-Llama-3.3-70B (9)
```

### PAID chain (fallback when all free providers fail)

```
gemini-paid: gemini-3.6-flash (52)
deepseek:    deepseek-chat
mistral:     mistral-large
grok:        grok-2
openai:      gpt-4o / gpt-5
claude:      claude-sonnet / claude-opus
```

---

## Dynamic Catalog Refresh

Several providers expose `/v1/models` endpoints that return their full model catalog.
The backend can call these on startup to build/update a local cache.

### Endpoints

| Provider | Endpoint | Auth Required | Returns Free/Paid Info? |
|---|---|---|---|
| NVIDIA | `GET https://integrate.api.nvidia.com/v1/models` | Yes | No — all models returned, must probe |
| Groq | `GET https://api.groq.com/openai/v1/models` | Yes | No — all models returned |
| SambaNova | `GET https://api.sambanova.ai/v1/models` | Yes | No |
| Gemini | `GET https://generativelanguage.googleapis.com/v1beta/models` | Yes | No |
| Cloudflare | `GET https://api.cloudflare.com/client/v4/accounts/{id}/ai/models/search` | Yes | No |
| OpenRouter | `GET https://openrouter.ai/api/v1/models` | No | **Yes** — includes pricing & free flag |

### OpenRouter as a meta-catalog
OpenRouter's `/v1/models` endpoint is the most useful for catalog building — it returns
all models across all providers with pricing info and a `:free` suffix for free models.
No auth required. Could be used to discover new free models automatically.

### NVIDIA Model Probing
NVIDIA's `/v1/models` returns 101 models but doesn't indicate which are free/hosted.
To determine which are available:
1. Send a minimal chat completion (`max_tokens: 8`) to each model
2. HTTP 200 = hosted and working
3. HTTP 429 = rate-limited but hosted
4. HTTP 404/403/500 = not available

### Recommended Refresh Strategy
1. **On startup**: Fetch all provider catalogs, build local cache
2. **Daily**: Refresh catalogs to detect new models or deprecations
3. **On 429**: Mark model as rate-limited, try next model in chain
4. **On 404**: Mark model as retired, remove from chain
5. **On success**: Track token usage against known TPD limits

---

## Summary: Best Free Models by Use Case

| Use Case | Best Model | Intelligence | Provider | Why |
|---|---|---|---|---|
| **Complex planning** | `z-ai/glm-5.2` | 53 | NVIDIA | Near Claude Opus, free 40 RPM |
| **Fast planning** | `gemini-3.5-flash-lite` | 37 | Gemini free | 397 t/s, 1,500 RPD |
| **General tasks** | `glm-4.7-flash` | 23 | z.ai | Completely free, no limits |
| **High-volume light** | `llama-3.1-8b-instant` | 5-10 | Groq | 14,400 RPD, 500K TPD, 840 t/s |
| **Heartbeats** | `llama-3.1-8b-instant` | 5-10 | Groq | Fastest, highest free RPD |
| **Fallback heavy** | `gpt-oss-120b` | 24 | Groq/SambaNova/NVIDIA | Available on 3 providers |
| **Last resort** | `llama-3.3-70b-versatile` | 9 | Groq | Always available, low intelligence |
