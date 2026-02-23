import Replicate from 'replicate';
import { createHash } from 'crypto';
import { logger } from '../utils/logger';
import Redis from 'ioredis';
import axios from 'axios';
import fs from 'fs';
import path from 'path';
import { LLMElementMatcher } from './llmElementMatcher';

const replicateClient = process.env.REPLICATE_API_TOKEN
  ? new Replicate({ auth: process.env.REPLICATE_API_TOKEN })
  : null;

const MODAL_API_KEY = process.env.MODAL_API_KEY;
const MODAL_ENDPOINT = process.env.MODAL_OMNIPARSER_ENDPOINT;
const USE_MODAL = !!MODAL_API_KEY && !!MODAL_ENDPOINT;

const redis = process.env.REDIS_HOST
  ? new Redis({
      host: process.env.REDIS_HOST,
      port: parseInt(process.env.REDIS_PORT || '6379'),
      retryStrategy: (times) => Math.min(times * 50, 2000),
    })
  : null;

interface OmniParserElement {
  type: 'text' | 'icon';
  bbox: [number, number, number, number];
  interactivity: boolean;
  content: string;
}

interface OmniParserResponse {
  img: string;
  elements: string;
}

interface ParsedElement {
  id: number;
  type: 'text' | 'icon';
  bbox: { x1: number; y1: number; x2: number; y2: number };
  normalizedBbox: [number, number, number, number];
  interactivity: boolean;
  content: string;
  confidence: number;
}

interface CachedElements {
  elements: ParsedElement[];
  timestamp: number;
  url?: string;
  screenshotHash: string;
  screenshotWidth: number;
  screenshotHeight: number;
  windowBounds?: { x: number; y: number; width: number; height: number };
}

export interface OmniParserDetectionResult {
  coordinates: { x: number; y: number };
  confidence: number;
  method: 'omniparser' | 'omniparser_cached' | 'vision_fallback';
  selectedElement?: string;
  cacheHit?: boolean;
  allElements?: ParsedElement[];
}

export class OmniParserService {
  private readonly CACHE_TTL_SECONDS = 3 * 24 * 60 * 60;
  private readonly CACHE_PREFIX = 'omniparser';
  private llmMatcher: LLMElementMatcher;

  constructor() {
    this.llmMatcher = new LLMElementMatcher();
  }

  async detectElement(
    screenshot: { base64: string; mimeType: string },
    description: string,
    context: any
  ): Promise<OmniParserDetectionResult> {
    if (!this.isAvailable()) {
      throw new Error('OmniParser not available — configure HUGGINGFACE_OMNIPARSER_ENDPOINT, MODAL_OMNIPARSER_ENDPOINT, or REPLICATE_API_TOKEN');
    }

    logger.info('🔍 [OMNIPARSER] Starting element detection', {
      description,
      hasWindowBounds: !!context?.windowBounds,
      fetchAllElements: description === 'fetch_all_elements',
    });

    // Ensure model is warm before expensive API call
    const { omniParserWarmup } = await import('./omniParserWarmup');
    if (!omniParserWarmup.isWarm()) {
      logger.warn('⚠️ [OMNIPARSER] Model is cold, triggering warmup first');
      const warmupResult = await omniParserWarmup.ensureWarm();
      logger.info('✅ [OMNIPARSER] Warmup complete', { wasWarm: warmupResult.wasWarm, latencyMs: warmupResult.latencyMs });
    }

    const screenshotHash = this.hashScreenshot(screenshot.base64);
    const url = context?.url || context?.activeUrl || 'unknown';
    const cacheKey = this.getCacheKey(url, screenshotHash);

    const cached = await this.getFromCache(cacheKey);
    if (cached) {
      logger.info('✅ [OMNIPARSER] Cache hit', { cacheKey, elementCount: cached.elements.length, age: Math.round((Date.now() - cached.timestamp) / 1000) + 's' });

      if (description === 'fetch_all_elements') {
        return { coordinates: { x: 0, y: 0 }, confidence: 1.0, method: 'omniparser_cached', cacheHit: true, allElements: cached.elements };
      }

      const result = await this.findElementInCache(cached, description, context);
      if (result) return { ...result, method: 'omniparser_cached', cacheHit: true };

      logger.warn('⚠️ [OMNIPARSER] Element not found in cache, calling API');
    }

    logger.info('📡 [OMNIPARSER] Calling API', { cacheKey, reason: cached ? 'element_not_found' : 'cache_miss' });

    const elements = await this.callOmniParserAPI(screenshot, context);

    await this.saveToCache(cacheKey, {
      elements,
      timestamp: Date.now(),
      url,
      screenshotHash,
      screenshotWidth: context?.screenshotWidth || context?.screenWidth || 1440,
      screenshotHeight: context?.screenshotHeight || context?.screenHeight || 900,
      windowBounds: context?.windowBounds,
    });

    if (description === 'fetch_all_elements') {
      return { coordinates: { x: 0, y: 0 }, confidence: 1.0, method: 'omniparser', cacheHit: false, allElements: elements };
    }

    const result = await this.findElementInCache(
      {
        elements,
        timestamp: Date.now(),
        url,
        screenshotHash,
        screenshotWidth: context?.screenshotWidth || context?.screenWidth || 1440,
        screenshotHeight: context?.screenshotHeight || context?.screenHeight || 900,
        windowBounds: context?.windowBounds,
      },
      description,
      context
    );

    if (!result) throw new Error(`Element not found: ${description}`);

    return { ...result, method: 'omniparser', cacheHit: false };
  }

  private async callOmniParserAPI(
    screenshot: { base64: string; mimeType: string },
    context: any
  ): Promise<ParsedElement[]> {
    const startTime = Date.now();

    // Priority 1: Hugging Face Gradio (no cold starts)
    const HF_ENDPOINT = process.env.HUGGINGFACE_OMNIPARSER_ENDPOINT;
    if (HF_ENDPOINT) {
      try {
        logger.info('📡 [OMNIPARSER] Calling Hugging Face Gradio API', { endpoint: HF_ENDPOINT });
        const imageDataUri = `data:${screenshot.mimeType};base64,${screenshot.base64}`;
        const response = await axios.post(
          HF_ENDPOINT,
          { data: [imageDataUri, 0.05, 0.1] },
          { headers: { 'Content-Type': 'application/json' }, timeout: 60000 }
        );
        const latencyMs = Date.now() - startTime;
        logger.info('✅ [OMNIPARSER] Hugging Face call successful', { latencyMs, provider: 'huggingface' });
        const elementsString = response.data.data[0];
        return this.parseOmniParserElements(
          elementsString,
          context?.screenshotWidth || context?.screenWidth || 1440,
          context?.screenshotHeight || context?.screenHeight || 900,
          context?.windowBounds
        );
      } catch (error: any) {
        logger.warn('⚠️ [OMNIPARSER] Hugging Face call failed, trying Modal.com', { error: error.message });
      }
    }

    // Priority 2: Modal.com serverless GPU
    if (USE_MODAL) {
      try {
        logger.info('📡 [OMNIPARSER] Calling Modal.com serverless endpoint', { endpoint: MODAL_ENDPOINT });
        const response = await axios.post(
          MODAL_ENDPOINT!,
          { image: screenshot.base64, imgsz: 640, box_threshold: 0.05, iou_threshold: 0.1 },
          { headers: { Authorization: `Bearer ${MODAL_API_KEY}`, 'Content-Type': 'application/json' }, timeout: 60000 }
        );
        const latencyMs = Date.now() - startTime;
        logger.info('✅ [OMNIPARSER] Modal.com call successful', { latencyMs, provider: 'modal' });
        const output = response.data as OmniParserResponse;
        return this.parseOmniParserElements(
          output.elements,
          context?.screenshotWidth || context?.screenWidth || 1440,
          context?.screenshotHeight || context?.screenHeight || 900,
          context?.windowBounds
        );
      } catch (error: any) {
        logger.warn('⚠️ [OMNIPARSER] Modal.com call failed, falling back to Replicate', { error: error.message });
      }
    }

    // Priority 3: Replicate (fallback — has cold starts, warmup mitigates this)
    if (!replicateClient) throw new Error('No OmniParser provider configured');

    logger.info('📡 [OMNIPARSER] Calling Replicate API (fallback)');
    const imageDataUri = `data:${screenshot.mimeType};base64,${screenshot.base64}`;
    const TIMEOUT_MS = 180000;

    const timeoutPromise = new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error(`OmniParser API timeout after ${TIMEOUT_MS}ms`)), TIMEOUT_MS)
    );

    const apiPromise = replicateClient.run(
      'microsoft/omniparser-v2:49cf3d41b8d3aca1360514e83be4c97131ce8f0d99abfc365526d8384caa88df',
      { input: { image: imageDataUri, imgsz: 640, box_threshold: 0.05, iou_threshold: 0.1 } }
    );

    const output = (await Promise.race([apiPromise, timeoutPromise])) as OmniParserResponse;
    const latencyMs = Date.now() - startTime;
    logger.info('✅ [OMNIPARSER] Replicate call successful', { latencyMs, provider: 'replicate' });

    // Save debug output
    try {
      const debugDir = path.join(process.cwd(), 'omniparser-debug');
      if (!fs.existsSync(debugDir)) fs.mkdirSync(debugDir, { recursive: true });
      const ts = new Date().toISOString().replace(/[:.]/g, '-');
      fs.writeFileSync(path.join(debugDir, `omniparser-${ts}.json`), JSON.stringify({ timestamp: new Date().toISOString(), latencyMs, rawResponse: output }, null, 2));
    } catch { /* non-fatal */ }

    return this.parseOmniParserElements(
      output.elements,
      context?.screenshotWidth || context?.screenWidth || 1440,
      context?.screenshotHeight || context?.screenHeight || 900,
      context?.windowBounds
    );
  }

  private parseOmniParserElements(
    elementsString: string,
    screenshotWidth: number,
    screenshotHeight: number,
    windowBounds?: { x: number; y: number; width: number; height: number }
  ): ParsedElement[] {
    const elements: ParsedElement[] = [];
    const lines = elementsString.split('\n');

    for (const line of lines) {
      const match = line.match(/icon (\d+): ({.*})/);
      if (!match) continue;

      const id = parseInt(match[1]);
      let jsonStr = match[2];

      jsonStr = jsonStr.replace(/\bFalse\b/g, 'false').replace(/\bTrue\b/g, 'true');

      const contentRegex = /'content':\s*'(.+?)'}$/;
      const contentMatch = jsonStr.match(contentRegex);
      if (contentMatch) {
        const rawContent = contentMatch[1];
        const escapedContent = rawContent
          .replace(/\\/g, '\\\\')
          .replace(/"/g, '\\"')
          .replace(/\n/g, '\\n')
          .replace(/\r/g, '\\r')
          .replace(/\t/g, '\\t');
        jsonStr = jsonStr.replace(contentRegex, `"content": "${escapedContent}"}`);
      }

      jsonStr = jsonStr.replace(/'/g, '"');

      try {
        const data = JSON.parse(jsonStr) as OmniParserElement;
        const normalizedBbox = data.bbox;
        const absoluteBbox = {
          x1: normalizedBbox[0] * screenshotWidth,
          y1: normalizedBbox[1] * screenshotHeight,
          x2: normalizedBbox[2] * screenshotWidth,
          y2: normalizedBbox[3] * screenshotHeight,
        };

        if (windowBounds) {
          absoluteBbox.x1 += windowBounds.x;
          absoluteBbox.y1 += windowBounds.y;
          absoluteBbox.x2 += windowBounds.x;
          absoluteBbox.y2 += windowBounds.y;
        }

        elements.push({ id, type: data.type, bbox: absoluteBbox, normalizedBbox, interactivity: data.interactivity, content: data.content, confidence: 0.9 });
      } catch (error: any) {
        logger.warn('⚠️ [OMNIPARSER] Failed to parse element line', { line, error: error.message });
      }
    }

    return this.mergeIconTextPairs(elements, screenshotWidth, screenshotHeight);
  }

  /**
   * Spatial element merging: pairs icon elements with nearby text elements.
   *
   * OmniParser returns icon and text label as separate elements. For desktop icons,
   * sidebar items, toolbar buttons etc., the icon and its label are spatially adjacent.
   * Merging them produces a single element with content = "icon_content (text_label)"
   * so the LLM matcher can find "hello-world folder" by matching the merged content
   * instead of having to choose between the icon (no label) and the text (no position).
   *
   * Rules:
   *  - Only merge icon → text (not text → text, not icon → icon)
   *  - Text must be within MERGE_DISTANCE normalized units of the icon center
   *  - Text must be below or overlapping the icon (label is usually below the icon)
   *  - Each text element can only be consumed by one icon (closest wins)
   *  - Merged icons get content = "icon_content (text_label)" or just text_label if
   *    icon content is generic ("unanswerable", "a folder.", etc.)
   *  - Consumed text elements are removed from the output
   */
  private mergeIconTextPairs(elements: ParsedElement[], screenshotWidth: number, screenshotHeight: number): ParsedElement[] {
    // Only merge when icon content is generic/unhelpful — if OmniParser already gave the icon
    // a meaningful name (e.g. "hello-world"), don't overwrite it with nearby text.
    const GENERIC_ICON_CONTENT = /^(unanswerable|a folder\.|a file folder\.|a bookmark\.|an arrow|a loading screen|a symbol|a stop button|the number|the "not" function|the power button|the time or date|the "refresh" function|adding a new|image blank|remote|a text box|a user profile|a low profile|the option to close|the 3-point view|a tool for writing)$/i;

    // Text elements that are NOT useful labels (clock, date, single chars, log noise)
    const NOISE_TEXT = /^(\d{1,2}:\d{2}(am|pm)?|mon|tue|wed|thu|fri|sat|sun|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d+|\[|\]|\{|\}|>|<|\/|\\|\|)$/i;

    const icons = elements.filter((e) => e.type === 'icon');
    const texts = elements.filter((e) => e.type === 'text');
    const consumedTextIds = new Set<number>();

    for (const icon of icons) {
      const iconContent = icon.content.trim();

      // Only attempt merge if icon content is generic — meaningful icon names are kept as-is
      if (!GENERIC_ICON_CONTENT.test(iconContent) && iconContent.length >= 4) continue;

      const iconCx = (icon.normalizedBbox[0] + icon.normalizedBbox[2]) / 2;
      const iconBottom = icon.normalizedBbox[3]; // bottom edge of icon

      // Find candidate text elements: must be DIRECTLY BELOW the icon (not beside it),
      // horizontally centered under the icon, and not noise text
      const LABEL_MAX_DX = 0.08;  // text center within 8% horizontally of icon center
      const LABEL_MAX_DY = 0.06;  // text top edge within 6% below icon bottom edge

      const candidates = texts
        .filter((t) => {
          if (consumedTextIds.has(t.id)) return false;
          const label = t.content.trim();
          if (NOISE_TEXT.test(label)) return false;
          if (label.length < 2) return false;

          const tCx = (t.normalizedBbox[0] + t.normalizedBbox[2]) / 2;
          const tTop = t.normalizedBbox[1];
          const dx = Math.abs(tCx - iconCx);
          const dy = tTop - iconBottom; // positive = text starts below icon bottom

          // Text must be horizontally centered under icon AND start just below icon bottom
          return dx <= LABEL_MAX_DX && dy >= -0.02 && dy <= LABEL_MAX_DY;
        })
        .sort((a, b) => {
          const aCx = (a.normalizedBbox[0] + a.normalizedBbox[2]) / 2;
          const bCx = (b.normalizedBbox[0] + b.normalizedBbox[2]) / 2;
          return Math.abs(aCx - iconCx) - Math.abs(bCx - iconCx);
        });

      if (candidates.length === 0) continue;

      const bestText = candidates[0];
      consumedTextIds.add(bestText.id);

      const textLabel = bestText.content.trim();
      icon.content = textLabel;

      logger.debug('🔗 [OMNIPARSER] Merged icon+text', {
        iconId: icon.id,
        textId: bestText.id,
        original: iconContent,
        merged: textLabel,
      });
    }

    // Remove consumed text elements — they are now encoded in their paired icon
    const result = elements.filter((e) => !consumedTextIds.has(e.id));

    const mergedCount = consumedTextIds.size;
    if (mergedCount > 0) {
      logger.info(`🔗 [OMNIPARSER] Merged ${mergedCount} icon+text pairs`, { before: elements.length, after: result.length });
    }

    return result;
  }

  private async findElementInCache(
    cached: CachedElements,
    description: string,
    context: any
  ): Promise<{ coordinates: { x: number; y: number }; confidence: number; selectedElement?: string } | null> {
    if (!description || typeof description !== 'string') return null;

    logger.info('🤖 [OMNIPARSER] Using LLM-based element matching', { description, elementCount: cached.elements.length });

    const matchResult = await this.llmMatcher.matchElementWithRetry(description, cached.elements, {
      intentType: context.intentType,
      activeApp: context.activeApp,
      activeUrl: context.activeUrl,
      screenshotWidth: cached.screenshotWidth,
      screenshotHeight: cached.screenshotHeight,
      maxRetries: 3,
    });

    if (!matchResult.element) {
      logger.warn('⚠️ [OMNIPARSER] LLM matcher found no suitable element', { description, reasoning: matchResult.reasoning });
      return null;
    }

    const center = {
      x: Math.round((matchResult.element.bbox.x1 + matchResult.element.bbox.x2) / 2),
      y: Math.round((matchResult.element.bbox.y1 + matchResult.element.bbox.y2) / 2),
    };

    logger.info('✅ [OMNIPARSER] LLM matched element', { description, matched: matchResult.element.content, coordinates: center, confidence: matchResult.confidence });

    return { coordinates: center, confidence: matchResult.confidence, selectedElement: matchResult.element.content };
  }

  async invalidateCache(url?: string, screenshotHash?: string): Promise<void> {
    if (!redis) return;
    try {
      if (url && screenshotHash) {
        await redis.del(this.getCacheKey(url, screenshotHash));
      } else if (url) {
        const keys = await redis.keys(`${this.CACHE_PREFIX}:${url}:*`);
        if (keys.length > 0) await redis.del(...keys);
      }
    } catch (error: any) {
      logger.error('❌ [OMNIPARSER] Cache invalidation failed', { error: error.message });
    }
  }

  private async getFromCache(cacheKey: string): Promise<CachedElements | null> {
    if (!redis) return null;
    try {
      const cached = await redis.get(cacheKey);
      if (!cached) return null;
      const data = JSON.parse(cached) as CachedElements;
      const age = Date.now() - data.timestamp;
      if (age > this.CACHE_TTL_SECONDS * 1000) {
        await redis.del(cacheKey);
        return null;
      }
      return data;
    } catch (error: any) {
      logger.error('❌ [OMNIPARSER] Cache read failed', { cacheKey, error: error.message });
      return null;
    }
  }

  private async saveToCache(cacheKey: string, data: CachedElements): Promise<void> {
    if (!redis) return;
    try {
      await redis.setex(cacheKey, this.CACHE_TTL_SECONDS, JSON.stringify(data));
      logger.info('💾 [OMNIPARSER] Cached elements', { cacheKey, elementCount: data.elements.length });
    } catch (error: any) {
      logger.error('❌ [OMNIPARSER] Cache write failed', { cacheKey, error: error.message });
    }
  }

  private getCacheKey(url: string, screenshotHash: string): string {
    return `${this.CACHE_PREFIX}:${url}:${screenshotHash}`;
  }

  private hashScreenshot(base64: string): string {
    return createHash('sha256').update(base64).digest('hex').substring(0, 16);
  }

  isAvailable(): boolean {
    return !!(process.env.HUGGINGFACE_OMNIPARSER_ENDPOINT || USE_MODAL || replicateClient);
  }
}

export const omniParserService = new OmniParserService();
