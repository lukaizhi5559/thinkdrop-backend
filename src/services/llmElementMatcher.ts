import { logger } from '../utils/logger';
import { LLMRouter } from '../utils/llmRouter';

interface ParsedElement {
  id: number;
  type: 'text' | 'icon';
  bbox: { x1: number; y1: number; x2: number; y2: number };
  normalizedBbox: [number, number, number, number];
  interactivity: boolean;
  content: string;
  confidence: number;
}

interface ElementMatchResult {
  element: ParsedElement | null;
  confidence: number;
  reasoning: string;
  attemptNumber?: number;
  excludedCount?: number;
}

interface MatchOptions {
  intentType?: string;
  activeApp?: string;
  activeUrl?: string;
  screenshotWidth?: number;
  screenshotHeight?: number;
  windowBounds?: { x: number; y: number; width: number; height: number };
  excludedElementIds?: number[];
  maxRetries?: number;
}

export class LLMElementMatcher {
  private llmRouter: LLMRouter;

  constructor() {
    this.llmRouter = new LLMRouter();
  }

  async matchElementWithRetry(
    description: string,
    elements: ParsedElement[],
    options: MatchOptions = {}
  ): Promise<ElementMatchResult> {
    const maxRetries = options.maxRetries || 3;
    const excludedIds: number[] = options.excludedElementIds || [];

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      logger.info('🔄 [LLM_MATCHER] Match attempt', { description, attempt, maxRetries, excludedCount: excludedIds.length });

      const result = await this.matchElement(description, elements, { ...options, excludedElementIds: excludedIds });

      if (result.element) {
        return { ...result, attemptNumber: attempt, excludedCount: excludedIds.length };
      }

      if (attempt < maxRetries) {
        logger.warn('⚠️ [LLM_MATCHER] No match on attempt, will retry', { description, attempt, remainingRetries: maxRetries - attempt });
      }
    }

    return { element: null, confidence: 0, reasoning: `No match found after ${maxRetries} attempts`, attemptNumber: maxRetries, excludedCount: excludedIds.length };
  }

  async matchElement(
    description: string,
    elements: ParsedElement[],
    options: MatchOptions = {}
  ): Promise<ElementMatchResult> {
    if (elements.length === 0) {
      return { element: null, confidence: 0, reasoning: 'No elements provided' };
    }

    const excludedIds = options.excludedElementIds || [];
    const availableElements = excludedIds.length > 0 ? elements.filter((e) => !excludedIds.includes(e.id)) : elements;

    if (availableElements.length === 0) {
      return { element: null, confidence: 0, reasoning: 'All elements have been excluded from previous attempts' };
    }

    const candidates = this.preFilterCandidates(description, availableElements, options);

    if (candidates.length === 0) {
      return { element: null, confidence: 0, reasoning: 'No matching candidates found' };
    }

    logger.info('🤖 [LLM_MATCHER] Starting LLM-based matching', { description, candidateCount: candidates.length, totalElements: elements.length });

    const prompt = this.buildMatchingPrompt(description, candidates, options);

    try {
      const response = await this.llmRouter.processPrompt(prompt, { skipCache: false, taskType: 'element_matching' });
      const result = this.parseLLMResponse(response.text, candidates);

      logger.info('✅ [LLM_MATCHER] Match found', { description, matched: result.element?.content, confidence: result.confidence });

      if (result.confidence < 0.5) {
        const hasWindowBounds = !!options.windowBounds;
        logger.warn('⚠️ [LLM_MATCHER] Low confidence match rejected', { description, matched: result.element?.content, confidence: result.confidence, hasWindowBounds });

        if (!hasWindowBounds) {
          throw new Error(`Element matching failed: Low confidence (${result.confidence}) - Frontend must send windowBounds in context. Matched: "${result.element?.content}"`);
        } else {
          throw new Error(`Element matching failed: Low confidence (${result.confidence}) - No good match for "${description}". Matched: "${result.element?.content}"`);
        }
      }

      return result;
    } catch (error: any) {
      logger.error('❌ [LLM_MATCHER] LLM matching failed', { description, error: error.message });
      return { element: candidates[0], confidence: 0.5, reasoning: `LLM matching failed, using first candidate: ${error.message}` };
    }
  }

  private preFilterCandidates(description: string, elements: ParsedElement[], context?: any): ParsedElement[] {
    const descLower = description.toLowerCase().trim();

    const cleanElements = elements.filter((elem) => {
      if (!elem.content) return false;
      const content = elem.content.toLowerCase().trim();
      if (content.match(/\d+\.\d+s/)) return false;
      if (content.includes('backend:') || content.includes('frontend:')) return false;
      if (content.includes('thinking') || content.includes('iteration')) return false;
      if (content.includes('llm:') || content.includes('action')) return false;
      return true;
    });

    const isSpotlight = context?.intentType === 'spotlight_search';
    let searchPool = cleanElements;

    if (isSpotlight) {
      searchPool = cleanElements.filter((elem) => {
        const centerY = (elem.bbox.y1 + elem.bbox.y2) / 2;
        return centerY > 0.22;
      });
    }

    const filenameMatch = description.match(/([\w\-\.]+\.[a-zA-Z]{2,4})/);
    if (filenameMatch) {
      const filename = filenameMatch[1].toLowerCase();
      const filenameMatches = searchPool.filter((elem) => {
        const contentLower = elem.content.toLowerCase().trim();
        const normalizedContent = contentLower.replace(/[\s\.]+/g, '');
        const normalizedFilename = filename.replace(/[\s\.]+/g, '');
        return normalizedContent.includes(normalizedFilename) || contentLower.includes(filename) || contentLower.replace(/\s+/g, '.').includes(filename);
      });

      if (filenameMatches.length > 0) {
        if (isSpotlight) {
          const compactMatches = filenameMatches.filter((elem) => {
            const width = elem.bbox.x2 - elem.bbox.x1;
            const height = elem.bbox.y2 - elem.bbox.y1;
            return width < 0.30 && height < 0.15;
          });
          if (compactMatches.length > 0) return compactMatches.slice(0, 10);
        }
        return filenameMatches.slice(0, 10);
      }

      if (isSpotlight) {
        const interactiveInResults = searchPool.filter((elem) => {
          const centerY = (elem.bbox.y1 + elem.bbox.y2) / 2;
          return elem.interactivity && centerY > 0.22 && centerY < 0.50;
        });
        if (interactiveInResults.length > 0) return interactiveInResults.slice(0, 20);
      }
    }

    const exactMatches = cleanElements.filter((elem) => elem.content.toLowerCase().trim() === descLower);
    if (exactMatches.length > 0) return exactMatches.slice(0, 10);

    const substringMatches = cleanElements.filter((elem) => {
      const contentLower = elem.content.toLowerCase().trim();
      return descLower.includes(contentLower) || contentLower.includes(descLower);
    });

    if (substringMatches.length > 0) {
      const interactiveKeywords = ['input', 'button', 'field', 'box', 'click', 'select', 'dropdown'];
      const needsInteractive = interactiveKeywords.some((kw) => descLower.includes(kw));
      if (needsInteractive) {
        return [...substringMatches.filter((e) => e.interactivity).slice(0, 20), ...substringMatches.filter((e) => !e.interactivity).slice(0, 10)];
      }
      return substringMatches.slice(0, 30);
    }

    const inputFieldKeywords = ['input field', 'text field', 'search field', 'search box', 'input box'];
    const isInputFieldDescription = inputFieldKeywords.some((kw) => descLower.includes(kw));
    if (isInputFieldDescription) {
      return [...cleanElements.filter((e) => !e.interactivity).slice(0, 30), ...cleanElements.filter((e) => e.interactivity).slice(0, 20)];
    }

    const interactive = cleanElements.filter((e) => e.interactivity);
    const nonInteractive = cleanElements.filter((e) => !e.interactivity);
    return [...interactive.slice(0, 40), ...nonInteractive.slice(0, 10)];
  }

  private buildMatchingPrompt(description: string, candidates: ParsedElement[], context: any): string {
    const spatialHints = this.getSpatialHints(candidates, context);
    const elementList = candidates
      .map((elem, idx) => {
        const spatial = this.getElementSpatialDescription(elem, context);
        const interactive = elem.interactivity ? '✓ interactive' : '✗ not interactive';
        return `${idx + 1}. "${elem.content}" (${elem.type}, ${spatial}, ${interactive})`;
      })
      .join('\n');

    const intentGuidance = this.getIntentSpecificGuidance(context.intentType);

    return `You are an expert UI element matcher. Select the BEST matching element for the description.

**Target Description:** "${description}"

**Context:**
- Active app: ${context.activeApp || 'unknown'}
- Intent type: ${context.intentType || 'unknown'}
- Screen size: ${context.screenshotWidth || '?'}x${context.screenshotHeight || '?'}

**Available Elements:**
${elementList}

**Spatial Context:**
${spatialHints}

**Matching Rules:**
1. Exact matches take priority over partial matches
2. Interactive elements are preferred for clickable actions
3. Spatial context matters: Consider element position
4. Avoid false positives: Skip menu items when looking for content
5. File extensions: "test.txt.rtf" should match "test.txt rtf"
6. Case insensitive

${intentGuidance}

**Output Format:**
Return ONLY a JSON object:
{
  "elementIndex": <number 1-${candidates.length}>,
  "confidence": <number 0.0-1.0>,
  "reasoning": "<brief explanation>"
}

Return ONLY the JSON object, no additional text.`;
  }

  private getIntentSpecificGuidance(intentType?: string): string {
    if (!intentType) return '';
    switch (intentType) {
      case 'spotlight_search':
        return `**CRITICAL - Spotlight Search Rules:**
- Prioritize TOP area elements (y < 30%) - Spotlight results appear at the top
- Text elements are valid targets
- Ignore browser/web elements
- Exact filename match in top area beats interactive icon in middle/bottom`;
      case 'browser_navigation':
        return `**Browser Navigation Rules:**
- Prioritize interactive elements in the address bar or navigation area`;
      case 'file_explorer':
        return `**File Explorer Rules:**
- File/folder names in the main content area are the target
- Avoid sidebar or menu bar elements`;
      case 'search':
      case 'type_text':
        return `**CRITICAL - Search Input Field Rules:**
- "search input field" = text placeholder like "Search Amazon", NOT a button/icon
- Input fields often have placeholder text and may be marked as non-interactive
- If description says "input field", prioritize text/placeholder elements over buttons`;
      default:
        return '';
    }
  }

  private getSpatialHints(candidates: ParsedElement[], context: any): string {
    const screenHeight = context.screenshotHeight || 900;
    const topThreshold = screenHeight * 0.1;
    const bottomThreshold = screenHeight * 0.9;
    const hints: string[] = [];
    const topCount = candidates.filter((e) => e.bbox.y1 < topThreshold).length;
    const centerCount = candidates.filter((e) => e.bbox.y1 >= topThreshold && e.bbox.y1 <= bottomThreshold).length;
    const bottomCount = candidates.filter((e) => e.bbox.y1 > bottomThreshold).length;
    if (topCount > 0) hints.push(`- Top area (menu bar): ${topCount} elements`);
    if (centerCount > 0) hints.push(`- Center area (main content): ${centerCount} elements`);
    if (bottomCount > 0) hints.push(`- Bottom area (status bar): ${bottomCount} elements`);
    return hints.join('\n') || '- No clear spatial distribution';
  }

  private getElementSpatialDescription(elem: ParsedElement, context: any): string {
    const screenHeight = context.screenshotHeight || 900;
    const screenWidth = context.screenshotWidth || 1440;
    const centerY = (elem.bbox.y1 + elem.bbox.y2) / 2;
    const centerX = (elem.bbox.x1 + elem.bbox.x2) / 2;
    let vertical = 'center';
    if (centerY < screenHeight * 0.1) vertical = 'top';
    else if (centerY > screenHeight * 0.9) vertical = 'bottom';
    let horizontal = 'center';
    if (centerX < screenWidth * 0.2) horizontal = 'left';
    else if (centerX > screenWidth * 0.8) horizontal = 'right';
    return `${vertical}-${horizontal}`;
  }

  private parseLLMResponse(response: string, candidates: ParsedElement[]): ElementMatchResult {
    try {
      let jsonStr = response.trim();
      if (jsonStr.startsWith('```')) {
        jsonStr = jsonStr.replace(/```json?\n?/g, '').replace(/```\n?$/g, '').trim();
      }
      const parsed = JSON.parse(jsonStr);
      const elementIndex = parsed.elementIndex;
      const confidence = parsed.confidence;
      const reasoning = parsed.reasoning;

      if (typeof elementIndex !== 'number' || elementIndex < 1 || elementIndex > candidates.length) {
        throw new Error(`Invalid elementIndex: ${elementIndex}`);
      }
      if (typeof confidence !== 'number' || confidence < 0 || confidence > 1) {
        throw new Error(`Invalid confidence: ${confidence}`);
      }

      return { element: candidates[elementIndex - 1], confidence, reasoning: reasoning || 'LLM match' };
    } catch (error: any) {
      logger.error('❌ [LLM_MATCHER] Failed to parse LLM response', { response, error: error.message });
      return { element: candidates[0], confidence: 0.5, reasoning: `Failed to parse LLM response: ${error.message}` };
    }
  }
}
