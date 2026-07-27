/**
 * Model profile system for handling provider-specific capabilities
 */

import type { ThinkingEffort } from "./settings";

export interface ModelProfile {
  /**
   * Whether the model should PREFER adaptive thinking (thinking.type: "adaptive") over manual
   * budget-based thinking, even though manual mode still works.
   * Claude Opus 4.6 and Sonnet 4.6 recommend adaptive thinking (manual budget_tokens is
   * deprecated-but-functional for them). This differs from {@link supportsAdaptiveThinkingOnly},
   * which means manual thinking returns a 400 error.
   * Adaptive-only models implicitly prefer adaptive as well.
   */
  prefersAdaptiveThinking: boolean;
  /**
   * Whether the model requires the interleaved-thinking beta header (Claude 4 models only)
   */
  requiresInterleavedThinkingHeader: boolean;
  /**
   * Ordered list of effort levels the model accepts via `output_config.effort`.
   * An empty array means the model does not support the effort parameter at all.
   * The ordering matches the picker display (highest first: max → xhigh → high → medium → low).
   * Not all models support all levels:
   * - Opus 5 / Opus 4.8 / Opus 4.7 / Sonnet 5: max, xhigh, high, medium, low
   *   - Opus 4.6 / Sonnet 4.6:                  max, high, medium, low  (no xhigh)
   *   - Opus 4.5 / Sonnet 4.5 / Haiku 4.5:      high, medium, low
   */
  supportedThinkingEfforts: readonly ThinkingEffort[];
  /**
   * Whether the model supports 1M context window
   */
  supports1MContext: boolean;
  /**
   * Whether the model uses adaptive thinking only (thinking.type: "adaptive"), without budget_tokens.
   * Claude Opus 5, Opus 4.8, Opus 4.7, and Sonnet 5 use this mode exclusively; manual thinking returns 400.
   * Temperature/top_p/top_k are also unsupported when extended thinking is active.
   * See:
   *   https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-5.html
   *   https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-8.html
   *   https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-7.html
   */
  supportsAdaptiveThinkingOnly: boolean;
  /**
   * Whether the model supports caching with tool results (cachePoint after toolResult blocks)
   * When false, cachePoint should only be added to messages WITHOUT toolResult
   * Reference: Amazon Nova models don't support cachePoint after toolResult
   */
  supportsCachingWithToolResults: boolean;
  /**
   * Whether the model supports prompt caching via cache points
   */
  supportsPromptCaching: boolean;
  /**
   * Whether the model supports extended thinking (Claude Opus 4.6, Opus 4.5, Opus 4.1, Opus 4, Sonnet 5, Sonnet 4.6, Sonnet 4.5, Sonnet 4, Sonnet 3.7)
   */
  supportsThinking: boolean;
  /**
   * Whether the model supports the thinking `display` field ("summarized" | "omitted").
   * "omitted" suppresses streamed thinking text (blocks come back empty but the encrypted
   * signature is retained), reducing time-to-first-text-token. Supported by Claude extended
   * thinking models.
   */
  supportsThinkingDisplay: boolean;
  /**
   * Whether the model supports the toolChoice parameter
   */
  supportsToolChoice: boolean;
  /**
   * Whether the model supports the status field in tool results (error/success)
   * Reference: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolResultBlock.html
   * Currently only Claude models support this field
   */
  supportsToolResultStatus: boolean;
  /**
   * Format to use for tool result content ('text' or 'json')
   */
  toolResultFormat: "json" | "text";
}

export interface ModelTokenLimits {
  /**
   * Maximum number of input tokens (context window)
   */
  maxInputTokens: number;
  /**
   * Maximum number of output tokens
   */
  maxOutputTokens: number;
}

export function getModelProfile(modelId: string): ModelProfile {
  const defaultProfile: ModelProfile = {
    prefersAdaptiveThinking: false,
    requiresInterleavedThinkingHeader: false,
    supportedThinkingEfforts: [],
    supports1MContext: false,
    supportsAdaptiveThinkingOnly: false,
    supportsCachingWithToolResults: false,
    supportsPromptCaching: false,
    supportsThinking: false,
    supportsThinkingDisplay: false,
    supportsToolChoice: false,
    supportsToolResultStatus: false,
    toolResultFormat: "text",
  };

  const normalizedId = normalizeModelId(modelId);
  const parts = normalizedId.split(".");

  if (parts.length < 2) {
    return defaultProfile;
  }

  const provider = parts[0];

  // Provider-specific profiles
  switch (provider) {
    case "ai21":

    case "cohere":
    case "meta": {
      // Older models don't support tool choice
      return defaultProfile;
    }

    case "amazon": {
      // Amazon Nova models support tool choice and prompt caching
      // Nova does NOT support cachePoint after toolResult blocks
      if (modelId.includes("nova")) {
        return {
          prefersAdaptiveThinking: false,
          requiresInterleavedThinkingHeader: false,
          supportedThinkingEfforts: [],
          supports1MContext: false,
          supportsAdaptiveThinkingOnly: false,
          supportsCachingWithToolResults: false,
          supportsPromptCaching: true,
          supportsThinking: false,
          supportsThinkingDisplay: false,
          supportsToolChoice: true,
          supportsToolResultStatus: false,
          toolResultFormat: "text",
        };
      }
      return defaultProfile;
    }
    case "anthropic": {
      return getAnthropicProfile(modelId);
    }
    case "mistral": {
      // Mistral models require JSON format for tool results
      return {
        prefersAdaptiveThinking: false,
        requiresInterleavedThinkingHeader: false,
        supportedThinkingEfforts: [],
        supports1MContext: false,
        supportsAdaptiveThinkingOnly: false,
        supportsCachingWithToolResults: false,
        supportsPromptCaching: false,
        supportsThinking: false,
        supportsThinkingDisplay: false,
        supportsToolChoice: false,
        supportsToolResultStatus: false,
        toolResultFormat: "json",
      };
    }

    case "openai": {
      // OpenAI models support tool choice but not prompt caching
      return {
        prefersAdaptiveThinking: false,
        requiresInterleavedThinkingHeader: false,
        supportedThinkingEfforts: [],
        supports1MContext: false,
        supportsAdaptiveThinkingOnly: false,
        supportsCachingWithToolResults: false,
        supportsPromptCaching: false,
        supportsThinking: false,
        supportsThinkingDisplay: false,
        supportsToolChoice: true,
        supportsToolResultStatus: false,
        toolResultFormat: "text",
      };
    }

    default: {
      return defaultProfile;
    }
  }
}

/**
 * Get token limits for a given Bedrock model ID
 * Returns model-specific token limits for known models, or conservative defaults for others
 * @param modelId The full Bedrock model ID (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0")
 * @param enable1MContext Whether to enable 1M context for supported models (default: false)
 * @returns Token limits with maxInputTokens and maxOutputTokens
 */
export function getModelTokenLimits(modelId: string, enable1MContext = false): ModelTokenLimits {
  const normalizedModelId = normalizeModelId(modelId);

  // Claude models have specific token limits based on model family
  if (normalizedModelId.startsWith("anthropic.claude")) {
    return getClaudeTokenLimits(normalizedModelId, enable1MContext);
  }

  // OpenAI GPT OSS models: 128K context, 16K output
  // See: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-oss-120b.html
  // See: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-oss-20b.html
  if (normalizedModelId.startsWith("openai.gpt-oss")) {
    return { maxInputTokens: 128_000 - 16_000, maxOutputTokens: 16_000 };
  }

  // Default for unknown models
  return {
    maxInputTokens: 196_000, // 200K context - 4K output
    maxOutputTokens: 4096,
  };
}

/**
 * Reserve output tokens the way GitHub Copilot does: cap the response reservation at
 * `min(model's native max output, 15% of the context window)`, then give the remainder to the
 * prompt. This mirrors Copilot's model-metadata hydration (vscode-copilot-chat
 * `modelMetadataFetcher._hydrateResolvedModel`), preventing a large native output ceiling
 * (e.g. 128K for Opus 4.8) from eating most of a 200K context window.
 *
 * @example copilotAlignedLimits(200_000, 128_000)   → { maxInputTokens: 170_000, maxOutputTokens: 30_000 }
 * @example copilotAlignedLimits(1_000_000, 128_000) → { maxInputTokens: 872_000, maxOutputTokens: 128_000 }
 */
function copilotAlignedLimits(contextWindow: number, nativeMaxOutput: number): ModelTokenLimits {
  const reserve = Math.floor(Math.min(nativeMaxOutput, contextWindow * 0.15));
  return { maxInputTokens: contextWindow - reserve, maxOutputTokens: reserve };
}

/**
 * Build the model profile for Anthropic Claude models.
 * Extracted from {@link getModelProfile} to keep that function's cognitive complexity low.
 */
function getAnthropicProfile(modelId: string): ModelProfile {
  // Claude models support tool choice and prompt caching
  // Extended thinking is supported by Claude Opus 5, Opus 4+, Sonnet 5, Sonnet 4+, Haiku 4.5+, and Sonnet 3.7
  const supportsThinking =
    modelId.includes("opus-5") ||
    modelId.includes("opus-4") ||
    modelId.includes("sonnet-5") ||
    modelId.includes("sonnet-4") ||
    modelId.includes("haiku-4-5") ||
    modelId.includes("haiku-4.5") ||
    modelId.includes("sonnet-3-7") ||
    modelId.includes("sonnet-3.7");

  // Interleaved thinking (beta header) is only for Claude 4 models.
  // Sonnet 5 uses adaptive-thinking-only (like Opus 4.8) and does not use the interleaved header.
  const requiresInterleavedThinkingHeader =
    modelId.includes("opus-4") ||
    modelId.includes("sonnet-4") ||
    modelId.includes("haiku-4-5") ||
    modelId.includes("haiku-4.5");

  // Claude models with extended thinking have issues with cachePoint after toolResult
  // When extended thinking is enabled, cachePoint should only be added to messages without toolResult
  const supportsCachingWithToolResults = !supportsThinking;

  // Opus 5, Opus 4.8, Opus 4.7, and Sonnet 5 use adaptive thinking only (thinking.type: "adaptive").
  // budget_tokens and temperature/top_p/top_k are not supported and will return a 400 error.
  // See:
  //   https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-5.html
  //   https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-8.html
  //   https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-7.html
  //   https://platform.claude.com/docs/en/docs/about-claude/models/overview (Claude Sonnet 5)
  const supportsAdaptiveThinkingOnly =
    modelId.includes("opus-5") ||
    modelId.includes("opus-4-8") ||
    modelId.includes("opus-4-7") ||
    modelId.includes("sonnet-5");

  // Claude Opus 4.6 and Sonnet 4.6 recommend adaptive thinking. Manual budget-based thinking
  // is deprecated-but-still-functional for them, so we treat adaptive as a *preference* rather
  // than the hard requirement that `supportsAdaptiveThinkingOnly` represents. Adaptive-only
  // models implicitly prefer adaptive too.
  const prefersAdaptiveThinking =
    supportsAdaptiveThinkingOnly || modelId.includes("opus-4-6") || modelId.includes("sonnet-4-6");

  // The thinking `display` field ("summarized" | "omitted") is supported by Claude extended
  // thinking models. "omitted" suppresses streamed thinking text for faster time-to-first-token.
  const supportsThinkingDisplay = supportsThinking;

  // Adaptive thinking / thinking effort parameter support.
  // Each model supports a specific subset of effort levels:
  //   - Opus 5 / Opus 4.8 / Opus 4.7 / Sonnet 5: max, xhigh, high, medium, low
  //   - Opus 4.6 / Sonnet 4.6:                    max, high, medium, low  (no xhigh)
  //   - Opus 4.5 / Sonnet 4.5 / Haiku 4.5:        high, medium, low
  // Ordered highest-to-lowest for the picker display.
  const ALL_EFFORTS: readonly ThinkingEffort[] = ["max", "xhigh", "high", "medium", "low"];
  const NO_XHIGH_EFFORTS: readonly ThinkingEffort[] = ["max", "high", "medium", "low"];
  const BASIC_EFFORTS: readonly ThinkingEffort[] = ["high", "medium", "low"];

  let supportedThinkingEfforts: readonly ThinkingEffort[];
  if (
    modelId.includes("opus-5") ||
    modelId.includes("opus-4-8") ||
    modelId.includes("opus-4-7") ||
    modelId.includes("sonnet-5")
  ) {
    supportedThinkingEfforts = ALL_EFFORTS; // max, xhigh, high, medium, low
  } else if (modelId.includes("opus-4-6") || modelId.includes("sonnet-4-6")) {
    supportedThinkingEfforts = NO_XHIGH_EFFORTS; // max, high, medium, low
  } else if (
    modelId.includes("opus-4-5") ||
    modelId.includes("sonnet-4-5") ||
    modelId.includes("haiku-4-5") ||
    modelId.includes("haiku-4.5")
  ) {
    supportedThinkingEfforts = BASIC_EFFORTS; // high, medium, low
  } else {
    supportedThinkingEfforts = [];
  }

  return {
    prefersAdaptiveThinking,
    requiresInterleavedThinkingHeader,
    supportedThinkingEfforts,
    supports1MContext: supports1MContext(modelId),
    supportsAdaptiveThinkingOnly,
    supportsCachingWithToolResults,
    supportsPromptCaching: true,
    supportsThinking,
    supportsThinkingDisplay,
    supportsToolChoice: true,
    supportsToolResultStatus: true, // Claude models support status field in tool results
    toolResultFormat: "text",
  };
}

/**
 * Get token limits for a Claude model based on its normalized model ID
 */
function getClaudeTokenLimits(
  normalizedModelId: string,
  enable1MContext: boolean,
): ModelTokenLimits {
  // Native output ceilings below feed Copilot's 15%-of-context reservation via
  // `copilotAlignedLimits`. At a 200K window the reserve collapses to 30K (15%) for every
  // model whose native output exceeds 30K; at 1M it uses the native ceiling (15% = 150K > 128K).

  // Claude Opus 5, Opus 4.8, Opus 4.7, and Sonnet 5: 200K default context (or 1M with the
  // `context-1m-2025-08-07` beta enabled), 128K native output. The model card headlines a 1M
  // context window, but — like the official Claude Code CLI — the *effective* window defaults
  // to 200K and opts into 1M via the beta header. temperature/top_p/top_k unsupported;
  // adaptive thinking only.
  // See:
  //   https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-5.html
  //   https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-8.html
  //   https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-7.html
  //   https://platform.claude.com/docs/en/docs/about-claude/models/overview (Claude Sonnet 5)
  if (
    normalizedModelId.includes("opus-5") ||
    normalizedModelId.includes("opus-4-8") ||
    normalizedModelId.includes("opus-4-7") ||
    normalizedModelId.includes("sonnet-5")
  ) {
    return copilotAlignedLimits(enable1MContext ? 1_000_000 : 200_000, 128_000);
  }

  // Claude Opus 4.6: 200K context (or 1M with setting enabled), 128K native output
  // https://platform.claude.com - Opus 4.6 supports 128K output and optional 1M context
  if (normalizedModelId.includes("opus-4-6")) {
    return copilotAlignedLimits(enable1MContext ? 1_000_000 : 200_000, 128_000);
  }

  // Claude Sonnet 4.6: 200K context (or 1M with setting enabled), 64K native output
  if (normalizedModelId.includes("sonnet-4-6")) {
    return copilotAlignedLimits(enable1MContext ? 1_000_000 : 200_000, 64_000);
  }

  // Claude Sonnet 4.5 and 4: 200K context (or 1M with setting enabled), 64K native output
  if (normalizedModelId.includes("sonnet-4")) {
    return copilotAlignedLimits(enable1MContext ? 1_000_000 : 200_000, 64_000);
  }

  // Claude Sonnet 3.7: 200K context, 64K native output
  if (normalizedModelId.includes("sonnet-3-7") || normalizedModelId.includes("sonnet-3.7")) {
    return copilotAlignedLimits(200_000, 64_000);
  }

  // Claude Opus 4.5, 4.1 and 4: 200K context, 64K native output
  if (normalizedModelId.includes("opus-4")) {
    return copilotAlignedLimits(200_000, 64_000);
  }

  // Claude Haiku 4.5: 200K context, 64K native output
  if (normalizedModelId.includes("haiku-4-5") || normalizedModelId.includes("haiku-4.5")) {
    return copilotAlignedLimits(200_000, 64_000);
  }

  // Claude Haiku 3.5: 200K context, 8,192 native output
  if (normalizedModelId.includes("haiku-3-5") || normalizedModelId.includes("haiku-3.5")) {
    return copilotAlignedLimits(200_000, 8192);
  }

  // Claude Haiku 3: 200K context, 4,096 native output
  if (normalizedModelId.includes("haiku-3")) {
    return copilotAlignedLimits(200_000, 4096);
  }

  // Claude 3.5 Sonnet (older): 200K context, 8,192 native output
  if (normalizedModelId.includes("sonnet-3-5") || normalizedModelId.includes("sonnet-3.5")) {
    return copilotAlignedLimits(200_000, 8192);
  }

  // Claude Opus 3: 200K context, 4,096 native output
  if (normalizedModelId.includes("opus-3")) {
    return copilotAlignedLimits(200_000, 4096);
  }

  // Default for unknown Claude models
  return copilotAlignedLimits(200_000, 4096);
}

/**
 * Normalize a Bedrock model ID by stripping inference profile prefixes.
 * Handles both regional prefixes (us., eu., ap., etc.) and global prefix (global.)
 * @param modelId The full Bedrock model ID with optional prefix
 * @returns Normalized model ID without prefix
 * @example
 * normalizeModelId("global.anthropic.claude-opus-4-5") → "anthropic.claude-opus-4-5"
 * normalizeModelId("us.anthropic.claude-opus-4-5") → "anthropic.claude-opus-4-5"
 * normalizeModelId("anthropic.claude-opus-4-5") → "anthropic.claude-opus-4-5"
 */
function normalizeModelId(modelId: string): string {
  const parts = modelId.split(".");
  if (parts.length > 2 && (parts[0].length === 2 || parts[0] === "global")) {
    return parts.slice(1).join(".");
  }
  return modelId;
}

/**
 * Check if a model supports 1M context window
 * Claude Opus 4.8/4.7/4.6, Sonnet 5, and Sonnet 4.x support an extended 1M context window that is
 * opted into via the `context-1m-2025-08-07` anthropic_beta parameter (default effective window is
 * 200K). This mirrors the official Claude Code CLI, which exposes a 200K↔1M toggle for all of
 * Opus 4.6/4.7/4.8 (via the `opus[1m]` model-string suffix) rather than running them at a fixed 1M.
 */
function supports1MContext(modelId: string): boolean {
  return (
    modelId.includes("opus-5") ||
    modelId.includes("opus-4-8") ||
    modelId.includes("opus-4-7") ||
    modelId.includes("opus-4-6") ||
    modelId.includes("sonnet-5") ||
    modelId.includes("sonnet-4")
  );
}

/**
 * Get the model profile for a given Bedrock model ID
 * @param modelId The full Bedrock model ID (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0")
 * @returns Model profile with capabilities
 */
