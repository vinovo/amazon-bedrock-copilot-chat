import * as assert from "node:assert";

import { getModelProfile, getModelTokenLimits } from "../profiles";

/**
 * Claude Sonnet 5 support.
 *
 * Sonnet 5 uses the dateless Bedrock ID `anthropic.claude-sonnet-5` and behaves like
 * Claude Opus 4.8: adaptive-thinking-only (no manual budget_tokens, no
 * temperature/top_p/top_k), supports the `effort` parameter, and offers a toggleable
 * 200K<->1M context window with 128K max output.
 * See: https://platform.claude.com/docs/en/docs/about-claude/models/overview
 */
suite("Claude Sonnet 5", () => {
  // Exercise both the bare base model ID and the inference-profile-prefixed variants so we know
  // normalization keeps working for the fallback inference profile path.
  const ids = [
    "anthropic.claude-sonnet-5",
    "us.anthropic.claude-sonnet-5",
    "eu.anthropic.claude-sonnet-5",
    "global.anthropic.claude-sonnet-5",
  ];

  suite("getModelProfile", () => {
    for (const id of ids) {
      test(`recognizes capabilities for ${id}`, () => {
        const profile = getModelProfile(id);

        // Adaptive-thinking-only, like Opus 4.8.
        assert.equal(profile.supportsAdaptiveThinkingOnly, true, "supportsAdaptiveThinkingOnly");
        assert.equal(profile.prefersAdaptiveThinking, true, "prefersAdaptiveThinking");
        assert.equal(profile.supportsThinking, true, "supportsThinking");

        // Effort parameter is supported (defaults to "high").
        assert.equal(profile.supportsThinkingEffort, true, "supportsThinkingEffort");

        // Toggleable 1M context.
        assert.equal(profile.supports1MContext, true, "supports1MContext");

        // Shared Claude capabilities.
        assert.equal(profile.supportsPromptCaching, true, "supportsPromptCaching");
        assert.equal(profile.supportsToolChoice, true, "supportsToolChoice");
        assert.equal(profile.supportsToolResultStatus, true, "supportsToolResultStatus");
        assert.equal(profile.toolResultFormat, "text", "toolResultFormat");
        assert.equal(profile.supportsThinkingDisplay, true, "supportsThinkingDisplay");

        // Adaptive-thinking models must NOT request the interleaved-thinking beta header.
        assert.equal(
          profile.requiresInterleavedThinkingHeader,
          false,
          "requiresInterleavedThinkingHeader",
        );
      });
    }
  });

  suite("getModelTokenLimits", () => {
    for (const id of ids) {
      test(`uses 200K/128K by default for ${id}`, () => {
        const limits = getModelTokenLimits(id);
        assert.equal(limits.maxOutputTokens, 128_000, "maxOutputTokens");
        assert.equal(limits.maxInputTokens, 200_000 - 128_000, "maxInputTokens (200K window)");
      });

      test(`uses 1M/128K when 1M context is enabled for ${id}`, () => {
        const limits = getModelTokenLimits(id, true);
        assert.equal(limits.maxOutputTokens, 128_000, "maxOutputTokens");
        assert.equal(limits.maxInputTokens, 1_000_000 - 128_000, "maxInputTokens (1M window)");
      });
    }
  });
});
