import * as assert from "node:assert";

import { getModelProfile, getModelTokenLimits } from "../profiles";

/**
 * Claude Opus 5 support.
 *
 * Opus 5 uses the dateless Bedrock ID `anthropic.claude-opus-5` and behaves like
 * Claude Opus 4.8: adaptive-thinking-only (no manual budget_tokens, no
 * temperature/top_p/top_k), supports the full `effort` parameter set, and offers a
 * toggleable 200K↔1M context window with 128K max output.
 * Geo prefixes are us/eu/au (no jp, unlike Opus 4.8).
 * See: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-5.html
 */
suite("Claude Opus 5", () => {
  // Exercise both the bare base model ID and the inference-profile-prefixed variants so we know
  // normalization keeps working for the fallback inference profile path.
  const ids = [
    "anthropic.claude-opus-5",
    "us.anthropic.claude-opus-5",
    "eu.anthropic.claude-opus-5",
    "global.anthropic.claude-opus-5",
  ];

  suite("getModelProfile", () => {
    for (const id of ids) {
      test(`recognizes capabilities for ${id}`, () => {
        const profile = getModelProfile(id);

        // Adaptive-thinking-only, like Opus 4.8.
        assert.equal(profile.supportsAdaptiveThinkingOnly, true, "supportsAdaptiveThinkingOnly");
        assert.equal(profile.prefersAdaptiveThinking, true, "prefersAdaptiveThinking");
        assert.equal(profile.supportsThinking, true, "supportsThinking");

        // Full effort set: max, xhigh, high, medium, low.
        assert.deepEqual(
          profile.supportedThinkingEfforts,
          ["max", "xhigh", "high", "medium", "low"],
          "supportedThinkingEfforts",
        );

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
      // Copilot-aligned reservation: min(128K native output, 15% of context window).
      // At 200K → 15% = 30K reserve; at 1M → 15% = 150K exceeds the 128K native cap → 128K.
      test(`reserves 30K output (15% of 200K) by default for ${id}`, () => {
        const limits = getModelTokenLimits(id);
        assert.equal(limits.maxOutputTokens, 30_000, "maxOutputTokens");
        assert.equal(limits.maxInputTokens, 200_000 - 30_000, "maxInputTokens (200K window)");
      });

      test(`reserves 128K output (native cap) when 1M context is enabled for ${id}`, () => {
        const limits = getModelTokenLimits(id, true);
        assert.equal(limits.maxOutputTokens, 128_000, "maxOutputTokens");
        assert.equal(limits.maxInputTokens, 1_000_000 - 128_000, "maxInputTokens (1M window)");
      });
    }
  });
});
