import * as assert from "node:assert";

import { fallbackMaxOutputTokens, resolveMaxOutputTokens } from "../custom/limits";

suite("Max-output-token resolution", () => {
  test("prefers a valid backend-reported limit over the fallback", () => {
    assert.strictEqual(resolveMaxOutputTokens("claude-opus-5", 128_000), 128_000);
    // Reported wins even when it is smaller than the family fallback (legacy models).
    assert.strictEqual(resolveMaxOutputTokens("claude-3-opus", 4096), 4096);
  });

  test("ignores an absent or non-positive reported limit", () => {
    assert.strictEqual(resolveMaxOutputTokens("claude-opus-5"), 128_000);
    assert.strictEqual(resolveMaxOutputTokens("claude-opus-5", 0), 128_000);
    assert.strictEqual(resolveMaxOutputTokens("claude-opus-5", Number.NaN), 128_000);
  });

  test("Claude fallback is generation-aware within a family (breeze id form)", () => {
    assert.strictEqual(fallbackMaxOutputTokens("claude-opus-5"), 128_000);
    assert.strictEqual(fallbackMaxOutputTokens("claude-sonnet-5"), 128_000);
    assert.strictEqual(fallbackMaxOutputTokens("claude-opus-4-8"), 128_000);
    assert.strictEqual(fallbackMaxOutputTokens("claude-opus-4-6"), 128_000);
    assert.strictEqual(fallbackMaxOutputTokens("claude-sonnet-4-6"), 64_000);
    assert.strictEqual(fallbackMaxOutputTokens("claude-opus-4-5-20251101"), 64_000);
    assert.strictEqual(fallbackMaxOutputTokens("claude-haiku-4-5-20251001"), 64_000);
    assert.strictEqual(fallbackMaxOutputTokens("claude-opus-4-1-20250805"), 32_000);
    assert.strictEqual(fallbackMaxOutputTokens("claude-opus-4-20250514"), 32_000);
    assert.strictEqual(fallbackMaxOutputTokens("claude-sonnet-4-20250514"), 64_000);
    assert.strictEqual(fallbackMaxOutputTokens("claude-3-5-sonnet-20241022"), 8192);
    assert.strictEqual(fallbackMaxOutputTokens("claude-3-opus-20240229"), 4096);
  });

  test("Claude fallback handles QGenie id form (version before tier, provider prefix, :1M)", () => {
    assert.strictEqual(fallbackMaxOutputTokens("anthropic::claude-5-sonnet"), 128_000);
    assert.strictEqual(fallbackMaxOutputTokens("anthropic::claude-4-8-opus"), 128_000);
    assert.strictEqual(fallbackMaxOutputTokens("anthropic::claude-4-6-sonnet:1M"), 64_000);
    assert.strictEqual(fallbackMaxOutputTokens("anthropic::claude-4-6-opus:1M"), 128_000);
    assert.strictEqual(fallbackMaxOutputTokens("anthropic::claude-4-5-sonnet"), 64_000);
  });

  test("OpenAI GPT / o-series fallback ceilings", () => {
    assert.strictEqual(fallbackMaxOutputTokens("gpt-5.4"), 128_000);
    assert.strictEqual(fallbackMaxOutputTokens("azure::gpt-5.6-sol"), 128_000);
    assert.strictEqual(fallbackMaxOutputTokens("o3"), 100_000);
    assert.strictEqual(fallbackMaxOutputTokens("gpt-4.1"), 32_000);
    assert.strictEqual(fallbackMaxOutputTokens("gpt-4o"), 16_384);
  });

  test("Gemini fallback is generation-aware (2.5/3.x = 64K, 2.0 = 8K)", () => {
    assert.strictEqual(fallbackMaxOutputTokens("vertexai::gemini-2.5-pro"), 65_536);
    assert.strictEqual(fallbackMaxOutputTokens("vertexai::gemini-2.5-flash"), 65_536);
    assert.strictEqual(fallbackMaxOutputTokens("vertexai::gemini-3.5-flash"), 65_536);
    assert.strictEqual(fallbackMaxOutputTokens("vertexai::gemini-3.1-pro-preview"), 65_536);
    assert.strictEqual(fallbackMaxOutputTokens("vertexai::gemini-3-flash-preview"), 65_536);
    assert.strictEqual(fallbackMaxOutputTokens("gemini-2.0-flash-001"), 8192);
  });

  test("Qwen / DeepSeek / Llama fall back to the 8K conservative ceiling", () => {
    assert.strictEqual(fallbackMaxOutputTokens("qwen3-235b"), 8192);
    assert.strictEqual(fallbackMaxOutputTokens("qwen2.5-coder-32b-128k"), 8192);
    assert.strictEqual(fallbackMaxOutputTokens("deepseek-coder-v2-lite-instruct"), 8192);
    assert.strictEqual(fallbackMaxOutputTokens("llama-4-maverick-17b"), 8192);
    assert.strictEqual(fallbackMaxOutputTokens("llama-3.3-70b"), 8192);
  });

  test("unknown models fall back to the conservative floor", () => {
    assert.strictEqual(fallbackMaxOutputTokens("some-unknown-model"), 8192);
    assert.strictEqual(fallbackMaxOutputTokens("google/gemma-4-31b-it"), 8192);
  });
});
