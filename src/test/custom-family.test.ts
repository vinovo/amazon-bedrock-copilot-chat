import * as assert from "node:assert";

import { isGeminiModelFamily, resolveModelFamily } from "../custom/family";

suite("Custom Provider - model family reporting", () => {
  test("keeps plain family ids unchanged", () => {
    assert.strictEqual(resolveModelFamily("gemini-3-pro"), "gemini-3-pro");
    assert.strictEqual(resolveModelFamily("claude-sonnet-4-6"), "claude-sonnet-4-6");
    assert.strictEqual(resolveModelFamily("gpt-5.4"), "gpt-5.4");
  });

  test("strips routing prefix, variant suffix and gateway path", () => {
    assert.strictEqual(resolveModelFamily("anthropic::claude-4-6-sonnet:1M"), "claude-4-6-sonnet");
    assert.strictEqual(resolveModelFamily("azure::gpt-5.4"), "gpt-5.4");
    assert.strictEqual(resolveModelFamily("vertex_ai/gemini-3-pro"), "gemini-3-pro");
  });

  test("falls back to an alias when the id is a codename", () => {
    assert.strictEqual(
      resolveModelFamily("team-model-a", ["vertex_ai/gemini-3-pro"]),
      "gemini-3-pro",
    );
  });

  test("falls back to the normalized id for unrecognized families", () => {
    assert.strictEqual(resolveModelFamily("Qwen3-235B"), "qwen3-235b");
    assert.strictEqual(resolveModelFamily("llama-4", ["some-alias"]), "llama-4");
  });

  test("never reports the constant 'custom' family for a real model", () => {
    assert.notStrictEqual(resolveModelFamily("gemini-3-pro"), "custom");
  });

  test("isGeminiModelFamily matches on the family prefix", () => {
    assert.ok(isGeminiModelFamily("gemini-3-pro"));
    assert.ok(isGeminiModelFamily("Gemini-2.5-flash"));
    assert.strictEqual(isGeminiModelFamily("claude-sonnet-4-6"), false);
  });
});
