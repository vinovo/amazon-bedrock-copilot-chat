import * as assert from "node:assert";
import * as vscode from "vscode";

import {
  _forceTokenizerInitFailureForTests,
  _resetTokenizerForTests,
  countMessageTokens,
  countStringTokens,
} from "../tokenizer";

/**
 * Build an Assistant-role message — the only role that may carry tool-call parts.
 */
function assistantMessage(
  content: (vscode.LanguageModelTextPart | vscode.LanguageModelToolCallPart)[],
): vscode.LanguageModelChatRequestMessage {
  return vscode.LanguageModelChatMessage.Assistant(
    content,
  ) as unknown as vscode.LanguageModelChatRequestMessage;
}

/**
 * Build a User-role chat-request message with the given content parts.
 *
 * `LanguageModelChatMessage.User(...)` produces a value structurally
 * compatible with `LanguageModelChatRequestMessage` (same `role` / `content`
 * shape), which is what the tokenizer accepts.
 */
function userMessage(
  content: (vscode.LanguageModelTextPart | vscode.LanguageModelToolResultPart)[],
): vscode.LanguageModelChatRequestMessage {
  return vscode.LanguageModelChatMessage.User(
    content,
  ) as unknown as vscode.LanguageModelChatRequestMessage;
}

suite("Tokenizer", () => {
  suite("countStringTokens (BPE-backed path)", () => {
    setup(() => {
      _resetTokenizerForTests();
    });

    test("returns 0 for empty string", () => {
      assert.equal(countStringTokens(""), 0);
    });

    test("returns a positive integer for non-empty text", () => {
      const tokens = countStringTokens("Hello world!");
      assert.ok(Number.isInteger(tokens), "result should be an integer");
      assert.ok(tokens > 0, "result should be > 0");
      // "Hello world!" is ~3 BPE tokens in o200k_base; sanity-bound the answer
      // generously so the test isn't brittle against tokenizer updates.
      assert.ok(tokens < 10, `unexpectedly large token count: ${tokens}`);
    });

    test("longer text yields more tokens than shorter text", () => {
      const short = countStringTokens("hi");
      const long = countStringTokens("The quick brown fox jumps over the lazy dog. ".repeat(20));
      assert.ok(long > short, `expected long (${long}) > short (${short})`);
    });

    test("BPE estimate beats naive chars/4 on JSON-heavy content", () => {
      // Code-and-JSON content is what the previous chars/4 heuristic
      // systematically under-counted by ~25–40%. The new BPE estimator
      // should produce a noticeably higher count for the same input.
      const json = JSON.stringify({
        repeat: Array.from({ length: 50 }, (_, i) => ({
          enabled: true,
          name: `field_${i}`,
          value: `some_string_${i}`,
        })),
      });
      const naive = Math.ceil(json.length / 4);
      const bpe = countStringTokens(json);
      assert.ok(bpe > naive, `BPE (${bpe}) should exceed chars/4 (${naive}) on JSON-heavy content`);
    });
  });

  suite("countStringTokens (LRU caching)", () => {
    setup(() => {
      _resetTokenizerForTests();
    });

    test("repeated calls for the same text return identical counts", () => {
      const text = "A consistent sample of text used repeatedly.";
      const first = countStringTokens(text);
      const second = countStringTokens(text);
      const third = countStringTokens(text);
      assert.deepStrictEqual({ first, second, third }, { first, second: first, third: first });
    });
  });

  suite("countMessageTokens (structured walker)", () => {
    setup(() => {
      _resetTokenizerForTests();
    });

    test("sums tokens across multiple text parts", () => {
      const a = "First chunk of content.";
      const b = "Second, distinct chunk.";

      const single = countStringTokens(a) + countStringTokens(b);
      const combined = countMessageTokens(
        userMessage([new vscode.LanguageModelTextPart(a), new vscode.LanguageModelTextPart(b)]),
      );

      // No per-message overhead is added, so the message count equals the
      // sum of its part counts.
      assert.equal(combined, single);
    });

    test("counts tool-call parts using name + JSON-stringified input", () => {
      const toolName = "search_repository";
      const toolInput = { limit: 10, query: "tokenizer" };
      const expected = countStringTokens(toolName) + countStringTokens(JSON.stringify(toolInput));

      const actual = countMessageTokens(
        assistantMessage([new vscode.LanguageModelToolCallPart("call-1", toolName, toolInput)]),
      );

      assert.equal(actual, expected);
    });

    test("counts tool-result parts: text items and structured items", () => {
      const textItem = new vscode.LanguageModelTextPart("plain text result");
      const structuredItem = new vscode.LanguageModelPromptTsxPart({
        kind: "wrapped",
        value: "ignored-by-walker",
      });

      const expected =
        countStringTokens("plain text result") + countStringTokens(JSON.stringify(structuredItem));

      const actual = countMessageTokens(
        userMessage([new vscode.LanguageModelToolResultPart("call-1", [textItem, structuredItem])]),
      );

      assert.equal(actual, expected);
    });

    test("returns 0 for an empty message", () => {
      assert.equal(countMessageTokens(userMessage([])), 0);
    });
  });

  suite("init-failure fallback", () => {
    teardown(() => {
      // Restore the real encoder for any subsequent tests in the run.
      _resetTokenizerForTests();
    });

    test("falls back to chars/3 when the encoder is unavailable", () => {
      _forceTokenizerInitFailureForTests();

      const text = "The quick brown fox jumps over the lazy dog.";
      const expectedFallback = Math.ceil(text.length / 3);

      assert.equal(countStringTokens(text), expectedFallback);
    });

    test("fallback path is conservative vs the previous chars/4 heuristic", () => {
      _forceTokenizerInitFailureForTests();

      const text = "Some content used for budget-shaping decisions.";
      const oldHeuristic = Math.ceil(text.length / 4);
      const newFallback = countStringTokens(text);

      assert.ok(
        newFallback > oldHeuristic,
        `fallback (${newFallback}) should exceed legacy chars/4 (${oldHeuristic})`,
      );
    });
  });
});
