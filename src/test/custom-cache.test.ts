import * as assert from "node:assert";

import type { OpenAIChatMessage, OpenAIContentPartWithCache } from "../custom/client";
import { addCacheBreakpoints } from "../custom/converter";
import { isClaudeFamily } from "../custom/reasoning";

/** Count how many messages carry a cache breakpoint on their last content block. */
function countBreakpoints(messages: OpenAIChatMessage[]): number {
  let n = 0;
  for (const msg of messages) {
    if (
      Array.isArray(msg.content) &&
      msg.content.some((p) => (p as OpenAIContentPartWithCache).cache_control !== undefined)
    ) {
      n++;
    }
  }
  return n;
}

/** True if the given message has a cache breakpoint on its last content block. */
function isMarked(msg: OpenAIChatMessage): boolean {
  if (!Array.isArray(msg.content) || msg.content.length === 0) return false;
  const last = msg.content[msg.content.length - 1] as OpenAIContentPartWithCache;
  return last.cache_control?.type === "ephemeral";
}

function text(role: OpenAIChatMessage["role"], value: string): OpenAIChatMessage {
  return { content: value, role };
}

suite("Claude family detection (real backend ids)", () => {
  test("matches every Claude id on breeze/LiteLLM", () => {
    for (const id of ["claude-sonnet-4-6", "claude-sonnet-5", "claude-opus-5", "claude-opus-4-8"]) {
      assert.ok(isClaudeFamily(id), `expected ${id} to be Claude`);
    }
  });

  test("matches every Claude id on QGenie (provider:: prefix + :1M variant)", () => {
    for (const id of [
      "anthropic::claude-4-6-sonnet",
      "anthropic::claude-4-6-sonnet:1M",
      "anthropic::claude-4-8-opus",
    ]) {
      assert.ok(isClaudeFamily(id), `expected ${id} to be Claude`);
    }
  });

  test("matches older cache-capable Claude 3.5+", () => {
    assert.ok(isClaudeFamily("claude-3-5-sonnet"));
    assert.ok(isClaudeFamily("anthropic::claude-3-7-sonnet"));
  });

  test("rejects non-Claude models", () => {
    for (const id of ["gpt-5.2", "gpt-4o", "azure::gpt-5.4", "llama-3-70b", "gemini-2.5-pro"]) {
      assert.strictEqual(isClaudeFamily(id), false, `expected ${id} not to be Claude`);
    }
  });
});

suite("Cache breakpoint placement", () => {
  test("marks the current user message and stays within the cap of 4", () => {
    const messages: OpenAIChatMessage[] = [
      text("system", "You are helpful."),
      text("user", "First question"),
      text("assistant", "First answer"),
      text("user", "Second question"),
    ];
    addCacheBreakpoints(messages);
    assert.ok(countBreakpoints(messages) <= 4);
    assert.ok(isMarked(messages[messages.length - 1]), "current user message should be marked");
  });

  test("marks the last tool result of each round in an agentic loop", () => {
    const messages: OpenAIChatMessage[] = [
      text("system", "sys"),
      text("user", "do a task"),
      {
        content: null,
        role: "assistant",
        tool_calls: [{ function: { arguments: "{}", name: "a" }, id: "1", type: "function" }],
      },
      { content: "result A", role: "tool", tool_call_id: "1" },
      {
        content: null,
        role: "assistant",
        tool_calls: [{ function: { arguments: "{}", name: "b" }, id: "2", type: "function" }],
      },
      { content: "result B", role: "tool", tool_call_id: "2" },
    ];
    addCacheBreakpoints(messages);
    // Both tool-result messages are the last of their round → both marked.
    assert.ok(isMarked(messages[3]), "first tool result should be marked");
    assert.ok(isMarked(messages[5]), "second tool result should be marked");
    assert.ok(countBreakpoints(messages) <= 4);
  });

  test("does not mark assistant messages that are only tool calls", () => {
    const messages: OpenAIChatMessage[] = [
      text("user", "q"),
      {
        content: null,
        role: "assistant",
        tool_calls: [{ function: { arguments: "{}", name: "a" }, id: "1", type: "function" }],
      },
      { content: "result", role: "tool", tool_call_id: "1" },
    ];
    addCacheBreakpoints(messages);
    assert.strictEqual(isMarked(messages[1]), false, "tool-call-only assistant must not be marked");
  });

  test("converts string content to a block and marks its last block", () => {
    const messages: OpenAIChatMessage[] = [text("user", "hello")];
    addCacheBreakpoints(messages);
    assert.ok(Array.isArray(messages[0].content));
    assert.ok(isMarked(messages[0]));
  });

  test("is idempotent (re-running does not exceed the cap)", () => {
    const messages: OpenAIChatMessage[] = [
      text("system", "sys"),
      text("user", "a"),
      text("assistant", "b"),
      text("user", "c"),
      text("assistant", "d"),
      text("user", "e"),
    ];
    addCacheBreakpoints(messages);
    const first = countBreakpoints(messages);
    addCacheBreakpoints(messages);
    assert.strictEqual(countBreakpoints(messages), first, "second run should not add more");
    assert.ok(first <= 4);
  });
});
