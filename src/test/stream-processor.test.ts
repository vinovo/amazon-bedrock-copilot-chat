import type { ConverseStreamOutput } from "@aws-sdk/client-bedrock-runtime";
import { StopReason } from "@aws-sdk/client-bedrock-runtime";
import * as assert from "node:assert";
import * as vscode from "vscode";

import { StreamProcessor, type StreamUsage } from "../stream-processor";

/**
 * Build a minimal `AsyncIterable<ConverseStreamOutput>` from a fixed list of
 * events. Mirrors the shape of `BedrockRuntimeClient.send(ConverseStreamCommand).stream`.
 */
function asAsyncIterable<T>(events: T[]): AsyncIterable<T> {
  return {
    [Symbol.asyncIterator]() {
      let i = 0;
      return {
        async next() {
          if (i < events.length) {
            return { done: false, value: events[i++] };
          }
          return { done: true, value: undefined };
        },
      };
    },
  };
}

/**
 * Construct a Bedrock stream that returns one text chunk plus a metadata event
 * with the supplied usage payload. The resulting `result.usage` is what
 * `processResponseStream` would forward to Copilot Chat as a
 * `LanguageModelDataPart` with MIME `"usage"`.
 */
function streamWithUsage(usage: NonNullable<ConverseStreamOutput["metadata"]>["usage"]) {
  const events: ConverseStreamOutput[] = [
    { messageStart: { role: "assistant" } },
    { contentBlockDelta: { contentBlockIndex: 0, delta: { text: "Hello" } } },
    { contentBlockStop: { contentBlockIndex: 0 } },
    { messageStop: { stopReason: StopReason.END_TURN } },
    { metadata: { metrics: { latencyMs: 0 }, usage } },
  ];
  return asAsyncIterable(events);
}

suite("StreamProcessor — usage capture", () => {
  test("captures inputTokens and outputTokens from metadata.usage", async () => {
    const proc = new StreamProcessor();
    const reported: vscode.LanguageModelResponsePart[] = [];

    const result = await proc.processStream(
      streamWithUsage({ inputTokens: 100, outputTokens: 50, totalTokens: 150 }),
      { report: (p) => reported.push(p) },
      new vscode.CancellationTokenSource().token,
    );

    assert.deepStrictEqual(result.usage, {
      cacheReadInputTokens: undefined,
      cacheWriteInputTokens: undefined,
      inputTokens: 100,
      outputTokens: 50,
      totalTokens: 150,
    } satisfies StreamUsage);
  });

  test("propagates cache-read and cache-write tokens when present", async () => {
    const proc = new StreamProcessor();

    const result = await proc.processStream(
      streamWithUsage({
        cacheReadInputTokens: 150,
        cacheWriteInputTokens: 50,
        inputTokens: 200,
        outputTokens: 80,
        totalTokens: 280,
      }),
      { report: () => {} },
      new vscode.CancellationTokenSource().token,
    );

    assert.deepStrictEqual(result.usage, {
      cacheReadInputTokens: 150,
      cacheWriteInputTokens: 50,
      inputTokens: 200,
      outputTokens: 80,
      totalTokens: 280,
    } satisfies StreamUsage);
  });

  test("leaves usage undefined when metadata reports no usage", async () => {
    const proc = new StreamProcessor();

    const result = await proc.processStream(
      asAsyncIterable<ConverseStreamOutput>([
        { messageStart: { role: "assistant" } },
        { contentBlockDelta: { contentBlockIndex: 0, delta: { text: "ok" } } },
        { contentBlockStop: { contentBlockIndex: 0 } },
        { messageStop: { stopReason: StopReason.END_TURN } },
        // No metadata event at all.
      ]),
      { report: () => {} },
      new vscode.CancellationTokenSource().token,
    );

    assert.equal(result.usage, undefined);
  });

  test("ignores malformed metadata.usage missing required token fields", async () => {
    const proc = new StreamProcessor();

    const result = await proc.processStream(
      asAsyncIterable<ConverseStreamOutput>([
        { messageStart: { role: "assistant" } },
        { contentBlockDelta: { contentBlockIndex: 0, delta: { text: "ok" } } },
        { contentBlockStop: { contentBlockIndex: 0 } },
        { messageStop: { stopReason: StopReason.END_TURN } },
        // Usage with only one of the two required fields — should not be captured.
        // Cast around the SDK's strict TokenUsage shape: the runtime contract
        // we're guarding against is exactly "fields missing despite the type",
        // which the SDK cannot express.
        {
          metadata: {
            metrics: { latencyMs: 0 },
            usage: { inputTokens: 10 } as unknown as NonNullable<
              ConverseStreamOutput["metadata"]
            >["usage"],
          },
        },
      ]),
      { report: () => {} },
      new vscode.CancellationTokenSource().token,
    );

    assert.equal(result.usage, undefined);
  });
});
