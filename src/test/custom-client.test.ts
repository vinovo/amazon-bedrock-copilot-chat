import * as assert from "node:assert";

import { CustomBackendClient } from "../custom/client";

/**
 * Structurally identical to the payload returned by a real OpenAI-compatible
 * gateway that reports models under a `models` array where `name` is a list of
 * routing aliases, the canonical id lives in `model_name`, and capability
 * metadata is expressed via `capabilities` and `model_type`.
 */
const GATEWAY_PAYLOAD = {
  models: [
    {
      capabilities: ["embedding"],
      is_finetuned: false,
      model_name: "vendor/embed-large",
      model_type: { is_chat: false, is_tool_supported: false, modality: "Embedding" },
      name: ["embed-large"],
    },
    {
      capabilities: ["chat", "completion"],
      is_finetuned: false,
      model_name: "vendor/Coder-30B",
      model_type: { is_chat: true, is_tool_supported: true, modality: "Llm" },
      name: ["coder-30b", "coder-alias", "coder-beta"],
    },
    {
      capabilities: ["realtime"],
      is_finetuned: false,
      model_name: "vendor/Voice-Realtime",
      model_type: { is_chat: false, is_tool_supported: false, modality: "Llm" },
      name: ["voice-realtime"],
    },
    {
      capabilities: ["completion", "chat"],
      is_finetuned: false,
      model_name: "vendor/Instruct-4B",
      model_type: { is_chat: true, is_tool_supported: true, modality: "Llm" },
      name: ["instruct-4b"],
    },
    {
      capabilities: ["completion", "chat"],
      is_finetuned: false,
      model_name: "vendor/Big-235B",
      model_type: { is_chat: true, is_tool_supported: true, modality: "Llm" },
      name: ["big-235b", "Pro", "pro"],
    },
    {
      capabilities: ["chat"],
      is_finetuned: false,
      model_name: "hosted-flash",
      model_type: { is_chat: true, is_tool_supported: true, modality: "Llm" },
      name: ["provider::hosted-flash"],
    },
  ],
};

function mockFetchReturning(payload: unknown): typeof globalThis.fetch {
  return (async () =>
    ({
      json: async () => payload,
      ok: true,
      status: 200,
    }) as unknown as Response) as typeof globalThis.fetch;
}

/** Build a mock streaming `fetch` whose response body replays the given raw SSE text. */
function mockFetchStreaming(sse: string): typeof globalThis.fetch {
  return (async () => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(sse));
        controller.close();
      },
    });
    return { body, ok: true, status: 200 } as unknown as Response;
  }) as typeof globalThis.fetch;
}

suite("Custom backend model discovery", () => {
  let originalFetch: typeof globalThis.fetch;

  setup(() => {
    originalFetch = globalThis.fetch;
  });

  teardown(() => {
    globalThis.fetch = originalFetch;
  });

  test("discovers chat models from an alias-array / capability-metadata payload", async () => {
    globalThis.fetch = mockFetchReturning(GATEWAY_PAYLOAD);

    const client = new CustomBackendClient({
      apiKey: "test-token",
      baseUrl: "https://example.test",
    });
    const models = await client.listModels();

    // Only the four chat-capable entries survive; embedding and realtime models
    // are filtered out.
    assert.strictEqual(models.length, 4, "should discover exactly the chat-capable models");
    assert.deepStrictEqual(
      models.map((model) => model.id),
      ["coder-30b", "instruct-4b", "big-235b", "provider::hosted-flash"],
      "id should be the first routing alias from the `name` array",
    );
    assert.ok(
      models.every((model) => model.toolCalling === true),
      "tool support should be derived from model_type.is_tool_supported",
    );
  });

  test("supports the standard OpenAI envelope and a bare array", async () => {
    globalThis.fetch = mockFetchReturning({ data: [{ id: "gpt-x" }, { id: "gpt-y" }] });
    const client = new CustomBackendClient({ apiKey: "t", baseUrl: "https://example.test" });
    const enveloped = await client.listModels();
    assert.deepStrictEqual(
      enveloped.map((model) => model.id),
      ["gpt-x", "gpt-y"],
    );

    globalThis.fetch = mockFetchReturning(["alpha", "beta"]);
    const bareArray = await client.listModels();
    assert.deepStrictEqual(
      bareArray.map((model) => model.id),
      ["alpha", "beta"],
    );
  });

  test("captures per-model context/output token limits from the OpenAI envelope", async () => {
    globalThis.fetch = mockFetchReturning({
      data: [
        { id: "big", max_input_tokens: 200_000, max_output_tokens: 8192 },
        { context_length: 1_000_000, id: "huge", max_tokens: 32_000 },
        { id: "bare" },
      ],
    });
    const client = new CustomBackendClient({ apiKey: "t", baseUrl: "https://example.test" });
    const models = await client.listModels();

    const byId = new Map(models.map((model) => [model.id, model]));
    assert.strictEqual(byId.get("big")?.maxInputTokens, 200_000);
    assert.strictEqual(byId.get("big")?.maxOutputTokens, 8192);
    assert.strictEqual(byId.get("huge")?.maxInputTokens, 1_000_000, "falls back to context_length");
    assert.strictEqual(byId.get("huge")?.maxOutputTokens, 32_000, "falls back to max_tokens");
    assert.strictEqual(byId.get("bare")?.maxInputTokens, undefined);
    assert.strictEqual(byId.get("bare")?.maxOutputTokens, undefined);
  });
});

suite("Custom backend streaming", () => {
  let originalFetch: typeof globalThis.fetch;

  setup(() => {
    originalFetch = globalThis.fetch;
  });

  teardown(() => {
    globalThis.fetch = originalFetch;
  });

  async function collect(sse: string) {
    globalThis.fetch = mockFetchStreaming(sse);
    const client = new CustomBackendClient({ apiKey: "t", baseUrl: "https://example.test" });
    const deltas = [];
    for await (const delta of client.streamChat({ messages: [], model: "m", stream: true })) {
      deltas.push(delta);
    }
    return deltas;
  }

  test("yields content deltas and captures the trailing usage chunk", async () => {
    const deltas = await collect(
      [
        `data: ${JSON.stringify({ choices: [{ delta: { content: "Hello" } }] })}\n\n`,
        `data: ${JSON.stringify({ choices: [{ delta: { content: " world" } }] })}\n\n`,
        `data: ${JSON.stringify({ choices: [], usage: { completion_tokens: 5, prompt_tokens: 1234 } })}\n\n`,
        `data: [DONE]\n\n`,
      ].join(""),
    );

    assert.deepStrictEqual(
      deltas.flatMap((d) => (d.content ? [d.content] : [])),
      ["Hello", " world"],
    );
    const usageDelta = deltas.find((d) => d.usage);
    assert.deepStrictEqual(usageDelta?.usage, {
      cacheWriteTokens: undefined,
      cachedTokens: undefined,
      completionTokens: 5,
      promptIncludesCached: false,
      promptTokens: 1234,
    });
  });

  test("recovers the final usage chunk when the stream ends without a trailing blank line", async () => {
    // No `\n\n` after the usage event and no `[DONE]`: the event would be lost
    // if the parser did not flush its trailing buffer.
    const deltas = await collect(
      `data: ${JSON.stringify({ choices: [{ delta: { content: "hi" } }] })}\n\n` +
        `data: ${JSON.stringify({ choices: [], usage: { completion_tokens: 7, prompt_tokens: 999 } })}`,
    );

    const usageDelta = deltas.find((d) => d.usage);
    assert.deepStrictEqual(
      usageDelta?.usage,
      {
        cacheWriteTokens: undefined,
        cachedTokens: undefined,
        completionTokens: 7,
        promptIncludesCached: false,
        promptTokens: 999,
      },
      "trailing usage event must be flushed when the stream closes unterminated",
    );
  });

  test("reconstructs the true prompt size from raw Anthropic top-level cache counters", async () => {
    // Raw Anthropic shape: cached tokens live in top-level
    // cache_read_input_tokens/cache_creation_input_tokens, outside prompt_tokens.
    const deltas = await collect(
      `data: ${JSON.stringify({
        choices: [],
        usage: {
          cache_creation_input_tokens: 200,
          cache_read_input_tokens: 93610,
          completion_tokens: 12,
          prompt_tokens: 2,
        },
      })}\n\n` + `data: [DONE]\n\n`,
    );
    const usage = deltas.find((d) => d.usage)?.usage;
    assert.strictEqual(usage?.cachedTokens, 93610);
    assert.strictEqual(usage?.cacheWriteTokens, 200);
    assert.strictEqual(usage?.promptTokens, 2);
    assert.strictEqual(usage?.promptIncludesCached, false, "raw Anthropic shape excludes cache");
  });

  test("reads QGenie nested cache_read_tokens/cache_write_tokens (outside prompt_tokens)", async () => {
    // Confirmed live QGenie shape: cache counters are NESTED in
    // prompt_tokens_details under cache_read_tokens/cache_write_tokens, and
    // prompt_tokens holds only the non-cached delta.
    const deltas = await collect(
      `data: ${JSON.stringify({
        choices: [],
        usage: {
          completion_tokens: 4,
          prompt_tokens: 3,
          prompt_tokens_details: { cache_read_tokens: 4403, cache_write_tokens: 11 },
          total_tokens: 4421,
        },
      })}\n\n` + `data: [DONE]\n\n`,
    );
    const usage = deltas.find((d) => d.usage)?.usage;
    assert.strictEqual(usage?.cachedTokens, 4403);
    assert.strictEqual(usage?.cacheWriteTokens, 11);
    assert.strictEqual(usage?.promptTokens, 3);
    assert.strictEqual(
      usage?.promptIncludesCached,
      false,
      "QGenie prompt_tokens excludes cached tokens",
    );
  });

  test("marks cached tokens as already included under the OpenAI convention", async () => {
    const deltas = await collect(
      `data: ${JSON.stringify({
        choices: [],
        usage: {
          completion_tokens: 12,
          prompt_tokens: 95000,
          prompt_tokens_details: { cached_tokens: 93610 },
        },
      })}\n\n` + `data: [DONE]\n\n`,
    );
    const usage = deltas.find((d) => d.usage)?.usage;
    assert.strictEqual(usage?.cachedTokens, 93610);
    assert.strictEqual(usage?.promptTokens, 95000);
    assert.strictEqual(usage?.promptIncludesCached, true, "OpenAI shape includes cache in prompt");
  });
});
