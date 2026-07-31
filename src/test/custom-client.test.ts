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
});
