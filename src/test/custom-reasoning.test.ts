import * as assert from "node:assert";

import { CustomBackendClient } from "../custom/client";
import { CustomChatModelProvider } from "../custom/provider";
import {
  defaultReasoningEffort,
  heuristicReasoningCapability,
  reasoningEffortLabel,
} from "../custom/reasoning";

function mockFetchReturning(payload: unknown): typeof globalThis.fetch {
  return (async () =>
    ({
      json: async () => payload,
      ok: true,
      status: 200,
    }) as unknown as Response) as typeof globalThis.fetch;
}

/** Route `/model/info` to `infoPayload` and everything else to `listPayload`. */
function mockFetchByPath(listPayload: unknown, infoPayload: unknown): typeof globalThis.fetch {
  return (async (url: string) => {
    const payload = String(url).endsWith("/model/info") ? infoPayload : listPayload;
    return { json: async () => payload, ok: true, status: 200 } as unknown as Response;
  }) as unknown as typeof globalThis.fetch;
}

suite("Reasoning-effort family heuristic", () => {
  test("GPT-5 / o-series models get the six-level GPT set", () => {
    const gpt = heuristicReasoningCapability("gpt-5.2");
    assert.deepStrictEqual(gpt?.levels, ["none", "minimal", "low", "medium", "high", "xhigh"]);
    assert.strictEqual(gpt?.toolsIncompatibleWithReasoning, undefined);
    const o = heuristicReasoningCapability("o3-mini");
    assert.deepStrictEqual(o?.levels, ["none", "minimal", "low", "medium", "high", "xhigh"]);
  });

  test("gpt-5.6-* variants set toolsIncompatibleWithReasoning (reject tools+effort)", () => {
    // Confirmed against the live QGenie API: gpt-5.6-sol, gpt-5.6-terra, and
    // gpt-5.6-luna all return HTTP 400 when reasoning_effort + tools are both
    // present in the same request. effort=none or tools-only requests succeed.
    for (const id of ["gpt-5.6-sol", "azure::gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"]) {
      const cap = heuristicReasoningCapability(id);
      assert.strictEqual(cap?.toolsIncompatibleWithReasoning, true, `expected flag for ${id}`);
      assert.deepStrictEqual(cap?.levels, ["none", "minimal", "low", "medium", "high", "xhigh"]);
    }
  });

  test("Claude 4.x / 5 models get the five-level Claude set", () => {
    const claude5 = heuristicReasoningCapability("claude-5-sonnet");
    assert.deepStrictEqual(claude5?.levels, ["low", "medium", "high", "xhigh", "max"]);
    const claude46 = heuristicReasoningCapability("claude-4-6-opus");
    assert.deepStrictEqual(claude46?.levels, ["low", "medium", "high", "xhigh", "max"]);
  });

  test("strips provider prefix and variant suffix before matching", () => {
    const prefixed = heuristicReasoningCapability("anthropic::claude-5-sonnet:1M");
    assert.deepStrictEqual(prefixed?.levels, ["low", "medium", "high", "xhigh", "max"]);
  });

  test("matches when only an alias identifies the family", () => {
    const viaAlias = heuristicReasoningCapability("internal-router-42", "gpt-5-mini");
    assert.deepStrictEqual(viaAlias?.levels, ["none", "minimal", "low", "medium", "high", "xhigh"]);
  });

  test("non-reasoning families return undefined (no picker)", () => {
    assert.strictEqual(heuristicReasoningCapability("gpt-4o"), undefined);
    assert.strictEqual(heuristicReasoningCapability("llama-3-70b"), undefined);
  });

  test("generic thinking models get the conservative three-level set", () => {
    const thinking = heuristicReasoningCapability("qwen3-235b-a22b-thinking");
    assert.deepStrictEqual(thinking?.levels, ["low", "medium", "high"]);
  });
});

suite("Reasoning-effort defaults", () => {
  test("Claude prefers high, others prefer medium", () => {
    assert.strictEqual(
      defaultReasoningEffort(["low", "medium", "high", "xhigh", "max"], "claude-5-sonnet"),
      "high",
    );
    assert.strictEqual(
      defaultReasoningEffort(["none", "minimal", "low", "medium", "high", "xhigh"], "gpt-5"),
      "medium",
    );
  });

  test("falls back to the middle level when preferred is absent", () => {
    assert.strictEqual(defaultReasoningEffort(["low", "high"], "gpt-5"), "low");
  });

  test("labels are human-readable", () => {
    assert.strictEqual(reasoningEffortLabel("xhigh"), "Extra High");
    assert.strictEqual(reasoningEffortLabel("max"), "Max");
  });
});

suite("Custom backend reasoning discovery", () => {
  let originalFetch: typeof globalThis.fetch;

  setup(() => {
    originalFetch = globalThis.fetch;
  });

  teardown(() => {
    globalThis.fetch = originalFetch;
  });

  test("derives reasoning support from list metadata", async () => {
    globalThis.fetch = mockFetchReturning({
      data: [
        { id: "thinker", supports_reasoning: true },
        { id: "plain", supports_reasoning: false },
        { id: "unknown" },
      ],
    });
    const client = new CustomBackendClient({ apiKey: "t", baseUrl: "https://example.test" });
    const models = await client.listModels();
    assert.strictEqual(models.find((m) => m.id === "thinker")?.reasoning, true);
    assert.strictEqual(models.find((m) => m.id === "plain")?.reasoning, false);
    assert.strictEqual(models.find((m) => m.id === "unknown")?.reasoning, undefined);
  });

  test("fetchReasoningSupport reads LiteLLM /model/info model_info.supports_reasoning", async () => {
    globalThis.fetch = mockFetchByPath(
      { data: [{ id: "x" }] },
      {
        data: [
          { model_info: { supports_reasoning: true }, model_name: "claude-5-sonnet" },
          { model_info: { supports_reasoning: false }, model_name: "gpt-4o" },
        ],
      },
    );
    const client = new CustomBackendClient({ apiKey: "t", baseUrl: "https://example.test" });
    const support = await client.fetchReasoningSupport();
    assert.strictEqual(support.get("claude-5-sonnet"), true);
    assert.strictEqual(support.get("gpt-4o"), false);
  });

  test("fetchReasoningSupport swallows non-LiteLLM 404s", async () => {
    globalThis.fetch = (async () =>
      ({ ok: false, status: 404 }) as unknown as Response) as typeof globalThis.fetch;
    const client = new CustomBackendClient({ apiKey: "t", baseUrl: "https://example.test" });
    const support = await client.fetchReasoningSupport();
    assert.strictEqual(support.size, 0);
  });
});

suite("Custom backend reasoning picker schema", () => {
  const fakeMemento = {
    get: (_key: string, defaultValue?: unknown) => defaultValue,
    update: async () => undefined,
  } as never;

  /** Invoke the private buildModelInfo to inspect the emitted picker schema. */
  const buildModelInfo = (id: string) => {
    const provider = new CustomChatModelProvider(fakeMemento);
    const settings = { models: [], name: "Test backend" };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (provider as any).buildModelInfo(settings, id) as {
      configurationSchema?: { properties?: Record<string, Record<string, unknown>> };
      reasoning?: { levels: string[] };
    };
  };

  test("reasoningEffort picker uses group 'navigation' so VS Code renders it", () => {
    // Regression guard: VS Code's model picker discovers the Thinking Effort
    // dropdown by scanning configurationSchema for group === "navigation".
    // Any other group name leaves the picker hidden (see modelPickerConfiguration.ts).
    const info = buildModelInfo("claude-5-sonnet");
    const reasoningEffort = info.configurationSchema?.properties?.reasoningEffort;
    assert.ok(reasoningEffort, "expected a reasoningEffort picker for a Claude model");
    assert.strictEqual(reasoningEffort.group, "navigation");
    assert.deepStrictEqual(reasoningEffort.enum, ["low", "medium", "high", "xhigh", "max"]);
  });

  test("non-reasoning models expose no reasoningEffort picker", () => {
    const info = buildModelInfo("llama-3-70b");
    assert.strictEqual(info.configurationSchema?.properties?.reasoningEffort, undefined);
  });
});
