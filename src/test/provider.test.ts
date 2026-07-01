import { ModelModality } from "@aws-sdk/client-bedrock";
import * as assert from "node:assert";
import * as vscode from "vscode";
import { convertMessages, stripThinkingContent } from "../converters/messages";
import { convertTools } from "../converters/tools";
import { logger } from "../logger";
import { BEDROCK_ERROR_SENTINEL_ID, BedrockChatModelProvider } from "../provider";
import type { BedrockModelSummary } from "../types";

// Mock implementations extracted to avoid function nesting depth issues
const mockSecretStorage = {
  delete: async () => {},
  get: async () => undefined as string | undefined,
  keys: async () => [],
  onDidChange: () => ({ dispose: () => {} }),
  store: async () => {},
} as vscode.SecretStorage;

const mockGlobalState: vscode.Memento = {
  get: async () => {},
  keys: () => [],
  update: async () => {},
} as unknown as vscode.Memento;

// Helper to cast non-standard content blocks that may appear from models with extended thinking
const nonStandardBlock = (b: unknown) => b as any;

// Helper to create a test model for buildModelCandidates and findAlternativeProfile tests
const createTestModel = (modelId: string): BedrockModelSummary => ({
  inputModalities: [ModelModality.TEXT],
  modelArn: `arn:aws:bedrock:us-west-2::foundation-model/${modelId}`,
  modelId,
  modelName: modelId,
  outputModalities: [ModelModality.TEXT],
  providerName: "Anthropic",
  responseStreamingSupported: true,
});

// Helper to create a mock provider with controlled base model accessibility
const createMockClientProvider = (baseModelAccessible: boolean) => {
  const mockClient = {
    isModelAccessible: async () => baseModelAccessible,
  };
  const provider = new BedrockChatModelProvider(mockSecretStorage, mockGlobalState);

  (provider as any).client = mockClient;
  return provider;
};

// Helper to call the private calculateThinkingConfig method
const callCalcThinkingConfig = (
  modelProfile: { supportsThinking: boolean },
  modelLimits: { maxOutputTokens: number },
  maxTokensForRequest: number,
  thinkingEnabled: boolean,
) => {
  const provider = new BedrockChatModelProvider(mockSecretStorage, mockGlobalState);
  return (provider as any).calculateThinkingConfig(
    modelProfile,
    modelLimits,
    maxTokensForRequest,
    thinkingEnabled,
  ) as { budgetTokens: number; extendedThinkingEnabled: boolean };
};

// Helper to call the private buildModelConfigurationSchema method
const buildModelConfigurationSchema = (modelId: string, context1MEnabled = true) => {
  const provider = new BedrockChatModelProvider(mockSecretStorage, mockGlobalState);
  return (provider as any).buildModelConfigurationSchema(modelId, context1MEnabled) as
    | undefined
    | { properties?: Record<string, Record<string, unknown>> };
};

// Helper to call the private applyModelConfigurationOverrides method against a minimal
// settings object, returning the mutated settings for assertions.
const applyConfigOverride = (modelConfiguration: Record<string, unknown> | undefined) => {
  const provider = new BedrockChatModelProvider(mockSecretStorage, mockGlobalState);
  const settings = {
    context1M: { enabled: true },
    thinking: { display: "summarized", effort: "high", enabled: true },
  } as any;
  (provider as any).applyModelConfigurationOverrides(
    "anthropic.claude-opus-4-8-20251101-v1:0",
    settings,
    modelConfiguration,
  );
  return settings as {
    context1M: { enabled: boolean };
    thinking: { display: string; effort: string; enabled: boolean };
  };
};

suite("Amazon Bedrock Chat Provider Extension", () => {
  suite("provider", () => {
    test("prepareLanguageModelChatInformation returns array (no key -> sentinel only)", async () => {
      // With no auth configured, the silent path now returns a single sentinel
      // entry instead of an empty array, so VS Code core does not silently swap
      // the user's selected Bedrock model to a default model.
      const provider = new BedrockChatModelProvider(mockSecretStorage, mockGlobalState);

      const infos = await provider.prepareLanguageModelChatInformation(
        { silent: true },
        new vscode.CancellationTokenSource().token,
      );
      assert.ok(Array.isArray(infos));
      assert.equal(infos.length, 1);
      assert.equal(infos[0].id, BEDROCK_ERROR_SENTINEL_ID);
    });

    test("buildSentinelModelList returns sentinel-only when no successful fetch has occurred", () => {
      const provider = new BedrockChatModelProvider(mockSecretStorage, mockGlobalState);
      const result = (provider as any).buildSentinelModelList(
        new Error("ExpiredToken: AWS credentials expired"),
      ) as vscode.LanguageModelChatInformation[];

      assert.equal(result.length, 1);
      assert.equal(result[0].id, BEDROCK_ERROR_SENTINEL_ID);
      assert.equal(result[0].family, "bedrock-error");
      assert.ok(result[0].name.includes("Bedrock unavailable"));
      // The failure reason must be surfaced via tooltip and detail so the
      // user can diagnose without sending a request.
      assert.ok((result[0] as any).detail.includes("ExpiredToken"));
      assert.ok(result[0].tooltip?.includes("ExpiredToken"));
      // Tool calling and image input are disabled so VS Code does not advertise
      // capabilities that the sentinel cannot service.
      assert.equal(result[0].capabilities?.toolCalling, false);
      assert.equal(result[0].capabilities?.imageInput, false);
    });

    test("buildSentinelModelList appends sentinel to last-known model list after a prior success", () => {
      const provider = new BedrockChatModelProvider(mockSecretStorage, mockGlobalState);
      const cached: vscode.LanguageModelChatInformation[] = [
        {
          capabilities: { toolCalling: true },
          family: "bedrock",
          id: "anthropic.claude-sonnet-4-5-20250929-v1:0",
          maxInputTokens: 200_000,
          maxOutputTokens: 8192,
          name: "Claude Sonnet 4.5",
          version: "1.0.0",
        } as unknown as vscode.LanguageModelChatInformation,
      ];
      (provider as any).lastKnownModels = cached;

      const result = (provider as any).buildSentinelModelList(
        new Error("network blip"),
      ) as vscode.LanguageModelChatInformation[];

      assert.equal(result.length, 2);
      assert.equal(result[0].id, "anthropic.claude-sonnet-4-5-20250929-v1:0");
      assert.equal(result[1].id, BEDROCK_ERROR_SENTINEL_ID);
      // Sentinel always last so picker ordering is stable across failures.
      assert.ok((result[1] as any).detail.includes("network blip"));
    });

    test("provideLanguageModelChatResponse rejects the error sentinel without contacting AWS", async () => {
      const provider = new BedrockChatModelProvider(mockSecretStorage, mockGlobalState);
      // Replace the AWS client with a throwing stub: if the sentinel guard fails
      // and any AWS call is attempted, the test fails with a distinguishable error.
      (provider as any).client = {
        countTokens: async () => {
          throw new Error("AWS should not be contacted for the sentinel");
        },
        fetchModels: async () => {
          throw new Error("AWS should not be contacted for the sentinel");
        },
        resolveModelId: async () => {
          throw new Error("AWS should not be contacted for the sentinel");
        },
        setAuthConfig: () => {},
        setProfile: () => {},
        setRegion: () => {},
      };

      const sentinelModel: vscode.LanguageModelChatInformation = {
        capabilities: { imageInput: false, toolCalling: false },
        detail: "ExpiredToken: AWS credentials expired",
        family: "bedrock-error",
        id: BEDROCK_ERROR_SENTINEL_ID,
        maxInputTokens: 1,
        maxOutputTokens: 1,
        name: "⚠ Bedrock unavailable",
        version: "1.0.0",
      } as unknown as vscode.LanguageModelChatInformation;

      const reportedParts: vscode.LanguageModelResponsePart[] = [];
      const progress: vscode.Progress<vscode.LanguageModelResponsePart> = {
        report: (part) => reportedParts.push(part),
      };

      await assert.rejects(
        async () =>
          provider.provideLanguageModelChatResponse(
            sentinelModel,
            [],
            {} as Parameters<typeof provider.provideLanguageModelChatResponse>[2],
            progress,
            new vscode.CancellationTokenSource().token,
          ),
        (err: unknown) => {
          assert.ok(err instanceof Error);
          // The thrown error must surface the underlying reason so the user
          // can diagnose without inspecting logs.
          assert.ok(err.message.includes("ExpiredToken"));
          assert.ok(err.message.includes("could not be loaded"));
          return true;
        },
      );
      // No content should be emitted for the sentinel.
      assert.equal(reportedParts.length, 0);
    });

    test("provideTokenCount short-circuits to 0 for the error sentinel", async () => {
      const provider = new BedrockChatModelProvider(mockSecretStorage, mockGlobalState);
      // Same throwing stub as above: the sentinel guard must run before AWS.
      (provider as any).client = {
        countTokens: async () => {
          throw new Error("AWS should not be contacted for the sentinel");
        },
        resolveModelId: async () => {
          throw new Error("AWS should not be contacted for the sentinel");
        },
      };

      const sentinelModel: vscode.LanguageModelChatInformation = {
        capabilities: { imageInput: false, toolCalling: false },
        family: "bedrock-error",
        id: BEDROCK_ERROR_SENTINEL_ID,
        maxInputTokens: 1,
        maxOutputTokens: 1,
        name: "⚠ Bedrock unavailable",
        version: "1.0.0",
      } as unknown as vscode.LanguageModelChatInformation;

      const count = await provider.provideTokenCount(
        sentinelModel,
        "some prompt text that would otherwise be tokenized",
        new vscode.CancellationTokenSource().token,
      );
      assert.equal(count, 0);
    });

    test("provideTokenCount counts simple string", async () => {
      const provider = new BedrockChatModelProvider(mockSecretStorage, mockGlobalState);

      const est = await provider.provideTokenCount(
        {
          capabilities: {},
          family: "bedrock",
          id: "m",
          maxInputTokens: 1000,
          maxOutputTokens: 1000,
          name: "m",
          version: "1.0.0",
        } as unknown as vscode.LanguageModelChatInformation,
        "hello world",
        new vscode.CancellationTokenSource().token,
      );
      assert.equal(typeof est, "number");
      assert.ok(est > 0);
    });
  });

  suite("utils/convertMessages", () => {
    test("converts basic user/assistant text messages to Bedrock format", () => {
      const messages: vscode.LanguageModelChatMessage[] = [
        {
          content: [new vscode.LanguageModelTextPart("hi")],
          name: undefined,
          role: vscode.LanguageModelChatMessageRole.User,
        },
        {
          content: [new vscode.LanguageModelTextPart("hello")],
          name: undefined,
          role: vscode.LanguageModelChatMessageRole.Assistant,
        },
      ];
      const out = convertMessages(messages, "test.model-id");

      // Check structure
      assert.ok(out.messages);
      assert.ok(Array.isArray(out.messages));
      assert.equal(out.messages.length, 2);

      // Check first message (user)
      assert.equal(out.messages[0].role, "user");
      assert.ok(Array.isArray(out.messages[0].content));
      assert.equal(out.messages[0].content?.length, 1);
      assert.equal(out.messages[0].content?.[0]?.text, "hi");

      // Check second message (assistant)
      assert.equal(out.messages[1].role, "assistant");
      assert.ok(Array.isArray(out.messages[1].content));
      assert.equal(out.messages[1].content?.length, 1);
      assert.equal(out.messages[1].content?.[0]?.text, "hello");
    });

    test("converts assistant message with tool call to Bedrock format", () => {
      const toolCall = new vscode.LanguageModelToolCallPart("call1", "search", { q: "hello" });
      const messages: vscode.LanguageModelChatMessage[] = [
        {
          content: [new vscode.LanguageModelTextPart("Let me search for that")],
          name: undefined,
          role: vscode.LanguageModelChatMessageRole.User,
        },
        {
          content: [new vscode.LanguageModelTextPart("I'll search for you."), toolCall],
          name: undefined,
          role: vscode.LanguageModelChatMessageRole.Assistant,
        },
      ];

      const out = convertMessages(messages, "test.model-id");

      assert.equal(out.messages.length, 2);
      assert.equal(out.messages[1].role, "assistant");
      assert.equal(out.messages[1].content?.length, 2);

      // Check text content
      assert.equal(out.messages[1].content?.[0]?.text, "I'll search for you.");

      // Check tool call
      const toolUse = out.messages[1].content?.[1]?.toolUse;
      assert.ok(toolUse);
      assert.equal(toolUse.toolUseId, "call1");
      assert.equal(toolUse.name, "search");
      assert.deepStrictEqual(toolUse.input, { q: "hello" });
    });

    test("converts user message with tool result to Bedrock format", () => {
      const toolResult = new vscode.LanguageModelToolResultPart("call1", [
        new vscode.LanguageModelTextPart("Search results: Found 5 items"),
      ]);
      const messages: vscode.LanguageModelChatMessage[] = [
        {
          content: [new vscode.LanguageModelTextPart("Search for cats")],
          name: undefined,
          role: vscode.LanguageModelChatMessageRole.User,
        },
        {
          content: [new vscode.LanguageModelToolCallPart("call1", "search", { q: "cats" })],
          name: undefined,
          role: vscode.LanguageModelChatMessageRole.Assistant,
        },
        {
          content: [toolResult],
          name: undefined,
          role: vscode.LanguageModelChatMessageRole.User,
        },
      ];

      const out = convertMessages(messages, "test.model-id");

      assert.equal(out.messages.length, 3);
      assert.equal(out.messages[2].role, "user");

      // Check tool result
      const toolResultBlock = out.messages[2].content?.[0]?.toolResult;
      assert.ok(toolResultBlock);
      assert.equal(toolResultBlock.toolUseId, "call1");
      assert.ok(Array.isArray(toolResultBlock.content));
      assert.equal(toolResultBlock.content?.length, 1);
      // Tool result content is wrapped in a text object
      const resultContent: any = toolResultBlock.content?.[0];
      assert.ok(resultContent.text);
      assert.equal(resultContent.text, "Search results: Found 5 items");
    });

    test("merges consecutive user messages", () => {
      const messages: vscode.LanguageModelChatMessage[] = [
        {
          content: [new vscode.LanguageModelTextPart("First user message")],
          name: undefined,
          role: vscode.LanguageModelChatMessageRole.User,
        },
        {
          content: [new vscode.LanguageModelTextPart("Second user message")],
          name: undefined,
          role: vscode.LanguageModelChatMessageRole.User,
        },
      ];

      const out = convertMessages(messages, "test.model-id");

      // Should merge consecutive user messages into one
      assert.equal(out.messages.length, 1);
      assert.equal(out.messages[0].role, "user");
      assert.equal(out.messages[0].content?.length, 2);
      assert.equal(out.messages[0].content?.[0]?.text, "First user message");
      assert.equal(out.messages[0].content?.[1]?.text, "Second user message");
    });
  });

  suite("utils/stripThinkingContent", () => {
    test("strips reasoningContent blocks from messages", () => {
      const messages = [
        {
          content: [
            { reasoningContent: { reasoningText: { signature: "sig123", text: "thinking..." } } },
            { text: "Hello" },
          ],
          role: "assistant" as const,
        },
      ];

      const result = stripThinkingContent(messages);

      assert.equal(result.length, 1);
      assert.equal(result[0].content?.length, 1);
      assert.equal(result[0].content?.[0]?.text, "Hello");
    });

    test("strips thinking and redacted_thinking blocks from messages", () => {
      const messages = [
        {
          content: [
            nonStandardBlock({ thinking: "internal thought process" }),
            { text: "Response" },
          ],
          role: "assistant" as const,
        },
        {
          content: [
            nonStandardBlock({ redacted_thinking: "redacted" }),
            { text: "Another response" },
          ],
          role: "assistant" as const,
        },
      ];

      const result = stripThinkingContent(messages);

      assert.equal(result.length, 2);
      assert.equal(result[0].content?.length, 1);
      assert.equal(result[0].content?.[0]?.text, "Response");
      assert.equal(result[1].content?.length, 1);
      assert.equal(result[1].content?.[0]?.text, "Another response");
    });

    test("preserves non-thinking content", () => {
      const messages = [
        {
          content: [{ text: "User message" }],
          role: "user" as const,
        },
        {
          content: [
            { text: "Assistant response" },
            { toolUse: { input: {}, name: "search", toolUseId: "123" } },
          ],
          role: "assistant" as const,
        },
      ];

      const result = stripThinkingContent(messages);

      assert.equal(result.length, 2);
      assert.equal(result[0].content?.length, 1);
      assert.equal(result[1].content?.length, 2);
    });

    test("removes empty messages after stripping", () => {
      const messages = [
        {
          content: [
            { reasoningContent: { reasoningText: { signature: "sig", text: "only thinking" } } },
          ],
          role: "assistant" as const,
        },
        {
          content: [{ text: "Keep this" }],
          role: "user" as const,
        },
      ];

      const result = stripThinkingContent(messages);

      assert.equal(result.length, 1);
      assert.equal(result[0].role, "user");
      assert.equal(result[0].content?.[0]?.text, "Keep this");
    });

    test("handles messages without content", () => {
      const messages = [
        { content: undefined, role: "user" as const },
        { content: [{ text: "Hello" }], role: "assistant" as const },
      ];

      const result = stripThinkingContent(messages as any);

      // Message without content is removed (empty content array check)
      assert.equal(result.length, 1);
      assert.equal(result[0].content?.[0]?.text, "Hello");
    });
  });

  suite("utils/tools", () => {
    test("convertTools creates Bedrock tool configuration", () => {
      const out = convertTools(
        {
          requestInitiator: "",
          toolMode: vscode.LanguageModelChatToolMode.Auto,
          tools: [
            {
              description: "Does something",
              inputSchema: {
                additionalProperties: false,
                properties: { x: { type: "number" } },
                required: ["x"],
                type: "object",
              },
              name: "do_something",
            },
          ],
        } as vscode.ProvideLanguageModelChatResponseOptions,
        "test.model-id",
      );

      assert.ok(out);
      assert.ok(out.tools);
      assert.ok(Array.isArray(out.tools));
      assert.equal(out.tools.length, 1);

      // Check tool spec
      const toolSpec = out.tools[0].toolSpec;
      assert.ok(toolSpec);
      assert.equal(toolSpec.name, "do_something");
      assert.equal(toolSpec.description, "Does something");
      assert.ok(toolSpec.inputSchema);
      assert.ok(toolSpec.inputSchema.json);

      // Tool choice should not be set for models that don't support it
      assert.equal(out.toolChoice, undefined);
    });

    test("convertTools sets toolChoice for models that support it", () => {
      // Test with Anthropic model (supports tool choice)
      const out = convertTools(
        {
          requestInitiator: "",
          toolMode: vscode.LanguageModelChatToolMode.Required,
          tools: [
            {
              description: "Only tool",
              inputSchema: { type: "object" },
              name: "only_tool",
            },
          ],
        } as vscode.ProvideLanguageModelChatResponseOptions,
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
      );

      assert.ok(out);
      assert.ok(out.toolChoice);
      assert.deepStrictEqual(out.toolChoice, { any: {} });
    });

    test("convertTools handles Auto tool mode", () => {
      const out = convertTools(
        {
          requestInitiator: "",
          toolMode: vscode.LanguageModelChatToolMode.Auto,
          tools: [
            {
              description: "Tool",
              inputSchema: { type: "object" },
              name: "my_tool",
            },
          ],
        } as vscode.ProvideLanguageModelChatResponseOptions,
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
      );

      assert.ok(out);
      assert.ok(out.toolChoice);
      assert.deepStrictEqual(out.toolChoice, { auto: {} });
    });

    test("convertTools returns undefined when no tools provided", () => {
      const out = convertTools(
        {
          requestInitiator: "",
          toolMode: vscode.LanguageModelChatToolMode.Auto,
          tools: [],
        } as vscode.ProvideLanguageModelChatResponseOptions,
        "test.model-id",
      );

      assert.equal(out, undefined);
    });
  });

  // Note: validation tests skipped - validateBedrockMessages now validates converted messages

  suite("logger", () => {
    test("logger supports all log levels", () => {
      const logs: { args: unknown[]; level: string }[] = [];
      const mockChannel = {
        debug: (msg: string, ...args: unknown[]) =>
          logs.push({ args: [msg, ...args], level: "debug" }),
        error: (msg: string, ...args: unknown[]) =>
          logs.push({ args: [msg, ...args], level: "error" }),
        info: (msg: string, ...args: unknown[]) =>
          logs.push({ args: [msg, ...args], level: "info" }),
        trace: (msg: string, ...args: unknown[]) =>
          logs.push({ args: [msg, ...args], level: "trace" }),
        warn: (msg: string, ...args: unknown[]) =>
          logs.push({ args: [msg, ...args], level: "warn" }),
      } as unknown as vscode.LogOutputChannel;

      logger.initialize(mockChannel, vscode.ExtensionMode.Development);

      logger.trace("Trace message");
      logger.debug("Debug message");
      logger.info("Info message");
      logger.warn("Warn message");
      logger.error("Error message");

      assert.equal(logs.length, 5);
      assert.equal(logs[0].level, "trace");
      assert.equal(logs[0].args[0], "Trace message");
      assert.equal(logs[1].level, "debug");
      assert.equal(logs[1].args[0], "Debug message");
      assert.equal(logs[2].level, "info");
      assert.equal(logs[2].args[0], "Info message");
      assert.equal(logs[3].level, "warn");
      assert.equal(logs[3].args[0], "Warn message");
      assert.equal(logs[4].level, "error");
      assert.equal(logs[4].args[0], "Error message");
    });

    test("logger.log (deprecated) uses info level", () => {
      const logs: { args: unknown[]; level: string }[] = [];
      const mockChannel = {
        debug: (msg: string, ...args: unknown[]) =>
          logs.push({ args: [msg, ...args], level: "debug" }),
        error: (msg: string, ...args: unknown[]) =>
          logs.push({ args: [msg, ...args], level: "error" }),
        info: (msg: string, ...args: unknown[]) =>
          logs.push({ args: [msg, ...args], level: "info" }),
        trace: (msg: string, ...args: unknown[]) =>
          logs.push({ args: [msg, ...args], level: "trace" }),
        warn: (msg: string, ...args: unknown[]) =>
          logs.push({ args: [msg, ...args], level: "warn" }),
      } as unknown as vscode.LogOutputChannel;

      logger.initialize(mockChannel, vscode.ExtensionMode.Production);
      logger.log("Deprecated log message", { key: "value" });

      assert.equal(logs.length, 1);
      assert.equal(logs[0].level, "info");
      assert.equal(logs[0].args[0], "Deprecated log message");
      assert.deepStrictEqual(logs[0].args[1], { key: "value" });
    });

    test("logger passes structured data directly", () => {
      const logs: { args: unknown[]; level: string }[] = [];
      const mockChannel = {
        debug: (msg: string, ...args: unknown[]) =>
          logs.push({ args: [msg, ...args], level: "debug" }),
        error: (msg: string, ...args: unknown[]) =>
          logs.push({ args: [msg, ...args], level: "error" }),
        info: (msg: string, ...args: unknown[]) =>
          logs.push({ args: [msg, ...args], level: "info" }),
        trace: (msg: string, ...args: unknown[]) =>
          logs.push({ args: [msg, ...args], level: "trace" }),
        warn: (msg: string, ...args: unknown[]) =>
          logs.push({ args: [msg, ...args], level: "warn" }),
      } as unknown as vscode.LogOutputChannel;

      logger.initialize(mockChannel, vscode.ExtensionMode.Development);
      logger.info("Object test", { nested: { key: "value" } });

      assert.equal(logs.length, 1);
      assert.equal(logs[0].args[0], "Object test");
      // Structured data is preserved as an object, not formatted
      assert.deepStrictEqual(logs[0].args[1], { nested: { key: "value" } });
    });
  });

  suite("buildModelCandidates", () => {
    test("preferRegional=false (default): prioritizes global profiles", () => {
      const provider = new BedrockChatModelProvider(mockSecretStorage, mockGlobalState);
      const models = [createTestModel("anthropic.claude-3-5-sonnet-20241022-v2:0")];
      const availableProfiles = new Set([
        "global.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us-west-2.anthropic.claude-3-5-sonnet-20241022-v2:0",
      ]);

      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      const candidates = (provider as any).buildModelCandidates(
        models,
        availableProfiles,
        "us-west-2",
        false,
      );

      assert.equal(candidates.length, 1);

      assert.equal(candidates[0].modelIdToUse, "global.anthropic.claude-3-5-sonnet-20241022-v2:0");

      assert.equal(candidates[0].hasInferenceProfile, true);
    });

    test("preferRegional=true: prioritizes regional profiles", () => {
      const provider = new BedrockChatModelProvider(mockSecretStorage, mockGlobalState);
      const models = [createTestModel("anthropic.claude-3-5-sonnet-20241022-v2:0")];
      const availableProfiles = new Set([
        "global.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us-west-2.anthropic.claude-3-5-sonnet-20241022-v2:0",
      ]);

      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      const candidates = (provider as any).buildModelCandidates(
        models,
        availableProfiles,
        "us-west-2",
        true,
      );

      assert.equal(candidates.length, 1);

      assert.equal(
        candidates[0].modelIdToUse,
        "us-west-2.anthropic.claude-3-5-sonnet-20241022-v2:0",
      );

      assert.equal(candidates[0].hasInferenceProfile, true);
    });

    test("preferRegional=true, regional unavailable: falls back to global", () => {
      const provider = new BedrockChatModelProvider(mockSecretStorage, mockGlobalState);
      const models = [createTestModel("anthropic.claude-3-5-sonnet-20241022-v2:0")];
      const availableProfiles = new Set(["global.anthropic.claude-3-5-sonnet-20241022-v2:0"]);

      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      const candidates = (provider as any).buildModelCandidates(
        models,
        availableProfiles,
        "us-west-2",
        true,
      );

      assert.equal(candidates.length, 1);

      assert.equal(candidates[0].modelIdToUse, "global.anthropic.claude-3-5-sonnet-20241022-v2:0");

      assert.equal(candidates[0].hasInferenceProfile, true);
    });

    test("preferRegional=false, global unavailable: uses regional profile", () => {
      const provider = new BedrockChatModelProvider(mockSecretStorage, mockGlobalState);
      const models = [createTestModel("anthropic.claude-3-5-sonnet-20241022-v2:0")];
      const availableProfiles = new Set(["us-west-2.anthropic.claude-3-5-sonnet-20241022-v2:0"]);

      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      const candidates = (provider as any).buildModelCandidates(
        models,
        availableProfiles,
        "us-west-2",
        false,
      );

      assert.equal(candidates.length, 1);

      assert.equal(
        candidates[0].modelIdToUse,
        "us-west-2.anthropic.claude-3-5-sonnet-20241022-v2:0",
      );

      assert.equal(candidates[0].hasInferenceProfile, true);
    });

    test("no profiles available: uses base model", () => {
      const provider = new BedrockChatModelProvider(mockSecretStorage, mockGlobalState);
      const models = [createTestModel("anthropic.claude-3-5-sonnet-20241022-v2:0")];
      const availableProfiles = new Set<string>();

      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      const candidates = (provider as any).buildModelCandidates(
        models,
        availableProfiles,
        "us-west-2",
        false,
      );

      assert.equal(candidates.length, 1);

      assert.equal(candidates[0].modelIdToUse, "anthropic.claude-3-5-sonnet-20241022-v2:0");

      assert.equal(candidates[0].hasInferenceProfile, false);
    });

    test("filters models without streaming support", () => {
      const provider = new BedrockChatModelProvider(mockSecretStorage, mockGlobalState);
      const models = [
        {
          ...createTestModel("no-streaming-model"),
          responseStreamingSupported: false,
        },
      ];
      const availableProfiles = new Set<string>();

      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      const candidates = (provider as any).buildModelCandidates(
        models,
        availableProfiles,
        "us-west-2",
        false,
      );

      assert.equal(candidates.length, 0);
    });

    test("filters models without TEXT output modality", () => {
      const provider = new BedrockChatModelProvider(mockSecretStorage, mockGlobalState);
      const models = [
        {
          ...createTestModel("embedding-only-model"),
          outputModalities: ["EMBEDDING"],
        },
      ];
      const availableProfiles = new Set<string>();

      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      const candidates = (provider as any).buildModelCandidates(
        models,
        availableProfiles,
        "us-west-2",
        false,
      );

      assert.equal(candidates.length, 0);
    });

    test("returns correct candidate structure", () => {
      const provider = new BedrockChatModelProvider(mockSecretStorage, mockGlobalState);
      const model = createTestModel("anthropic.claude-3-5-sonnet-20241022-v2:0");
      const models = [model];
      const availableProfiles = new Set(["global.anthropic.claude-3-5-sonnet-20241022-v2:0"]);

      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      const candidates = (provider as any).buildModelCandidates(
        models,
        availableProfiles,
        "us-west-2",
        false,
      );

      assert.equal(candidates.length, 1);
      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      const candidate = candidates[0];
      assert.ok(candidate);

      assert.equal(candidate.modelIdToUse, "global.anthropic.claude-3-5-sonnet-20241022-v2:0");

      assert.equal(candidate.hasInferenceProfile, true);

      assert.deepStrictEqual(candidate.model, model);
    });
  });

  suite("findAlternativeProfile", () => {
    test("preferRegional=false, global fails: falls back to regional profile", async () => {
      const provider = createMockClientProvider(false);
      const model = createTestModel("anthropic.claude-3-5-sonnet-20241022-v2:0");
      const candidate = {
        hasInferenceProfile: true,
        model,
        modelIdToUse: "global.anthropic.claude-3-5-sonnet-20241022-v2:0",
      };
      const availableProfiles = new Set([
        "global.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us-west-2.anthropic.claude-3-5-sonnet-20241022-v2:0",
      ]);
      const abortController = new AbortController();

      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      const result = await (provider as any).findAlternativeProfile(
        candidate,
        "us-west-2",
        availableProfiles,
        false,
        abortController.signal,
      );

      assert.equal(result.isAccessible, true);

      assert.equal(result.hasInferenceProfile, true);

      assert.equal(result.modelIdToUse, "us-west-2.anthropic.claude-3-5-sonnet-20241022-v2:0");
    });

    test("preferRegional=false, regional fails: falls back to global profile", async () => {
      const provider = createMockClientProvider(false);
      const model = createTestModel("anthropic.claude-3-5-sonnet-20241022-v2:0");
      const candidate = {
        hasInferenceProfile: true,
        model,
        modelIdToUse: "us-west-2.anthropic.claude-3-5-sonnet-20241022-v2:0",
      };
      const availableProfiles = new Set([
        "global.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us-west-2.anthropic.claude-3-5-sonnet-20241022-v2:0",
      ]);
      const abortController = new AbortController();

      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      const result = await (provider as any).findAlternativeProfile(
        candidate,
        "us-west-2",
        availableProfiles,
        false,
        abortController.signal,
      );

      assert.equal(result.isAccessible, true);

      assert.equal(result.hasInferenceProfile, true);

      assert.equal(result.modelIdToUse, "global.anthropic.claude-3-5-sonnet-20241022-v2:0");
    });

    test("preferRegional=true, regional fails: skips global fallback, tries base model", async () => {
      const provider = createMockClientProvider(true);
      const model = createTestModel("anthropic.claude-3-5-sonnet-20241022-v2:0");
      const candidate = {
        hasInferenceProfile: true,
        model,
        modelIdToUse: "us-west-2.anthropic.claude-3-5-sonnet-20241022-v2:0",
      };
      const availableProfiles = new Set([
        "global.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us-west-2.anthropic.claude-3-5-sonnet-20241022-v2:0",
      ]);
      const abortController = new AbortController();

      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      const result = await (provider as any).findAlternativeProfile(
        candidate,
        "us-west-2",
        availableProfiles,
        true,
        abortController.signal,
      );

      // Should skip global profile and go directly to base model

      assert.equal(result.isAccessible, true);

      assert.equal(result.hasInferenceProfile, false);

      assert.equal(result.modelIdToUse, "anthropic.claude-3-5-sonnet-20241022-v2:0");
    });

    test("preferRegional=true, global fails: still tries regional fallback", async () => {
      const provider = createMockClientProvider(false);
      const model = createTestModel("anthropic.claude-3-5-sonnet-20241022-v2:0");
      const candidate = {
        hasInferenceProfile: true,
        model,
        modelIdToUse: "global.anthropic.claude-3-5-sonnet-20241022-v2:0",
      };
      const availableProfiles = new Set([
        "global.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us-west-2.anthropic.claude-3-5-sonnet-20241022-v2:0",
      ]);
      const abortController = new AbortController();

      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      const result = await (provider as any).findAlternativeProfile(
        candidate,
        "us-west-2",
        availableProfiles,
        true,
        abortController.signal,
      );

      // Should try regional profile even when preferRegional=true

      assert.equal(result.isAccessible, true);

      assert.equal(result.hasInferenceProfile, true);

      assert.equal(result.modelIdToUse, "us-west-2.anthropic.claude-3-5-sonnet-20241022-v2:0");
    });

    test("no alternative profile available: falls back to base model", async () => {
      const provider = createMockClientProvider(true);
      const model = createTestModel("anthropic.claude-3-5-sonnet-20241022-v2:0");
      const candidate = {
        hasInferenceProfile: true,
        model,
        modelIdToUse: "global.anthropic.claude-3-5-sonnet-20241022-v2:0",
      };
      const availableProfiles = new Set(["global.anthropic.claude-3-5-sonnet-20241022-v2:0"]);
      const abortController = new AbortController();

      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      const result = await (provider as any).findAlternativeProfile(
        candidate,
        "us-west-2",
        availableProfiles,
        false,
        abortController.signal,
      );

      assert.equal(result.isAccessible, true);

      assert.equal(result.hasInferenceProfile, false);

      assert.equal(result.modelIdToUse, "anthropic.claude-3-5-sonnet-20241022-v2:0");
    });

    test("no accessible alternative or base model: returns inaccessible", async () => {
      const provider = createMockClientProvider(false);
      const model = createTestModel("anthropic.claude-3-5-sonnet-20241022-v2:0");
      const candidate = {
        hasInferenceProfile: true,
        model,
        modelIdToUse: "global.anthropic.claude-3-5-sonnet-20241022-v2:0",
      };
      const availableProfiles = new Set(["global.anthropic.claude-3-5-sonnet-20241022-v2:0"]);
      const abortController = new AbortController();

      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      const result = await (provider as any).findAlternativeProfile(
        candidate,
        "us-west-2",
        availableProfiles,
        false,
        abortController.signal,
      );

      assert.equal(result.isAccessible, false);

      assert.equal(result.hasInferenceProfile, true);

      assert.equal(result.modelIdToUse, "global.anthropic.claude-3-5-sonnet-20241022-v2:0");
    });
  });

  suite("calculateThinkingConfig", () => {
    const thinkingProfile = { supportsThinking: true };
    const nonThinkingProfile = { supportsThinking: false };

    test("large model defaults produce 16k budget (base budget is the binding constraint)", () => {
      // Claude Sonnet 4: maxOutputTokens = 64000
      // baseBudget = 16000, maxBudgetFromOutput = 16000, visibleReserve = max(100, 16000) = 16000
      // maxTokensForRequest - visibleReserve = 64000 - 16000 = 48000
      // budgetTokens = min(16000, 16000, 48000) = 16000
      const result = callCalcThinkingConfig(
        thinkingProfile,
        { maxOutputTokens: 64_000 },
        64_000,
        true,
      );
      assert.equal(result.budgetTokens, 16_000);
      assert.equal(result.extendedThinkingEnabled, true);
    });

    test("small explicit max_tokens disables thinking when budget < 1024", () => {
      // max_tokens = 500 → visibleReserve = max(100, 125) = 125
      // maxTokensForRequest - visibleReserve = 500 - 125 = 375
      // budgetTokens = min(16000, 16000, 375) = 375 → < 1024 → thinking disabled
      const result = callCalcThinkingConfig(
        thinkingProfile,
        { maxOutputTokens: 64_000 },
        500,
        true,
      );
      assert.ok(
        result.budgetTokens < 1024,
        `Expected budgetTokens < 1024, got ${result.budgetTokens}`,
      );
      assert.equal(result.extendedThinkingEnabled, false);
    });

    test("budgetTokens never goes negative when maxTokensForRequest < visibleReserve", () => {
      // max_tokens = 50 → visibleReserve = max(100, 12) = 100
      // maxTokensForRequest - visibleReserve = 50 - 100 = -50
      // budgetTokens = max(0, min(16000, 16000, -50)) = 0
      const result = callCalcThinkingConfig(thinkingProfile, { maxOutputTokens: 64_000 }, 50, true);
      assert.equal(result.budgetTokens, 0);
      assert.equal(result.extendedThinkingEnabled, false);
    });

    test("thinking disabled when model does not support it", () => {
      const result = callCalcThinkingConfig(
        nonThinkingProfile,
        { maxOutputTokens: 64_000 },
        64_000,
        true,
      );
      assert.equal(result.extendedThinkingEnabled, false);
      // budgetTokens is still computed but thinking is disabled
      assert.equal(result.budgetTokens, 16_000);
    });

    test("thinking disabled when setting is off", () => {
      const result = callCalcThinkingConfig(
        thinkingProfile,
        { maxOutputTokens: 64_000 },
        64_000,
        false,
      );
      assert.equal(result.extendedThinkingEnabled, false);
    });

    test("reserve leaves room for visible output with moderate max_tokens", () => {
      // max_tokens = 2000 → visibleReserve = max(100, 500) = 500
      // maxTokensForRequest - visibleReserve = 2000 - 500 = 1500
      // budgetTokens = min(16000, 16000, 1500) = 1500
      // Remaining for visible output = 2000 - 1500 = 500 ≥ visibleReserve
      const result = callCalcThinkingConfig(
        thinkingProfile,
        { maxOutputTokens: 64_000 },
        2000,
        true,
      );
      assert.equal(result.budgetTokens, 1500);
      assert.ok(
        2000 - result.budgetTokens >= 500,
        "Should leave at least visibleReserve tokens for visible output",
      );
      assert.equal(result.extendedThinkingEnabled, true);
    });

    test("maxBudgetFromOutput caps budget for small-output models", () => {
      // Model with maxOutputTokens = 4096
      // maxBudgetFromOutput = floor(4096 * 0.25) = 1024
      // visibleReserve = max(100, floor(4096 * 0.25)) = 1024
      // maxTokensForRequest - visibleReserve = 4096 - 1024 = 3072
      // budgetTokens = min(16000, 1024, 3072) = 1024
      const result = callCalcThinkingConfig(thinkingProfile, { maxOutputTokens: 4096 }, 4096, true);
      assert.equal(result.budgetTokens, 1024);
      assert.equal(result.extendedThinkingEnabled, true);
    });

    test("very small model output disables thinking (budget below 1024 threshold)", () => {
      // Model with maxOutputTokens = 2000
      // maxBudgetFromOutput = floor(2000 * 0.25) = 500
      // budgetTokens = min(16000, 500, ...) = at most 500 → < 1024
      const result = callCalcThinkingConfig(thinkingProfile, { maxOutputTokens: 2000 }, 2000, true);
      assert.ok(result.budgetTokens < 1024);
      assert.equal(result.extendedThinkingEnabled, false);
    });

    test("budget math when maxTokensForRequest falls back to maxOutputTokens (no explicit max_tokens)", () => {
      // Simulates the case where VSCode doesn't provide max_tokens
      // and maxTokensForRequest falls back to modelLimits.maxOutputTokens
      const maxOutput = 128_000; // Claude Opus 4.6
      const result = callCalcThinkingConfig(
        thinkingProfile,
        { maxOutputTokens: maxOutput },
        maxOutput,
        true,
      );
      // baseBudget = 16000, maxBudgetFromOutput = 32000, visibleReserve = 32000
      // maxTokensForRequest - visibleReserve = 128000 - 32000 = 96000
      // budgetTokens = min(16000, 32000, 96000) = 16000
      assert.equal(result.budgetTokens, 16_000);
      assert.equal(result.extendedThinkingEnabled, true);
    });
  });

  suite("model configuration schema (picker controls)", () => {
    test("includes contextLength tokens picker for toggleable-1M models", () => {
      // Sonnet 4.5 has a toggleable 1M context window (standard 200K).
      const schema = buildModelConfigurationSchema("anthropic.claude-sonnet-4-5-20250929-v1:0");
      const contextLength = schema?.properties?.contextLength;
      assert.ok(contextLength, "expected contextLength property");
      // `group: "tokens"` makes VS Code render this as the dedicated "Context Size" picker.
      assert.equal(contextLength.group, "tokens");
      assert.equal(contextLength.type, "number");
      // Enum values MUST be numeric token counts (VS Code formats them via formatTokenCount).
      assert.deepEqual(contextLength.enum, [200_000, 1_000_000]);
      assert.deepEqual(contextLength.enumItemLabels, ["200K", "1M"]);
      assert.equal(contextLength.default, 1_000_000);
    });

    test("includes contextLength tokens picker for Opus 4.8 (toggleable 200K<->1M)", () => {
      // Opus 4.8 exposes a toggleable 1M context window (200K default), matching the official
      // Claude Code CLI which offers an `opus[1m]` toggle for Opus 4.6/4.7/4.8.
      const schema = buildModelConfigurationSchema("anthropic.claude-opus-4-8-20251101-v1:0");
      const contextLength = schema?.properties?.contextLength;
      assert.ok(contextLength, "expected contextLength property for Opus 4.8");
      assert.equal(contextLength.group, "tokens");
      assert.deepEqual(contextLength.enum, [200_000, 1_000_000]);
    });

    test("includes contextLength tokens picker for Opus 4.6 (toggleable 200K<->1M)", () => {
      const schema = buildModelConfigurationSchema("anthropic.claude-opus-4-6-v1");
      assert.ok(schema?.properties?.contextLength, "expected contextLength property for Opus 4.6");
    });

    test("includes contextLength tokens picker for Sonnet 5 (toggleable 200K<->1M)", () => {
      // Sonnet 5 exposes a toggleable 1M context window (200K default), like Opus 4.6/4.7/4.8.
      const schema = buildModelConfigurationSchema("anthropic.claude-sonnet-5");
      const contextLength = schema?.properties?.contextLength;
      assert.ok(contextLength, "expected contextLength property for Sonnet 5");
      assert.equal(contextLength.group, "tokens");
      assert.deepEqual(contextLength.enum, [200_000, 1_000_000]);
    });

    test("contextLength default follows the persisted/build-time selection", () => {
      // The picker default is the advertised window: 1M when enabled, 200K otherwise. This is what
      // keeps the picker UI in sync with the badge denominator after a refresh.
      const enabled = buildModelConfigurationSchema(
        "anthropic.claude-opus-4-8-20251101-v1:0",
        true,
      );
      assert.equal(enabled?.properties?.contextLength?.default, 1_000_000);

      const disabled = buildModelConfigurationSchema(
        "anthropic.claude-opus-4-8-20251101-v1:0",
        false,
      );
      assert.equal(disabled?.properties?.contextLength?.default, 200_000);
    });

    test("omits contextLength for Opus 4.5 (200K only, no 1M)", () => {
      // Opus 4.5 has a fixed 200K window on Bedrock — no 200K<->1M choice, so no picker.
      const schema = buildModelConfigurationSchema("anthropic.claude-opus-4-5-20251101-v1:0");
      assert.ok(!schema?.properties?.contextLength);
    });

    test("omits contextLength for models without 1M context support", () => {
      // Haiku 3.5 does not support the 1M context window.
      const schema = buildModelConfigurationSchema("anthropic.claude-3-5-haiku-20241022-v1:0");
      assert.ok(!schema?.properties?.contextLength);
    });

    test("returns undefined for models with no configurable capabilities", () => {
      // Mistral has neither thinking nor 1M context.
      const schema = buildModelConfigurationSchema("mistral.mistral-large-2407-v1:0");
      assert.equal(schema, undefined);
    });
  });

  suite("model configuration overrides (picker -> request)", () => {
    test("contextLength=200000 disables the 1M context window", () => {
      const settings = applyConfigOverride({ contextLength: 200_000 });
      assert.equal(settings.context1M.enabled, false);
    });

    test("contextLength=1000000 enables the 1M context window", () => {
      const settings = applyConfigOverride({ contextLength: 1_000_000 });
      assert.equal(settings.context1M.enabled, true);
    });

    test("absent contextLength leaves the fallback setting intact", () => {
      const settings = applyConfigOverride({ thinkingEffort: "low" });
      assert.equal(settings.context1M.enabled, true);
      assert.equal(settings.thinking.effort, "low");
    });

    test("non-numeric contextLength value is ignored", () => {
      const settings = applyConfigOverride({ contextLength: "bogus" });
      assert.equal(settings.context1M.enabled, true);
    });
  });
});
