import type {
  CancellationToken,
  LanguageModelChatInformation,
  LanguageModelChatProvider,
  LanguageModelChatRequestMessage,
  LanguageModelResponsePart,
  Progress,
} from "vscode";
import * as vscode from "vscode";

import { logger } from "../logger";
import { countMessageTokens, countStringTokens } from "../tokenizer";
import {
  type ChatCompletionRequest,
  CustomBackendClient,
  CustomBackendError,
  type StreamDelta,
} from "./client";
import { convertMessages, convertTools } from "./converter";
import { type CustomBackendSettings, getCustomBackendSettings, toBackendConfig } from "./settings";

type CustomLanguageModelChatInformation = LanguageModelChatInformation & {
  readonly category?: { label: string; order: number };
  readonly isUserSelectable?: boolean;
};

const CUSTOM_MODEL_PICKER_CATEGORY = { label: "Custom Backend", order: 60 } as const;

/**
 * Sentinel model surfaced when the backend is unconfigured or unreachable, so
 * the picker keeps a Custom entry visible (mirrors the Bedrock provider's
 * behavior) instead of silently reverting the user's selection.
 */
export const CUSTOM_ERROR_SENTINEL_ID = "__custom_error_sentinel__";

const DEFAULT_MAX_OUTPUT_TOKENS = 8192;

export class CustomChatModelProvider implements vscode.Disposable, LanguageModelChatProvider {
  private static readonly CONTEXT_SELECTION_KEY = "custom.contextSelection";

  private readonly _onDidChange = new vscode.EventEmitter<void>();
  readonly onDidChangeLanguageModelChatInformation = this._onDidChange.event;

  private readonly client = new CustomBackendClient({ apiKey: "", baseUrl: "" });

  constructor(
    private readonly secrets: vscode.SecretStorage,
    private readonly globalState: vscode.Memento,
  ) {}

  dispose(): void {
    this._onDidChange.dispose();
  }

  notifyModelInformationChanged(reason?: string): void {
    const suffix = reason ? `: ${reason}` : "";
    logger.debug(`[Custom Provider] Signaling model info refresh${suffix}`);
    this._onDidChange.fire();
  }

  async provideLanguageModelChatInformation(
    options: { silent: boolean },
    token: CancellationToken,
  ): Promise<LanguageModelChatInformation[]> {
    const settings = await getCustomBackendSettings(this.secrets);
    const config = toBackendConfig(settings);

    if (!config) {
      if (!options.silent) {
        vscode.window.showInformationMessage(
          "Custom backend is not configured. Run 'Manage Custom Model Provider' to set a base URL and access token.",
        );
      }
      return this.buildSentinel("Backend not configured (missing base URL or access token)");
    }

    this.client.setConfig(config);

    try {
      const models = await this.resolveModels(settings, token);
      if (models.length === 0) {
        return this.buildSentinel("No models discovered and none configured manually");
      }
      return models;
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        return [];
      }
      const message = error instanceof Error ? error.message : String(error);
      if (!options.silent) {
        vscode.window.showErrorMessage(`Failed to load custom backend models: ${message}`);
      }
      logger.error("[Custom Provider] Failed to load models", error);
      return this.buildSentinel(message);
    }
  }

  async provideLanguageModelChatResponse(
    model: LanguageModelChatInformation,
    messages: readonly LanguageModelChatRequestMessage[],
    options: Parameters<LanguageModelChatProvider["provideLanguageModelChatResponse"]>[2],
    progress: Progress<LanguageModelResponsePart>,
    token: CancellationToken,
  ): Promise<void> {
    if (model.id === CUSTOM_ERROR_SENTINEL_ID) {
      const reason = (model as CustomLanguageModelChatInformation).detail ?? "unknown error";
      throw new Error(
        `Custom backend is unavailable: ${reason}. Run 'Manage Custom Model Provider', then re-open the model picker.`,
      );
    }

    const settings = await getCustomBackendSettings(this.secrets);
    const config = toBackendConfig(settings);
    if (!config) {
      throw new Error("Custom backend is not configured (missing base URL or access token).");
    }
    this.client.setConfig(config);

    // Persist a new context-length picker selection, then refresh the model
    // list so the badge denominator updates on the next turn.
    const rawContextLength = (options.modelConfiguration as Record<string, unknown> | undefined)
      ?.contextLength;
    const pickedContext = typeof rawContextLength === "number" ? rawContextLength : undefined;
    if (typeof pickedContext === "number" && Number.isFinite(pickedContext)) {
      const stored = this.getPersistedContextLength(model.id);
      if (pickedContext !== stored) {
        await this.persistContextLength(model.id, pickedContext);
        this.notifyModelInformationChanged("context length changed");
      }
    }

    const abortController = new AbortController();
    const cancellation = token.onCancellationRequested(() => abortController.abort());

    try {
      const request: ChatCompletionRequest = {
        messages: convertMessages(messages),
        model: model.id,
        stream: true,
        // Ask the backend to include token counts in the final streaming chunk so
        // we can feed them into VS Code's context-window tracker badge.
        stream_options: { include_usage: true },
      };

      const tools = convertTools(options.tools);
      if (tools) {
        request.tools = tools;
        request.tool_choice = mapToolChoice(options.toolMode);
      }
      if (typeof options.modelOptions?.max_tokens === "number") {
        request.max_tokens = options.modelOptions.max_tokens;
      }
      if (typeof options.modelOptions?.temperature === "number") {
        request.temperature = options.modelOptions.temperature;
      }

      await this.streamResponse(request, progress, abortController.signal);
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        logger.info("[Custom Provider] Request cancelled by user");
        return;
      }
      logger.error("[Custom Provider] Chat request failed", {
        error: error instanceof Error ? error.message : String(error),
        modelId: model.id,
        status: error instanceof CustomBackendError ? error.status : undefined,
      });
      throw error;
    } finally {
      cancellation.dispose();
    }
  }

  async provideTokenCount(
    model: LanguageModelChatInformation,
    text: LanguageModelChatRequestMessage | string,
    _token: CancellationToken,
  ): Promise<number> {
    if (model.id === CUSTOM_ERROR_SENTINEL_ID) {
      return 0;
    }
    return typeof text === "string" ? countStringTokens(text) : countMessageTokens(text);
  }

  private accumulateToolCall(
    tc: NonNullable<StreamDelta["toolCalls"]>[number],
    accumulators: Map<number, { args: string; id: string; name: string }>,
  ): void {
    const acc = accumulators.get(tc.index) ?? { args: "", id: "", name: "" };
    if (tc.id) acc.id = tc.id;
    if (tc.name) acc.name = tc.name;
    if (tc.argumentsFragment) acc.args += tc.argumentsFragment;
    accumulators.set(tc.index, acc);
  }

  private buildModelInfo(
    id: string,
    defaultMaxInputTokens: number,
    caps?: { toolCalling?: boolean; vision?: boolean },
  ): CustomLanguageModelChatInformation {
    const maxInputTokens = this.getPersistedContextLength(id) ?? defaultMaxInputTokens;
    const contextLengthSchema = {
      default: maxInputTokens,
      description: "Context window size for this model.",
      enum: [128_000, 200_000, 1_000_000],
      enumDescriptions: [
        "128K token context window.",
        "200K token context window.",
        "1M token context window.",
      ],
      enumItemLabels: ["128K", "200K", "1M"],
      group: "tokens",
      type: "number",
    };
    return {
      // Default to advertising tool calling; most OpenAI-compatible backends
      // support it, and the picker needs it enabled for agent mode.
      capabilities: {
        imageInput: caps?.vision ?? false,
        toolCalling: caps?.toolCalling ?? true,
      },
      category: CUSTOM_MODEL_PICKER_CATEGORY,
      configurationSchema: { properties: { contextLength: contextLengthSchema } },
      family: "custom",
      id,
      isUserSelectable: true,
      maxInputTokens,
      maxOutputTokens: DEFAULT_MAX_OUTPUT_TOKENS,
      name: id,
      tooltip: `Custom backend model: ${id}`,
      version: "1.0.0",
    };
  }

  private buildSentinel(detail: string): LanguageModelChatInformation[] {
    const sentinel: CustomLanguageModelChatInformation = {
      capabilities: { imageInput: false, toolCalling: false },
      category: CUSTOM_MODEL_PICKER_CATEGORY,
      detail,
      family: "custom",
      id: CUSTOM_ERROR_SENTINEL_ID,
      isUserSelectable: true,
      maxInputTokens: 1,
      maxOutputTokens: 1,
      name: "⚠ Custom backend unavailable",
      tooltip: detail,
      version: "1.0.0",
    };
    return [sentinel];
  }

  private flushToolCalls(
    accumulators: Map<number, { args: string; id: string; name: string }>,
    progress: Progress<LanguageModelResponsePart>,
  ): void {
    for (const acc of accumulators.values()) {
      if (!acc.name) continue;
      let input: object;
      try {
        input = acc.args ? (JSON.parse(acc.args) as object) : {};
      } catch (error) {
        logger.warn("[Custom Provider] Failed to parse tool call arguments; sending empty input", {
          error: error instanceof Error ? error.message : String(error),
          name: acc.name,
        });
        input = {};
      }
      progress.report(new vscode.LanguageModelToolCallPart(acc.id || acc.name, acc.name, input));
    }
  }

  private getPersistedContextLength(modelId: string): number | undefined {
    const map = this.globalState.get<Record<string, number>>(
      CustomChatModelProvider.CONTEXT_SELECTION_KEY,
      {},
    );
    const value = map[modelId];
    return typeof value === "number" && Number.isFinite(value) ? value : undefined;
  }

  private async persistContextLength(modelId: string, maxInputTokens: number): Promise<void> {
    const map = {
      ...this.globalState.get<Record<string, number>>(
        CustomChatModelProvider.CONTEXT_SELECTION_KEY,
        {},
      ),
      [modelId]: maxInputTokens,
    };
    await this.globalState.update(CustomChatModelProvider.CONTEXT_SELECTION_KEY, map);
  }

  private reportUsage(
    usage: undefined | { completionTokens?: number; promptTokens?: number },
    progress: Progress<LanguageModelResponsePart>,
  ): void {
    if (usage?.promptTokens === undefined || usage.completionTokens === undefined) return;
    try {
      const completionTokens = usage.completionTokens;
      const promptTokens = usage.promptTokens;
      progress.report(
        vscode.LanguageModelDataPart.json(
          {
            completion_tokens: completionTokens,
            prompt_tokens: promptTokens,
            prompt_tokens_details: { cached_tokens: 0 },
            total_tokens: promptTokens + completionTokens,
          },
          "usage",
        ),
      );
      logger.debug("[Custom Provider] Reported token usage", {
        completionTokens,
        promptTokens,
        totalTokens: promptTokens + completionTokens,
      });
    } catch (error) {
      logger.debug("[Custom Provider] Failed to report usage data part", {
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  private async resolveModels(
    settings: CustomBackendSettings,
    token: CancellationToken,
  ): Promise<LanguageModelChatInformation[]> {
    const abortController = new AbortController();
    const cancellation = token.onCancellationRequested(() => abortController.abort());

    try {
      // Manual model list, when provided, is authoritative and skips discovery.
      if (settings.models.length > 0) {
        return settings.models.map((id) => this.buildModelInfo(id, settings.maxInputTokens));
      }

      const discovered = await this.client.listModels(abortController.signal);
      return discovered.map((m) =>
        this.buildModelInfo(m.id, settings.maxInputTokens, {
          toolCalling: m.toolCalling,
          vision: m.vision,
        }),
      );
    } finally {
      cancellation.dispose();
    }
  }

  private async streamResponse(
    request: ChatCompletionRequest,
    progress: Progress<LanguageModelResponsePart>,
    signal: AbortSignal,
  ): Promise<void> {
    const toolAccumulators = new Map<number, { args: string; id: string; name: string }>();
    let lastUsage: undefined | { completionTokens?: number; promptTokens?: number };

    for await (const delta of this.client.streamChat(request, signal)) {
      if (delta.content) {
        progress.report(new vscode.LanguageModelTextPart(delta.content));
      }
      if (delta.toolCalls) {
        for (const tc of delta.toolCalls) {
          this.accumulateToolCall(tc, toolAccumulators);
        }
      }
      if (delta.usage) {
        lastUsage = delta.usage;
      }
    }

    this.flushToolCalls(toolAccumulators, progress);

    // Report token usage to VS Code's context-window tracker badge.
    // Convention: emit a LanguageModelDataPart with MIME "usage" whose JSON
    // payload follows the OpenAI APIUsage shape. Copilot Chat ≥ 1.120.0
    // consumes this and updates the badge numerator accordingly.
    // See: https://github.com/microsoft/vscode/pull/315394
    this.reportUsage(lastUsage, progress);
  }
}

function mapToolChoice(
  mode: undefined | vscode.LanguageModelChatToolMode,
): ChatCompletionRequest["tool_choice"] {
  return mode === vscode.LanguageModelChatToolMode.Required ? "required" : "auto";
}
