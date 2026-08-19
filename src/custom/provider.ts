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
import {
  type CustomBackendSettings,
  DEFAULT_MAX_INPUT_TOKENS,
  parseBackendSettings,
  toBackendConfig,
} from "./settings";

type CustomLanguageModelChatInformation = LanguageModelChatInformation & {
  readonly category?: { label: string; order: number };
  readonly isUserSelectable?: boolean;
  /**
   * The backend group's raw configuration (baseUrl/apiKey/...), embedded at
   * discovery time so VS Code round-trips it back to
   * {@link CustomChatModelProvider.provideLanguageModelChatResponse} as
   * `model.configuration` — the response options only carry per-model
   * `modelConfiguration`, never the group config.
   */
  readonly configuration?: Record<string, unknown>;
};

const CUSTOM_MODEL_PICKER_CATEGORY = { label: "Custom Backend", order: 60 } as const;

/**
 * Sentinel model surfaced when a configured backend is unreachable, so the
 * picker keeps a visible entry (mirrors the Bedrock provider's behavior)
 * instead of silently reverting the user's selection.
 */
export const CUSTOM_ERROR_SENTINEL_ID = "__custom_error_sentinel__";

const DEFAULT_MAX_OUTPUT_TOKENS = 8192;

/**
 * Options passed to {@link CustomChatModelProvider.provideLanguageModelChatInformation}.
 * The pinned proposed API only types `configuration`; `silent` is present at
 * runtime but not in the d.ts, so it is declared here defensively.
 */
type PrepareOptions = {
  readonly configuration?: Record<string, unknown>;
  readonly silent?: boolean;
};

export class CustomChatModelProvider implements vscode.Disposable, LanguageModelChatProvider {
  private static readonly CONTEXT_SELECTION_KEY = "custom.contextSelection";

  private readonly _onDidChange = new vscode.EventEmitter<void>();
  readonly onDidChangeLanguageModelChatInformation = this._onDidChange.event;

  private readonly client = new CustomBackendClient({ apiKey: "", baseUrl: "" });

  constructor(private readonly globalState: vscode.Memento) {}

  dispose(): void {
    this._onDidChange.dispose();
  }

  notifyModelInformationChanged(reason?: string): void {
    const suffix = reason ? `: ${reason}` : "";
    logger.debug(`[Custom Provider] Signaling model info refresh${suffix}`);
    this._onDidChange.fire();
  }

  async provideLanguageModelChatInformation(
    options: PrepareOptions,
    token: CancellationToken,
  ): Promise<LanguageModelChatInformation[]> {
    const settings = parseBackendSettings(options.configuration);
    const config = toBackendConfig(settings);

    // No configuration yet (no group added, or missing required fields): return
    // nothing so VS Code shows its native "add a backend" affordance rather
    // than a bogus error entry.
    if (!config) {
      return [];
    }

    this.client.setConfig(config);

    try {
      const models = await this.resolveModels(settings, token);
      if (models.length === 0) {
        return this.buildSentinel(settings, "No models discovered and none configured manually");
      }
      // Embed the backend group config in each model so it round-trips back to
      // `provideLanguageModelChatResponse`, which only receives per-model
      // `modelConfiguration` from VS Code, never the group's `configuration`.
      return models.map((model) => ({ ...model, configuration: options.configuration }));
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        return [];
      }
      const message = error instanceof Error ? error.message : String(error);
      logger.error("[Custom Provider] Failed to load models", error);
      return this.buildSentinel(settings, message);
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
        `Custom backend is unavailable: ${reason}. Reconfigure the backend, then re-open the model picker.`,
      );
    }

    // The backend group config (baseUrl/apiKey/...) is embedded in the model at
    // discovery time and round-tripped here as `model.configuration`. VS Code's
    // `options.modelConfiguration` only carries per-model schema values (e.g.
    // the context-size picker), so the group config must come from the model.
    const modelConfig = options.modelConfiguration as Record<string, unknown> | undefined;
    const settings = parseBackendSettings(
      (model as CustomLanguageModelChatInformation).configuration,
    );
    const config = toBackendConfig(settings);
    if (!config) {
      throw new Error("Custom backend is not configured (missing base URL or access token).");
    }
    this.client.setConfig(config);

    // Persist a new context-size picker selection, then refresh the model
    // list so the badge denominator updates on the next turn.
    const rawContextSize = modelConfig?.contextSize;
    const pickedContext = typeof rawContextSize === "number" ? rawContextSize : undefined;
    if (typeof pickedContext === "number" && Number.isFinite(pickedContext)) {
      const stored = this.getPersistedContextSize(settings, model.id);
      if (pickedContext !== stored) {
        await this.persistContextSize(settings, model.id, pickedContext);
        this.notifyModelInformationChanged("context size changed");
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
      const status = error instanceof CustomBackendError ? error.status : undefined;
      if (status === 401) {
        // The key has likely expired. We cannot refresh it (interactive OIDC),
        // so log a clear hint. Per requirements, this is not surfaced to the UI.
        logger.error(
          "[Custom Provider] Authentication failed (401): the backend rejected the API key. " +
            "It may have expired — re-issue it (e.g. run `breeze aws auth`) and update the backend's key.",
          { modelId: model.id },
        );
      } else {
        logger.error("[Custom Provider] Chat request failed", {
          error: error instanceof Error ? error.message : String(error),
          modelId: model.id,
          status,
        });
      }
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
    settings: CustomBackendSettings,
    id: string,
    caps?: {
      maxInputTokens?: number;
      maxOutputTokens?: number;
      toolCalling?: boolean;
      vision?: boolean;
    },
  ): CustomLanguageModelChatInformation {
    // Precedence for the context window: user's context-size override →
    // backend-reported per-model limit → group's configured fallback → default.
    const reported = caps?.maxInputTokens;
    const fallback = reported ?? settings.maxInputTokens ?? DEFAULT_MAX_INPUT_TOKENS;
    const maxInputTokens = this.getPersistedContextSize(settings, id) ?? fallback;
    const maxOutputTokens = caps?.maxOutputTokens ?? DEFAULT_MAX_OUTPUT_TOKENS;
    // The property key MUST be `contextSize`: VS Code's context-usage badge reads the window
    // denominator from `modelConfiguration.contextSize` / `properties.contextSize.default`.
    const contextSizeSchema = {
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
      category: this.pickerCategory(settings),
      configurationSchema: { properties: { contextSize: contextSizeSchema } },
      family: "custom",
      id,
      isUserSelectable: true,
      maxInputTokens,
      maxOutputTokens,
      name: id,
      tooltip: `${settings.name ?? "Custom backend"} model: ${id}`,
      version: "1.0.0",
    };
  }

  private buildSentinel(
    settings: CustomBackendSettings,
    detail: string,
  ): LanguageModelChatInformation[] {
    const sentinel: CustomLanguageModelChatInformation = {
      capabilities: { imageInput: false, toolCalling: false },
      category: this.pickerCategory(settings),
      detail,
      family: "custom",
      id: CUSTOM_ERROR_SENTINEL_ID,
      isUserSelectable: true,
      maxInputTokens: 1,
      maxOutputTokens: 1,
      name: `⚠ ${settings.name ?? "Custom backend"} unavailable`,
      tooltip: detail,
      version: "1.0.0",
    };
    return [sentinel];
  }

  /** Picker category labeled with the backend's friendly name, when set. */
  private pickerCategory(settings: CustomBackendSettings): { label: string; order: number } {
    return settings.name
      ? { label: settings.name, order: CUSTOM_MODEL_PICKER_CATEGORY.order }
      : CUSTOM_MODEL_PICKER_CATEGORY;
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

  private getPersistedContextSize(
    settings: CustomBackendSettings,
    modelId: string,
  ): number | undefined {
    const map = this.globalState.get<Record<string, number>>(
      CustomChatModelProvider.CONTEXT_SELECTION_KEY,
      {},
    );
    const value = map[contextSelectionKey(settings, modelId)];
    return typeof value === "number" && Number.isFinite(value) ? value : undefined;
  }

  private async persistContextSize(
    settings: CustomBackendSettings,
    modelId: string,
    maxInputTokens: number,
  ): Promise<void> {
    const map = {
      ...this.globalState.get<Record<string, number>>(
        CustomChatModelProvider.CONTEXT_SELECTION_KEY,
        {},
      ),
      [contextSelectionKey(settings, modelId)]: maxInputTokens,
    };
    await this.globalState.update(CustomChatModelProvider.CONTEXT_SELECTION_KEY, map);
  }

  private reportUsage(
    usage: undefined | { completionTokens?: number; promptTokens?: number },
    progress: Progress<LanguageModelResponsePart>,
  ): void {
    // The prompt count is the dominant term in the badge numerator; report as
    // long as it is known, defaulting completion to 0 rather than suppressing
    // the whole report when a backend omits `completion_tokens`.
    if (usage?.promptTokens === undefined) return;
    try {
      const completionTokens = usage.completionTokens ?? 0;
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
        return settings.models.map((id) => this.buildModelInfo(settings, id));
      }

      const discovered = await this.client.listModels(abortController.signal);
      return discovered.map((m) =>
        this.buildModelInfo(settings, m.id, {
          maxInputTokens: m.maxInputTokens,
          maxOutputTokens: m.maxOutputTokens,
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
    // Accumulate usage field-by-field rather than replacing wholesale: gateways
    // may split `prompt_tokens` and `completion_tokens` across chunks, or send a
    // partial usage object mid-stream. Keeping the last defined value of each
    // field ensures the final report carries the turn's full prompt size, which
    // is what VS Code's context badge treats as the (absolute) numerator.
    const usage: { completionTokens?: number; promptTokens?: number } = {};
    let sawUsage = false;

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
        if (delta.usage.promptTokens !== undefined) {
          usage.promptTokens = delta.usage.promptTokens;
          sawUsage = true;
        }
        if (delta.usage.completionTokens !== undefined) {
          usage.completionTokens = delta.usage.completionTokens;
          sawUsage = true;
        }
      }
    }

    this.flushToolCalls(toolAccumulators, progress);

    // Report token usage to VS Code's context-window tracker badge.
    // Convention: emit a LanguageModelDataPart with MIME "usage" whose JSON
    // payload follows the OpenAI APIUsage shape. Copilot Chat ≥ 1.120.0
    // consumes this and updates the badge numerator accordingly.
    // See: https://github.com/microsoft/vscode/pull/315394
    this.reportUsage(sawUsage ? usage : undefined, progress);
  }
}

function mapToolChoice(
  mode: undefined | vscode.LanguageModelChatToolMode,
): ChatCompletionRequest["tool_choice"] {
  return mode === vscode.LanguageModelChatToolMode.Required ? "required" : "auto";
}

/**
 * Namespace context-size overrides by backend so the same model id at
 * different backends does not share a stored value.
 */
function contextSelectionKey(settings: CustomBackendSettings, modelId: string): string {
  return `${settings.baseUrl ?? ""}::${modelId}`;
}
