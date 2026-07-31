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
import { type ChatCompletionRequest, CustomBackendClient, CustomBackendError } from "./client";
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

// Default context window advertised when the backend does not report limits.
const DEFAULT_MAX_INPUT_TOKENS = 128_000;
const DEFAULT_MAX_OUTPUT_TOKENS = 8192;

export class CustomChatModelProvider implements vscode.Disposable, LanguageModelChatProvider {
  private readonly _onDidChange = new vscode.EventEmitter<void>();
  readonly onDidChangeLanguageModelChatInformation = this._onDidChange.event;

  private readonly client = new CustomBackendClient({ apiKey: "", baseUrl: "" });

  constructor(private readonly secrets: vscode.SecretStorage) {}

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

    const abortController = new AbortController();
    const cancellation = token.onCancellationRequested(() => abortController.abort());

    try {
      const request: ChatCompletionRequest = {
        messages: convertMessages(messages),
        model: model.id,
        stream: true,
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

  private buildModelInfo(
    id: string,
    caps?: { toolCalling?: boolean; vision?: boolean },
  ): CustomLanguageModelChatInformation {
    return {
      // Default to advertising tool calling; most OpenAI-compatible backends
      // support it, and the picker needs it enabled for agent mode.
      capabilities: {
        imageInput: caps?.vision ?? false,
        toolCalling: caps?.toolCalling ?? true,
      },
      category: CUSTOM_MODEL_PICKER_CATEGORY,
      family: "custom",
      id,
      isUserSelectable: true,
      maxInputTokens: DEFAULT_MAX_INPUT_TOKENS,
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

  private async resolveModels(
    settings: CustomBackendSettings,
    token: CancellationToken,
  ): Promise<LanguageModelChatInformation[]> {
    const abortController = new AbortController();
    const cancellation = token.onCancellationRequested(() => abortController.abort());

    try {
      // Manual model list, when provided, is authoritative and skips discovery.
      if (settings.models.length > 0) {
        return settings.models.map((id) => this.buildModelInfo(id));
      }

      const discovered = await this.client.listModels(abortController.signal);
      return discovered.map((m) =>
        this.buildModelInfo(m.id, { toolCalling: m.toolCalling, vision: m.vision }),
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

    for await (const delta of this.client.streamChat(request, signal)) {
      if (delta.content) {
        progress.report(new vscode.LanguageModelTextPart(delta.content));
      }
      if (delta.toolCalls) {
        for (const tc of delta.toolCalls) {
          const acc = toolAccumulators.get(tc.index) ?? { args: "", id: "", name: "" };
          if (tc.id) acc.id = tc.id;
          if (tc.name) acc.name = tc.name;
          if (tc.argumentsFragment) acc.args += tc.argumentsFragment;
          toolAccumulators.set(tc.index, acc);
        }
      }
    }

    this.flushToolCalls(toolAccumulators, progress);
  }
}

function mapToolChoice(
  mode: undefined | vscode.LanguageModelChatToolMode,
): ChatCompletionRequest["tool_choice"] {
  return mode === vscode.LanguageModelChatToolMode.Required ? "required" : "auto";
}
