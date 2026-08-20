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
  type TokenUsage,
} from "./client";
import { addCacheBreakpoints, convertMessages, convertTools } from "./converter";
import {
  defaultReasoningEffort,
  heuristicReasoningCapability,
  isClaudeFamily,
  type ReasoningCapability,
  reasoningEffortDescription,
  reasoningEffortLabel,
  type ReasoningEffortLevel,
} from "./reasoning";
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
  /**
   * Resolved reasoning-effort capability for this model, embedded at discovery
   * so it round-trips to {@link CustomChatModelProvider.provideLanguageModelChatResponse}
   * (VS Code re-supplies the whole information object). Absent when the model is
   * not reasoning-capable.
   */
  readonly reasoning?: ReasoningCapability;
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

      // Copilot never sends cache breakpoints to `custom`-vendor providers
      // (issue #313920), so add them ourselves for Claude models — the only
      // family on our backends (breeze/LiteLLM, QGenie) that supports prompt
      // caching. LiteLLM forwards `cache_control` to Anthropic; other backends
      // ignore the unknown field.
      if (isClaudeFamily(model.id)) {
        addCacheBreakpoints(request.messages);
      }

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

      // Forward the picked reasoning effort only when the model declared it and
      // the value is within the declared set — mirrors Copilot, which scrubs any
      // effort a model didn't advertise rather than passing it through blindly.
      const reasoning = (model as CustomLanguageModelChatInformation).reasoning;
      const pickedEffort = modelConfig?.reasoningEffort;
      if (typeof pickedEffort === "string" && reasoning) {
        if ((reasoning.levels as readonly string[]).includes(pickedEffort)) {
          request.reasoning_effort = pickedEffort as ReasoningEffortLevel;
        } else {
          logger.warn("[Custom Provider] Dropping unsupported reasoning effort", {
            allowed: reasoning.levels,
            modelId: model.id,
            requested: pickedEffort,
          });
        }
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
      aliases?: readonly string[];
      maxInputTokens?: number;
      maxOutputTokens?: number;
      reasoning?: boolean;
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
    const reasoning = this.resolveReasoning(id, caps);
    const properties: Record<string, object> = { contextSize: contextSizeSchema };
    if (reasoning) {
      properties.reasoningEffort = this.buildReasoningEffortSchema(id, reasoning);
    }
    return {
      // Default to advertising tool calling; most OpenAI-compatible backends
      // support it, and the picker needs it enabled for agent mode.
      capabilities: {
        imageInput: caps?.vision ?? false,
        toolCalling: caps?.toolCalling ?? true,
      },
      category: this.pickerCategory(settings),
      configurationSchema: { properties },
      family: "custom",
      id,
      isUserSelectable: true,
      maxInputTokens,
      maxOutputTokens,
      name: id,
      reasoning,
      tooltip: `${settings.name ?? "Custom backend"} model: ${id}`,
      version: "1.0.0",
    };
  }

  /**
   * Resolve reasoning capability for a model, most authoritative first:
   * backend-declared support (`/model/info` or list metadata) gates whether a
   * picker appears at all; the concrete level *set* comes from the family
   * heuristic (matched against the id and any aliases). A backend that declares
   * support for an unrecognized family still gets the conservative level set.
   */
  private resolveReasoning(
    id: string,
    caps?: { aliases?: readonly string[]; reasoning?: boolean },
  ): ReasoningCapability | undefined {
    const heuristic = heuristicReasoningCapability(id, ...(caps?.aliases ?? []));
    if (caps?.reasoning === false) {
      return undefined;
    }
    if (caps?.reasoning === true) {
      return heuristic ?? { format: "chat-completions", levels: ["low", "medium", "high"] };
    }
    return heuristic;
  }

  /** VS Code picker schema for the reasoning-effort levels of a model. */
  private buildReasoningEffortSchema(id: string, reasoning: ReasoningCapability): object {
    const levels = reasoning.levels;
    return {
      default: defaultReasoningEffort(levels, id),
      description: "How much reasoning effort the model should apply.",
      enum: [...levels],
      enumDescriptions: levels.map(reasoningEffortDescription),
      enumItemLabels: levels.map(reasoningEffortLabel),
      // MUST be "navigation": VS Code's model picker surfaces the Thinking
      // Effort dropdown by scanning configurationSchema for the property whose
      // group is "navigation" (see modelPickerConfiguration.ts). "reasoning"
      // or any other group name leaves the picker hidden.
      group: "navigation",
      type: "string",
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
    usage: TokenUsage | undefined,
    progress: Progress<LanguageModelResponsePart>,
  ): void {
    // The prompt count is the dominant term in the badge numerator; report as
    // long as it is known, defaulting completion to 0 rather than suppressing
    // the whole report when a backend omits `completion_tokens`.
    if (usage?.promptTokens === undefined) return;
    try {
      const completionTokens = usage.completionTokens ?? 0;
      const cachedTokens = usage.cachedTokens ?? 0;
      const cacheWriteTokens = usage.cacheWriteTokens ?? 0;

      // Reconstruct the true prompt size so the context badge accumulates
      // instead of shrinking on a cache hit. With Anthropic-style caching the
      // backend may report only the *non-cached* input in `prompt_tokens`
      // (QGenie) while the bulk of the context sits in cache-read/-write
      // counters; those must be added back. Under the OpenAI convention
      // (`prompt_tokens_details.cached_tokens`) the cached tokens are already
      // inside `prompt_tokens`, so only cache *writes* are added.
      const promptTokens =
        usage.promptTokens + (usage.promptIncludesCached ? 0 : cachedTokens) + cacheWriteTokens;

      progress.report(
        vscode.LanguageModelDataPart.json(
          {
            completion_tokens: completionTokens,
            prompt_tokens: promptTokens,
            prompt_tokens_details: { cached_tokens: cachedTokens },
            total_tokens: promptTokens + completionTokens,
          },
          "usage",
        ),
      );
      logger.debug("[Custom Provider] Reported token usage", {
        cacheWriteTokens,
        cachedTokens,
        completionTokens,
        promptIncludesCached: usage.promptIncludesCached ?? false,
        promptTokens,
        rawPromptTokens: usage.promptTokens,
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
      // LiteLLM-style backends expose per-model `supports_reasoning` only via a
      // separate `/model/info` probe; merge it in (best-effort, may be empty).
      const reasoningSupport = await this.client.fetchReasoningSupport(abortController.signal);
      return discovered.map((m) =>
        this.buildModelInfo(settings, m.id, {
          aliases: m.aliases,
          maxInputTokens: m.maxInputTokens,
          maxOutputTokens: m.maxOutputTokens,
          reasoning: reasoningSupport.get(m.id) ?? m.reasoning,
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
    // may split fields across chunks, or send a partial usage object mid-stream.
    // Keeping the last defined value of each field ensures the final report
    // carries the turn's full prompt size, which is what VS Code's context badge
    // treats as the (absolute) numerator.
    const usage: TokenUsage = {};
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
        if (delta.usage.cachedTokens !== undefined) {
          usage.cachedTokens = delta.usage.cachedTokens;
          sawUsage = true;
        }
        if (delta.usage.cacheWriteTokens !== undefined) {
          usage.cacheWriteTokens = delta.usage.cacheWriteTokens;
          sawUsage = true;
        }
        if (delta.usage.promptIncludesCached !== undefined) {
          usage.promptIncludesCached = delta.usage.promptIncludesCached;
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
