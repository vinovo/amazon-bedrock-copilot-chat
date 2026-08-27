import { readFileSync } from "node:fs";
import { Agent } from "undici";

import { logger } from "../logger";
import type { ReasoningEffortLevel } from "./reasoning";

/**
 * Minimal client for OpenAI-compatible chat backends.
 *
 * Targets any HTTP endpoint that speaks the OpenAI `/v1/chat/completions`
 * protocol (streaming SSE) and, optionally, `GET /v1/models` for discovery.
 * The only required configuration is a base URL and a bearer token, which
 * makes it usable against a wide range of self-hosted gateways and proxies.
 */

/**
 * Anthropic-style prompt-cache breakpoint. Attached to the last content block
 * of a message (or a tool result) to mark "cache everything up to here".
 * LiteLLM/breeze forwards this verbatim to the Anthropic Messages API; backends
 * that don't understand it ignore the unknown field.
 */
export interface CacheControl {
  type: "ephemeral";
}

export interface ChatCompletionRequest {
  max_tokens?: number;
  messages: OpenAIChatMessage[];
  model: string;
  /** Reasoning effort forwarded to reasoning-capable models (Chat Completions form). */
  reasoning_effort?: ReasoningEffortLevel;
  stream: true;
  /** Request that the backend include token-usage data in the final streaming chunk. */
  stream_options?: { include_usage?: boolean };
  temperature?: number;
  tool_choice?: "auto" | "none" | "required";
  tools?: OpenAITool[];
}

export interface CustomBackendConfig {
  /** When true, TLS certificate verification is skipped (for backends with private CAs). */
  readonly allowInsecureTls?: boolean;
  /** Bearer token sent as both `Authorization: Bearer` and `x-api-key`. */
  readonly apiKey: string;
  /** Root URL of the backend, e.g. `https://gateway.example.com` (trailing `/v1` optional). */
  readonly baseUrl: string;
  /** Optional PEM CA bundle path to trust for TLS (e.g. a corporate proxy CA). */
  readonly caBundlePath?: string;
}

/** A single model as reported by `GET /v1/models`. */
export interface CustomModel {
  /** Alternate ids/aliases the backend lists for this model (used for family matching). */
  readonly aliases?: readonly string[];
  readonly id: string;
  /** Context window (input tokens) reported by the backend, if any. */
  readonly maxInputTokens?: number;
  /** Maximum output tokens reported by the backend, if any. */
  readonly maxOutputTokens?: number;
  /**
   * Whether the backend explicitly declares reasoning/thinking support for this
   * model. `undefined` means the backend said nothing (fall back to heuristics).
   */
  readonly reasoning?: boolean;
  /** Whether the backend reports tool/function-calling support for this model. */
  readonly toolCalling?: boolean;
  /** Whether the backend reports image input support for this model. */
  readonly vision?: boolean;
}

/** OpenAI chat message shape sent on the wire. */
export interface OpenAIChatMessage {
  content: null | OpenAIContentPartWithCache[] | string;
  name?: string;
  role: "assistant" | "system" | "tool" | "user";
  tool_call_id?: string;
  tool_calls?: OpenAIToolCall[];
}

export type OpenAIContentPart =
  | { image_url: { url: string }; type: "image_url" }
  | { text: string; type: "text" };

/** A content part that may carry a cache breakpoint (text/image on the wire). */
export type OpenAIContentPartWithCache = OpenAIContentPart & { cache_control?: CacheControl };

export interface OpenAITool {
  function: {
    description?: string;
    name: string;
    parameters: unknown;
  };
  type: "function";
}

export interface OpenAIToolCall {
  function: { arguments: string; name: string };
  id: string;
  type: "function";
}

/** A parsed streaming delta from a `chat.completion.chunk` event. */
export interface StreamDelta {
  content?: string;
  finishReason?: string;
  toolCalls?: {
    argumentsFragment?: string;
    id?: string;
    index: number;
    name?: string;
  }[];
  usage?: TokenUsage;
}

/**
 * Normalized token usage. With Anthropic-style prompt caching, backends differ
 * in whether `prompt_tokens` already includes cached tokens: LiteLLM normalizes
 * it to include them, but some gateways (e.g. QGenie) report only the
 * *non-cached* input in `prompt_tokens` and surface the rest separately. We
 * capture the cache fields so the context numerator can be reconstructed as the
 * true total instead of shrinking on a cache hit.
 */
export interface TokenUsage {
  /** Cache-read (hit) input tokens, when the backend reports them. */
  cachedTokens?: number;
  /** Cache-creation (write) input tokens, when the backend reports them. */
  cacheWriteTokens?: number;
  completionTokens?: number;
  /**
   * Whether `promptTokens` already includes {@link cachedTokens}. The OpenAI
   * convention (`prompt_tokens_details.cached_tokens`) counts cached tokens
   * inside `prompt_tokens`; Anthropic's raw shape (`cache_read_input_tokens`)
   * reports them *outside* `prompt_tokens`. This tells the numerator math
   * whether to add the cached tokens or not.
   */
  promptIncludesCached?: boolean;
  /** Prompt tokens as reported; may or may not already include cached tokens. */
  promptTokens?: number;
}

export class CustomBackendClient {
  private dispatcherCache: undefined | { agent: Agent; key: string };

  constructor(private config: CustomBackendConfig) {}

  /**
   * Probe LiteLLM's `/v1/model/info` for per-model capability metadata that the
   * plain `/models` list omits — notably `supports_reasoning`. Returns a map of
   * model id → declared reasoning support. Non-LiteLLM backends 404/err here;
   * failures are swallowed (returns an empty map) so discovery still succeeds.
   */
  async fetchReasoningSupport(signal?: AbortSignal): Promise<Map<string, boolean>> {
    const support = new Map<string, boolean>();
    let response: Response;
    try {
      response = await fetch(this.endpoint("/model/info"), {
        dispatcher: this.dispatcher(),
        headers: this.headers(),
        signal,
      } as RequestInit);
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") throw error;
      return support;
    }
    if (!response.ok) return support;

    let json: unknown;
    try {
      json = await response.json();
    } catch {
      return support;
    }

    for (const entry of extractModelList(json)) {
      if (!entry || typeof entry !== "object") continue;
      const obj = entry as Record<string, unknown>;
      const id = extractModelId(obj);
      const info = obj.model_info as Record<string, unknown> | undefined;
      const supports = info?.supports_reasoning ?? obj.supports_reasoning;
      if (id && typeof supports === "boolean") {
        support.set(id, supports);
      }
    }
    return support;
  }

  /**
   * Fetch available models via `GET /v1/models`. Tolerates both the standard
   * OpenAI envelope (`{ data: [{ id }] }`) and a bare array, and derives
   * capability hints when the backend supplies them.
   */
  async listModels(signal?: AbortSignal): Promise<CustomModel[]> {
    const url = this.endpoint("/models");
    const response = await fetch(url, {
      dispatcher: this.dispatcher(),
      headers: this.headers(),
      signal,
    } as RequestInit);

    if (!response.ok) {
      const body = await safeText(response);
      throw new CustomBackendError(
        `Model listing failed (${response.status}): ${body}`,
        response.status,
      );
    }

    const json = await response.json();
    const models = parseModelList(json);
    logger.debug("[Custom Provider] Model listing response", {
      discovered: models.length,
      // Truncated raw body to reveal the actual shape when discovery yields 0.
      raw: JSON.stringify(json)?.slice(0, 2000),
      topLevelKeys:
        json && typeof json === "object"
          ? Object.keys(json as Record<string, unknown>)
          : typeof json,
      url,
    });
    return models;
  }

  setConfig(config: CustomBackendConfig): void {
    this.config = config;
  }

  /**
   * Open a streaming chat completion and yield parsed deltas as they arrive.
   */
  async *streamChat(
    request: ChatCompletionRequest,
    signal?: AbortSignal,
  ): AsyncGenerator<StreamDelta> {
    const url = this.endpoint("/chat/completions");
    const startedAt = Date.now();
    logger.debug("[Custom Provider] Opening chat stream", {
      hasTools: (request.tools?.length ?? 0) > 0,
      messageCount: request.messages.length,
      model: request.model,
      reasoningEffort: request.reasoning_effort,
      url,
    });

    let response: Response;
    try {
      response = await fetch(url, {
        body: JSON.stringify(request),
        dispatcher: this.dispatcher(),
        headers: { ...this.headers(), "content-type": "application/json" },
        method: "POST",
        signal,
      } as RequestInit);
    } catch (error) {
      // A `fetch()` rejection is a transport failure (DNS, TLS, connection
      // reset, socket timeout) — never an HTTP status. undici collapses the
      // real reason to the generic message "fetch failed" and stashes it on
      // `error.cause`, so surface the whole chain or the failure is undebuggable.
      if (error instanceof Error && error.name === "AbortError") throw error;
      logger.error("[Custom Provider] Chat stream transport failure", {
        detail: describeError(error),
        elapsedMs: Date.now() - startedAt,
        model: request.model,
        url,
      });
      throw error;
    }

    logger.debug("[Custom Provider] Chat stream response headers", {
      contentType: response.headers?.get("content-type"),
      elapsedMs: Date.now() - startedAt,
      ok: response.ok,
      status: response.status,
    });

    if (!response.ok || !response.body) {
      const body = await safeText(response);
      logger.error("[Custom Provider] Chat request returned error status", {
        body: body.slice(0, 2000),
        hasBody: Boolean(response.body),
        model: request.model,
        status: response.status,
      });
      throw new CustomBackendError(
        `Chat request failed (${response.status}): ${body}`,
        response.status,
      );
    }

    let eventCount = 0;
    for await (const delta of parseSseStream(response.body)) {
      eventCount++;
      yield delta;
    }

    // Outcome classification (empty vs. truncated vs. content) is owned by the
    // provider, which sees the parsed finish reason; here we only record the
    // raw shape of the completed stream for debugging.
    logger.debug("[Custom Provider] Chat stream completed", {
      elapsedMs: Date.now() - startedAt,
      events: eventCount,
      model: request.model,
    });
  }

  /**
   * Build (and cache) an undici dispatcher matching the current TLS settings.
   * Returns `undefined` for the default case so the process-global dispatcher
   * is used. A custom CA bundle keeps verification on; `allowInsecureTls`
   * disables it entirely. `VERIFY_X509_STRICT` is relaxed for custom CAs
   * because some corporate CAs use non-critical Basic Constraints that
   * OpenSSL 3.x rejects under strict verification.
   */
  private dispatcher(): Agent | undefined {
    const { allowInsecureTls, caBundlePath } = this.config;
    if (!allowInsecureTls && !caBundlePath) {
      this.dispatcherCache = undefined;
      return undefined;
    }

    const key = `${allowInsecureTls ? "insecure" : "secure"}|${caBundlePath ?? ""}`;
    if (this.dispatcherCache?.key === key) {
      return this.dispatcherCache.agent;
    }

    const connect: Record<string, unknown> = {};
    if (allowInsecureTls) {
      connect.rejectUnauthorized = false;
    } else if (caBundlePath) {
      try {
        connect.ca = readFileSync(caBundlePath);
      } catch (error) {
        logger.error("[Custom Provider] Failed to read CA bundle", {
          caBundlePath,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }

    const agent = new Agent({ connect });
    this.dispatcherCache = { agent, key };
    return agent;
  }

  private endpoint(path: string): string {
    let base = this.config.baseUrl;
    while (base.endsWith("/")) {
      base = base.slice(0, -1);
    }
    const root = base.endsWith("/v1") ? base : `${base}/v1`;
    return `${root}${path}`;
  }

  private headers(): Record<string, string> {
    return {
      accept: "application/json",
      authorization: `Bearer ${this.config.apiKey}`,
      "x-api-key": this.config.apiKey,
    };
  }
}

export class CustomBackendError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "CustomBackendError";
  }
}

/**
 * Render an error for logging, walking the `cause` chain. undici reports every
 * transport failure as a `TypeError: fetch failed` and hides the actionable
 * reason (ECONNRESET, TLS alert, ENOTFOUND, UND_ERR_*) on `error.cause`, so a
 * bare `error.message` is useless for diagnosing network problems.
 */
function describeError(error: unknown): Record<string, unknown> {
  if (!(error instanceof Error)) {
    return { value: String(error) };
  }
  const chain: string[] = [];
  let current: unknown = error;
  const seen = new Set<unknown>();
  while (current instanceof Error && !seen.has(current)) {
    seen.add(current);
    const code = (current as { code?: unknown }).code;
    const codeSuffix = code ? ` (${String(code)})` : "";
    chain.push(`${current.name}: ${current.message}${codeSuffix}`);
    current = (current as { cause?: unknown }).cause;
  }
  const root = error as { cause?: { code?: unknown } };
  return {
    causeChain: chain,
    code: (error as { code?: unknown }).code ?? root.cause?.code,
    message: error.message,
    name: error.name,
  };
}

/** Collect alternate ids/aliases for family matching, excluding the primary id. */
function extractAliases(obj: Record<string, unknown>, primaryId: string): string[] | undefined {
  const aliases = new Set<string>();
  for (const key of ["model", "name", "model_name"] as const) {
    const value = obj[key];
    if (typeof value === "string" && value && value !== primaryId) {
      aliases.add(value);
    } else if (Array.isArray(value)) {
      for (const item of value) {
        if (typeof item === "string" && item && item !== primaryId) aliases.add(item);
      }
    }
  }
  return aliases.size > 0 ? [...aliases] : undefined;
}

function extractModelId(obj: Record<string, unknown>): string | undefined {
  // Standard OpenAI uses a string `id`. Some gateways instead expose `name`
  // (which may be an array of routing aliases) and/or a canonical `model_name`.
  for (const key of ["id", "model", "name", "model_name"] as const) {
    const id = firstString(obj[key]);
    if (id) return id;
  }
  return undefined;
}

function extractModelList(json: unknown): unknown[] {
  if (Array.isArray(json)) {
    return json;
  }
  const data = (json as { data?: unknown })?.data;
  if (Array.isArray(data)) {
    return data;
  }
  const models = (json as { models?: unknown })?.models;
  if (Array.isArray(models)) {
    return models;
  }
  return [];
}

/** Return the value if it's a non-empty string, or the first such string in an array. */
function firstString(value: unknown): string | undefined {
  if (typeof value === "string" && value.length > 0) return value;
  if (Array.isArray(value)) {
    for (const item of value as unknown[]) {
      if (typeof item === "string" && item.length > 0) return item;
    }
  }
  return undefined;
}

/** Whether a model entry advertises the given capability in its `capabilities` array. */
function hasCapability(obj: Record<string, unknown>, capability: string): boolean {
  const capabilities = obj.capabilities;
  return Array.isArray(capabilities) && capabilities.includes(capability);
}

/**
 * Decide whether a model entry is usable for chat. Prefers explicit capability
 * metadata (`model_type.is_chat` or a `capabilities` array); when a backend
 * returns no such metadata, the model is assumed to be chat-capable.
 */
function isChatModel(obj: Record<string, unknown>): boolean {
  const modelType = obj.model_type as Record<string, unknown> | undefined;
  if (modelType && typeof modelType.is_chat === "boolean") {
    return modelType.is_chat;
  }
  if (Array.isArray(obj.capabilities)) {
    return hasCapability(obj, "chat");
  }
  return true;
}

/** Coerce a value to a finite non-negative number, or `undefined`. */
function numberOrUndefined(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : undefined;
}

/** First positive-number field found among the given keys, or `undefined`. */
function numericField(obj: Record<string, unknown>, keys: string[]): number | undefined {
  for (const key of keys) {
    const value = obj[key];
    if (typeof value === "number" && Number.isFinite(value) && value > 0) {
      return value;
    }
  }
  return undefined;
}

function parseChoice(choice: Record<string, unknown>, result: StreamDelta): void {
  if (typeof choice.finish_reason === "string") {
    result.finishReason = choice.finish_reason;
  }
  const delta = choice.delta as Record<string, unknown> | undefined;
  if (!delta) return;

  if (typeof delta.content === "string") {
    result.content = delta.content;
  }
  const toolCalls = delta.tool_calls as undefined | unknown[];
  if (Array.isArray(toolCalls)) {
    result.toolCalls = parseToolCallDeltas(toolCalls);
  }
}

function parseChunk(chunk: unknown): StreamDelta | undefined {
  if (!chunk || typeof chunk !== "object") return undefined;
  const obj = chunk as Record<string, unknown>;

  const result: StreamDelta = {
    usage: parseUsage(obj.usage as Record<string, unknown> | undefined),
  };

  const choice = (obj.choices as undefined | unknown[])?.[0] as Record<string, unknown> | undefined;
  if (choice) {
    parseChoice(choice, result);
  }

  const hasSomething =
    result.content !== undefined ||
    result.toolCalls !== undefined ||
    result.finishReason !== undefined ||
    result.usage !== undefined;
  return hasSomething ? result : undefined;
}

function parseModelList(json: unknown): CustomModel[] {
  const models: CustomModel[] = [];
  const seen = new Set<string>();
  const entries = extractModelList(json);
  for (const entry of entries) {
    if (typeof entry === "string") {
      if (!seen.has(entry)) {
        seen.add(entry);
        models.push({ id: entry });
      }
      continue;
    }
    if (entry && typeof entry === "object") {
      const obj = entry as Record<string, unknown>;
      const id = extractModelId(obj);
      // Skip entries we can't identify or that aren't usable for chat
      // (e.g. embedding-only or realtime-only models).
      if (!id || seen.has(id) || !isChatModel(obj)) continue;
      seen.add(id);
      models.push({
        aliases: extractAliases(obj, id),
        id,
        maxInputTokens: numericField(obj, ["max_input_tokens", "context_length", "context_window"]),
        maxOutputTokens: numericField(obj, ["max_output_tokens", "max_tokens"]),
        reasoning: supportsReasoning(obj),
        toolCalling: supportsTools(obj),
        vision: supportsVision(obj),
      });
    }
  }
  return models;
}

/** Parse one SSE event block into a delta, `"done"` on `[DONE]`, or `undefined` to skip. */
function parseSseEvent(rawEvent: string): "done" | StreamDelta | undefined {
  const dataLine = rawEvent.split("\n").find((line) => line.startsWith("data:"));
  if (!dataLine) return undefined;

  const payload = dataLine.slice("data:".length).trim();
  if (payload === "[DONE]") return "done";
  if (!payload) return undefined;

  let chunk: unknown;
  try {
    chunk = JSON.parse(payload);
  } catch (error) {
    logger.warn("[Custom Provider] Failed to parse SSE chunk", {
      error: error instanceof Error ? error.message : String(error),
      payload: payload.slice(0, 200),
    });
    return undefined;
  }

  return parseChunk(chunk);
}

/**
 * Parse an OpenAI-style SSE stream (`data: {json}\n\n`, terminated by
 * `data: [DONE]`) into normalized {@link StreamDelta} objects.
 */
async function* parseSseStream(body: ReadableStream<Uint8Array>): AsyncGenerator<StreamDelta> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const flush = function* (rawEvent: string): Generator<StreamDelta> {
    const outcome = parseSseEvent(rawEvent);
    if (outcome && outcome !== "done") yield outcome;
  };

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);

        logger.trace("[Custom Provider] SSE event", { raw: rawEvent.slice(0, 500) });
        const outcome = parseSseEvent(rawEvent);
        if (outcome === "done") return;
        if (outcome) yield outcome;
      }
    }

    // Some gateways close the stream after the final usage event without a
    // trailing blank line, leaving it unterminated in `buffer`. Emit it so the
    // final `usage` chunk (which carries the turn's real `prompt_tokens`) is not
    // silently dropped — that loss is what makes the context badge under-report.
    if (buffer.trim().length > 0) {
      yield* flush(buffer);
    }
  } finally {
    reader.releaseLock();
  }
}

function parseToolCallDeltas(toolCalls: unknown[]): NonNullable<StreamDelta["toolCalls"]> {
  return toolCalls.map((tc) => {
    const t = tc as Record<string, unknown>;
    const fn = t.function as Record<string, unknown> | undefined;
    return {
      argumentsFragment: typeof fn?.arguments === "string" ? fn.arguments : undefined,
      id: typeof t.id === "string" ? t.id : undefined,
      index: typeof t.index === "number" ? t.index : 0,
      name: typeof fn?.name === "string" ? fn.name : undefined,
    };
  });
}

function parseUsage(usage: Record<string, unknown> | undefined): StreamDelta["usage"] {
  if (!usage) return undefined;
  const details = usage.prompt_tokens_details as Record<string, unknown> | undefined;
  // Cache accounting varies by backend, in both key names AND whether cached
  // tokens are already counted inside `prompt_tokens`:
  //   - OpenAI/LiteLLM: `prompt_tokens_details.cached_tokens`, INSIDE prompt_tokens.
  //   - QGenie:         `prompt_tokens_details.cache_read_tokens` /
  //                     `.cache_write_tokens`, OUTSIDE prompt_tokens
  //                     (prompt_tokens holds only the non-cached delta).
  //   - Anthropic raw:  top-level `cache_read_input_tokens` /
  //                     `cache_creation_input_tokens`, OUTSIDE prompt_tokens.
  // `promptIncludesCached` is true only for the OpenAI convention so the
  // numerator math knows whether to add cache reads back.
  const openAiCached = numberOrUndefined(details?.cached_tokens);
  const externalCachedRead =
    numberOrUndefined(details?.cache_read_tokens) ??
    numberOrUndefined(usage.cache_read_input_tokens) ??
    numberOrUndefined(usage.cache_read_tokens);
  const cachedTokens = openAiCached ?? externalCachedRead;
  const cacheWriteTokens =
    numberOrUndefined(details?.cache_write_tokens) ??
    numberOrUndefined(usage.cache_creation_input_tokens) ??
    numberOrUndefined(usage.cache_creation_tokens);
  return {
    cachedTokens,
    cacheWriteTokens,
    completionTokens: numberOrUndefined(usage.completion_tokens),
    promptIncludesCached: openAiCached !== undefined,
    promptTokens: numberOrUndefined(usage.prompt_tokens),
  };
}

async function safeText(response: Response): Promise<string> {
  try {
    const text = await response.text();
    return text.slice(0, 500);
  } catch {
    return "<no body>";
  }
}

/** Derive reasoning/thinking support from capability metadata, if present. */
function supportsReasoning(obj: Record<string, unknown>): boolean | undefined {
  const info = obj.model_info as Record<string, unknown> | undefined;
  if (info && typeof info.supports_reasoning === "boolean") {
    return info.supports_reasoning;
  }
  if (typeof obj.supports_reasoning === "boolean") {
    return obj.supports_reasoning;
  }
  const modelType = obj.model_type as Record<string, unknown> | undefined;
  if (modelType && typeof modelType.is_reasoning === "boolean") {
    return modelType.is_reasoning;
  }
  if (hasCapability(obj, "reasoning") || hasCapability(obj, "thinking")) {
    return true;
  }
  return undefined;
}

/** Derive tool/function-calling support from capability metadata, if present. */
function supportsTools(obj: Record<string, unknown>): boolean | undefined {
  const modelType = obj.model_type as Record<string, unknown> | undefined;
  if (modelType && typeof modelType.is_tool_supported === "boolean") {
    return modelType.is_tool_supported;
  }
  if (typeof obj.tool_calling === "boolean") {
    return obj.tool_calling;
  }
  return undefined;
}

/** Derive image-input support from capability metadata, if present. */
function supportsVision(obj: Record<string, unknown>): boolean | undefined {
  if (typeof obj.vision === "boolean") {
    return obj.vision;
  }
  if (hasCapability(obj, "vision") || hasCapability(obj, "image")) {
    return true;
  }
  return undefined;
}
