import { logger } from "../logger";

/**
 * Minimal client for OpenAI-compatible chat backends.
 *
 * Targets any HTTP endpoint that speaks the OpenAI `/v1/chat/completions`
 * protocol (streaming SSE) and, optionally, `GET /v1/models` for discovery.
 * The only required configuration is a base URL and a bearer token, which
 * makes it usable against a wide range of self-hosted gateways and proxies.
 */

export interface ChatCompletionRequest {
  max_tokens?: number;
  messages: OpenAIChatMessage[];
  model: string;
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
}

/** A single model as reported by `GET /v1/models`. */
export interface CustomModel {
  readonly id: string;
  /** Whether the backend reports tool/function-calling support for this model. */
  readonly toolCalling?: boolean;
  /** Whether the backend reports image input support for this model. */
  readonly vision?: boolean;
}

/** OpenAI chat message shape sent on the wire. */
export interface OpenAIChatMessage {
  content: null | OpenAIContentPart[] | string;
  name?: string;
  role: "assistant" | "system" | "tool" | "user";
  tool_call_id?: string;
  tool_calls?: OpenAIToolCall[];
}

export type OpenAIContentPart =
  | { image_url: { url: string }; type: "image_url" }
  | { text: string; type: "text" };

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
  usage?: { completionTokens?: number; promptTokens?: number };
}

export class CustomBackendClient {
  constructor(private config: CustomBackendConfig) {}

  /**
   * Fetch available models via `GET /v1/models`. Tolerates both the standard
   * OpenAI envelope (`{ data: [{ id }] }`) and a bare array, and derives
   * capability hints when the backend supplies them.
   */
  async listModels(signal?: AbortSignal): Promise<CustomModel[]> {
    const url = this.endpoint("/models");
    const response = await withInsecureTls(this.config.allowInsecureTls, async () =>
      fetch(url, { headers: this.headers(), signal }),
    );

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
    const response = await withInsecureTls(this.config.allowInsecureTls, async () =>
      fetch(url, {
        body: JSON.stringify(request),
        headers: { ...this.headers(), "content-type": "application/json" },
        method: "POST",
        signal,
      }),
    );

    if (!response.ok || !response.body) {
      const body = await safeText(response);
      throw new CustomBackendError(
        `Chat request failed (${response.status}): ${body}`,
        response.status,
      );
    }

    yield* parseSseStream(response.body);
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
  for (const entry of extractModelList(json)) {
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
        id,
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

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);

        const outcome = parseSseEvent(rawEvent);
        if (outcome === "done") return;
        if (outcome) yield outcome;
      }
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
  return {
    completionTokens:
      typeof usage.completion_tokens === "number" ? usage.completion_tokens : undefined,
    promptTokens: typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : undefined,
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

/** Node's fetch honors this env var to disable TLS verification for the process. */
async function withInsecureTls<T>(allow: boolean | undefined, fn: () => Promise<T>): Promise<T> {
  if (!allow) {
    return fn();
  }
  const previous = process.env.NODE_TLS_REJECT_UNAUTHORIZED;
  process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";
  return fn().finally(() => {
    if (previous === undefined) {
      delete process.env.NODE_TLS_REJECT_UNAUTHORIZED;
    } else {
      process.env.NODE_TLS_REJECT_UNAUTHORIZED = previous;
    }
  });
}
