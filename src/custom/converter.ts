import { MIMEType } from "node:util";
import * as vscode from "vscode";

import { logger } from "../logger";
import type {
  OpenAIChatMessage,
  OpenAIContentPart,
  OpenAIContentPartWithCache,
  OpenAITool,
  OpenAIToolCall,
} from "./client";

/**
 * Convert VS Code chat messages/tools into OpenAI Chat Completions payloads.
 *
 * The mapping is intentionally lossless for the parts the OpenAI protocol
 * models natively (text, images, tool calls, tool results) and drops anything
 * else (e.g. provider-specific data parts) rather than guessing.
 */

interface DataPart {
  readonly data: Uint8Array;
  readonly mimeType: string;
}

export function convertMessages(
  messages: readonly vscode.LanguageModelChatRequestMessage[],
): OpenAIChatMessage[] {
  const result: OpenAIChatMessage[] = [];

  for (const msg of messages) {
    switch (msg.role) {
      case vscode.LanguageModelChatMessageRole.Assistant: {
        result.push(...convertAssistantMessage(msg));
        break;
      }
      case vscode.LanguageModelChatMessageRole.User: {
        result.push(...convertUserMessage(msg));
        break;
      }
      default: {
        // Any other role (e.g. system) maps to an OpenAI system message.
        result.push({ content: extractText(msg), role: "system" });
        break;
      }
    }
  }

  return result.filter((msg) => isNonEmptyMessage(msg));
}

/** Anthropic's hard cap on the number of `cache_control` breakpoints per request. */
const MAX_CACHE_BREAKPOINTS = 4;

/**
 * Insert Anthropic-style prompt-cache breakpoints into an already-converted
 * message list, mirroring VS Code Copilot's placement strategy
 * (`addCacheBreakpoints`). Copilot never sends these to `custom`-vendor
 * providers (the sentinel is gated to a built-in allow-list, issue #313920), so
 * we reproduce the placement here for Claude models, which are the only ones on
 * our backends that support prompt caching.
 *
 * Strategy (max 4 breakpoints, matching Anthropic's limit):
 *  - below the current (last) user message: the last tool-result of each round
 *    and the current user message — these are the stable prefix of an agentic
 *    loop, so the next request hits the cache on the previous tool result;
 *  - above it: terminal assistant messages (a turn's final response), giving a
 *    hit on the previous turn when no tools were called;
 *  - any remaining breakpoints go to the leading system/user messages.
 *
 * Markers are attached to the LAST content block of the chosen message (the
 * message is converted to block form if it was a plain string), which is how
 * Anthropic scopes "cache everything up to and including this block".
 */
export function addCacheBreakpoints(messages: OpenAIChatMessage[]): void {
  // Only allocate breakpoints not already present, mirroring Copilot's
  // `MaxCacheBreakpoints - countCacheBreakpoints(...)`. This keeps the function
  // idempotent and never exceeds Anthropic's cap across repeated calls.
  let remaining = MAX_CACHE_BREAKPOINTS - countCacheBreakpoints(messages);
  let belowCurrentUserMessage = true;

  for (let i = messages.length - 1; i >= 0 && remaining > 0; i--) {
    const msg = messages[i];
    const prev = messages[i - 1];

    if (hasCacheBreakpoint(msg)) {
      if (msg.role === "user") {
        belowCurrentUserMessage = false;
      }
      continue;
    }

    const isLastToolResultInRound = msg.role === "tool" && prev?.role !== "tool";
    const isTerminalAssistant = msg.role === "assistant" && !msg.tool_calls?.length;
    const shouldMark =
      (belowCurrentUserMessage && (isLastToolResultInRound || msg.role === "user")) ||
      isTerminalAssistant;

    if (shouldMark && markCacheBreakpoint(msg)) {
      remaining--;
    }
    if (msg.role === "user") {
      belowCurrentUserMessage = false;
    }
  }

  // Spend any leftover breakpoints on the leading system/user messages (the
  // static preamble: system prompt + custom instructions), which change least.
  for (let i = 0; i < messages.length && remaining > 0; i++) {
    const msg = messages[i];
    if (msg.role !== "system" && msg.role !== "user") {
      break;
    }
    if (!hasCacheBreakpoint(msg) && markCacheBreakpoint(msg)) {
      remaining--;
    }
  }
}

/** Whether a message already carries a cache breakpoint on any content block. */
function hasCacheBreakpoint(msg: OpenAIChatMessage): boolean {
  return Array.isArray(msg.content) && msg.content.some((part) => part.cache_control !== undefined);
}

/** Total cache breakpoints already present across all messages. */
function countCacheBreakpoints(messages: OpenAIChatMessage[]): number {
  let count = 0;
  for (const msg of messages) {
    if (Array.isArray(msg.content)) {
      count += msg.content.filter((part) => part.cache_control !== undefined).length;
    }
  }
  return count;
}

/**
 * Attach a cache breakpoint to the last content block of a message, converting
 * string content into a single text block first. Returns `false` (no-op) when
 * the message has no cacheable content (e.g. an assistant message that is only
 * tool calls), so the caller doesn't waste a breakpoint.
 */
function markCacheBreakpoint(msg: OpenAIChatMessage): boolean {
  if (hasCacheBreakpoint(msg)) {
    return false;
  }
  if (typeof msg.content === "string") {
    if (msg.content.length === 0) {
      return false;
    }
    msg.content = [{ text: msg.content, type: "text" }];
  }
  if (!Array.isArray(msg.content) || msg.content.length === 0) {
    return false;
  }
  const last = msg.content[msg.content.length - 1] as OpenAIContentPartWithCache;
  last.cache_control = { type: "ephemeral" };
  return true;
}

export function convertTools(
  tools: readonly vscode.LanguageModelChatTool[] | undefined,
): OpenAITool[] | undefined {
  if (!tools || tools.length === 0) {
    return undefined;
  }
  return tools.map((tool) => ({
    function: {
      description: tool.description,
      name: tool.name,
      parameters: tool.inputSchema ?? { properties: {}, type: "object" },
    },
    type: "function",
  }));
}

function convertAssistantMessage(msg: vscode.LanguageModelChatRequestMessage): OpenAIChatMessage[] {
  let text = "";
  const toolCalls: OpenAIToolCall[] = [];

  for (const part of msg.content) {
    if (part instanceof vscode.LanguageModelTextPart) {
      text += part.value;
    } else if (part instanceof vscode.LanguageModelToolCallPart) {
      toolCalls.push({
        function: { arguments: JSON.stringify(part.input ?? {}), name: part.name },
        id: part.callId,
        type: "function",
      });
    }
  }

  const message: OpenAIChatMessage = { content: text || null, role: "assistant" };
  if (toolCalls.length > 0) {
    message.tool_calls = toolCalls;
  }
  return [message];
}

/**
 * A VS Code user message can carry both plain input and tool results. Tool
 * results must become separate `role: "tool"` messages in the OpenAI protocol,
 * so this may expand into multiple messages.
 */
function convertUserMessage(msg: vscode.LanguageModelChatRequestMessage): OpenAIChatMessage[] {
  const contentParts: OpenAIContentPart[] = [];
  const toolMessages: OpenAIChatMessage[] = [];

  for (const part of msg.content) {
    if (part instanceof vscode.LanguageModelTextPart) {
      if (part.value.trim()) {
        contentParts.push({ text: part.value, type: "text" });
      }
    } else if (part instanceof vscode.LanguageModelToolResultPart) {
      toolMessages.push({
        content: extractToolResultText(part.content),
        role: "tool",
        tool_call_id: part.callId,
      });
    } else if (isDataPart(part)) {
      const imagePart = toImagePart(part);
      if (imagePart) contentParts.push(imagePart);
    }
  }

  const messages: OpenAIChatMessage[] = [];
  if (contentParts.length > 0) {
    // Collapse a lone text part to a plain string for maximum backend compatibility.
    const onlyText =
      contentParts.length === 1 && contentParts[0].type === "text"
        ? contentParts[0].text
        : undefined;
    messages.push({ content: onlyText ?? contentParts, role: "user" });
  }
  messages.push(...toolMessages);
  return messages;
}

function extractText(msg: vscode.LanguageModelChatRequestMessage): string {
  return msg.content
    .filter((p): p is vscode.LanguageModelTextPart => p instanceof vscode.LanguageModelTextPart)
    .map((p) => p.value)
    .join("");
}

function extractToolResultText(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((item) => {
        if (item instanceof vscode.LanguageModelTextPart) {
          return item.value;
        }
        if (item && typeof item === "object" && "value" in item) {
          return String((item as { value: unknown }).value);
        }
        return typeof item === "string" ? item : JSON.stringify(item);
      })
      .join("");
  }
  if (content && typeof content === "object") {
    return JSON.stringify(content);
  }
  return String(content ?? "");
}

function isDataPart(part: unknown): part is DataPart {
  return (
    !!part &&
    typeof part === "object" &&
    "data" in part &&
    "mimeType" in part &&
    (part as DataPart).data instanceof Uint8Array
  );
}

function isNonEmptyMessage(msg: OpenAIChatMessage): boolean {
  if (msg.tool_calls && msg.tool_calls.length > 0) {
    return true;
  }
  if (typeof msg.content === "string") {
    return msg.role === "tool" || msg.content.length > 0;
  }
  return Array.isArray(msg.content) && msg.content.length > 0;
}

function toImagePart(part: DataPart): OpenAIContentPart | undefined {
  try {
    const mime = new MIMEType(part.mimeType);
    if (mime.type !== "image") {
      return undefined;
    }
    const base64 = Buffer.from(part.data).toString("base64");
    return {
      image_url: { url: `data:${part.mimeType};base64,${base64}` },
      type: "image_url",
    };
  } catch (error) {
    logger.warn("[Custom Provider] Skipping image with invalid MIME type", {
      error: error instanceof Error ? error.message : String(error),
      mimeType: part.mimeType,
    });
    return undefined;
  }
}
