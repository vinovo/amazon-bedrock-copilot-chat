import { MIMEType } from "node:util";
import * as vscode from "vscode";

import { logger } from "../logger";
import type { OpenAIChatMessage, OpenAIContentPart, OpenAITool, OpenAIToolCall } from "./client";

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
