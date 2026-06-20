import type { JsonValue } from "type-fest" with { "resolution-mode": "import" };

import { logger } from "./logger";

interface ToolCall {
  id: string;
  input: unknown;
  name: string;
}

export class ToolBuffer {
  private readonly emittedIndices = new Set<number>();
  private readonly inputBuffers = new Map<number, string>();
  private readonly tools = new Map<number, ToolCall>();

  appendInput(index: number, inputChunk: string): void {
    const current = this.inputBuffers.get(index) ?? "";
    this.inputBuffers.set(index, current + inputChunk);
  }

  /**
   * Clear all tracking state. Should be called at the start of each new request.
   */
  clear(): void {
    this.tools.clear();
    this.inputBuffers.clear();
    this.emittedIndices.clear();
  }

  finalizeTool(index: number): ToolCall | undefined {
    const tool = this.tools.get(index);
    const inputStr = this.inputBuffers.get(index);

    // If the tool was never started there is nothing to finalize.
    if (!tool) {
      return undefined;
    }

    // A zero-argument tool call (e.g. an MCP "whoami" tool) streams a tool_use block
    // with no input deltas, leaving the buffer empty. Treat that as an empty-object input
    // rather than dropping the (valid) tool call. Only genuinely malformed JSON is skipped.
    const rawInput = inputStr && inputStr.trim().length > 0 ? inputStr : "{}";

    try {
      tool.input = JSON.parse(rawInput) as JsonValue;
    } catch {
      logger.warn("[ToolBuffer] Failed to parse tool input JSON, skipping tool call", {
        inputLength: rawInput.length,
        toolId: tool.id,
        toolName: tool.name,
      });
      logger.trace("[ToolBuffer] Raw input preview for failed tool parse", {
        rawInputPreview: rawInput.slice(0, 200).replaceAll("\n", String.raw`\n`),
        toolId: tool.id,
        toolName: tool.name,
      });
      this.tools.delete(index);
      this.inputBuffers.delete(index);
      return undefined;
    }

    this.tools.delete(index);
    this.inputBuffers.delete(index);

    return tool;
  }

  /**
   * Check if a tool at this index has already been emitted to prevent duplicates.
   */
  isEmitted(index: number): boolean {
    return this.emittedIndices.has(index);
  }

  /**
   * Mark a tool as emitted to prevent duplicate emissions.
   */
  markEmitted(index: number): void {
    this.emittedIndices.add(index);
  }

  startTool(index: number, id: string, name: string): void {
    this.tools.set(index, { id, input: {}, name });
    this.inputBuffers.set(index, "");
  }

  /**
   * Try to parse and return the tool if JSON is valid, without removing from buffer.
   * Useful for early emission while continuing to accumulate more input.
   * Returns undefined if JSON is not yet valid or tool doesn't exist.
   */
  tryGetValidTool(index: number): ToolCall | undefined {
    const tool = this.tools.get(index);
    const inputStr = this.inputBuffers.get(index);

    if (!tool) {
      return undefined;
    }

    // An empty buffer represents a complete zero-argument call ({}), not incomplete JSON.
    const rawInput = inputStr && inputStr.trim().length > 0 ? inputStr : "{}";

    try {
      const parsed = JSON.parse(rawInput) as JsonValue;
      return {
        id: tool.id,
        input: parsed,
        name: tool.name,
      };
    } catch {
      // JSON not yet valid - this is expected during streaming
      return undefined;
    }
  }
}
