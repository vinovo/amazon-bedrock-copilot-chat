/**
 * Reasoning-effort capability resolution for custom OpenAI-compatible backends.
 *
 * Custom gateways vary wildly in what they report: LiteLLM (breeze) exposes a
 * `/model/info` endpoint with a `supports_reasoning` flag, while others (QGenie)
 * report nothing about reasoning at all. To surface VS Code's native "Thinking
 * Effort" picker consistently, capability is resolved per model id with this
 * precedence (most authoritative first):
 *
 *   1. user override    — explicit per-model config the user supplies
 *   2. backend-declared — e.g. breeze `/model/info` `supports_reasoning`
 *   3. family heuristic — infer from the model id/family (GPT, Claude, ...)
 *   4. none             — no picker, and `reasoning_effort` is never sent
 *
 * The level *sets* mirror VS Code Copilot's own picker: GPT reasoning models
 * expose six levels (`none`..`xhigh`) and Claude models five (`low`..`max`).
 * These are not sniffed from an id in Copilot — the CAPI server declares them —
 * but the observed sets are reproduced here as the heuristic fallback for
 * backends that declare nothing.
 */

/** The full ordered vocabulary of reasoning-effort levels VS Code understands. */
export const REASONING_EFFORT_LEVELS = [
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
] as const;

/**
 * The wire shape a backend expects the reasoning effort in. Mirrors VS Code's
 * `reasoningEffortFormat`: Chat Completions puts it top-level, the Responses API
 * nests it under `reasoning.effort`, and Anthropic Messages under
 * `output_config.effort`. Custom OpenAI-compatible gateways (breeze/QGenie)
 * normalize to Chat Completions, which is the default.
 */
export type ReasoningEffortFormat = "chat-completions" | "messages" | "responses";

export type ReasoningEffortLevel = (typeof REASONING_EFFORT_LEVELS)[number];

/** GPT reasoning models: six levels, matching Copilot's picker. */
const GPT_LEVELS: ReasoningEffortLevel[] = ["none", "minimal", "low", "medium", "high", "xhigh"];

/** Claude reasoning models: five levels, matching Copilot's picker. */
const CLAUDE_LEVELS: ReasoningEffortLevel[] = ["low", "medium", "high", "xhigh", "max"];

/** Gemini / generic thinking models: the conservative three-level set. */
const GENERIC_LEVELS: ReasoningEffortLevel[] = ["low", "medium", "high"];

export interface ReasoningCapability {
  /** Wire format for forwarding the effort; defaults to `chat-completions`. */
  readonly format: ReasoningEffortFormat;
  /** Allowed effort levels for this model, in picker order. Never empty. */
  readonly levels: ReasoningEffortLevel[];
}

/**
 * Pick the default (pre-selected) effort for a set of levels, mirroring
 * Copilot's `pickDefaultReasoningEffort`: Claude/Kimi families prefer `high`,
 * everything else `medium`; falling back to the middle level if neither is
 * offered.
 */
export function defaultReasoningEffort(
  levels: readonly ReasoningEffortLevel[],
  familyId: string,
): ReasoningEffortLevel | undefined {
  if (levels.length === 0) return undefined;
  const family = normalizeId(familyId);
  const preferred = family.startsWith("claude") || family.includes("kimi") ? "high" : "medium";
  if (levels.includes(preferred)) return preferred;
  return levels[Math.floor((levels.length - 1) / 2)];
}

/**
 * Infer reasoning capability from a model id/family, reproducing Copilot's
 * observed GPT/Claude/Gemini picker sets. Also matches any alias the backend
 * lists for the model. Returns `undefined` when no family is recognized, which
 * means "no picker" unless a higher-precedence source declares support.
 */
export function heuristicReasoningCapability(
  ...idsOrAliases: readonly string[]
): ReasoningCapability | undefined {
  for (const raw of idsOrAliases) {
    if (!raw) continue;
    const id = normalizeId(raw);

    // OpenAI reasoning families: GPT-5+ and the o-series.
    if (id.startsWith("gpt-5") || /^o[1-9]/.test(id)) {
      return { format: "chat-completions", levels: GPT_LEVELS };
    }
    // Anthropic Claude 4.x / 5 (Opus/Sonnet) support extended thinking.
    if (/^claude-([45]|opus-[45]|sonnet-[45])/.test(id) || /^claude-\d+-[45678]/.test(id)) {
      return { format: "chat-completions", levels: CLAUDE_LEVELS };
    }
    // Gemini 2.5 / 3.x thinking models.
    if (/^gemini-(2\.5|3)/.test(id)) {
      return { format: "chat-completions", levels: GENERIC_LEVELS };
    }
    // Generic "thinking" models (e.g. Qwen3-235B-A22B-Thinking).
    if (id.includes("think")) {
      return { format: "chat-completions", levels: GENERIC_LEVELS };
    }
  }
  return undefined;
}

/**
 * Whether a model id/alias identifies an Anthropic Claude model that supports
 * prompt caching (Claude 3.5 and all 4.x/5 families). Matched against the id
 * and any aliases after {@link normalizeId} strips provider prefixes/variants.
 *
 * Verified against real backend ids: breeze/LiteLLM (`claude-sonnet-4-6`,
 * `claude-opus-5`, ...) and QGenie (`anthropic::claude-4-6-sonnet`,
 * `anthropic::claude-4-8-opus`, `...:1M`).
 */
export function isClaudeFamily(...idsOrAliases: readonly string[]): boolean {
  for (const raw of idsOrAliases) {
    if (!raw) continue;
    const id = normalizeId(raw);
    if (
      /^claude-([45]|opus-[45]|sonnet-[45])/.test(id) ||
      /^claude-\d+-[3-8]/.test(id) ||
      /^claude-3-[5-9]/.test(id)
    ) {
      return true;
    }
  }
  return false;
}

/**
 * Normalize an id for family matching: drop a `provider::` routing prefix
 * (QGenie uses `anthropic::claude-5-sonnet`, `azure::gpt-5.4`) and a trailing
 * `:variant` suffix (e.g. `claude-4-6-opus:1M`), and lowercase.
 */
export function normalizeId(id: string): string {
  let s = id.toLowerCase();
  const sep = s.indexOf("::");
  if (sep !== -1) s = s.slice(sep + 2);
  const colon = s.indexOf(":");
  if (colon !== -1) s = s.slice(0, colon);
  return s;
}

/** Picker hover description for an effort level (mirrors Copilot wording). */
export function reasoningEffortDescription(level: string): string {
  switch (level) {
    case "high": {
      return "Greater reasoning depth but slower";
    }
    case "low": {
      return "Faster responses with less reasoning";
    }
    case "max": {
      return "Absolute maximum capability with no constraints";
    }
    case "medium": {
      return "Balanced reasoning and speed";
    }
    case "minimal": {
      return "Minimal reasoning for fastest responses";
    }
    case "none": {
      return "No reasoning applied";
    }
    case "xhigh": {
      return "Highest reasoning depth but slowest";
    }
    default: {
      return level;
    }
  }
}

/** Human-readable picker label for an effort level (mirrors Copilot wording). */
export function reasoningEffortLabel(level: string): string {
  switch (level) {
    case "high": {
      return "High";
    }
    case "low": {
      return "Low";
    }
    case "max": {
      return "Max";
    }
    case "medium": {
      return "Medium";
    }
    case "minimal": {
      return "Minimal";
    }
    case "none": {
      return "None";
    }
    case "xhigh": {
      return "Extra High";
    }
    default: {
      return level.charAt(0).toUpperCase() + level.slice(1);
    }
  }
}
