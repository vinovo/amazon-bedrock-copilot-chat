/**
 * Max-output-token resolution for custom OpenAI-compatible backends.
 *
 * A backend-reported limit is authoritative when present (breeze/LiteLLM exposes
 * `max_output_tokens` on `/v1/models`). When it is absent (QGenie reports no
 * limits at all), fall back to a per-model ceiling keyed on model family *and*
 * generation — a family alone is insufficient, since limits differ sharply
 * within a family (Claude Opus 5 = 128K, Opus 4.5 = 64K, Opus 4.1 = 32K).
 *
 * Fallback values must never exceed a model's real ceiling: over-requesting
 * `max_tokens` makes the backend reject the request outright, so unknown models
 * resolve to a conservative floor rather than an optimistic guess.
 *
 * Sources: Anthropic model overview (platform.claude.com) and the breeze
 * `/v1/models` capability probe; OpenAI model reference for GPT-5/o-series.
 */

import { normalizeId } from "./reasoning";

/** Conservative ceiling for models we can't identify — safe on any backend. */
const CONSERVATIVE_MAX_OUTPUT_TOKENS = 8192;

/** Per-model fallback ceiling used when the backend reports no limit. */
export function fallbackMaxOutputTokens(rawId: string): number {
  const id = normalizeId(rawId);
  return (
    claudeMaxOutputTokens(id) ??
    openAiMaxOutputTokens(id) ??
    otherFamilyMaxOutputTokens(id) ??
    CONSERVATIVE_MAX_OUTPUT_TOKENS
  );
}

/**
 * Resolve a model's max output tokens: the backend-reported value wins when
 * present and positive, otherwise the per-model fallback ceiling.
 */
export function resolveMaxOutputTokens(rawId: string, reported?: number): number {
  if (typeof reported === "number" && Number.isFinite(reported) && reported > 0) {
    return reported;
  }
  return fallbackMaxOutputTokens(rawId);
}

/**
 * Anthropic Claude synchronous max-output ceiling, by tier and generation.
 * Returns undefined for non-Claude ids so the OpenAI table can be tried next.
 */
function claudeMaxOutputTokens(id: string): number | undefined {
  const tier = /\b(opus|sonnet|haiku|fable)\b/.exec(id)?.[1];
  if (!tier) return undefined;

  // Version digits appear before or after the tier depending on the backend
  // (breeze `claude-opus-4-8`, QGenie `claude-4-8-opus`); date stamps like
  // `20251001` are >2 digits and dropped, leaving major/minor as the first two.
  const nums = id
    .split(/\D+/)
    .filter((s) => s.length > 0 && s.length <= 2)
    .map(Number);
  const gen = (nums[0] ?? 0) + (nums[1] ?? 0) / 10;

  if (tier === "fable") return 128_000;
  if (tier === "haiku") {
    if (gen >= 4.5) return 64_000;
    if (gen >= 3.5) return 8192;
    return 4096;
  }
  if (tier === "opus") {
    if (gen >= 4.6) return 128_000; // 4.6, 4.7, 4.8, 5.x
    if (gen >= 4.5) return 64_000;
    if (gen >= 4) return 32_000; // 4.0, 4.1
    if (gen >= 3.5) return 8192;
    return 4096;
  }
  // sonnet
  if (gen >= 5) return 128_000;
  if (gen >= 4) return 64_000; // 4.0, 4.5, 4.6
  if (gen >= 3.5) return 8192; // 3.5, 3.7 (base, without the 64K beta header)
  return 4096;
}

/**
 * OpenAI GPT / o-series max-output ceiling. Returns undefined for non-OpenAI
 * ids. GPT-5 Pro's larger 272K output is Responses-API-only, so the chat
 * ceiling of 128K is used for the whole GPT-5 line.
 */
function openAiMaxOutputTokens(id: string): number | undefined {
  if (id.startsWith("gpt-5")) return 128_000;
  if (/^o[1-9]/.test(id)) return 100_000;
  if (id.startsWith("gpt-4.1")) return 32_000;
  if (id.startsWith("gpt-4o")) return 16_384;
  if (id.startsWith("gpt-4")) return 8192;
  return undefined;
}

/**
 * Fallback ceilings for the remaining families seen on these backends: Gemini
 * (QGenie), Qwen/DeepSeek (QGenie), and Llama (breeze, though breeze reports its
 * real limit so this is only a safety net). Returns undefined for anything else.
 */
function otherFamilyMaxOutputTokens(id: string): number | undefined {
  if (id.startsWith("gemini")) {
    // 2.5 and 3.x cap at 64K output; 2.0 (and earlier) at 8K.
    const nums = id
      .split(/\D+/)
      .filter((s) => s.length > 0 && s.length <= 2)
      .map(Number);
    const gen = (nums[0] ?? 0) + (nums[1] ?? 0) / 10;
    return gen >= 2.5 ? 65_536 : 8192;
  }
  // Qwen3, DeepSeek V3/R1, and Llama cap output at 8K on the backends we target;
  // erring low is safe since an over-requested max_tokens is rejected outright.
  if (id.startsWith("qwen") || id.startsWith("deepseek") || id.startsWith("llama")) {
    return 8192;
  }
  return undefined;
}
