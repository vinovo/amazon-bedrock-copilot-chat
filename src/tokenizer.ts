/**
 * Local BPE tokenizer for the Bedrock provider.
 *
 * Used by `provideTokenCount` as the fallback estimator when the Bedrock
 * `CountTokens` API is unavailable (e.g. IAM lacks `bedrock:CountTokens`,
 * which is the common case). The exact-count path via `CountTokens` is still
 * preferred when permitted.
 *
 * Implementation choices
 * ----------------------
 * - **Encoding: `o200k_base`.** This is the GPT-4o BPE vocabulary, and it is
 *   what the `vscode-copilot-chat` extension itself uses to budget prompts for
 *   its first-party Claude offering. `o200k_base` is not the *exact* tokenizer
 *   any Bedrock model uses (Claude has its own proprietary BPE; Llama uses
 *   SentencePiece; Nova uses its own). However, no offline tokenizer exists
 *   for Claude/Nova, and `o200k_base` produces token counts within ~5–10% of
 *   the true count for English prose and within ~10–15% for code/JSON — which
 *   is markedly more accurate than the previous `chars / 4` heuristic
 *   (which under-counted code/JSON by 25–40%).
 *
 * - **Library: `@microsoft/tiktokenizer`.** Pure-JS BPE implementation, same
 *   library Copilot Chat ships. No native deps, no WASM — bundles cleanly.
 *
 * - **Singleton + LRU.** The encoder is initialized lazily on first use and
 *   cached for the life of the extension host. An LRU(5000) of `string →
 *   tokenLength` mirrors Copilot Chat's `BPETokenizer._cache` and amortizes
 *   the per-encode cost across the many `provideTokenCount` calls Copilot
 *   makes per turn.
 *
 * - **No OpenAI-style per-message overhead constants.** Copilot Chat adds
 *   3-token-per-message and 8-token-per-tool framing because its prompts are
 *   serialized for the OpenAI Chat Completions wire format. Bedrock Converse
 *   uses different framing, so adding those constants would systematically
 *   over-count. We only count the actual content bytes.
 *
 * - **Fallback on init failure.** If the rank file is missing or
 *   `@microsoft/tiktokenizer` fails to construct the tokenizer, we fall back
 *   to `Math.ceil(text.length / 3)`. This is conservative (slightly over-counts
 *   for plain English; matches truth for code/JSON) and is preferable to
 *   `chars / 4` because over-counting is the safe direction in the
 *   prompt-shaping budget — it just leaves a small headroom.
 */
import type { TikTokenizer } from "@microsoft/tiktokenizer";
import { createTokenizer, getRegexByEncoder, getSpecialTokensByEncoder } from "@microsoft/tiktokenizer";
import { existsSync } from "node:fs";
import { join } from "node:path";
import * as vscode from "vscode";

import { logger } from "./logger";

const ENCODER = "o200k_base";
const RANK_FILE_NAME = "o200k_base.tiktoken";
/** Maximum number of (text → tokenLength) entries cached in memory. */
const CACHE_SIZE = 5000;
/** Tiktokenizer's own internal LRU; bigger than ours since it caches by token sequence. */
const TIKTOKENIZER_INTERNAL_CACHE = 64_000;

/**
 * Resolve the path to the vendored rank file. Two layouts:
 *
 *  - **Bundled (production / test):** `dist/extension.js` and `dist/o200k_base.tiktoken`
 *    sit next to each other. `__dirname` resolves to `dist/`.
 *  - **Source (running .ts directly, e.g. unit tests under `src/`):** the rank
 *    file lives at `src/tokenizer/o200k_base.tiktoken`. We probe a few common
 *    relative paths.
 */
function findRankFile(): string | undefined {
  // `__dirname` is correct here: the extension is bundled as CommonJS (see
  // `compile` script in package.json: `--target=node --format=cjs`). The
  // unicorn/prefer-module suggestion to use `import.meta.url` does not apply.
  // eslint-disable-next-line unicorn/prefer-module -- CJS bundle target
  const here = __dirname;
  const candidates = [
    join(here, RANK_FILE_NAME),
    join(here, "tokenizer", RANK_FILE_NAME),
    join(here, "..", "src", "tokenizer", RANK_FILE_NAME),
  ];
  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  return undefined;
}

let tokenizerInstance: TikTokenizer | undefined;
let tokenizerInitFailed = false;

function getTokenizer(): TikTokenizer | undefined {
  if (tokenizerInstance !== undefined) {
    return tokenizerInstance;
  }
  if (tokenizerInitFailed) {
    return undefined;
  }
  try {
    const rankFile = findRankFile();
    if (rankFile === undefined) {
      tokenizerInitFailed = true;
      logger.warn(
        "[Tokenizer] Could not locate o200k_base.tiktoken; falling back to char-based estimation",
      );
      return undefined;
    }
    tokenizerInstance = createTokenizer(
      rankFile,
      getSpecialTokensByEncoder(ENCODER),
      getRegexByEncoder(ENCODER),
      TIKTOKENIZER_INTERNAL_CACHE,
    );
    logger.debug("[Tokenizer] Initialized o200k_base tokenizer");
    return tokenizerInstance;
  } catch (error) {
    tokenizerInitFailed = true;
    logger.warn("[Tokenizer] Failed to initialize o200k_base tokenizer", {
      error: error instanceof Error ? error.message : String(error),
    });
    return undefined;
  }
}

// ---------------------------------------------------------------------------
// String → token-length cache (FIFO eviction; deterministic with same encoder)
// ---------------------------------------------------------------------------
//
// `Map` preserves insertion order, so we use it as a simple bounded LRU:
// re-inserting an existing key promotes it. Eviction removes the oldest
// (front-of-iteration) entry.
const lengthCache = new Map<string, number>();

/** @internal Force the encoder-init path to fail. Used only by unit tests. */
export function _forceTokenizerInitFailureForTests(): void {
  tokenizerInstance = undefined;
  tokenizerInitFailed = true;
}

/** @internal Reset cache + initialization state. Used only by unit tests. */
export function _resetTokenizerForTests(): void {
  lengthCache.clear();
  tokenizerInstance = undefined;
  tokenizerInitFailed = false;
}

/**
 * Count tokens in a structured `LanguageModelChatRequestMessage`, walking each part.
 *
 * Mirrors the previous `estimateTokens` shape so behaviour is unchanged for
 * non-text leaves (images use the existing `min(bytes / 50, 1600)` heuristic;
 * non-image binary uses `bytes / 4`; unserializable items use `100` as a safe
 * minimum). Only the text-leaf path is upgraded to use the BPE tokenizer.
 *
 * No per-message or per-tool overhead is added: Bedrock's Converse API has
 * different framing semantics than OpenAI Chat Completions, and adding OpenAI-
 * style overhead constants would systematically over-count.
 *
 * Structural walker by design: splitting the part-kind discrimination into
 * smaller functions reduces clarity for marginal complexity-score gain.
 * Mirrors the original `estimateTokens` shape from `provider.ts`.
 */
// eslint-disable-next-line sonarjs/cognitive-complexity -- Structural part-kind walker; see JSDoc.
export function countMessageTokens(message: vscode.LanguageModelChatRequestMessage): number {
  let total = 0;
  for (const part of message.content) {
    if (part instanceof vscode.LanguageModelTextPart) {
      total += cachedTokenLength(part.value);
      continue;
    }
    if (part instanceof vscode.LanguageModelToolCallPart) {
      const inputStr = JSON.stringify(part.input) ?? "";
      total += cachedTokenLength(part.name) + cachedTokenLength(inputStr);
      continue;
    }
    if (part instanceof vscode.LanguageModelToolResultPart) {
      for (const item of part.content) {
        if (item instanceof vscode.LanguageModelTextPart) {
          total += cachedTokenLength(item.value);
          continue;
        }
        try {
          total += cachedTokenLength(JSON.stringify(item));
        } catch {
          total += 100; // safe minimum for unserializable content
        }
      }
      continue;
    }
    // LanguageModelDataPart (duck-typed to avoid runtime class availability issues).
    if (typeof part === "object" && part !== null && "data" in part && "mimeType" in part) {
      const dataPart = part as { data: Uint8Array; mimeType: string };
      // Amortized image estimate: (bytes × 15 px/byte) / 750 px/token = bytes / 50.
      // Capped at 1600 — Claude's hard maximum per image regardless of dimensions.
      // Non-image binary: treat bytes as text chars (BPE doesn't help here).
      total += dataPart.mimeType.startsWith("image/")
        ? Math.min(Math.ceil(dataPart.data.length / 50), 1600)
        : Math.ceil(dataPart.data.length / 4);
    }
  }
  return total;
}

/**
 * Count tokens in a plain string using `o200k_base`.
 *
 * Async-friendly signature for parity with `provideTokenCount`, even though
 * the underlying call is synchronous (the tokenizer is in-process JS).
 */
export function countStringTokens(text: string): number {
  return cachedTokenLength(text);
}

// ---------------------------------------------------------------------------
// Test-only hooks
// ---------------------------------------------------------------------------

function cachedTokenLength(text: string): number {
  if (text.length === 0) {
    return 0;
  }
  const cached = lengthCache.get(text);
  if (cached !== undefined) {
    // Promote: reinsert moves the key to the back.
    lengthCache.delete(text);
    lengthCache.set(text, cached);
    return cached;
  }
  const length = encodeOrEstimate(text);
  if (lengthCache.size >= CACHE_SIZE) {
    // Evict the least recently used entry.
    const oldestKey = lengthCache.keys().next().value;
    if (oldestKey !== undefined) {
      lengthCache.delete(oldestKey);
    }
  }
  lengthCache.set(text, length);
  return length;
}

function encodeOrEstimate(text: string): number {
  const tokenizer = getTokenizer();
  if (tokenizer === undefined) {
    // Fallback: chars / 3 (conservative; matches code/JSON ratios reasonably).
    return Math.ceil(text.length / 3);
  }
  try {
    return tokenizer.encode(text).length;
  } catch (error) {
    logger.warn("[Tokenizer] encode() failed; falling back to char-based estimate", {
      error: error instanceof Error ? error.message : String(error),
      textLength: text.length,
    });
    return Math.ceil(text.length / 3);
  }
}
