import type { CustomBackendConfig } from "./client";

/**
 * Per-group configuration parsing for the custom (OpenAI-compatible) provider.
 *
 * With VS Code's native language-model *provider groups*, each backend is a
 * group the user adds in the model picker. VS Code stores the group's
 * configuration (including secrets) and passes it to the provider on every
 * `provideLanguageModelChatInformation` / `provideLanguageModelChatResponse`
 * call. This module validates that raw configuration object into a typed shape;
 * it no longer reads workspace settings or SecretStorage directly.
 */

export const DEFAULT_MAX_INPUT_TOKENS = 128_000;

/** Typed view of one backend group's configuration. */
export interface CustomBackendSettings {
  readonly allowInsecureTls: boolean;
  readonly apiKey: string | undefined;
  readonly baseUrl: string | undefined;
  /** Optional PEM CA bundle path to trust for TLS (e.g. a corporate proxy CA). */
  readonly caBundlePath: string | undefined;
  readonly maxInputTokens: number;
  /** Explicit model IDs to expose when auto-discovery is unavailable. */
  readonly models: string[];
  /** Friendly name for this backend, shown as its picker heading. */
  readonly name: string | undefined;
}

/**
 * Parse the raw per-group configuration object VS Code hands the provider into
 * a typed {@link CustomBackendSettings}. Tolerates missing/misshapen fields.
 */
export function parseBackendSettings(
  configuration: Record<string, unknown> | undefined,
): CustomBackendSettings {
  const config = configuration ?? {};
  return {
    allowInsecureTls: config.allowInsecureTls === true,
    apiKey: nonEmptyString(config.apiKey),
    baseUrl: nonEmptyString(config.baseUrl),
    caBundlePath: nonEmptyString(config.caBundlePath),
    maxInputTokens: positiveNumber(config.maxInputTokens) ?? DEFAULT_MAX_INPUT_TOKENS,
    models: parseModels(config.models),
    name: nonEmptyString(config.name),
  };
}

/** Returns a ready-to-use client config, or `undefined` if not fully configured. */
export function toBackendConfig(settings: CustomBackendSettings): CustomBackendConfig | undefined {
  if (!settings.baseUrl || !settings.apiKey) {
    return undefined;
  }
  return {
    allowInsecureTls: settings.allowInsecureTls,
    apiKey: settings.apiKey,
    baseUrl: settings.baseUrl,
    caBundlePath: settings.caBundlePath,
  };
}

function nonEmptyString(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

/**
 * Accept the schema's array-of-objects (`[{ id }]`) shape, and also tolerate a
 * bare string array or a legacy comma-separated string for compatibility.
 */
function parseModels(value: unknown): string[] {
  const ids: string[] = [];
  const seen = new Set<string>();
  const push = (id: string | undefined): void => {
    if (id && !seen.has(id)) {
      seen.add(id);
      ids.push(id);
    }
  };

  if (Array.isArray(value)) {
    for (const entry of value) {
      if (typeof entry === "string") {
        push(entry.trim());
      } else if (entry && typeof entry === "object") {
        push(nonEmptyString((entry as Record<string, unknown>).id));
      }
    }
  } else if (typeof value === "string") {
    for (const part of value.split(",")) {
      push(part.trim());
    }
  }

  return ids;
}

function positiveNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : undefined;
}
