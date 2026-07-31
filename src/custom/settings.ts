import * as vscode from "vscode";

import type { CustomBackendConfig } from "./client";

/**
 * Persistence for the generic (OpenAI-compatible) backend provider.
 *
 * Only two things are strictly required to reach a backend: a base URL and an
 * access token. The base URL and non-secret preferences live in workspace/user
 * settings; the token lives in {@link vscode.SecretStorage}. An optional
 * comma-separated model list lets users target backends whose `/v1/models`
 * endpoint is absent or non-standard.
 */

export const CUSTOM_API_KEY_SECRET = "custom.apiKey";

export interface CustomBackendSettings {
  readonly allowInsecureTls: boolean;
  readonly apiKey: string | undefined;
  readonly baseUrl: string | undefined;
  /** Explicit model IDs to expose when auto-discovery is unavailable. */
  readonly models: string[];
}

export async function getCustomBackendSettings(
  secrets: vscode.SecretStorage,
): Promise<CustomBackendSettings> {
  const config = vscode.workspace.getConfiguration("custom");
  const baseUrl = normalizeBaseUrl(config.get<null | string>("baseUrl"));
  const allowInsecureTls = config.get<boolean>("allowInsecureTls") ?? false;
  const rawModels = config.get<string>("models") ?? "";
  const models = rawModels
    .split(",")
    .map((m) => m.trim())
    .filter((m) => m.length > 0);

  const apiKey = (await secrets.get(CUSTOM_API_KEY_SECRET)) || undefined;

  return { allowInsecureTls, apiKey, baseUrl, models };
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
  };
}

function normalizeBaseUrl(value: null | string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed || undefined;
}
