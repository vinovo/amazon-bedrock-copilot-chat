import * as vscode from "vscode";

import { CustomBackendClient } from "./client";
import { CUSTOM_API_KEY_SECRET, getCustomBackendSettings, toBackendConfig } from "./settings";

/**
 * Interactive configuration for the generic backend provider, invoked via the
 * `custom.manage` command (Command Palette → "Manage Custom Model Provider").
 */
export async function manageCustomSettings(secrets: vscode.SecretStorage): Promise<void> {
  const settings = await getCustomBackendSettings(secrets);

  const action = await vscode.window.showQuickPick(
    [
      {
        description: settings.baseUrl ? `Current: ${settings.baseUrl}` : "Not set",
        label: "Set Base URL",
        value: "baseUrl" as const,
      },
      {
        description: settings.apiKey ? "Configured" : "Not set",
        label: "Set Access Token",
        value: "token" as const,
      },
      {
        description: settings.models.length > 0 ? settings.models.join(", ") : "Auto-discover",
        label: "Set Models (manual, comma-separated)",
        value: "models" as const,
      },
      {
        description: `Current: ${settings.allowInsecureTls ? "enabled" : "disabled"}`,
        label: "Toggle Allow Insecure TLS",
        value: "tls" as const,
      },
      { label: "Test Connection", value: "test" as const },
      { label: "Clear Settings", value: "clear" as const },
    ],
    { placeHolder: "Choose an action", title: "Manage Custom Model Provider" },
  );

  if (!action) return;

  switch (action.value) {
    case "baseUrl": {
      await setBaseUrl(settings.baseUrl);
      break;
    }
    case "clear": {
      await clearSettings(secrets);
      break;
    }
    case "models": {
      await setModels(settings.models);
      break;
    }
    case "test": {
      await testConnection(secrets);
      break;
    }
    case "tls": {
      await toggleTls(settings.allowInsecureTls);
      break;
    }
    case "token": {
      await setToken(secrets);
      break;
    }
  }
}

async function askScope(): Promise<undefined | vscode.ConfigurationTarget> {
  const scope = await vscode.window.showQuickPick(
    [
      {
        description: "Save globally for all workspaces",
        label: "$(globe) User Settings",
        value: vscode.ConfigurationTarget.Global,
      },
      {
        description: "Save for this workspace only",
        label: "$(folder) Workspace Settings",
        value: vscode.ConfigurationTarget.Workspace,
      },
    ],
    { placeHolder: "Where do you want to save this setting?", title: "Configuration Scope" },
  );
  return scope?.value;
}

async function clearSettings(secrets: vscode.SecretStorage): Promise<void> {
  const config = vscode.workspace.getConfiguration("custom");
  await Promise.all([
    config.update("baseUrl", undefined, vscode.ConfigurationTarget.Global),
    config.update("baseUrl", undefined, vscode.ConfigurationTarget.Workspace),
    config.update("models", undefined, vscode.ConfigurationTarget.Global),
    config.update("models", undefined, vscode.ConfigurationTarget.Workspace),
    config.update("allowInsecureTls", undefined, vscode.ConfigurationTarget.Global),
    config.update("allowInsecureTls", undefined, vscode.ConfigurationTarget.Workspace),
    secrets.delete(CUSTOM_API_KEY_SECRET),
  ]);
  vscode.window.showInformationMessage("Custom backend settings cleared.");
}

async function setBaseUrl(current: string | undefined): Promise<void> {
  const value = await vscode.window.showInputBox({
    ignoreFocusOut: true,
    prompt: "Backend base URL (e.g. https://gateway.example.com). A '/v1' suffix is optional.",
    title: "Custom Backend Base URL",
    validateInput: (v) => (v.trim().startsWith("http") ? undefined : "Must be an http(s) URL"),
    value: current ?? "",
  });
  if (value === undefined) return;

  const target = await askScope();
  if (target === undefined) return;
  await vscode.workspace
    .getConfiguration("custom")
    .update("baseUrl", value.trim() || undefined, target);
  vscode.window.showInformationMessage("Custom backend base URL saved.");
}

async function setModels(current: string[]): Promise<void> {
  const value = await vscode.window.showInputBox({
    ignoreFocusOut: true,
    prompt:
      "Comma-separated model IDs to expose. Leave empty to auto-discover via the backend's /v1/models endpoint.",
    title: "Custom Backend Models",
    value: current.join(", "),
  });
  if (value === undefined) return;

  const target = await askScope();
  if (target === undefined) return;
  await vscode.workspace
    .getConfiguration("custom")
    .update("models", value.trim() || undefined, target);
  vscode.window.showInformationMessage("Custom backend model list saved.");
}

async function setToken(secrets: vscode.SecretStorage): Promise<void> {
  const value = await vscode.window.showInputBox({
    ignoreFocusOut: true,
    password: true,
    prompt: "Access token / API key for the backend",
    title: "Custom Backend Access Token",
  });
  if (value === undefined) return;

  if (value.trim()) {
    await secrets.store(CUSTOM_API_KEY_SECRET, value.trim());
    vscode.window.showInformationMessage("Custom backend access token saved.");
  } else {
    await secrets.delete(CUSTOM_API_KEY_SECRET);
    vscode.window.showInformationMessage("Custom backend access token cleared.");
  }
}

async function testConnection(secrets: vscode.SecretStorage): Promise<void> {
  const settings = await getCustomBackendSettings(secrets);
  const config = toBackendConfig(settings);
  if (!config) {
    vscode.window.showErrorMessage("Cannot test: set a base URL and access token first.");
    return;
  }

  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: "Testing custom backend..." },
    async () => {
      try {
        const client = new CustomBackendClient(config);
        const models = await client.listModels();
        vscode.window.showInformationMessage(
          `Connection OK. Discovered ${models.length} model(s).`,
        );
      } catch (error) {
        vscode.window.showErrorMessage(
          `Connection failed: ${error instanceof Error ? error.message : String(error)}`,
        );
      }
    },
  );
}

async function toggleTls(current: boolean): Promise<void> {
  const target = await askScope();
  if (target === undefined) return;
  await vscode.workspace.getConfiguration("custom").update("allowInsecureTls", !current, target);
  vscode.window.showInformationMessage(
    `Insecure TLS ${current ? "disabled" : "enabled"} for the custom backend.`,
  );
}
