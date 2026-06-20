import { ModelModality } from "@aws-sdk/client-bedrock";
import type {
  ConverseStreamCommandInput,
  CountTokensCommandInput,
  Message,
  SystemContentBlock,
  ToolConfiguration,
} from "@aws-sdk/client-bedrock-runtime";
import { inspect, MIMEType } from "node:util";
import type {
  CancellationToken,
  LanguageModelChatInformation,
  LanguageModelChatProvider,
  LanguageModelChatRequestMessage,
  LanguageModelResponsePart,
  Progress,
} from "vscode";
import * as vscode from "vscode";

import { getRegionPrefix } from "./aws-partition";
import { BedrockAPIClient, ListFoundationModelsDeniedError } from "./bedrock-client";
import { convertMessages, stripThinkingContent } from "./converters/messages";
import { convertTools } from "./converters/tools";
import { logger } from "./logger";
import { getModelProfile, getModelTokenLimits } from "./profiles";
import { getBedrockSettings } from "./settings";
import { StreamProcessor, type ThinkingBlock } from "./stream-processor";
import { countMessageTokens, countStringTokens } from "./tokenizer";
import type { AuthConfig, AuthMethod, BedrockModelSummary } from "./types";
import { validateBedrockMessages } from "./validation";

/**
 * Extends the stable `LanguageModelChatInformation` with fields that VS Code's chat
 * model picker reads from the proposed `chatProvider` API surface but does not actually
 * gate behind a proposal check at runtime.
 *
 * - `isUserSelectable` is required for the model to appear in the model picker dropdown.
 *   Since VS Code refactored the picker (~1.116+), `filterModelsForSession` filters out
 *   any model whose metadata.isUserSelectable is not truthy, for ALL vendors. Without it
 *   our models still show up in the Manage Models view but are missing from the picker,
 *   so users cannot select/pin them.
 * - `category` groups models under a named section in the picker. Without it, our models
 *   fall under "Other Models" (collapsed, ordered last).
 *
 * `configurationSchema` and `isUserSelectable` are part of the proposed `chatProvider` API and
 * are declared on the base `LanguageModelChatInformation` once `vscode.proposed.chatProvider.d.ts`
 * is vendored; we re-declare `isUserSelectable` here only to keep this type self-documenting.
 * `category` is NOT part of any proposal — it is duck-typed and read by the picker at runtime.
 */
type BedrockLanguageModelChatInformation = LanguageModelChatInformation & {
  readonly category?: { label: string; order: number };
  readonly isUserSelectable?: boolean;
};

/**
 * Thinking display mode passed via `thinking.display` for Claude extended thinking.
 * "omitted" suppresses streamed thinking text (faster time-to-first-token).
 */
type ThinkingDisplay = "omitted" | "summarized";

/**
 * Effort level passed via `output_config.effort` for adaptive thinking and Claude
 * effort-aware models. See https://platform.claude.com/docs/en/build-with-claude/effort
 */
type ThinkingEffort = "high" | "low" | "medium";

class NoAccessibleModelsError extends Error {
  constructor() {
    super("No accessible Bedrock models detected");
    this.name = "NoAccessibleModelsError";
  }
}

const BEDROCK_MODEL_PICKER_CATEGORY = { label: "Amazon Bedrock", order: 50 } as const;

/**
 * Stable id used for the synthetic "⚠ Bedrock unavailable" entry that we surface
 * in the model picker when fetching the real Bedrock model list fails. Sending
 * a chat against this id is rejected up-front in `provideLanguageModelChatResponse`.
 *
 * The sentinel exists to defeat VS Code core's silent "reset selected model to
 * default" behavior in `chatInputPart.shouldResetOnModelListChange` — by always
 * keeping at least one Bedrock entry in the list, the user's selection is never
 * silently swapped to a non-Bedrock model just because our fetch failed.
 */
export const BEDROCK_ERROR_SENTINEL_ID = "__bedrock_error_sentinel__";

export class BedrockChatModelProvider implements vscode.Disposable, LanguageModelChatProvider {
  // Event to notify VS Code that model information has changed (stable API name)
  private readonly _onDidChangeLanguageModelInformation = new vscode.EventEmitter<void>();
  readonly onDidChangeLanguageModelChatInformation =
    this._onDidChangeLanguageModelInformation.event;

  private chatEndpoints: { model: string; modelMaxPromptTokens: number }[] = [];
  private readonly client: BedrockAPIClient;
  /** Tracks whether the initial model fetch has completed (for avoiding startup feedback loops) */
  private initialFetchComplete = false;
  /**
   * Snapshot of the last successfully-fetched Bedrock model list. Used together
   * with the error sentinel to keep at least one Bedrock entry visible in the
   * model picker after a fetch failure, so VS Code core does not silently switch
   * the user's selection to a non-Bedrock default model.
   */
  private lastKnownModels: BedrockLanguageModelChatInformation[] = [];
  private lastThinkingBlock?: ThinkingBlock;
  private readonly streamProcessor: StreamProcessor;

  constructor(
    private readonly secrets: vscode.SecretStorage,
    private readonly globalState: vscode.Memento,
  ) {
    // Initialize with default region - will be updated on first use
    this.client = new BedrockAPIClient("us-east-1", undefined);
    this.streamProcessor = new StreamProcessor();
  }

  /**
   * Dispose resources held by the provider
   */
  public dispose(): void {
    try {
      this._onDidChangeLanguageModelInformation.dispose();
    } catch {
      // ignore
    }
  }

  /**
   * Returns true if the initial model fetch has completed.
   * Used to avoid feedback loops when responding to onDidChangeChatModels during startup.
   */
  public isInitialFetchComplete(): boolean {
    return this.initialFetchComplete;
  }

  /**
   * Notify the workbench that the available model information should be refreshed.
   * Hooked up from extension activation to configuration, secrets, and model selection changes.
   */
  public notifyModelInformationChanged(reason?: string): void {
    const suffix = reason ? `: ${reason}` : "";
    logger.debug(`[Bedrock Model Provider] Signaling model info refresh${suffix}`);
    // Snapshot the cached endpoints right before the refresh so we can correlate
    // any subsequent context-window-tracker ("X / Y tokens" badge) denominator
    // changes with the trigger that caused them. The next call to
    // `prepareLanguageModelChatInformation` will rebuild this list and emit a
    // "models rebuilt" log; comparing the two reveals which model(s) changed
    // their max input/output limits and why the badge denominator moved.
    logger.debug("[Bedrock Model Provider] Cached endpoints at refresh time:", {
      endpoints: this.chatEndpoints.map((e) => ({
        model: e.model,
        modelMaxPromptTokens: e.modelMaxPromptTokens,
      })),
      endpointsCount: this.chatEndpoints.length,
      reason: reason ?? "unspecified",
    });
    this._onDidChangeLanguageModelInformation.fire();
  }

  // eslint-disable-next-line sonarjs/cognitive-complexity -- Provider bootstrapping requires multiple guarded flows
  async prepareLanguageModelChatInformation(
    options: { silent: boolean },
    token: CancellationToken,
  ): Promise<LanguageModelChatInformation[]> {
    const settings = await getBedrockSettings(this.globalState);

    // Check if this is the first run by checking if we've shown the welcome prompt before
    const hasRunBefore = this.globalState.get<boolean>("bedrock.hasRunBefore", false);

    if (!hasRunBefore && !options.silent) {
      const action = await vscode.window.showInformationMessage(
        "Amazon Bedrock integration requires AWS credentials. Would you like to configure your AWS profile and region first?",
        "Configure Settings",
        "Use Default Credentials",
      );

      // Mark that we've shown the prompt
      await this.globalState.update("bedrock.hasRunBefore", true);

      if (action === "Configure Settings") {
        await vscode.commands.executeCommand("bedrock.manage");
        // Return empty array - user will need to refresh after configuring
        return [];
      } else if (action !== "Use Default Credentials") {
        // User cancelled
        return [];
      }
      // If "Use Default Credentials" was selected, continue with the fetch
    }

    const authConfig = await this.getAuthConfig(options.silent);
    if (!authConfig) {
      if (!options.silent) {
        vscode.window.showErrorMessage(
          "AWS Bedrock authentication not configured. Please run 'Manage Amazon Bedrock Provider'.",
        );
      }
      // Surface a sentinel rather than an empty list so VS Code's chat UI does not
      // silently switch the user's selection to a non-Bedrock default model.
      return this.buildSentinelModelList(new Error("AWS Bedrock authentication not configured"));
    }

    this.client.setRegion(settings.region);
    if (authConfig.method === "profile") {
      this.client.setProfile(settings.profile);
    }
    this.client.setAuthConfig(authConfig);

    try {
      // Create AbortController for cancellation support
      const abortController = new AbortController();

      // Set up cancellation handling
      const cancellationListener = token.onCancellationRequested(() => {
        abortController.abort();
      });

      try {
        const fetchModels = async (
          progress?: vscode.Progress<{ message?: string }>,
        ): Promise<LanguageModelChatInformation[]> => {
          progress?.report({ message: "Fetching model list..." });

          const [models, apiProfileIds] = await Promise.all([
            this.client.fetchModels(abortController.signal),
            this.client.fetchInferenceProfiles(abortController.signal),
          ]);

          // Merge normal profile detection with any fallback profiles we detected when ListFoundationModels is blocked
          const availableProfileIds = new Set<string>(apiProfileIds);
          for (const fallbackId of this.client.getFallbackInferenceProfileIds()) {
            availableProfileIds.add(fallbackId);
          }

          // Fetch application inference profiles after we have foundation models
          const applicationProfiles = await this.client.fetchApplicationInferenceProfiles(
            models,
            abortController.signal,
          );

          // Extract region prefix for inference profile IDs (handles GovCloud, China, and commercial regions)
          const regionPrefix = getRegionPrefix(settings.region);
          const candidates = this.buildModelCandidates(
            models,
            availableProfileIds,
            regionPrefix,
            settings.inferenceProfiles.preferRegional,
          );

          progress?.report({
            message: `Checking availability of ${candidates.length} models...`,
          });

          // Check model accessibility in parallel using allSettled to handle failures gracefully
          const accessibilityChecks = await Promise.allSettled(
            candidates.map(async (candidate) =>
              this.evaluateCandidateAccessibility(
                candidate,
                regionPrefix,
                availableProfileIds,
                settings.inferenceProfiles.preferRegional,
                abortController.signal,
              ),
            ),
          );

          progress?.report({ message: "Building model list..." });

          // Build final list of accessible models
          const infos: LanguageModelChatInformation[] = [];
          for (const result of accessibilityChecks) {
            // If the check failed, treat as inaccessible
            if (result.status === "rejected") {
              logger.error("[Bedrock Model Provider] Accessibility check failed", result.reason);
              continue;
            }

            const { hasInferenceProfile, isAccessible, model: m, modelIdToUse } = result.value;

            if (!isAccessible) {
              logger.debug(
                `[Bedrock Model Provider] Excluding inaccessible model: ${modelIdToUse} (not authorized or not available)`,
              );
              continue;
            }

            const limits = getModelTokenLimits(modelIdToUse, settings.context1M.enabled);
            const maxInput = limits.maxInputTokens;
            const maxOutput = limits.maxOutputTokens;
            const vision = m.inputModalities.includes(ModelModality.IMAGE);

            // Determine tooltip suffix based on inference profile type
            let tooltipSuffix = "";
            if (hasInferenceProfile) {
              tooltipSuffix = modelIdToUse.startsWith("global.")
                ? " (Global Inference Profile)"
                : " (Regional Inference Profile)";
            }

            const modelInfo: BedrockLanguageModelChatInformation = {
              capabilities: {
                imageInput: vision,
                toolCalling: true,
              },
              category: BEDROCK_MODEL_PICKER_CATEGORY,
              configurationSchema: this.buildThinkingConfigurationSchema(modelIdToUse),
              family: "bedrock",
              id: modelIdToUse,
              isUserSelectable: true,
              maxInputTokens: maxInput,
              maxOutputTokens: maxOutput,
              name: m.modelName,
              tooltip: `Amazon Bedrock - ${m.providerName}${tooltipSuffix}`,
              version: "1.0.0",
            };
            infos.push(modelInfo);
          }

          // Add application inference profiles
          progress?.report({
            message: `Processing ${applicationProfiles.length} application profiles...`,
          });

          for (const profile of applicationProfiles) {
            // Filter profiles similar to foundation models - must support streaming and text output
            if (
              !profile.responseStreamingSupported ||
              !profile.outputModalities.includes(ModelModality.TEXT)
            ) {
              logger.debug(
                `[Bedrock Model Provider] Excluding application profile: ${profile.modelId} (no streaming or text output)`,
              );
              continue;
            }

            // Use base model ID for token limits (falls back to profile ID if not available)
            const modelIdForLimits = profile.baseModelId ?? profile.modelId;
            const limits = getModelTokenLimits(modelIdForLimits, settings.context1M.enabled);
            const maxInput = limits.maxInputTokens;
            const maxOutput = limits.maxOutputTokens;
            const vision = profile.inputModalities.includes(ModelModality.IMAGE);

            const profileInfo: BedrockLanguageModelChatInformation = {
              capabilities: {
                imageInput: vision,
                toolCalling: true,
              },
              category: BEDROCK_MODEL_PICKER_CATEGORY,
              configurationSchema: this.buildThinkingConfigurationSchema(modelIdForLimits),
              family: "bedrock",
              id: profile.modelArn,
              isUserSelectable: true,
              maxInputTokens: maxInput,
              maxOutputTokens: maxOutput,
              name: profile.modelName,
              tooltip: `Amazon Bedrock - ${profile.providerName} (Application Inference Profile)`,
              version: "1.0.0",
            };
            infos.push(profileInfo);
          }

          // Sort models: inference profiles by updatedAt/createdAt (newest first), then others
          progress?.report({ message: "Sorting models..." });

          // Build lookup map for O(1) access during sorting
          const modelDateMap = new Map<string, Date | undefined>();
          for (const c of candidates) {
            const date = c.model.updatedAt ?? c.model.createdAt;
            modelDateMap.set(c.model.modelId, date);
            modelDateMap.set(c.model.modelArn, date);
          }
          for (const p of applicationProfiles) {
            const date = p.updatedAt ?? p.createdAt;
            modelDateMap.set(p.modelId, date);
            modelDateMap.set(p.modelArn, date);
          }

          infos.sort((a, b) => {
            const aDate = modelDateMap.get(a.id);
            const bDate = modelDateMap.get(b.id);

            // If both have dates, sort by date (newest first)
            if (aDate && bDate) {
              return bDate.getTime() - aDate.getTime();
            }

            // Models with dates come before models without dates
            if (aDate) return -1;
            if (bDate) return 1;

            // If neither has a date, maintain original order
            return 0;
          });

          if (infos.length === 0) {
            throw new NoAccessibleModelsError();
          }

          this.chatEndpoints = infos.map((info) => ({
            model: info.id,
            modelMaxPromptTokens: info.maxInputTokens + info.maxOutputTokens,
          }));
          // Cache the successful fetch so subsequent failures can re-emit the
          // same model entries (alongside an error sentinel) and avoid the
          // VS Code core silent-fallback behavior.
          this.lastKnownModels = infos as BedrockLanguageModelChatInformation[];

          // Mark initial fetch as complete to allow onDidChangeChatModels handling
          this.initialFetchComplete = true;

          // Log the effective per-model limits after the rebuild so we can
          // diagnose context-window badge denominator changes. The badge
          // denominator in VS Code is `maxInputTokens + maxOutputTokens`
          // looked up from the model registered with this provider, so any
          // shift in these numbers between rebuilds will visibly move the
          // badge on the next request.
          logger.debug("[Bedrock Model Provider] Models rebuilt with effective token limits:", {
            context1MEnabled: settings.context1M.enabled,
            models: infos.map((info) => ({
              id: info.id,
              maxInputTokens: info.maxInputTokens,
              maxOutputTokens: info.maxOutputTokens,
              totalContextWindow: info.maxInputTokens + info.maxOutputTokens,
            })),
          });

          return infos;
        };

        // Show progress notification only if not silent
        if (options.silent) {
          return await fetchModels();
        }

        return await vscode.window.withProgress(
          {
            cancellable: true,
            location: vscode.ProgressLocation.Notification,
            title: "Loading Bedrock models",
          },
          fetchModels,
        );
      } finally {
        cancellationListener.dispose();
      }
    } catch (error) {
      // Don't log or show errors if the operation was cancelled by the user
      if (error instanceof Error && error.name === "AbortError") {
        logger.info("[Bedrock Model Provider] Model fetch cancelled by user");
        return [];
      }

      if (!options.silent) {
        logger.error("[Bedrock Model Provider] Failed to fetch models", error);
        if (error instanceof ListFoundationModelsDeniedError) {
          const manualModelId = await vscode.window.showInputBox({
            placeHolder: "global.anthropic.claude-sonnet-4-6",
            prompt:
              "Model listing is blocked by AWS permissions. Enter a Bedrock model ID or inference profile ID to use.",
          });

          if (manualModelId) {
            const manualInfo = await this.buildManualModelInformation(
              manualModelId,
              settings,
              token,
            );

            if (manualInfo) {
              this.chatEndpoints = [
                {
                  model: manualInfo.id,
                  modelMaxPromptTokens: manualInfo.maxInputTokens + manualInfo.maxOutputTokens,
                },
              ];
              return [manualInfo];
            }
          }

          vscode.window.showErrorMessage(
            "Could not detect any Bedrock models with current permissions. Please update your AWS policy or provide a reachable model ID.",
          );
          return this.buildSentinelModelList(error);
        } else if (error instanceof NoAccessibleModelsError) {
          const manualModelId = await vscode.window.showInputBox({
            placeHolder: "global.anthropic.claude-sonnet-4-6",
            prompt:
              "No accessible Bedrock models were detected. Enter a Bedrock model ID or inference profile ID to use.",
          });

          if (manualModelId) {
            const manualInfo = await this.buildManualModelInformation(
              manualModelId,
              settings,
              token,
            );

            if (manualInfo) {
              this.chatEndpoints = [
                {
                  model: manualInfo.id,
                  modelMaxPromptTokens: manualInfo.maxInputTokens + manualInfo.maxOutputTokens,
                },
              ];
              return [manualInfo];
            }
          }

          vscode.window.showErrorMessage(
            "Could not detect any accessible Bedrock models. Please update your AWS policy or provide a reachable model ID.",
          );
          return this.buildSentinelModelList(error);
        } else {
          vscode.window.showErrorMessage(
            `Failed to fetch Bedrock models. Please check your AWS profile and region settings. Error: ${error instanceof Error ? error.message : String(error)}`,
          );
          return this.buildSentinelModelList(error);
        }
      }
      // Silent fetch (no UI prompts allowed): still surface a sentinel so the
      // user's selected Bedrock model is not silently replaced.
      return this.buildSentinelModelList(error);
    }
  }

  async provideLanguageModelChatInformation(
    options: { silent: boolean },
    token: CancellationToken,
  ): Promise<LanguageModelChatInformation[]> {
    return this.prepareLanguageModelChatInformation({ silent: options.silent ?? false }, token);
  }

  // eslint-disable-next-line sonarjs/cognitive-complexity -- Chat response handling requires validation of thinking config and error handling
  async provideLanguageModelChatResponse(
    model: LanguageModelChatInformation,
    messages: readonly LanguageModelChatRequestMessage[],
    options: Parameters<LanguageModelChatProvider["provideLanguageModelChatResponse"]>[2],
    progress: Progress<LanguageModelResponsePart>,
    token: CancellationToken,
  ): Promise<void> {
    const trackingProgress: Progress<LanguageModelResponsePart> = {
      report: (part) => {
        try {
          progress.report(part);
        } catch (error) {
          logger.warn("[Bedrock Model Provider] Progress.report failed", {
            error:
              error instanceof Error ? { message: error.message, name: error.name } : String(error),
            modelId: model.id,
          });
          // Re-throw so callers can detect emission failures (e.g. stream-processor
          // uses try-catch around ThinkingPart emission to track hasEmittedThinking).
          throw error;
        }
      },
    };

    // Reject the synthetic error sentinel up-front. The sentinel exists only to
    // keep a Bedrock entry in the picker after a model-list fetch failure (see
    // `BEDROCK_ERROR_SENTINEL_ID`); it never represents a real model.
    if (model.id === BEDROCK_ERROR_SENTINEL_ID) {
      const reason = (model as BedrockLanguageModelChatInformation).detail ?? "unknown error";
      throw new Error(
        `Bedrock model list could not be loaded: ${reason}. ` +
          `Please verify your AWS profile/region (run 'Manage Amazon Bedrock Provider'), ` +
          `then re-open the model picker to retry.`,
      );
    }

    try {
      // Get authentication configuration (silent to avoid prompting during active chat)
      const authConfig = await this.getAuthConfig(true);
      if (!authConfig) {
        throw new Error("AWS Bedrock authentication not configured");
      }

      // Configure client with authentication
      this.client.setAuthConfig(authConfig);

      // Resolve model ID for application inference profiles (ARNs) to base model ID
      // This is needed because internal logic (getModelProfile, getModelTokenLimits) expects base model IDs
      // Note: For the actual API call, we still use the original model.id (ARN for app profiles)
      const abortController = new AbortController();
      const cancellationListener = token.onCancellationRequested(() => {
        abortController.abort();
      });

      let baseModelId: string;
      try {
        baseModelId = await this.client.resolveModelId(model.id, abortController.signal);
        logger.info("[Bedrock Model Provider] Resolved model ID", {
          originalModelId: model.id,
          resolvedBaseModelId: baseModelId,
        });
      } catch (error) {
        // If resolution fails, use the original model ID
        baseModelId = model.id;
        logger.warn("[Bedrock Model Provider] Failed to resolve model ID, using original", {
          error: error instanceof Error ? error.message : String(error),
          modelId: model.id,
        });
      } finally {
        cancellationListener.dispose();
      }

      // Log incoming messages
      this.logIncomingMessages(messages);

      // Get settings and model configuration
      const settings = await getBedrockSettings(this.globalState);
      const modelProfile = getModelProfile(baseModelId);
      const modelLimits = getModelTokenLimits(baseModelId, settings.context1M.enabled);

      // Per-model picker configuration (proposed `chatProvider` API) takes precedence over the
      // workspace `bedrock.thinking.*` fallback settings. Whatever the user selects in the model
      // picker arrives on `options.modelConfiguration`, keyed to our `configurationSchema`.
      this.applyModelConfigurationOverrides(settings, options.modelConfiguration);

      // Calculate thinking configuration
      // Use model's maxOutputTokens as default when VSCode doesn't provide max_tokens.
      // This prevents thinking budget starvation that causes MAX_TOKENS errors
      // (GitHub Copilot uses server-configured large values + 16K thinking budget by default)
      const maxTokensForRequest =
        typeof options.modelOptions?.max_tokens === "number"
          ? options.modelOptions.max_tokens
          : modelLimits.maxOutputTokens;
      const { budgetTokens, extendedThinkingEnabled: initialThinkingEnabled } =
        this.calculateThinkingConfig(
          modelProfile,
          modelLimits,
          maxTokensForRequest,
          settings.thinking.enabled,
        );
      let extendedThinkingEnabled = initialThinkingEnabled;

      // Check if we can actually use extended thinking with the current conversation history
      // When thinking is enabled, ALL assistant messages must have thinking blocks.
      // VSCode doesn't preserve thinking blocks, so we can only inject our stored lastThinkingBlock.
      // This means we can only support thinking when:
      // - There are no previous assistant messages (first turn), OR
      // - There is exactly one previous assistant message AND we have a stored thinking block
      // If there are 2+ assistant messages, we can't provide thinking blocks for all of them.
      if (extendedThinkingEnabled) {
        const assistantMsgCount = messages.filter(
          (m) => m.role === vscode.LanguageModelChatMessageRole.Assistant,
        ).length;

        if (assistantMsgCount > 1) {
          // Can't inject thinking blocks for multiple previous assistant messages
          // Each assistant message needs its own unique thinking block, but we only have one stored
          logger.debug(
            "[Bedrock Model Provider] Disabling extended thinking - multiple assistant messages in history require individual thinking blocks",
            { assistantMsgCount },
          );
          extendedThinkingEnabled = false;
          // Clear stale thinking block to prevent it from being misapplied if conversation
          // history later truncates back to a single assistant message (signatures are
          // integrity-bound to specific thinking blocks)
          this.lastThinkingBlock = undefined;
        } else if (assistantMsgCount === 1 && !this.lastThinkingBlock?.signature) {
          // Have one assistant message but no thinking block to inject
          logger.debug(
            "[Bedrock Model Provider] Disabling extended thinking - no stored thinking block available for previous assistant message",
          );
          extendedThinkingEnabled = false;
        }
      }

      // Convert messages with thinking configuration
      const converted = convertMessages(messages, baseModelId, {
        extendedThinkingEnabled,
        lastThinkingBlock: this.lastThinkingBlock,
        promptCachingEnabled: settings.promptCaching.enabled,
      });

      // Log converted messages
      this.logConvertedMessages(converted.messages);

      // Validate messages and tools
      validateBedrockMessages(converted.messages);

      const toolConfig = convertTools(
        options,
        baseModelId,
        extendedThinkingEnabled,
        settings.promptCaching.enabled,
      );

      if (options.tools && options.tools.length > 128) {
        throw new Error("Cannot have more than 128 tools per request.");
      }

      // Determine if thinking effort should be applied (only for Opus 4.5 and Sonnet 4.6)
      const thinkingEffortEnabled = modelProfile.supportsThinkingEffort;

      // Build beta headers
      const betaHeaders = this.buildBetaHeaders(
        modelProfile,
        extendedThinkingEnabled,
        settings.context1M.enabled,
        thinkingEffortEnabled,
      );

      // Build request input
      const requestInput = this.buildRequestInput(
        model,
        modelProfile,
        converted,
        options,
        toolConfig,
        extendedThinkingEnabled,
        budgetTokens,
        betaHeaders,
        thinkingEffortEnabled ? settings.thinking.effort : undefined,
        modelProfile.supportsThinkingDisplay ? settings.thinking.display : undefined,
      );

      // Log request details
      this.logRequestDetails(requestInput);

      // Validate token count
      await this.validateTokenCount(model, requestInput, token);

      // Process the stream
      await this.processResponseStream(
        requestInput,
        trackingProgress,
        extendedThinkingEnabled,
        token,
      );
    } catch (error) {
      // Check for context window overflow errors and provide better error messages
      // Reference: https://github.com/strands-agents/sdk-python/blob/dbf6200d104539217dddfc7bd729c53f46e2ec56/src/strands/models/bedrock.py#L852-L860
      if (isContextWindowOverflowError(error)) {
        const errorMessage =
          "Input exceeds model context window. " +
          "Consider reducing conversation history, removing tool results, or adjusting model parameters.";
        logger.error("[Bedrock Model Provider] Context window overflow", {
          messageCount: messages.length,
          modelId: model.id,
          originalError: error instanceof Error ? error.message : String(error),
        });
        throw new Error(errorMessage, { cause: error });
      }

      // Extract detailed error information from AWS SDK error
      const errorDetails: Record<string, unknown> = {
        messageCount: messages.length,
        modelId: model.id,
      };

      if (error instanceof Error) {
        errorDetails.error = {
          message: error.message,
          name: error.name,
          stack: error.stack,
        };

        // AWS SDK errors have additional metadata in hidden fields
        const awsError = error as unknown as Record<string, unknown>;

        // Extract $metadata
        if (awsError.$metadata) {
          errorDetails.awsMetadata = awsError.$metadata;
        }

        // Use util.format with %O to capture hidden fields like $response
        // This properly shows non-enumerable properties that inspect might miss
        errorDetails.fullErrorWithFormat = inspect(error, {
          depth: 10,
          getters: true,
          maxArrayLength: 100,
          maxStringLength: 1000,
          showHidden: true,
        });
      } else {
        errorDetails.error = String(error);
      }

      logger.error("[Bedrock Model Provider] Chat request failed", errorDetails);
      throw error;
    }
  }

  async provideTokenCount(
    model: LanguageModelChatInformation,
    text: LanguageModelChatRequestMessage | string,
    token: CancellationToken,
  ): Promise<number> {
    // The chat context window tracker (the "X / Y tokens" badge in Copilot Chat)
    // is populated separately from this method.
    //
    // Until VS Code 1.120.0, third-party `LanguageModelChatProvider` extensions
    // could not report usage to the badge — Copilot Chat hardcoded
    // `usage: { prompt_tokens: 0, ... }` for all non-Copilot providers. That
    // limitation was lifted in VS Code 1.120.0 / Copilot Chat 1.120.0:
    //
    //   Issue:    https://github.com/microsoft/vscode/issues/291100
    //   Fix PR:   https://github.com/microsoft/vscode/pull/315394
    //
    // We now emit a `LanguageModelDataPart` with MIME `"usage"` from
    // `processResponseStream` once Bedrock's stream metadata event arrives
    // (see `processResponseStream` and `StreamProcessor.handleMetadata`). That
    // data part feeds the badge with the *exact* token counts reported by
    // Bedrock, so this method's return value is **not** displayed in the badge.
    //
    // `provideTokenCount` IS still called many times per turn by Copilot's
    // prompt-shaping logic (it's the only thing that decides what fits in the
    // context window before the request is sent). Accuracy here matters for
    // avoiding context-overflow errors from Bedrock at request time — see the
    // `isContextWindowOverflowError` defense in `provideLanguageModelChatResponse`.

    // The error sentinel is not a real model — short-circuit before any AWS
    // calls or tokenization work. Returning 0 keeps Copilot's prompt-shaping
    // logic from doing pointless work for an entry that cannot be sent against.
    if (model.id === BEDROCK_ERROR_SENTINEL_ID) {
      return 0;
    }

    // Fallback estimation when the Bedrock CountTokens API is unavailable
    // (e.g. IAM lacks `bedrock:CountTokens`). Delegates to the local
    // `o200k_base` BPE tokenizer in `src/tokenizer.ts` — same approach
    // Copilot Chat takes for non-Copilot first-party models.
    const estimateTokens = (input: LanguageModelChatRequestMessage | string): number => {
      return typeof input === "string" ? countStringTokens(input) : countMessageTokens(input);
    };

    try {
      const abortController = new AbortController();
      const cancellationListener = token.onCancellationRequested(() => {
        abortController.abort();
      });

      // Resolve model ID for application inference profiles (ARNs) to base model ID.
      // convertMessages calls getModelProfile which expects base model IDs.
      let baseModelId: string;
      try {
        baseModelId = await this.client.resolveModelId(model.id, abortController.signal);
      } catch (error) {
        baseModelId = model.id;
        logger.warn("[Bedrock Model Provider] Failed to resolve model ID, using original", {
          error: error instanceof Error ? error.message : String(error),
          modelId: model.id,
        });
      }

      try {
        // CountTokens API expects structured messages; fall back to char-based estimation
        // for simple string inputs.
        if (typeof text === "string") {
          return estimateTokens(text);
        }

        const settings = await getBedrockSettings(this.globalState);
        const converted = convertMessages([text], baseModelId, {
          extendedThinkingEnabled: false,
          lastThinkingBlock: undefined,
          promptCachingEnabled: settings.promptCaching.enabled,
        });

        const tokenCount = await this.client.countTokens(
          model.id,
          {
            converse: {
              messages: converted.messages,
              ...(converted.system.length > 0 ? { system: converted.system } : {}),
            },
          },
          abortController.signal,
        );

        if (tokenCount !== undefined) {
          return tokenCount;
        }

        // CountTokens API unavailable (e.g. IAM denial) — use local estimation.
        return estimateTokens(text);
      } finally {
        cancellationListener.dispose();
      }
    } catch (error) {
      if (!(error instanceof Error && error.name === "AbortError")) {
        logger.warn("[Bedrock Model Provider] Token count failed, using estimation", error);
      }
      return estimateTokens(text);
    }
  }

  /**
   * Build additional model request fields for adaptive-thinking-only models (Opus 4.8 / 4.7).
   * No budget_tokens, no temperature override. The `effort` parameter is honored when supported
   * (Opus 4.8 defaults to "high").
   * See:
   *   https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-8.html
   *   https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-7.html
   */
  private applyAdaptiveThinkingFields(
    requestInput: ConverseStreamCommandInput,
    modelId: string,
    modelProfile: ReturnType<typeof getModelProfile>,
    betaHeaders: string[],
    thinkingEffort?: ThinkingEffort,
    thinkingDisplay?: ThinkingDisplay,
  ): void {
    const includeDisplay = Boolean(thinkingDisplay) && modelProfile.supportsThinkingDisplay;
    requestInput.additionalModelRequestFields = {
      thinking: {
        type: "adaptive",
        ...(includeDisplay ? { display: thinkingDisplay } : {}),
      },
      ...(betaHeaders.length > 0 ? { anthropic_beta: betaHeaders } : {}),
      ...(thinkingEffort && modelProfile.supportsThinkingEffort
        ? { output_config: { effort: thinkingEffort } }
        : {}),
    };
    logger.debug("[Bedrock Model Provider] Adaptive thinking enabled", {
      anthropicBeta: betaHeaders.length > 0 ? betaHeaders : undefined,
      modelId,
      thinkingDisplay: includeDisplay ? thinkingDisplay : undefined,
      thinkingEffort:
        thinkingEffort && modelProfile.supportsThinkingEffort ? thinkingEffort : undefined,
    });
  }

  /**
   * Apply per-model picker configuration (proposed `chatProvider` API) on top of the workspace
   * `bedrock.thinking.*` fallback settings. Values present in `modelConfiguration` win; absent
   * values leave the fallback intact. Unknown/invalid values are ignored.
   */
  private applyModelConfigurationOverrides(
    settings: Awaited<ReturnType<typeof getBedrockSettings>>,
    modelConfiguration: Readonly<Record<string, unknown>> | undefined,
  ): void {
    if (!modelConfiguration) {
      return;
    }

    if (typeof modelConfiguration.thinkingEnabled === "boolean") {
      settings.thinking.enabled = modelConfiguration.thinkingEnabled;
    }

    const effort = modelConfiguration.thinkingEffort;
    if (effort === "high" || effort === "medium" || effort === "low") {
      settings.thinking.effort = effort;
    }

    const display = modelConfiguration.thinkingDisplay;
    if (display === "summarized" || display === "omitted") {
      settings.thinking.display = display;
    }
  }

  /**
   * Build additional model request fields for standard extended thinking models
   * (Opus 4.6, 4.5, 4.1, Sonnet 4.6/4.5/4, Haiku 4.5, Sonnet 3.7).
   * Requires temperature 1.0 and an explicit budget_tokens value.
   */
  private applyStandardExtendedThinkingFields(
    requestInput: ConverseStreamCommandInput,
    modelId: string,
    budgetTokens: number,
    betaHeaders: string[],
    thinkingEffort?: ThinkingEffort,
    thinkingDisplay?: ThinkingDisplay,
  ): void {
    requestInput.inferenceConfig!.temperature = 1;
    requestInput.additionalModelRequestFields = {
      thinking: {
        budget_tokens: budgetTokens,
        type: "enabled",
        ...(thinkingDisplay ? { display: thinkingDisplay } : {}),
      },
      ...(betaHeaders.length > 0 ? { anthropic_beta: betaHeaders } : {}),
      // Add thinking effort for Claude Opus 4.5 and Sonnet 4.6 (controls token expenditure)
      ...(thinkingEffort ? { output_config: { effort: thinkingEffort } } : {}),
    };
    logger.debug("[Bedrock Model Provider] Extended thinking enabled", {
      anthropicBeta: betaHeaders.length > 0 ? betaHeaders : undefined,
      budgetTokens,
      interleavedThinking: betaHeaders.includes("interleaved-thinking-2025-05-14"),
      modelId,
      supports1MContext: betaHeaders.includes("context-1m-2025-08-07"),
      temperature: 1,
      thinkingDisplay: thinkingDisplay ?? "(not applicable)",
      thinkingEffort: thinkingEffort ?? "(not applicable)",
    });
  }

  /**
   * Build beta headers array for the request
   */
  private buildBetaHeaders(
    modelProfile: ReturnType<typeof getModelProfile>,
    extendedThinkingEnabled: boolean,
    context1MEnabled: boolean,
    thinkingEffortEnabled: boolean,
  ): string[] {
    const anthropicBeta: string[] = [];

    if (extendedThinkingEnabled) {
      // Add interleaved-thinking beta header for Claude 4 models
      if (modelProfile.requiresInterleavedThinkingHeader) {
        anthropicBeta.push("interleaved-thinking-2025-05-14");
      }

      // Add 1M context beta header for models that support it and setting is enabled
      if (modelProfile.supports1MContext && context1MEnabled) {
        anthropicBeta.push("context-1m-2025-08-07");
      }
    } else if (modelProfile.supports1MContext && context1MEnabled) {
      // Even if thinking is not enabled, add 1M context beta header
      anthropicBeta.push("context-1m-2025-08-07");
    }

    // Add effort beta header for Claude Opus 4.5 and Sonnet 4.6 when thinking effort is configured
    if (thinkingEffortEnabled) {
      anthropicBeta.push("effort-2025-11-24");
    }

    return anthropicBeta;
  }

  /**
   * Allow users with restricted permissions to manually supply a model or inference profile ID.
   */
  private async buildManualModelInformation(
    modelId: string,
    settings: Awaited<ReturnType<typeof getBedrockSettings>>,
    token: CancellationToken,
  ): Promise<LanguageModelChatInformation | undefined> {
    const abortController = new AbortController();
    const cancellationListener = token.onCancellationRequested(() => abortController.abort());

    try {
      let baseModelId = modelId;
      try {
        baseModelId = await this.client.resolveModelId(modelId, abortController.signal);
      } catch (resolveError) {
        logger.warn("[Bedrock Model Provider] Manual model resolution failed, using provided ID", {
          error:
            resolveError instanceof Error
              ? { message: resolveError.message, name: resolveError.name }
              : String(resolveError),
          modelId,
        });
      }

      const limits = getModelTokenLimits(baseModelId, settings.context1M.enabled);
      const likelyVisionCapable = /anthropic\.|nova\.|llama\.|pixtral|gpt-oss/i.test(baseModelId);

      return {
        capabilities: {
          imageInput: likelyVisionCapable,
          toolCalling: true,
        },
        family: "bedrock",
        id: modelId,
        maxInputTokens: limits.maxInputTokens,
        maxOutputTokens: limits.maxOutputTokens,
        name: modelId,
        tooltip: "Amazon Bedrock - manual model entry",
        version: "1.0.0",
      };
    } catch (error) {
      if (!(error instanceof Error && error.name === "AbortError")) {
        logger.error("[Bedrock Model Provider] Manual model setup failed", error);
      }
      return undefined;
    } finally {
      cancellationListener.dispose();
    }
  }

  private buildModelCandidates(
    models: BedrockModelSummary[],
    availableProfileIds: Set<string>,
    regionPrefix: string,
    preferRegional = false,
  ): {
    hasInferenceProfile: boolean;
    model: BedrockModelSummary;
    modelIdToUse: string;
  }[] {
    const candidates: {
      hasInferenceProfile: boolean;
      model: BedrockModelSummary;
      modelIdToUse: string;
    }[] = [];

    for (const m of models) {
      if (!m.responseStreamingSupported || !m.outputModalities.includes(ModelModality.TEXT)) {
        continue;
      }

      // Determine which model ID to use (with or without inference profile)
      // By default, prefer global inference profiles for best availability, then regional, then base model
      // When preferRegional is enabled, check regional profiles first (for Control Tower compliance)
      const globalProfileId = `global.${m.modelId}`;
      const regionalProfileId = `${regionPrefix}.${m.modelId}`;

      let modelIdToUse = m.modelId;
      let hasInferenceProfile = false;

      if (preferRegional) {
        // Prefer regional profiles first
        if (availableProfileIds.has(regionalProfileId)) {
          modelIdToUse = regionalProfileId;
          hasInferenceProfile = true;
          logger.trace(
            `[Bedrock Model Provider] Using regional inference profile for ${m.modelId}`,
          );
        } else if (availableProfileIds.has(globalProfileId)) {
          modelIdToUse = globalProfileId;
          hasInferenceProfile = true;
          logger.trace(
            `[Bedrock Model Provider] Using global inference profile for ${m.modelId} (regional not available)`,
          );
        }
      } else {
        // Default behavior: prefer global profiles first
        if (availableProfileIds.has(globalProfileId)) {
          modelIdToUse = globalProfileId;
          hasInferenceProfile = true;
          logger.trace(`[Bedrock Model Provider] Using global inference profile for ${m.modelId}`);
        } else if (availableProfileIds.has(regionalProfileId)) {
          modelIdToUse = regionalProfileId;
          hasInferenceProfile = true;
          logger.trace(
            `[Bedrock Model Provider] Using regional inference profile for ${m.modelId}`,
          );
        }
      }

      candidates.push({ hasInferenceProfile, model: m, modelIdToUse });
    }

    return candidates;
  }

  /**
   * Build and configure the request input for Bedrock API
   */
  private buildRequestInput(
    model: LanguageModelChatInformation,
    modelProfile: ReturnType<typeof getModelProfile>,
    converted: { messages: Message[]; system: SystemContentBlock[] },
    options: Parameters<LanguageModelChatProvider["provideLanguageModelChatResponse"]>[2],
    toolConfig: ToolConfiguration | undefined,
    extendedThinkingEnabled: boolean,
    budgetTokens: number,
    betaHeaders: string[],
    thinkingEffort?: ThinkingEffort,
    thinkingDisplay?: ThinkingDisplay,
  ): ConverseStreamCommandInput {
    // When adaptive thinking is active, Anthropic disallows temperature/top_p/top_k.
    // - Adaptive-only models (Opus 4.8/4.7) never accept them.
    // - Adaptive-preferring models (Opus 4.6 / Sonnet 4.6) only forbid them while extended
    //   thinking is enabled; without thinking they behave like normal sampling models.
    const usesAdaptiveThinking =
      modelProfile.supportsAdaptiveThinkingOnly ||
      (modelProfile.prefersAdaptiveThinking && extendedThinkingEnabled);

    const requestInput: ConverseStreamCommandInput = {
      inferenceConfig: {
        maxTokens: Math.min(
          typeof options.modelOptions?.max_tokens === "number"
            ? options.modelOptions.max_tokens
            : model.maxOutputTokens,
          model.maxOutputTokens,
        ),
        // Adaptive thinking does not support temperature/top_p/top_k — omit entirely.
        ...(!usesAdaptiveThinking && {
          temperature:
            typeof options.modelOptions?.temperature === "number"
              ? options.modelOptions?.temperature
              : 0.7,
        }),
      },
      messages: converted.messages,
      modelId: model.id,
    };

    if (converted.system.length > 0) {
      requestInput.system = converted.system;
    }

    if (options.modelOptions && !usesAdaptiveThinking) {
      const mo = options.modelOptions;
      if (typeof mo.top_p === "number") {
        requestInput.inferenceConfig!.topP = mo.top_p;
      }
      if (typeof mo.stop === "string") {
        requestInput.inferenceConfig!.stopSequences = [mo.stop];
      } else if (Array.isArray(mo.stop)) {
        requestInput.inferenceConfig!.stopSequences = mo.stop;
      }
    }

    if (toolConfig) {
      requestInput.toolConfig = toolConfig;
    }

    // Add additional model request fields (thinking, effort, beta headers)
    this.configureAdditionalModelFields(
      requestInput,
      model.id,
      modelProfile,
      extendedThinkingEnabled,
      budgetTokens,
      betaHeaders,
      thinkingEffort,
      thinkingDisplay,
    );

    return requestInput;
  }

  /**
   * Returns the previously-known Bedrock model list (if any) plus a single
   * synthetic "⚠ Bedrock unavailable" sentinel describing the failure. Keeping
   * at least one Bedrock entry in the list prevents VS Code core's chat input
   * part from silently resetting the user's selection to a non-Bedrock default
   * model when our fetch fails. See `BEDROCK_ERROR_SENTINEL_ID` for context.
   */
  private buildSentinelModelList(error: unknown): BedrockLanguageModelChatInformation[] {
    const sentinel = makeBedrockErrorSentinel(error);
    return this.lastKnownModels.length > 0 ? [...this.lastKnownModels, sentinel] : [sentinel];
  }

  /**
   * Build the per-model `configurationSchema` (proposed `chatProvider` API) that drives the
   * model picker's thinking controls. Returns `undefined` for models without any thinking
   * capability so no picker UI is shown for them.
   *
   * The schema is keyed off the model's capability profile:
   * - `thinkingEnabled` (boolean) — toggle extended thinking, when the model supports thinking.
   * - `thinkingEffort` (enum, `group: "navigation"`) — effort level, surfaced as the dedicated
   *   "Thinking Effort" picker button for effort-capable models.
   * - `thinkingDisplay` (enum) — summarized vs omitted thinking output, for display-capable models.
   *
   * Whatever the user picks arrives back on `options.modelConfiguration` and takes precedence
   * over the workspace `bedrock.thinking.*` fallback settings.
   */
  private buildThinkingConfigurationSchema(
    modelId: string,
  ): BedrockLanguageModelChatInformation["configurationSchema"] {
    const profile = getModelProfile(modelId);

    if (!profile.supportsThinking) {
      return undefined;
    }

    type SchemaProperties = NonNullable<
      BedrockLanguageModelChatInformation["configurationSchema"]
    >["properties"];
    type SchemaProperty = NonNullable<SchemaProperties>[string];

    const thinkingEnabled: SchemaProperty = {
      default: true,
      description: "Enable extended thinking for this model.",
      type: "boolean",
    };

    const thinkingEffort: SchemaProperty | undefined = profile.supportsThinkingEffort
      ? {
          default: "high",
          description: "Thinking effort level.",
          enum: ["high", "medium", "low"],
          enumDescriptions: [
            "Maximum capability — Claude uses as many tokens as needed for the best outcome.",
            "Balanced approach with moderate token savings.",
            "Most efficient — significant token savings with some capability reduction.",
          ],
          enumItemLabels: ["High", "Medium", "Low"],
          // `group: "navigation"` surfaces this as the dedicated picker button (UBB mode).
          group: "navigation",
          type: "string",
        }
      : undefined;

    const thinkingDisplay: SchemaProperty | undefined = profile.supportsThinkingDisplay
      ? {
          default: "summarized",
          description: "How thinking content is returned.",
          enum: ["summarized", "omitted"],
          enumDescriptions: [
            "Stream summarized thinking text (default).",
            "Suppress streamed thinking text for faster time-to-first-token (still billed in full).",
          ],
          enumItemLabels: ["Summarized", "Omitted"],
          type: "string",
        }
      : undefined;

    return {
      properties: {
        thinkingEnabled,
        ...(thinkingEffort ? { thinkingEffort } : {}),
        ...(thinkingDisplay ? { thinkingDisplay } : {}),
      },
    };
  }

  /**
   * Calculate thinking configuration parameters
   */
  private calculateThinkingConfig(
    modelProfile: ReturnType<typeof getModelProfile>,
    modelLimits: ReturnType<typeof getModelTokenLimits>,
    maxTokensForRequest: number,
    thinkingEnabled: boolean,
  ): { budgetTokens: number; extendedThinkingEnabled: boolean } {
    // Use a base budget of 16,000 tokens (aligned with GitHub Copilot's default),
    // capped at 25% of maxOutputTokens and constrained by maxTokensForRequest.
    // Reserve at least 25% of maxTokensForRequest (minimum 100 tokens) for visible
    // response content so that small explicit max_tokens values still produce output.
    const baseBudget = 16_000;
    const maxBudgetFromOutput = Math.floor(modelLimits.maxOutputTokens * 0.25);
    const visibleReserve = Math.max(100, Math.floor(maxTokensForRequest * 0.25));
    const budgetTokens = Math.max(
      0,
      Math.min(baseBudget, maxBudgetFromOutput, maxTokensForRequest - visibleReserve),
    );
    const extendedThinkingEnabled =
      thinkingEnabled && modelProfile.supportsThinking && budgetTokens >= 1024;

    return { budgetTokens, extendedThinkingEnabled };
  }

  /**
   * Configure additional model request fields for thinking, effort, and beta headers
   */
  private configureAdditionalModelFields(
    requestInput: ConverseStreamCommandInput,
    modelId: string,
    modelProfile: ReturnType<typeof getModelProfile>,
    extendedThinkingEnabled: boolean,
    budgetTokens: number,
    betaHeaders: string[],
    thinkingEffort?: ThinkingEffort,
    thinkingDisplay?: ThinkingDisplay,
  ): void {
    if (extendedThinkingEnabled) {
      // Route to adaptive thinking for models that require it (Opus 4.8/4.7) AND models that
      // merely prefer it (Opus 4.6 / Sonnet 4.6, where manual budget thinking is deprecated).
      if (modelProfile.supportsAdaptiveThinkingOnly || modelProfile.prefersAdaptiveThinking) {
        this.applyAdaptiveThinkingFields(
          requestInput,
          modelId,
          modelProfile,
          betaHeaders,
          thinkingEffort,
          thinkingDisplay,
        );
      } else {
        this.applyStandardExtendedThinkingFields(
          requestInput,
          modelId,
          budgetTokens,
          betaHeaders,
          thinkingEffort,
          thinkingDisplay,
        );
      }
      return;
    }

    if (thinkingEffort) {
      // Claude Opus 4.5 and Sonnet 4.6 effort parameter can be used even without extended thinking
      // This affects all token spend including tool calls
      requestInput.additionalModelRequestFields = {
        ...(betaHeaders.length > 0 ? { anthropic_beta: betaHeaders } : {}),
        output_config: { effort: thinkingEffort },
      };

      logger.debug("[Bedrock Model Provider] Thinking effort enabled (without extended thinking)", {
        anthropicBeta: betaHeaders.length > 0 ? betaHeaders : undefined,
        modelId,
        thinkingEffort,
      });
      return;
    }

    if (betaHeaders.length > 0) {
      // Add beta headers even without thinking or effort
      requestInput.additionalModelRequestFields = {
        anthropic_beta: betaHeaders,
      };

      logger.debug("[Bedrock Model Provider] 1M context enabled", { modelId });
    }
  }

  /**
   * Count tokens for a complete request using the CountTokens API.
   * Falls back to estimation if the API is unavailable or fails.
   * @param modelId The model ID to count tokens for
   * @param input The complete input structure (messages, system, toolConfig)
   * @param token Cancellation token
   * @returns The number of input tokens
   */
  private async countRequestTokens(
    modelId: string,
    input: {
      messages: Message[];
      system?: SystemContentBlock[];
      toolConfig?: ToolConfiguration;
    },
    token: CancellationToken,
  ): Promise<number> {
    // Fallback estimation function
    const estimateTokens = (): number => {
      let total = 0;

      // Estimate messages tokens
      for (const msg of input.messages) {
        for (const content of msg.content ?? []) {
          if ("text" in content && content.text) {
            total += Math.ceil(content.text.length / 4);
          }
        }
      }

      // Estimate system tokens
      if (input.system) {
        for (const sys of input.system) {
          if ("text" in sys && sys.text) {
            total += Math.ceil(sys.text.length / 4);
          }
        }
      }

      // Estimate tool tokens
      if ((input.toolConfig?.tools?.length ?? 0) > 0) {
        try {
          const json = JSON.stringify(input.toolConfig);
          total += Math.ceil(json.length / 4);
        } catch {
          // Ignore serialization errors
        }
      }

      return total;
    };

    try {
      // Create AbortController for cancellation support
      const abortController = new AbortController();
      const cancellationListener = token.onCancellationRequested(() => {
        abortController.abort();
      });

      try {
        // Deep copy messages and strip thinking content for CountTokens API
        // The CountTokens API doesn't support thinking blocks when thinking mode is not enabled,
        // but our messages may contain thinking blocks from previous responses (injected via lastThinkingBlock)
        const messagesForCounting = structuredClone(input.messages);
        stripThinkingContent(messagesForCounting);

        // Build the CountTokens API input
        const countInput: CountTokensCommandInput["input"] = {
          converse: {
            messages: messagesForCounting,
            ...(input.system && input.system.length > 0 ? { system: input.system } : {}),
            ...(input.toolConfig ? { toolConfig: input.toolConfig } : {}),
          },
        };

        // Use the CountTokens API
        const tokenCount = await this.client.countTokens(
          modelId,
          countInput,
          abortController.signal,
        );

        // If CountTokens API is available, use its result
        if (tokenCount !== undefined) {
          logger.debug(`[Bedrock Model Provider] Request token count from API: ${tokenCount}`);
          return tokenCount;
        }

        // Fall back to estimation if CountTokens is not available
        logger.debug(
          "[Bedrock Model Provider] CountTokens not available for request, using estimation",
        );
        return estimateTokens();
      } finally {
        cancellationListener.dispose();
      }
    } catch (error) {
      // If there's any error (including cancellation), fall back to estimation
      if (error instanceof Error && error.name === "AbortError") {
        logger.debug("[Bedrock Model Provider] Request token count cancelled, using estimation");
      } else {
        logger.warn("[Bedrock Model Provider] Request token count failed, using estimation", error);
      }
      return estimateTokens();
    }
  }

  private async evaluateCandidateAccessibility(
    candidate: {
      hasInferenceProfile: boolean;
      model: BedrockModelSummary;
      modelIdToUse: string;
    },
    regionPrefix: string,
    availableProfileIds: Set<string>,
    preferRegional: boolean,
    abortSignal: AbortSignal,
  ): Promise<{
    hasInferenceProfile: boolean;
    isAccessible: boolean;
    model: BedrockModelSummary;
    modelIdToUse: string;
  }> {
    if (candidate.hasInferenceProfile) {
      // If the profile was returned by ListInferenceProfiles, trust it
      // This avoids expensive Converse API validation calls
      if (availableProfileIds.has(candidate.modelIdToUse)) {
        logger.trace(
          `[Bedrock Model Provider] Trusting inference profile from ListInferenceProfiles: ${candidate.modelIdToUse}`,
        );
        return { ...candidate, isAccessible: true };
      }

      // Profile not in list, validate with Converse as last resort
      const profileAccessible = await this.client.testInferenceProfileAccess(
        candidate.modelIdToUse,
        abortSignal,
      );

      if (profileAccessible) {
        return { ...candidate, isAccessible: true };
      }

      // Profile is denied, try to find an alternative
      return this.findAlternativeProfile(
        candidate,
        regionPrefix,
        availableProfileIds,
        preferRegional,
        abortSignal,
      );
    }

    // No inference profile; check base model directly
    const baseModelAccessible = await this.client.isModelAccessible(
      candidate.model.modelId,
      abortSignal,
    );

    return { ...candidate, isAccessible: baseModelAccessible };
  }

  /**
   * Try to find an accessible alternative inference profile when the initially selected one is denied.
   * When preferRegional=false (default), attempts opposite profile type (regional when global denied, or vice versa).
   * When preferRegional=true, skips global fallback when regional profile is denied (honors regional-only preference).
   * Falls back to base model if no profiles are accessible.
   */
  private async findAlternativeProfile(
    candidate: {
      hasInferenceProfile: boolean;
      model: BedrockModelSummary;
      modelIdToUse: string;
    },
    regionPrefix: string,
    availableProfileIds: Set<string>,
    preferRegional: boolean,
    abortSignal: AbortSignal,
  ): Promise<{
    hasInferenceProfile: boolean;
    isAccessible: boolean;
    model: BedrockModelSummary;
    modelIdToUse: string;
  }> {
    logger.info(
      `[Bedrock Model Provider] Inference profile ${candidate.modelIdToUse} denied, trying alternatives for ${candidate.model.modelId}`,
    );

    // If this was a global profile, try regional
    if (candidate.modelIdToUse.startsWith("global.")) {
      const regionalProfileId = `${regionPrefix}.${candidate.model.modelId}`;
      if (availableProfileIds.has(regionalProfileId)) {
        // Profile is in ListInferenceProfiles, trust it
        logger.info(
          `[Bedrock Model Provider] Using regional profile ${regionalProfileId} instead of global profile`,
        );
        return {
          ...candidate,
          hasInferenceProfile: true,
          isAccessible: true,
          modelIdToUse: regionalProfileId,
        };
      }
    } else if (candidate.modelIdToUse.startsWith(`${regionPrefix}.`)) {
      // If this was a regional profile and preferRegional=true, skip global fallback
      // (honors user preference for regional-only in Control Tower/SCP environments)
      if (preferRegional) {
        logger.info(
          `[Bedrock Model Provider] Regional profile denied and preferRegional=true, skipping global fallback`,
        );
      } else {
        const globalProfileId = `global.${candidate.model.modelId}`;
        if (availableProfileIds.has(globalProfileId)) {
          // Profile is in ListInferenceProfiles, trust it
          logger.info(
            `[Bedrock Model Provider] Using global profile ${globalProfileId} instead of regional profile`,
          );
          return {
            ...candidate,
            hasInferenceProfile: true,
            isAccessible: true,
            modelIdToUse: globalProfileId,
          };
        }
      }
    }

    // No accessible profile found, fall back to base model
    const baseModelAccessible = await this.client.isModelAccessible(
      candidate.model.modelId,
      abortSignal,
    );
    if (baseModelAccessible) {
      logger.info(
        `[Bedrock Model Provider] No accessible inference profile found for ${candidate.model.modelId}, using base model`,
      );
      return {
        ...candidate,
        hasInferenceProfile: false,
        isAccessible: true,
        modelIdToUse: candidate.model.modelId,
      };
    }

    logger.info(
      `[Bedrock Model Provider] No accessible inference profile or base model for ${candidate.model.modelId}`,
    );
    return { ...candidate, isAccessible: false };
  }

  /**
   * Get authentication configuration based on the stored auth method.
   * Retrieves credentials from SecretStorage for sensitive data (API keys, access keys)
   * and from globalState for non-sensitive data (profile name, auth method).
   * @param silent If true, don't prompt for missing credentials
   * @returns AuthConfig or undefined if authentication is not configured
   */
  private async getAuthConfig(silent = false): Promise<AuthConfig | undefined> {
    const method = this.globalState.get<AuthMethod>("bedrock.authMethod") ?? "profile";

    if (method === "api-key") {
      let apiKey = await this.secrets.get("bedrock.apiKey");
      if (!apiKey && !silent) {
        const entered = await vscode.window.showInputBox({
          ignoreFocusOut: true,
          password: true,
          prompt: "Enter your AWS Bedrock API key",
          title: "AWS Bedrock API Key",
        });
        if (entered?.trim()) {
          apiKey = entered.trim();
          await this.secrets.store("bedrock.apiKey", apiKey);
        }
      }
      if (!apiKey) {
        return undefined;
      }
      return { apiKey, method: "api-key" };
    }

    if (method === "profile") {
      const settings = await getBedrockSettings(this.globalState);
      return { method: "profile", profile: settings.profile };
    }

    if (method === "access-keys") {
      const accessKeyId = await this.secrets.get("bedrock.accessKeyId");
      const secretAccessKey = await this.secrets.get("bedrock.secretAccessKey");
      const sessionToken = await this.secrets.get("bedrock.sessionToken");

      if (!accessKeyId || !secretAccessKey) {
        if (!silent) {
          vscode.window.showErrorMessage(
            "AWS access keys not configured. Please run 'Manage Amazon Bedrock Provider'.",
          );
        }
        return undefined;
      }

      const result: AuthConfig = {
        accessKeyId,
        method: "access-keys",
        secretAccessKey,
      };
      if (sessionToken) {
        result.sessionToken = sessionToken;
      }
      return result;
    }

    return undefined;
  }

  /**
   * Log converted Bedrock messages for debugging
   */
  private logConvertedMessages(messages: Message[]): void {
    logger.debug("[Bedrock Model Provider] Converted to Bedrock messages:", messages.length);
    for (const [idx, msg] of messages.entries()) {
      const contentTypes = msg.content?.map((c) => {
        if ("text" in c) return "text";
        if ("image" in c) return "image";
        if ("toolUse" in c) return "toolUse";
        if ("toolResult" in c) return "toolResult";
        if ("reasoningContent" in c) return "reasoningContent";
        if ("thinking" in c || "redacted_thinking" in c) return "thinking";
        if ("cachePoint" in c) return "cachePoint";
        return "unknown";
      });
      logger.debug(`[Bedrock Model Provider] Bedrock message ${idx} (${msg.role}):`, contentTypes);
    }
  }

  /**
   * Log incoming VSCode messages for debugging and reproduction
   */
  private logIncomingMessages(messages: readonly LanguageModelChatRequestMessage[]): void {
    logger.info("[Bedrock Model Provider] === NEW REQUEST ===");
    logger.info("[Bedrock Model Provider] Converting messages, count:", messages.length);

    // Log full incoming VSCode messages at trace level for reproduction
    logger.trace("[Bedrock Model Provider] Full VSCode messages for reproduction:", {
      messages: messages.map((msg) => ({
        content: msg.content.map((part) => {
          if (part instanceof vscode.LanguageModelTextPart) {
            return { type: "text", value: part.value };
          }
          if (part instanceof vscode.LanguageModelToolCallPart) {
            return { callId: part.callId, input: part.input, name: part.name, type: "toolCall" };
          }
          if (part instanceof vscode.LanguageModelToolResultPart) {
            return { callId: part.callId, content: part.content, type: "toolResult" };
          }
          if (typeof part === "object" && part != null && "mimeType" in part && "data" in part) {
            const dataPart = part as { data: Uint8Array; mimeType: string };
            return {
              dataLength: dataPart.data.length,
              mimeType: dataPart.mimeType,
              type: "data",
            };
          }
          return { type: "unknown" };
        }),
        role: msg.role,
      })),
    });

    for (const [idx, msg] of messages.entries()) {
      const partTypes = msg.content.map((p) => {
        if (p instanceof vscode.LanguageModelTextPart) return "text";
        if (p instanceof vscode.LanguageModelToolCallPart) {
          return `toolCall(${p.name})`;
        }
        if (p instanceof vscode.LanguageModelToolResultPart) {
          return `toolResult(${p.callId})`;
        }
        if (typeof p === "object" && p != null && "mimeType" in p) {
          try {
            const dataPart = p as { mimeType: string };
            const mime = new MIMEType(dataPart.mimeType);
            if (mime.type === "image") {
              return `image(${mime.essence})`;
            }
            return `data(${mime.essence})`;
          } catch {
            // Invalid MIME type, skip
          }
        }
        return "unknown";
      });
      logger.debug(`[Bedrock Model Provider] Message ${idx} (${msg.role}):`, partTypes);
      // Log tool result details
      for (const part of msg.content) {
        if (part instanceof vscode.LanguageModelToolResultPart) {
          let contentPreview = "[Unable to preview]";
          try {
            const contentStr =
              typeof part.content === "string" ? part.content : JSON.stringify(part.content);
            contentPreview = contentStr.slice(0, 100);
          } catch {
            // Keep default
          }
          logger.debug(`[Bedrock Model Provider]   Tool Result:`, {
            callId: part.callId,
            contentPreview,
            contentType: typeof part.content,
            isError: "isError" in part ? part.isError : false,
          });
        }
      }
    }
  }

  /**
   * Log request details for debugging
   */
  private logRequestDetails(requestInput: ConverseStreamCommandInput): void {
    logger.info("[Bedrock Model Provider] Starting streaming request", {
      hasTools: !!requestInput.toolConfig,
      messageCount: requestInput.messages?.length,
      modelId: requestInput.modelId,
      systemMessageCount: requestInput.system?.length,
      toolCount: requestInput.toolConfig?.tools?.length,
    });

    // Log the actual request for debugging
    logger.debug("[Bedrock Model Provider] Request details:", {
      messages: requestInput.messages?.map((m) => ({
        contentBlocks: Array.isArray(m.content)
          ? m.content.map((c) => {
              if (c.text) return "text";
              if (c.image) return `image(${c.image.format})`;
              if (c.toolResult) {
                const preview =
                  c.toolResult.content?.[0]?.text?.slice(0, 100) ??
                  (JSON.stringify(c.toolResult.content?.[0]?.json)?.slice(0, 100) || "[empty]");
                return `toolResult(${c.toolResult.toolUseId},preview:${preview})`;
              }
              if (c.toolUse) return `toolUse(${c.toolUse.name})`;
              if ("reasoningContent" in c) return "reasoningContent";
              if ("thinking" in c) return "thinking";
              if ("redacted_thinking" in c) return "redacted_thinking";
              if ("cachePoint" in c) return "cachePoint";
              return "unknown";
            })
          : undefined,
        role: m.role,
      })),
    });

    // Log full message structures at trace level for detailed debugging
    logger.trace("[Bedrock Model Provider] Full request structure for reproduction:", {
      messages: requestInput.messages,
      system: requestInput.system,
      toolConfig: requestInput.toolConfig
        ? {
            toolChoice: requestInput.toolConfig.toolChoice,
            toolCount: requestInput.toolConfig.tools?.length,
          }
        : undefined,
    });
  }

  /**
   * Process the response stream and handle thinking blocks
   */
  private async processResponseStream(
    requestInput: ConverseStreamCommandInput,
    trackingProgress: Progress<LanguageModelResponsePart>,
    extendedThinkingEnabled: boolean,
    token: CancellationToken,
  ): Promise<void> {
    const abortController = new AbortController();
    const cancellationListener = token.onCancellationRequested(() => {
      abortController.abort();
    });

    try {
      const stream = await this.client.startConversationStream(
        requestInput,
        abortController.signal,
      );

      logger.info("[Bedrock Model Provider] Processing stream events");
      const result = await this.streamProcessor.processStream(stream, trackingProgress, token);

      // Store thinking block for next request ONLY if it has a signature
      // API requires signatures for interleaved thinking, so we only store blocks we can inject
      if (extendedThinkingEnabled && result.thinkingBlock?.signature) {
        this.lastThinkingBlock = result.thinkingBlock;
        logger.info(
          "[Bedrock Model Provider] Stored thinking block with signature for next request:",
          {
            signatureLength: result.thinkingBlock.signature.length,
            textLength: result.thinkingBlock.text.length,
          },
        );
      } else if (extendedThinkingEnabled && result.thinkingBlock) {
        logger.info(
          "[Bedrock Model Provider] Discarding thinking block without signature (cannot be reused):",
          {
            textLength: result.thinkingBlock.text.length,
          },
        );
      }

      // Log actual token usage from the stream metadata for observability
      if (result.usage) {
        // Look up the request-time effective context window for this model
        // so the badge numerator (this turn's prompt_tokens) and denominator
        // (modelMaxPromptTokens = maxInputTokens + maxOutputTokens) can be
        // correlated 1:1 in the logs. If the badge "resets" to a smaller
        // value but the denominator here is unchanged, the numerator must
        // have legitimately shrunk (Copilot trimmed history). If the
        // denominator changed, the model list was rebuilt with different
        // limits between this turn and the previous.
        const cachedEndpoint = this.chatEndpoints.find((e) => e.model === requestInput.modelId);
        // Bedrock's `inputTokens` excludes cached prompt tokens (those are
        // surfaced separately under `cacheReadInputTokens` / `cacheWriteInputTokens`).
        // The model still "sees" the cached portion of the prompt, so include
        // it when reporting the badge numerator — otherwise the tracker shrinks
        // dramatically on cache-hit legs of an agent loop and inflates again on
        // legs that re-encode large new content. This mirrors OpenAI's
        // convention where `prompt_tokens` is the full prompt size and
        // `prompt_tokens_details.cached_tokens` is a sub-figure.
        const cachedInputTokens =
          (result.usage.cacheReadInputTokens ?? 0) + (result.usage.cacheWriteInputTokens ?? 0);
        const effectivePromptTokens = result.usage.inputTokens + cachedInputTokens;
        const usedTokens = effectivePromptTokens + result.usage.outputTokens;
        const denominator = cachedEndpoint?.modelMaxPromptTokens;
        const percentage =
          typeof denominator === "number" && denominator > 0
            ? Math.round((usedTokens / denominator) * 1000) / 10
            : undefined;
        logger.info("[Bedrock Model Provider] Actual token usage from stream:", {
          cacheReadInputTokens: result.usage.cacheReadInputTokens,
          cacheWriteInputTokens: result.usage.cacheWriteInputTokens,
          // Mirrors what VS Code's chatContextUsageWidget computes: the badge
          // shows `(prompt_tokens + completion_tokens) / (maxInputTokens + maxOutputTokens)`.
          // Track these here so log scrubs of the badge resetting can be
          // matched directly against provider-reported numbers.
          contextWindow_effectivePromptTokens: effectivePromptTokens,
          contextWindow_modelId: requestInput.modelId,
          contextWindow_modelMaxPromptTokens: denominator,
          contextWindow_percentage: percentage,
          contextWindow_usedTokens: usedTokens,
          inputTokens: result.usage.inputTokens,
          outputTokens: result.usage.outputTokens,
          totalTokens: result.usage.totalTokens,
        });

        // Report usage to Copilot Chat for the context-window tracker badge.
        //
        // Convention: emit a `LanguageModelDataPart` whose payload is a UTF-8
        // JSON-encoded `APIUsage`-shaped object (OpenAI naming) with MIME type
        // "usage". Copilot Chat ≥ 1.120.0 (`ExtensionContributedChatEndpoint`)
        // recognises this MIME type and feeds the numbers into the badge.
        // Older clients silently drop unknown MIME types, so this is safe.
        //
        // See: https://github.com/microsoft/vscode/pull/315394
        // See: extensions/copilot/src/platform/endpoint/common/endpointTypes.ts
        //      (`CustomDataPartMimeTypes.Usage = 'usage'`)
        try {
          // OpenAI convention (which Copilot Chat's badge follows):
          //   prompt_tokens                  = full prompt the model sees
          //   prompt_tokens_details.cached   = cached subset of prompt_tokens
          //   completion_tokens              = generated tokens
          //   total_tokens                   = prompt_tokens + completion_tokens
          // Bedrock instead reports `inputTokens` as the *non-cached* fresh
          // input only, with cache hits/writes broken out separately. Sum them
          // back together so the badge tracks total conversation size, not
          // per-leg deltas (otherwise an agent loop sees the badge swing
          // wildly between cache-write legs (~tiny) and cache-miss legs
          // (~large) even though the conversation is monotonically growing).
          const promptTokens = effectivePromptTokens;
          const apiUsage = {
            completion_tokens: result.usage.outputTokens,
            prompt_tokens: promptTokens,
            prompt_tokens_details: {
              cached_tokens: result.usage.cacheReadInputTokens ?? 0,
            },
            total_tokens: promptTokens + result.usage.outputTokens,
          };
          // `LanguageModelDataPart.json` UTF-8-encodes the value and creates a
          // data part with the given MIME. Equivalent to constructing a part
          // from `TextEncoder().encode(JSON.stringify(value))` but uses the
          // public static factory API.
          trackingProgress.report(vscode.LanguageModelDataPart.json(apiUsage, "usage"));
        } catch (error) {
          // Defensive: never let a usage-reporting issue break the response.
          // This can happen on older VS Code builds where LanguageModelDataPart
          // is unavailable, or if the progress channel has already been closed.
          logger.debug("[Bedrock Model Provider] Failed to report usage data part", {
            error: error instanceof Error ? error.message : String(error),
          });
        }
      }

      logger.info("[Bedrock Model Provider] Finished processing stream");
    } finally {
      cancellationListener.dispose();
    }
  }

  /**
   * Validate token count against model limits
   */
  private async validateTokenCount(
    model: LanguageModelChatInformation,
    requestInput: ConverseStreamCommandInput,
    token: CancellationToken,
  ): Promise<void> {
    const inputTokenCount = await this.countRequestTokens(
      model.id,
      {
        messages: requestInput.messages!,
        system: requestInput.system,
        toolConfig: requestInput.toolConfig,
      },
      token,
    );

    const tokenLimit = Math.max(1, model.maxInputTokens);
    if (inputTokenCount > tokenLimit) {
      logger.error("[Bedrock Model Provider] Message exceeds token limit", {
        inputTokenCount,
        tokenLimit,
      });
      throw new Error(
        `Message exceeds token limit. Input: ${inputTokenCount} tokens, Limit: ${tokenLimit} tokens.`,
      );
    }

    logger.debug("[Bedrock Model Provider] Token count validation passed", {
      inputTokenCount,
      tokenLimit,
    });
  }
}

/**
 * Known error messages that indicate context window overflow from Bedrock API
 * Reference: https://github.com/strands-agents/sdk-python/blob/dbf6200d104539217dddfc7bd729c53f46e2ec56/src/strands/models/bedrock.py#L28-L32
 */
const CONTEXT_WINDOW_OVERFLOW_MESSAGES = [
  "Input is too long for requested model",
  "input length and `max_tokens` exceed context limit",
  "too many total text bytes",
];

/**
 * Check if an error is due to context window overflow
 * @param error The error to check
 * @returns true if the error is due to context window overflow
 */
function isContextWindowOverflowError(error: unknown): boolean {
  if (!error) {
    return false;
  }

  const errorMessage = error instanceof Error ? error.message : inspect(error);
  return CONTEXT_WINDOW_OVERFLOW_MESSAGES.some((msg) => errorMessage.includes(msg));
}

/**
 * Build the synthetic error-sentinel model entry shown in the picker when the
 * real Bedrock model list cannot be fetched. The failure reason is surfaced via
 * `name` (visible) and `tooltip`/`detail` (hover) so the user can diagnose.
 */
function makeBedrockErrorSentinel(error: unknown): BedrockLanguageModelChatInformation {
  const reason = error instanceof Error ? error.message : String(error);
  return {
    capabilities: { imageInput: false, toolCalling: false },
    category: BEDROCK_MODEL_PICKER_CATEGORY,
    detail: reason,
    family: "bedrock-error",
    id: BEDROCK_ERROR_SENTINEL_ID,
    isUserSelectable: true,
    maxInputTokens: 1,
    maxOutputTokens: 1,
    name: "⚠ Bedrock unavailable",
    tooltip:
      `Failed to load Bedrock models: ${reason}\n\n` +
      `Sending a request will fail. Run 'Manage Amazon Bedrock Provider' to fix your AWS profile/region, ` +
      `then re-open the model picker to retry.`,
    version: "1.0.0",
  };
}
