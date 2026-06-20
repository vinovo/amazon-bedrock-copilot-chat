# Slowness Fix Summary

## Problem

The extension was experiencing slow response times, especially after the first few turns of conversation. The issue was caused by repeated `ListInferenceProfiles` API calls that were failing or being denied.

## Root Cause

1. Every time the model list was refreshed (which happens frequently), the extension called `fetchInferenceProfiles()`
2. If the API call failed due to permission issues or rate limiting, there was no mechanism to prevent retrying on the next refresh
3. This led to repeated failed API calls that slowed down the extension significantly

## Solution Implemented

Added a **circuit breaker pattern with retry logic** to the `ListInferenceProfiles` API, similar to the existing pattern for `CountTokens` API:

### Key Changes in `bedrock-client.ts`:

1. **Added circuit breaker state tracking:**
   - `listInferenceProfilesAvailable`: tracks API availability (undefined/true/false)
   - `listInferenceProfilesFailureCount`: counts consecutive failures
   - `listInferenceProfilesCooldownUntil`: timestamp for cooldown expiry
   - `LIST_INFERENCE_PROFILES_MAX_RETRIES`: 3 retry attempts before cooldown
   - `LIST_INFERENCE_PROFILES_COOLDOWN_MS`: 30-minute cooldown period

2. **Modified `fetchInferenceProfiles()` method:**
   - Checks circuit breaker state before attempting API call
   - If API is known to be unavailable and within cooldown period, skips the call immediately
   - On `AccessDeniedException`, increments failure count
   - After 3 consecutive failures, enters 30-minute cooldown
   - Logs informative messages about retry attempts remaining
   - Automatically retries after cooldown expires

3. **Reset circuit breaker on credential/region changes:**
   - Added reset logic in `recreateClients()` method
   - Resets failure count and cooldown state
   - Ensures new credentials/regions get fresh retry attempts

### Previous Fixes (already in place):

- Added echo suppression in `extension.ts` to prevent feedback loops
- Added debouncing for configuration changes
- Added flag to skip events during initial model fetch

## Behavior

- **Attempt 1-3:** Tries `ListInferenceProfiles` on each model refresh
- **On each failure:** Logs info message with attempt count (e.g., "attempt 1/3")
- **After 3 failures:** Enters 30-minute cooldown, logs warning
- **During cooldown:** Immediately returns empty set, no API call made (logs remaining time)
- **After cooldown:** Resets and allows 3 fresh retry attempts
- **On success:** Marks API as available, resets failure count, uses normally
- **On credential/region change:** Resets circuit breaker completely for fresh attempts

## Benefits

1. **Eliminates repeated failed API calls** that were causing slowdown
2. **Gives a few retry chances** for transient failures (rate limits, temporary issues)
3. **Long cooldown period** (30 minutes) prevents excessive API calls
4. **Preserves functionality** by using fallback Anthropic profiles
5. **Automatic recovery** after cooldown period
6. **Minimal impact** on users with proper permissions
7. **Clear logging** for debugging permission issues

## Testing

Test the extension by:

1. Starting VSCode with the extension
2. Opening Copilot Chat with a Bedrock model
3. Verifying first few turns are fast
4. Verifying subsequent turns remain fast (no repeated API calls)
5. Checking logs for "ListInferenceProfiles" messages to confirm circuit breaker is working
6. Logs should show "attempt 1/3", "attempt 2/3", "attempt 3/3" on consecutive failures
7. After 3rd failure, logs should show "entering 30min cooldown"
8. During cooldown, logs should show "X minutes cooldown remaining"

## Follow-up: ListFoundationModels circuit breaker + fetchModels cache

The repeated `No accessible inference profile or base model for openai.gpt-oss-*` log lines
revealed a second, larger slowness source. Those lines are emitted by
`detectAnthropicFallbackModels()`, which only runs when `ListFoundationModels` is denied.
Each model-list refresh re-ran the full fallback fan-out (9 candidates × 1-3 live remote
calls each: global profile probe → regional probe → `GetFoundationModelAvailability` →
`Converse` probe) with no caching and no circuit breaker.

### Changes in `bedrock-client.ts`:

1. **Circuit breaker on `ListFoundationModels` denial** (mirrors the `ListInferenceProfiles`
   pattern): 3 denial attempts, then a 30-minute cooldown during which the call is skipped
   and fallback detection is used directly. Counter resets on success.
2. **Short-lived `fetchModels` result cache (60s TTL)** covering BOTH successful results and
   fallback-detection results. This was the key requirement — even **successful** requests are
   served from cache within the TTL window, so frequent refreshes don't trigger repeated
   remote requests regardless of whether `ListFoundationModels` succeeds or falls back.
3. **Cache + circuit-breaker reset** on region/profile/credential change (in `recreateClients()`),
   alongside the existing resets.

### Setter guards (also addressed earlier)

`setRegion` / `setProfile` / `setAuthConfig` now no-op when the value is unchanged, so
`recreateClients()` (and all circuit-breaker resets) only fire on a genuine config change —
this was what kept the retry counters stuck at "1/3".

## Follow-up: shared breaker for fetchApplicationInferenceProfiles

`fetchApplicationInferenceProfiles()` also calls `bedrock:ListInferenceProfiles` (with
`typeEquals: "APPLICATION"`) but had no circuit breaker, so it logged
`Failed to fetch application inference profiles ... not authorized to perform:
bedrock:ListInferenceProfiles` as an **error** on every model-list refresh.

Since both `fetchInferenceProfiles()` and `fetchApplicationInferenceProfiles()` are gated by
the **same IAM action**, the breaker was unified:

1. Extracted the breaker into shared helpers — `shouldSkipListInferenceProfiles()`,
   `recordListInferenceProfilesSuccess()`, `recordListInferenceProfilesDenial()`.
2. Both methods now check `shouldSkipListInferenceProfiles()` up-front and record
   success/denial through the shared state. A denial on either trips the same breaker, so
   after the retry budget is exhausted **both** calls are skipped during the 30-minute
   cooldown.
3. The application-profile path now treats `AccessDeniedException` as a breaker denial
   (info/warn with attempt count) instead of an unconditional `logger.error` every refresh.
