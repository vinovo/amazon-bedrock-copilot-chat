import { normalizeId } from "./reasoning";

/**
 * Model families Copilot branches on (prompt selection, edit-tool choice,
 * schema normalization, Gemini/Anthropic message handling). Copilot matches the
 * reported `family` by prefix, so the family we report must *start with* one of
 * these for the matching behavior to kick in.
 */
const COPILOT_FAMILY_PATTERN = /^(claude|gemini|gpt|grok|o[1-9](-|$))/;

/** Whether a reported family is a Gemini model (what Copilot's `isGeminiFamily` checks). */
export function isGeminiModelFamily(family: string): boolean {
  return family.toLowerCase().startsWith("gemini");
}

/**
 * Resolve the family reported to VS Code for a custom-backend model.
 *
 * Copilot treats `family` as the model family and keys its model-specific
 * handling on the prefix, so reporting the real family lets it do the right
 * thing for us. The id is normalized (routing prefix, variant suffix, gateway
 * `provider/` path), and when the id itself isn't a recognizable family — a
 * gateway alias or codename — any alias the backend lists is tried before
 * falling back to the normalized id.
 */
export function resolveModelFamily(id: string, aliases: readonly string[] = []): string {
  const primary = stripGatewayPath(normalizeId(id));
  if (COPILOT_FAMILY_PATTERN.test(primary)) {
    return primary;
  }
  for (const alias of aliases) {
    const candidate = stripGatewayPath(normalizeId(alias));
    if (COPILOT_FAMILY_PATTERN.test(candidate)) {
      return candidate;
    }
  }
  return primary || id;
}

/** Drop a gateway routing path such as `vertex_ai/` or `openrouter/google/`. */
function stripGatewayPath(id: string): string {
  const slash = id.lastIndexOf("/");
  return slash === -1 ? id : id.slice(slash + 1);
}
