/**
 * JSON Schema sanitizer for Gemini / Vertex AI function declarations.
 *
 * Vertex validates `function_declarations[].parameters` against a restricted
 * OpenAPI-style `Schema` message and rejects any unknown field (for example
 * `$comment`, `$schema`, `additionalProperties`, `const`, `$ref`) with a 400
 * "Unknown name ... Cannot find field". Copilot's own tool-schema normalizer
 * handles several Gemini quirks (nullable unions, dangling `required`, top-level
 * combinators) but leaves these keywords alone, so they are removed here.
 */

type JsonObject = Record<string, unknown>;

/** Keywords accepted by the Vertex `Schema` message. */
const ALLOWED_KEYWORDS: ReadonlySet<string> = new Set([
  "anyOf",
  "default",
  "description",
  "enum",
  "example",
  "format",
  "items",
  "maximum",
  "maxItems",
  "maxLength",
  "maxProperties",
  "minimum",
  "minItems",
  "minLength",
  "minProperties",
  "nullable",
  "pattern",
  "properties",
  "propertyOrdering",
  "required",
  "title",
  "type",
]);

/** Keywords that need recursive or custom handling rather than a plain copy. */
const STRUCTURAL_KEYWORDS: ReadonlySet<string> = new Set([
  "anyOf",
  "enum",
  "items",
  "properties",
  "required",
  "type",
]);

const MAX_DEPTH = 32;

/**
 * Reduce an arbitrary JSON Schema to the subset Vertex AI / Gemini accepts for
 * function declaration parameters. Returns `undefined` for a tool that takes no
 * arguments so the caller can omit `parameters` entirely (Vertex rejects an
 * OBJECT schema with empty `properties`).
 */
export function sanitizeSchemaForGemini(schema: unknown): JsonObject | undefined {
  if (!isObject(schema)) {
    return undefined;
  }
  const result = sanitizeNode(schema, schema, [], 0);
  if (!isObject(result.properties)) {
    return undefined;
  }
  result.type = "object";
  return result;
}

function copyPlainKeywords(out: JsonObject, node: JsonObject): void {
  for (const [key, value] of Object.entries(node)) {
    if (ALLOWED_KEYWORDS.has(key) && !STRUCTURAL_KEYWORDS.has(key)) {
      out[key] = value;
    }
  }
}

function inferType(out: JsonObject): void {
  if (out.type !== undefined || out.anyOf !== undefined) return;
  if (out.properties !== undefined) out.type = "object";
  else if (out.items !== undefined) out.type = "array";
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Fold `allOf` members into the parent (properties and required are unioned). */
function mergeAllOf(
  out: JsonObject,
  node: JsonObject,
  root: JsonObject,
  refs: readonly string[],
  depth: number,
): void {
  if (!Array.isArray(node.allOf)) return;
  for (const member of node.allOf) {
    const { properties, required, ...rest } = sanitizeNode(member, root, refs, depth + 1);
    Object.assign(out, rest);
    mergeProperties(out, properties);
    mergeRequired(out, required);
  }
}

function mergeProperties(out: JsonObject, properties: unknown): void {
  if (isObject(properties)) {
    out.properties = { ...(out.properties as JsonObject | undefined), ...properties };
  }
}

function mergeRequired(out: JsonObject, required: unknown): void {
  if (Array.isArray(required)) {
    const existing = Array.isArray(out.required) ? (out.required as unknown[]) : [];
    out.required = [...new Set([...existing, ...(required as unknown[])])];
  }
}

/** Vertex only supports string enums; `const` becomes a one-value enum. */
function normalizeEnum(out: JsonObject, node: JsonObject): void {
  let values: undefined | unknown[];
  if (Array.isArray(node.enum)) values = node.enum;
  else if ("const" in node) values = [node.const];
  if (!values) return;

  out.enum = values.map((v) => (typeof v === "string" ? v : JSON.stringify(v)));
  out.type = "string";
  delete out.minimum;
  delete out.maximum;
}

/**
 * Vertex only accepts a fixed set of `format` values per type (e.g. `date-time`
 * for strings, `int32`/`int64` for integers); anything else, like `uri` or
 * `email`, is a 400.
 */
function normalizeFormat(out: JsonObject): void {
  if (typeof out.format !== "string") return;
  let allowed: string[] = [];
  if (out.type === "string" && out.enum === undefined) allowed = ["date-time"];
  else if (out.type === "integer") allowed = ["int32", "int64"];
  else if (out.type === "number") allowed = ["float", "double"];
  if (!allowed.includes(out.format)) delete out.format;
}

/** `type: ["string", "null"]` becomes `type: "string", nullable: true`. */
function normalizeType(out: JsonObject, node: JsonObject): void {
  const { type } = node;
  if (Array.isArray(type)) {
    const names = type.filter((t): t is string => typeof t === "string");
    if (names.includes("null")) out.nullable = true;
    const first = names.find((t) => t !== "null");
    if (first !== undefined) out.type = first;
  } else if (typeof type === "string" && type !== "null") {
    out.type = type;
  }
}

/** Vertex rejects an OBJECT schema whose `properties` is present but empty. */
function pruneEmptyProperties(out: JsonObject): void {
  if (isObject(out.properties) && Object.keys(out.properties).length === 0) {
    delete out.properties;
  }
}

/** `required` may only reference declared properties. */
function pruneRequired(out: JsonObject): void {
  if (!Array.isArray(out.required)) return;
  const declared = isObject(out.properties) ? out.properties : {};
  const required = (out.required as unknown[]).filter(
    (name): name is string => typeof name === "string" && name in declared,
  );
  if (required.length > 0) out.required = required;
  else delete out.required;
}

/** Resolve a local `#/...` JSON pointer against the root schema. */
function resolveRef(root: JsonObject, ref: string): JsonObject | undefined {
  if (!ref.startsWith("#")) return undefined;
  let current: unknown = root;
  for (const raw of ref.slice(1).split("/").filter(Boolean)) {
    const key = decodeURIComponent(raw).replaceAll("~1", "/").replaceAll("~0", "~");
    if (!isObject(current) || !(key in current)) return undefined;
    current = current[key];
  }
  return isObject(current) ? current : undefined;
}

function sanitizeChildren(
  out: JsonObject,
  node: JsonObject,
  root: JsonObject,
  refs: readonly string[],
  depth: number,
): void {
  const child = (value: unknown): JsonObject => sanitizeNode(value, root, refs, depth + 1);

  if (isObject(node.properties)) {
    const properties: JsonObject = {};
    for (const [name, value] of Object.entries(node.properties)) {
      properties[name] = child(value);
    }
    mergeProperties(out, properties);
  }
  if (node.items !== undefined) {
    // Tuple-style `items: [...]` is unsupported; use the first entry.
    out.items = child(Array.isArray(node.items) ? node.items[0] : node.items);
  }
  // `oneOf` is treated as `anyOf`.
  const alternatives = Array.isArray(node.anyOf) ? node.anyOf : node.oneOf;
  if (Array.isArray(alternatives)) {
    out.anyOf = alternatives.map((value) => child(value));
  }
  mergeRequired(out, node.required);
}

function sanitizeNode(
  node: unknown,
  root: JsonObject,
  refs: readonly string[],
  depth: number,
): JsonObject {
  if (!isObject(node) || depth > MAX_DEPTH) {
    return {};
  }
  if (typeof node.$ref === "string") {
    return sanitizeRef(node, node.$ref, root, refs, depth);
  }

  const out: JsonObject = {};
  mergeAllOf(out, node, root, refs, depth);
  copyPlainKeywords(out, node);
  sanitizeChildren(out, node, root, refs, depth);
  normalizeType(out, node);
  normalizeEnum(out, node);
  normalizeFormat(out);
  pruneRequired(out);
  pruneEmptyProperties(out);
  inferType(out);
  return out;
}

/** Inline a local `$ref` (Vertex has no reference support), breaking cycles. */
function sanitizeRef(
  node: JsonObject,
  ref: string,
  root: JsonObject,
  refs: readonly string[],
  depth: number,
): JsonObject {
  const target = refs.includes(ref) ? undefined : resolveRef(root, ref);
  if (!target) {
    return typeof node.description === "string" ? { description: node.description } : {};
  }
  const siblings = Object.fromEntries(Object.entries(node).filter(([key]) => key !== "$ref"));
  return sanitizeNode({ ...target, ...siblings }, root, [...refs, ref], depth + 1);
}
