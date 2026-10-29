import * as assert from "node:assert";

import { sanitizeSchemaForGemini } from "../custom/schema";

suite("Custom Provider - Gemini schema sanitizer", () => {
  test("strips unsupported keywords such as $comment and $schema", () => {
    const out = sanitizeSchemaForGemini({
      $comment: "note",
      $schema: "http://json-schema.org/draft-07/schema#",
      additionalProperties: false,
      properties: {
        path: { $comment: "x", description: "file", type: "string" },
      },
      required: ["path"],
      type: "object",
    });
    assert.deepStrictEqual(out, {
      properties: { path: { description: "file", type: "string" } },
      required: ["path"],
      type: "object",
    });
  });

  test("converts nullable type arrays, const, and local $ref", () => {
    const out = sanitizeSchemaForGemini({
      $defs: { Name: { type: "string" } },
      properties: {
        kind: { const: "a" },
        maybe: { type: ["string", "null"] },
        name: { $ref: "#/$defs/Name" },
      },
      type: "object",
    });
    assert.deepStrictEqual(out?.properties, {
      kind: { enum: ["a"], type: "string" },
      maybe: { nullable: true, type: "string" },
      name: { type: "string" },
    });
  });

  test("omits parameters for missing or argument-less schemas", () => {
    assert.strictEqual(sanitizeSchemaForGemini(undefined), undefined);
    assert.strictEqual(sanitizeSchemaForGemini({ properties: {}, type: "object" }), undefined);
    assert.strictEqual(sanitizeSchemaForGemini({ $comment: "x", type: "object" }), undefined);
  });

  test("drops formats Vertex does not support", () => {
    const out = sanitizeSchemaForGemini({
      properties: {
        at: { format: "date-time", type: "string" },
        n: { format: "int64", type: "integer" },
        site: { format: "uri", type: "string" },
      },
      type: "object",
    });
    assert.deepStrictEqual(out?.properties, {
      at: { format: "date-time", type: "string" },
      n: { format: "int64", type: "integer" },
      site: { type: "string" },
    });
  });

  test("breaks recursive $ref cycles", () => {
    const out = sanitizeSchemaForGemini({
      $defs: { Node: { properties: { next: { $ref: "#/$defs/Node" } }, type: "object" } },
      properties: { root: { $ref: "#/$defs/Node" } },
      type: "object",
    });
    assert.strictEqual(out?.type, "object");
  });
});
