import * as assert from "node:assert";

import {
  DEFAULT_MAX_INPUT_TOKENS,
  parseBackendSettings,
  toBackendConfig,
} from "../custom/settings";

suite("Custom backend per-group settings", () => {
  test("parses a fully-specified group configuration", () => {
    const settings = parseBackendSettings({
      allowInsecureTls: true,
      apiKey: " sk-secret ",
      baseUrl: " https://gw.example.test ",
      caBundlePath: " /etc/ssl/corp.pem ",
      maxInputTokens: 200_000,
      models: [{ id: "coder-30b" }, { id: "big-235b" }],
      name: " Breeze Gateway ",
    });

    assert.strictEqual(settings.allowInsecureTls, true);
    assert.strictEqual(settings.apiKey, "sk-secret", "strings are trimmed");
    assert.strictEqual(settings.baseUrl, "https://gw.example.test");
    assert.strictEqual(settings.caBundlePath, "/etc/ssl/corp.pem");
    assert.strictEqual(settings.maxInputTokens, 200_000);
    assert.deepStrictEqual(settings.models, ["coder-30b", "big-235b"]);
    assert.strictEqual(settings.name, "Breeze Gateway");
  });

  test("applies defaults for a minimal / empty configuration", () => {
    const settings = parseBackendSettings(undefined);
    assert.strictEqual(settings.allowInsecureTls, false);
    assert.strictEqual(settings.apiKey, undefined);
    assert.strictEqual(settings.baseUrl, undefined);
    assert.strictEqual(settings.caBundlePath, undefined);
    assert.strictEqual(settings.maxInputTokens, DEFAULT_MAX_INPUT_TOKENS);
    assert.deepStrictEqual(settings.models, []);
    assert.strictEqual(settings.name, undefined);
  });

  test("treats blank strings as unset", () => {
    const settings = parseBackendSettings({ apiKey: "   ", baseUrl: "", name: "\t" });
    assert.strictEqual(settings.apiKey, undefined);
    assert.strictEqual(settings.baseUrl, undefined);
    assert.strictEqual(settings.name, undefined);
  });

  test("rejects non-positive or non-numeric maxInputTokens", () => {
    assert.strictEqual(
      parseBackendSettings({ maxInputTokens: 0 }).maxInputTokens,
      DEFAULT_MAX_INPUT_TOKENS,
    );
    assert.strictEqual(
      parseBackendSettings({ maxInputTokens: -5 }).maxInputTokens,
      DEFAULT_MAX_INPUT_TOKENS,
    );
    assert.strictEqual(
      parseBackendSettings({ maxInputTokens: "128000" }).maxInputTokens,
      DEFAULT_MAX_INPUT_TOKENS,
    );
  });

  suite("model list parsing", () => {
    test("accepts a bare string array and de-duplicates", () => {
      const { models } = parseBackendSettings({ models: ["a", "b", "a", " c "] });
      assert.deepStrictEqual(models, ["a", "b", "c"]);
    });

    test("accepts a legacy comma-separated string", () => {
      const { models } = parseBackendSettings({ models: "alpha, beta ,gamma" });
      assert.deepStrictEqual(models, ["alpha", "beta", "gamma"]);
    });

    test("ignores malformed entries", () => {
      const { models } = parseBackendSettings({ models: [{ id: "keep" }, {}, 42, null, ""] });
      assert.deepStrictEqual(models, ["keep"]);
    });
  });

  suite("toBackendConfig", () => {
    test("returns undefined until both base URL and key are present", () => {
      assert.strictEqual(toBackendConfig(parseBackendSettings({})), undefined);
      assert.strictEqual(
        toBackendConfig(parseBackendSettings({ baseUrl: "https://x.test" })),
        undefined,
      );
      assert.strictEqual(toBackendConfig(parseBackendSettings({ apiKey: "k" })), undefined);
    });

    test("passes TLS options through when fully configured", () => {
      const config = toBackendConfig(
        parseBackendSettings({
          allowInsecureTls: true,
          apiKey: "k",
          baseUrl: "https://x.test",
          caBundlePath: "/ca.pem",
        }),
      );
      assert.deepStrictEqual(config, {
        allowInsecureTls: true,
        apiKey: "k",
        baseUrl: "https://x.test",
        caBundlePath: "/ca.pem",
      });
    });
  });
});
