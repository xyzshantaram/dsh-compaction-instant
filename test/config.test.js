/**
 * Configuration resolution tests for the instant compaction engine.
 * @module dsh-compaction-instant/test/config
 */
import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync, rmSync } from "node:fs";
import { compileNoisePatterns, DEFAULT_ARG_TOOLS, DEFAULT_NOISE_PATTERNS } from "../src/compiler.js";
import { isWorthCompacting, resolveCompactSpec, resolveConfig, resolveTargetPolicy, TargetPressureConfigError } from "../src/index.js";

test("resolveConfig applies the documented defaults", () => {
  const config = resolveConfig({});
  assert.equal(config.thresholdRatio, 0.8);
  // Unconfigured: null propagates so the window tier chooses the absolute.
  assert.equal(config.compactAtTokens, null);
  assert.equal(config.retainTurns, 1);
  assert.equal(config.retainTokens, 5120);
  assert.equal(config.maxTokens, 8192);
  assert.equal(config.checkpointScale, 0.1);
  assert.equal(config.checkpointCap, 65536);
  assert.equal(config.textTokens, 512);
  assert.equal(config.userTextTokens, 1024);
  assert.equal(config.toolCallTokens, 128);
  assert.equal(config.toolResultExcerptTokens, 256);
  assert.equal(config.includeReasoning, false);
  assert.equal(config.stripNoiseXml, true);
  assert.equal(config.auto, true);
  assert.equal(config.noisePatterns.length, 5);
  assert.deepEqual(config.modelPolicies, []);
});

test("resolveConfig rejects unknown keys", () => {
  assert.throws(() => resolveConfig({ textToken: 1 }), /unknown key "textToken"/);
  assert.throws(() => resolveConfig({ retainRatio: 0.05 }), /unknown key "retainRatio"/);
  assert.throws(() => resolveConfig({ manualRetainRatio: 0.05 }), /unknown key "manualRetainRatio"/);
  assert.throws(() => resolveConfig({ manualRetainTokens: 100 }), /unknown key "manualRetainTokens"/);
});

test("resolveConfig validates ratios, term retention, and threshold", () => {
  assert.throws(() => resolveConfig({ thresholdRatio: 0 }), /thresholdRatio/);
  assert.throws(() => resolveConfig({ thresholdRatio: 1.5 }), /thresholdRatio/);
  assert.throws(() => resolveConfig({ retainTurns: 0 }), /retainTurns/);
  assert.throws(() => resolveConfig({ retainTurns: 1.5 }), /retainTurns/);
  assert.throws(() => resolveConfig({ retainTokens: -1 }), /retainTokens/);
  assert.throws(() => resolveConfig({ retainTokens: 1.5 }), /retainTokens/);
});

test("resolveConfig validates compiler budgets and flags", () => {
  assert.throws(() => resolveConfig({ textTokens: 0 }), /textTokens/);
  assert.throws(() => resolveConfig({ toolResultExcerptTokens: -1 }), /toolResultExcerptTokens/);
  assert.throws(() => resolveConfig({ includeReasoning: "yes" }), /includeReasoning/);
  assert.throws(() => resolveConfig({ stripNoiseXml: 1 }), /stripNoiseXml/);
  assert.throws(() => resolveConfig({ noisePatterns: ["("] }), /noisePatterns\[0\]/);
  assert.throws(() => resolveConfig({ noisePatterns: "x" }), /noisePatterns/);
  assert.throws(() => resolveConfig({ toolKeyFields: { read: 5 } }), /toolKeyFields/);
});

test("resolveConfig validates tool whitelists and hides, and deduplicates", () => {
  assert.ok(resolveConfig({}).toolArgTools.includes("read"));
  assert.ok(resolveConfig({}).toolArgTools.includes("bash"));
  // todo_write is bookkeeping: only the latest list is true, so earlier
  // writes are superseded noise the checkpoint never carries.
  assert.deepEqual(resolveConfig({}).hideTools, ["todo_write"]);
  assert.throws(() => resolveConfig({ toolArgTools: "read" }), /toolArgTools/);
  assert.throws(() => resolveConfig({ toolArgTools: [""] }), /toolArgTools/);
  assert.throws(() => resolveConfig({ hideTools: ["job_kill", 5] }), /hideTools/);
  const config = resolveConfig({ toolArgTools: ["read", "read", "bash"], hideTools: ["job_kill", "job_kill"] });
  assert.deepEqual(config.toolArgTools, ["read", "bash"]);
  assert.deepEqual(config.hideTools, ["job_kill"]);
});

test("resolveConfig treats schemastery-injected empty arrays as unset", () => {
  // The cordis config pipeline validates rows through the plugin's schemastery
  // schema, whose `~standard` adapter injects `[]` for every absent array key.
  // Regression: that shape must fall back to the defaults, not disable them.
  const injected = { toolArgTools: [], hideTools: [], noisePatterns: [], toolKeyFields: {}, debug: true };
  const config = resolveConfig(injected);
  assert.deepEqual(config.toolArgTools, [...DEFAULT_ARG_TOOLS]);
  assert.deepEqual(config.hideTools, ["todo_write"]);
  assert.deepEqual(config.noisePatterns, compileNoisePatterns(DEFAULT_NOISE_PATTERNS));
});

test("resolveConfig debug defaults off and installs a file sink when on", () => {
  assert.equal(resolveConfig({}).debug, false);
  assert.equal(resolveConfig({}).debugSink, undefined);
  assert.throws(() => resolveConfig({ debug: "yes" }), /debug/);
  assert.throws(() => resolveConfig({ debugLogPath: 5 }), /debugLogPath/);
  const debug = resolveConfig({ debug: true, debugLogPath: "/tmp/dsh-compaction-debug-test.log" });
  assert.equal(debug.debug, true);
  assert.equal(typeof debug.debugSink, "function");
  debug.debugSink(`test line ${Date.now()}`);
  const lines = readFileSync("/tmp/dsh-compaction-debug-test.log", "utf8").trim().split("\n");
  assert.match(lines[lines.length - 1], /test line/);
  rmSync("/tmp/dsh-compaction-debug-test.log", { force: true });
});

test("resolveConfig validates retained-term budgets", () => {
  assert.throws(() => resolveConfig({ retainTurns: 0 }), /retainTurns/);
  assert.equal(resolveConfig({ retainTurns: 3 }).retainTurns, 3);
  assert.equal(resolveConfig({ retainTokens: 50000 }).retainTokens, 50000);
  assert.throws(() => resolveConfig({ checkpointScale: 0 }), /checkpointScale/);
  assert.throws(() => resolveConfig({ checkpointCap: 0 }), /checkpointCap/);
  const config = resolveConfig({ retainTurns: 2, retainTokens: 3000, checkpointScale: 0.25, checkpointCap: 32768 });
  assert.equal(config.retainTurns, 2);
  assert.equal(config.retainTokens, 3000);
  assert.equal(config.checkpointScale, 0.25);
  assert.equal(config.checkpointCap, 32768);
});

test("resolveConfig validates the inert summarization pair for drop-in parity", () => {
  assert.throws(() => resolveConfig({ summarizationProvider: "p" }), /set together/);
  const config = resolveConfig({ summarizationProvider: "", summarizationModel: "" });
  assert.equal(config.maxTokens, 8192);
});

test("resolveConfig validates modelPolicies and rejects duplicates", () => {
  assert.throws(() => resolveConfig({ modelPolicies: [{ provider: "p" }] }), /modelPolicies\[0\]/);
  assert.throws(() => resolveConfig({ modelPolicies: [{ provider: "p", model: "m", retainRatio: 0.9 }] }), /modelPolicies\[0\].*unknown key "retainRatio"/);
  assert.throws(() => resolveConfig({
    modelPolicies: [
      { provider: "p", model: "m" },
      { provider: "p", model: "m" }
    ]
  }), /duplicate model policy/);
});

test("resolveTargetPolicy overlays exact-target fields over defaults", () => {
  const config = resolveConfig({ retainTurns: 2, retainTokens: 3000 });
  const policy = resolveTargetPolicy(config, { provider: "p", model: "m" });
  assert.equal(policy.thresholdRatio, 0.8);
  assert.equal(policy.retainTurns, 2);
  assert.equal(policy.retainTokens, 3000);
  const overridden = resolveConfig({
    modelPolicies: [{ provider: "p", model: "m", thresholdRatio: 0.9, retainTurns: 5, retainTokens: 9000 }]
  });
  const targeted = resolveTargetPolicy(overridden, { provider: "p", model: "m" });
  assert.equal(targeted.thresholdRatio, 0.9);
  assert.equal(targeted.retainTurns, 5);
  assert.equal(targeted.retainTokens, 9000);
});

test("resolveCompactSpec scales budgets and rejects invalid windows", () => {
  const policy = resolveTargetPolicy(resolveConfig({}), { provider: "p", model: "m" });
  const spec = resolveCompactSpec(policy, 1000);
  // A 1000-token window cannot hold the 200000 small-window tier, so the 0.8
  // ratio guard binds; the spec still carries the resolved tier absolute.
  assert.equal(spec.thresholdTokens, 800);
  assert.equal(spec.compactAtTokens, 200000);
  assert.equal(spec.retainTurns, 1);
  assert.equal(spec.retainTokens, 5120);
  assert.throws(() => resolveCompactSpec(policy, 0), TargetPressureConfigError);
  const configured = resolveTargetPolicy(resolveConfig({ retainTurns: 2, retainTokens: 4000 }), { provider: "p", model: "m" });
  const scaled = resolveCompactSpec(configured, 1000);
  assert.equal(scaled.retainTurns, 2);
  assert.equal(scaled.retainTokens, 4000);
});

test("resolveCompactSpec triggers on the smaller of the tier absolute and the ratio", () => {
  // Unconfigured, the window tier picks the absolute (null propagates from
  // resolveConfig); the ratio still guards a window too small to hold it.
  const config = resolveConfig({});
  assert.equal(config.compactAtTokens, null);
  assert.equal(config.compactToTokens, 15000);
  const policy = resolveTargetPolicy(config, { provider: "p", model: "m" });
  // A one-million-token window: the big-window tier binds.
  assert.equal(resolveCompactSpec(policy, 1_000_000).thresholdTokens, 250000);
  assert.equal(resolveCompactSpec(policy, 1_000_000).compactAtTokens, 250000);
  // A 128k window: the small-window tier loses to the ratio guard.
  assert.equal(resolveCompactSpec(policy, 128_000).thresholdTokens, 102400);
  assert.equal(resolveCompactSpec(policy, 128_000).compactAtTokens, 200000);
});

test("resolveCompactSpec picks the trigger tier from the context window", () => {
  const policy = resolveTargetPolicy(resolveConfig({}), { provider: "p", model: "m" });
  const threshold = (contextWindow) => resolveCompactSpec(policy, contextWindow).thresholdTokens;
  const absolute = (contextWindow) => resolveCompactSpec(policy, contextWindow).compactAtTokens;
  // A 250k window sits on the small tier (the ratio agrees exactly here).
  assert.equal(threshold(250_000), 200_000);
  assert.equal(absolute(250_000), 200_000);
  // The 262144 boundary itself stays on the small tier.
  assert.equal(threshold(262_144), 200_000);
  assert.equal(absolute(262_144), 200_000);
  // One token above the boundary flips the tier, but the 0.8 ratio guard
  // binds first: min(250000, 209716). Same deliberate class as 300k → 240k.
  assert.equal(threshold(262_145), 209_716);
  assert.equal(absolute(262_145), 250_000);
  // A 300k window is ratio-bounded by design, not tier-bounded.
  assert.equal(threshold(300_000), 240_000);
  // A one-million-token window holds the big tier outright.
  assert.equal(threshold(1_000_000), 250_000);
  // An explicitly configured absolute always wins over the tier.
  const explicit = resolveTargetPolicy(resolveConfig({ compactAtTokens: 100_000 }), { provider: "p", model: "m" });
  assert.equal(resolveCompactSpec(explicit, 262_144).thresholdTokens, 100_000);
  assert.equal(resolveCompactSpec(explicit, 262_144).compactAtTokens, 100_000);
  // An explicit ratio is still honored: min(200000, 125000).
  const guarded = resolveTargetPolicy(resolveConfig({ thresholdRatio: 0.5 }), { provider: "p", model: "m" });
  assert.equal(resolveCompactSpec(guarded, 250_000).thresholdTokens, 125_000);
});

test("isWorthCompacting rejects a span that cannot pay for the checkpoint framing", () => {
  // Every checkpoint carries a fixed framing cost, so a short span can only
  // grow the surface. The engine declines it instead of compiling, failing the
  // shrink gate, and repeating that on every step. Overflow recovery and an
  // explicit manual compaction must still be able to force one reduction, so
  // the framing floor is the only bar they have to clear.
  for (const trigger of ["context-overflow", "manual"]) {
    assert.equal(isWorthCompacting(null, trigger), false);
    assert.equal(isWorthCompacting({ start: 1, end: 2, spanTokens: 0 }, trigger), false);
    assert.equal(isWorthCompacting({ start: 1, end: 2, spanTokens: 287 }, trigger), false);
    assert.equal(isWorthCompacting({ start: 1, end: 2, spanTokens: 1023 }, trigger), false);
    assert.equal(isWorthCompacting({ start: 1, end: 2, spanTokens: 1024 }, trigger), true);
    assert.equal(isWorthCompacting({ start: 1, end: 2, spanTokens: 6336 }, trigger), true);
  }
});

test("automatic pressure compaction declines a span holding too little new material", () => {
  // Regression: selection starts at the surface head, which is the previous
  // checkpoint once one has landed. A span that is almost all checkpoint frees
  // close to nothing, so pressure stays above the threshold and compaction runs
  // again at once. One live session compacted its own previous checkpoint and
  // nothing else, freeing about 535 tokens on a 1924-token span.
  assert.equal(isWorthCompacting(null), false);
  // Large span, but nearly all of it is the checkpoint being rewritten.
  assert.equal(isWorthCompacting({ start: 1, end: 2, spanTokens: 60000, compilableTokens: 1924 }), false);
  assert.equal(isWorthCompacting({ start: 1, end: 2, spanTokens: 8000, compilableTokens: 4095 }), false);
  assert.equal(isWorthCompacting({ start: 1, end: 2, spanTokens: 8000, compilableTokens: 4096 }), true);
  // A span that is all tool results compiles to nothing, whatever its size.
  assert.equal(isWorthCompacting({ start: 1, end: 2, spanTokens: 300000, compilableTokens: 0 }), false);
  // A range from a custom selector that omits the field falls back to the total.
  assert.equal(isWorthCompacting({ start: 1, end: 2, spanTokens: 4095 }), false);
  assert.equal(isWorthCompacting({ start: 1, end: 2, spanTokens: 4096 }), true);
});
