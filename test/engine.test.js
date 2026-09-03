/**
 * Engine-level test: the real `InstantCompactionEngine.compile` hook runs the
 * deterministic compiler over a real detached session without any model call.
 * @module dsh-compaction-instant/test/engine
 */
import assert from "node:assert/strict";
import test from "node:test";
import { Context } from "@deepseek-ai/cordis";
import { createAssistantMessage, createToolResultMessage, createUserMessage } from "@deepseek-ai/dsh-llm";
import { Session } from "@deepseek-ai/dsh-session";
import { InstantCompactionEngine } from "../src/index.js";
import { compactSurfaceRegion } from "../src/region.js";

/** Build one detached session with a complete (idle) turn bracket. */
function makeIdleSession() {
  const user = createUserMessage({
    content: [{ type: "text", text: "please fix the bug" }],
    source: { kind: "user" }
  });
  const assistant = createAssistantMessage({
    content: [
      { type: "text", text: "on it" },
      { type: "tool-call", id: "call-1", name: "read", arguments: '{"file_path":"a.js"}' }
    ],
    source: { provider: "p", model: "m" }
  });
  const result = createToolResultMessage({
    callId: "call-1",
    content: [{ type: "text", text: "file content" }],
    isError: false
  });
  const assistant2 = createAssistantMessage({
    content: [{ type: "text", text: "done" }],
    source: { provider: "p", model: "m" }
  });
  const user2 = createUserMessage({
    content: [{ type: "text", text: "thank you" }],
    source: { kind: "user" }
  });
  const seed = [
    { type: "turn/start", seq: 0, time: 1, data: { turn: 1 } },
    { type: "user/message", seq: 1, time: 2, data: user, surfaceOp: "append" },
    { type: "assistant/message", seq: 2, time: 3, data: { message: assistant }, surfaceOp: "append" },
    { type: "tool/result", seq: 3, time: 4, data: { message: result }, surfaceOp: "append" },
    { type: "assistant/message", seq: 4, time: 5, data: { message: assistant2 }, surfaceOp: "append" },
    { type: "user/message", seq: 5, time: 6, data: user2, surfaceOp: "append" },
    { type: "turn/end", seq: 6, time: 7, data: { turn: 1 } }
  ];
  return Session.create("session-1", seed);
}

test("compile() compiles a real session deterministically with seq provenance", async () => {
  const session = makeIdleSession();
  const ctx = new Context();
  ctx.provide("tokenMeter", {
    measure: () => ({ nodes: session.surface.nodes.map((seq) => ({ seq, tokens: 100 })) }),
    estimateMessage: () => 10
  });
  const engine = new InstantCompactionEngine(ctx, {
    auto: false,
    textTokens: 64,
    userTextTokens: 64,
    toolCallTokens: 32,
    toolResultExcerptTokens: 32,
    maxTokens: 512
  });
  const prepared = {
    shadowedSeqs: [1, 2, 3, 4],
    session,
    start: 1,
    end: 4,
    startIdx: 0,
    endIdx: 3,
    measurement: { nodes: [] },
    selectedNodes: [],
    shadowedTokenCount: 400
  };
  const result = await engine.compile(prepared, undefined, undefined);
  assert.equal(result.provider, "dsh-compaction-instant");
  assert.equal(result.model, "vcc-compiler");
  const text = result.entries.map((entry) => entry.text).join("\n");
  assert.match(text, /please fix the bug/);
  assert.match(text, /on it/);
  assert.match(text, /\* read "a\.js" \(seq 2 -> result 3\)/);
  assert.doesNotMatch(text, /-> read: ok/);
  assert.doesNotMatch(text, /file content/);
  assert.match(text, /done/);
  // Deterministic: compiling twice yields identical output.
  const again = await engine.compile(prepared, undefined, undefined);
  assert.deepEqual(result.entries, again.entries);
  assert.deepEqual(result.stats, again.stats);
});

test("compile() aborts on a cancelled signal", async () => {
  const session = makeIdleSession();
  const ctx = new Context();
  ctx.provide("tokenMeter", {
    measure: () => ({ nodes: [] }),
    estimateMessage: () => 10
  });
  const engine = new InstantCompactionEngine(ctx, { auto: false });
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    () => engine.compile({ shadowedSeqs: [], session }, undefined, controller.signal),
    (error) => error.name === "AbortError"
  );
});

/**
 * Price every surface node identically and report a request total that the
 * caller controls, so a test can separate surface size from request pressure.
 */
function makeControlledMeter(perNode, totalTokens) {
  return {
    measure(session) {
      const nodes = session.surface.nodes.map((seq) => ({ seq, tokens: perNode }));
      const surfaceTokens = nodes.length * perNode;
      return {
        logRevision: session.events.length,
        baseline: { kind: "usage", tokens: totalTokens ?? surfaceTokens },
        surfaceDeltaTokens: 0,
        totalTokens: totalTokens ?? surfaceTokens,
        surfaceTokens,
        nodes
      };
    },
    estimateMessage: () => perNode
  };
}

/** Build a routed session of `turns` complete turns plus one open turn. */
function makeRoutedSession(id, turns) {
  const seed = [{ type: "request/header", seq: 0, time: 1, data: { header: { config: { provider: "p", model: "m" } } } }];
  let seq = 1;
  const push = (type, data, surfaceOp) => {
    seed.push({ type, seq, time: seq + 1, data, ...surfaceOp === undefined ? {} : { surfaceOp } });
    seq += 1;
  };
  for (let turn = 1; turn <= turns; turn += 1) {
    push("turn/start", { turn });
    push("user/message", createUserMessage({ content: [{ type: "text", text: `ask ${turn} ${"word ".repeat(40)}` }], source: { kind: "user" } }), "append");
    push("assistant/message", { message: createAssistantMessage({ content: [{ type: "text", text: `answer ${turn} ${"word ".repeat(40)}` }], source: { provider: "p", model: "m" } }) }, "append");
    push("turn/end", { turn });
  }
  const session = Session.create(id, seed);
  session.append("turn/start", { turn: turns + 1 });
  return session;
}

/** Wire an engine over a controlled meter and a one-million-token window. */
function makeEngine(perNode, totalTokens, config = {}) {
  const ctx = new Context();
  ctx.provide("tokenMeter", makeControlledMeter(perNode, totalTokens));
  ctx.provide("llm", { resolveModelInfo: async () => ({ context: { contextWindow: 1_000_000 } }) });
  return { ctx, engine: new InstantCompactionEngine(ctx, { auto: false, retainTurns: 1, retainTokens: 0, ...config }) };
}

test("automatic pressure reads the surface, not the cache-inflated request total", async () => {
  // Regression from live session a0872a7d. The token meter reports
  // `totalTokens` as the provider's usage for the last request plus the signed
  // surface delta since then, and that usage sums inputTokens, outputTokens,
  // cacheReadTokens, and cacheWriteTokens. One real request reported
  // inputTokens 2 with cacheReadTokens 493321, so the anchor sat near 495000
  // while the surface it described was about 350000. The difference became a
  // fixed floor of roughly 145000 tokens that no compaction could remove, so
  // pressure stayed above the threshold and compaction fired again at the very
  // next step boundary, compiling away the checkpoint it had just written.
  const { ctx, engine } = makeEngine(40000, 900000);
  const session = makeRoutedSession("session-inflated", 2);
  // Four surface nodes at 40000 is 160000 tokens, well under the trigger,
  // while the reported request total is 900000.
  const measured = ctx.tokenMeter.measure(session);
  assert.equal(measured.surfaceTokens, 160000);
  assert.equal(measured.totalTokens, 900000);
  assert.equal(await engine.compactIfNeeded({ session }, "pressure", undefined), null);
});

test("automatic pressure fires once at the trigger and then stays quiet", async () => {
  // The contract: cross 250000 surface tokens, compact exactly once, land well
  // under the post-compaction budget, and do not run again until the surface
  // climbs back to the trigger.
  const { ctx, engine } = makeEngine(40000, undefined);
  const session = makeRoutedSession("session-trigger", 5);
  const agent = { session };
  assert.equal(ctx.tokenMeter.measure(session).surfaceTokens, 400000);
  const first = await engine.compactIfNeeded(agent, "pressure", undefined);
  assert.notEqual(first, null, "crossing the trigger runs one compaction");
  const after = ctx.tokenMeter.measure(session).surfaceTokens;
  assert.ok(after < 250000, `the surface fell below the trigger (${after})`);
  // Every later step boundary must decline while the surface stays small.
  assert.equal(await engine.compactIfNeeded(agent, "pressure", undefined), null);
  assert.equal(await engine.compactIfNeeded(agent, "pressure", undefined), null);
});

test("a landed compaction is never reported as a failure", async () => {
  // Regression: the retry loop broke out and fell into a throw whenever the
  // surface stayed above the threshold, and the pre-step listener logged that
  // throw as "step compaction failed" even though the checkpoint was durably
  // written. A compaction that landed is a success.
  const { engine } = makeEngine(40000, undefined, { compactAtTokens: 1000, compactionRetries: 0 });
  const session = makeRoutedSession("session-stubborn", 5);
  const result = await engine.compactIfNeeded({ session }, "pressure", undefined);
  assert.notEqual(result, null, "the compaction landed and is returned, not thrown");
});

test("the checkpoint budget holds the trigger's compression ratio", () => {
  const ctx = new Context();
  ctx.provide("tokenMeter", { measure: () => ({ nodes: [] }), estimateMessage: () => 10 });
  // Compaction only runs at a step boundary, so the surface can overshoot the
  // trigger before it fires. The budget scales with the surface actually
  // found, which keeps the promised compression constant.
  const engine = new InstantCompactionEngine(ctx, {
    auto: false,
    compactAtTokens: 250000,
    compactToTokens: 15000,
    retainTokens: 5120,
    checkpointCap: 65536
  });
  // Exactly at the trigger: the whole surface must fit in 15000 tokens, and
  // the retained tail is verbatim, so the checkpoint gets what is left.
  assert.equal(engine.effectiveMaxTokens(244880), 9880);
  // Overshot to 340000: 340000 / 250000 * 15000 = 20400, less the tail.
  assert.equal(engine.effectiveMaxTokens(334880), 15280);
  // Below the trigger the budget stays flat rather than shrinking away.
  assert.equal(engine.effectiveMaxTokens(50000), 9880);
});

test("no compile attempt caps below the incoming checkpoint's own size", () => {
  const ctx = new Context();
  ctx.provide("tokenMeter", { measure: () => ({ nodes: [] }), estimateMessage: () => 10 });
  // Regression: the cap was a fraction of the span, and a fat checkpoint at
  // the head of a small span could not fit inside it. That is what forced the
  // elision which then destroyed the checkpoint.
  const engine = new InstantCompactionEngine(ctx, { auto: false, retainTokens: 5120, checkpointCap: 65536 });
  assert.equal(engine.effectiveMaxTokens(20000, 0, 0), 9880);
  // The retry ladder tightens the target on each attempt.
  assert.equal(engine.effectiveMaxTokens(20000, 3, 0), 1235);
  // No attempt caps below what the incoming checkpoint needs to survive.
  assert.equal(engine.effectiveMaxTokens(20000, 3, 8000), 8000);
  // The floor never breaks the post-compaction budget. A 61626-token
  // checkpoint written under the old 65536 cap is truncated by the compiler
  // instead of ratcheting the cap up for every later compaction.
  assert.equal(engine.effectiveMaxTokens(20000, 0, 61626), 9880);
  // A configured cap smaller than the checkpoint still wins.
  const smallCtx = new Context();
  smallCtx.provide("tokenMeter", { measure: () => ({ nodes: [] }), estimateMessage: () => 10 });
  const capped = new InstantCompactionEngine(smallCtx, { auto: false, retainTokens: 5120, checkpointCap: 4096 });
  assert.equal(capped.effectiveMaxTokens(20000, 0, 8000), 4096);
});

test("compile() caps the checkpoint at the smaller of checkpointCap and a span fraction", async () => {
  const session = makeIdleSession();
  const ctx = new Context();
  ctx.provide("tokenMeter", {
    measure: () => ({ nodes: [] }),
    estimateMessage: () => 10
  });
  // The total budget for one checkpoint is `checkpointCap`, bounded by a
  // fraction of the span being replaced. A cap above the span would let the
  // compiler elide nothing, and the fixed framing cost would then make the
  // replacement larger than its source.
  const engine = new InstantCompactionEngine(ctx, {
    auto: false,
    checkpointCap: 4096,
    textTokens: 512,
    userTextTokens: 1024,
    toolCallTokens: 128,
    toolResultExcerptTokens: 256
  });
  const long = `long text ${"word ".repeat(300)}`;
  const seeded = Session.create("session-2", [
    { type: "turn/start", seq: 0, time: 1, data: { turn: 1 } },
    { type: "user/message", seq: 1, time: 2, data: createUserMessage({ content: [{ type: "text", text: long }], source: { kind: "user" } }), surfaceOp: "append" },
    { type: "assistant/message", seq: 2, time: 3, data: { message: createAssistantMessage({ content: [{ type: "text", text: "ok" }], source: { provider: "p", model: "m" } }) }, surfaceOp: "append" },
    { type: "turn/end", seq: 3, time: 4, data: { turn: 1 } }
  ]);
  const prepared = {
    shadowedSeqs: [1, 2],
    session: seeded,
    start: 1,
    end: 2,
    startIdx: 0,
    endIdx: 1,
    measurement: { nodes: [] },
    selectedNodes: [],
    shadowedTokenCount: 2000
  };
  // A small span binds on its own size, a huge span binds on the cap.
  assert.equal(engine.effectiveMaxTokens(2000), 1200);
  assert.equal(engine.effectiveMaxTokens(1_000_000), 4096);
  // Each retry halves the target again, down to the floor.
  assert.equal(engine.effectiveMaxTokens(2000, 1), 600);
  assert.equal(engine.effectiveMaxTokens(2000, 2), 300);
  assert.equal(engine.effectiveMaxTokens(10), 128);
  const result = await engine.compile(prepared, undefined, undefined);
  // The compiled body stays well under the 1200-token budget, so the long
  // user text survives untruncated.
  const text = result.entries.map((entry) => entry.text).join("\n");
  assert.match(text, /word word word/);
  assert.ok(text.length > 400, `long text survived (${text.length} chars)`);
  assert.equal(result.capped, false);
});

test("engine regionDependencies drive a real manual transaction end-to-end", async () => {
  const session = makeIdleSession();
  const ctx = new Context();
  ctx.provide("tokenMeter", {
    measure: (target) => ({
      logRevision: 0,
      baseline: { kind: "none", tokens: 0 },
      surfaceDeltaTokens: 0,
      totalTokens: 0,
      surfaceTokens: 0,
      nodes: target.surface.nodes.map((seq) => ({ seq, tokens: 100 }))
    }),
    estimateMessage: (message) => message.content.reduce((total, block) => total + Math.ceil((block.text?.length ?? 0) / 4), 0)
  });
  const engine = new InstantCompactionEngine(ctx, {
    auto: false,
    textTokens: 64,
    userTextTokens: 64,
    toolCallTokens: 32,
    toolResultExcerptTokens: 32,
    maxTokens: 512
  });
  const result = await compactSurfaceRegion(engine.regionDependencies(), session, 1, 4, undefined, {
    owner: null,
    stability: "selected-span"
  }, undefined);
  assert.deepEqual(result.shadowedSeqs, [1, 2, 3, 4]);
  const summaryEvent = session.events[result.summarySeq];
  assert.equal(summaryEvent.data.provider, "dsh-compaction-instant");
  assert.equal(summaryEvent.data.model, "vcc-compiler");
  const checkpoint = session.events[result.summarySeq + 1];
  assert.equal(checkpoint.data.source.plugin, "compact");
  const text = checkpoint.data.content.map((block) => block.text).join("\n");
  assert.match(text, /<compacted-checkpoint>/);
  assert.match(text, /Automatic compaction: compiled 4 nodes \/ ~400 tokens into \d+ entries \/ ~\d+ tokens/);
  assert.match(text, /## Compiled checkpoint: 4 nodes \(seqs 1-4/);
  assert.match(text, /Verbatim tail retained: 1 nodes \/ ~100 tokens/);
  assert.match(text, /RECALL: /);
  assert.match(text, /recall/);
  assert.match(text, /search/);
  assert.match(text, /\* read "a\.js" \(seq 2 -> result 3\)/);
  assert.doesNotMatch(text, /-> read: ok/);
  assert.deepEqual(session.surface.nodes, [checkpoint.seq, 5]);
});



test("compactSurfaceRegion commits a plain-prose checkpoint smaller than the content it replaces", async () => {
  // Build one detached session: one complete turn holding 30 alternating
  // user and assistant prose messages. No tool calls, so the compiler has no
  // cheap rows to drop.
  const sentence = (index) => {
    const topics = [
      "The release plan lists the migration steps in order.",
      "The review found one open question about the retry policy.",
      "The team recorded the outage window in the shared calendar.",
      "The new cache layer removes the repeated database reads.",
      "The billing report now reconciles against the ledger each night."
    ];
    return `${topics[index % topics.length]} Entry ${index} notes the follow-up owner, the target date, and the confirmation that the change passed review.`;
  };
  const messages = [];
  for (let index = 0; index < 30; index += 1) {
    const text = sentence(index);
    messages.push(index % 2 === 0
      ? createUserMessage({ content: [{ type: "text", text }], source: { kind: "user" } })
      : createAssistantMessage({ content: [{ type: "text", text }], source: { provider: "p", model: "m" } }));
  }
  const seed = [{ type: "turn/start", seq: 0, time: 1, data: { turn: 1 } }];
  let seq = 1;
  for (const message of messages) {
    const type = message.role === "user" ? "user/message" : "assistant/message";
    const data = message.role === "user" ? message : { message };
    seed.push({ type, seq, time: seq + 1, data, surfaceOp: "append" });
    seq += 1;
  }
  seed.push({ type: "turn/end", seq, time: seq + 1, data: { turn: 1 } });
  const session = Session.create("session-prose", seed);

  // Honest meter: ceil(text.length / 4) per text block, applied to both
  // estimateMessage and the per-node surface pricing.
  const estimate = (message) => message.content.reduce(
    (total, block) => block.type === "text" ? total + Math.ceil(block.text.length / 4) : total,
    0
  );
  const meter = {
    estimateMessage: estimate,
    measure: (target) => {
      const nodes = target.surface.nodes.map((nodeSeq) => ({
        seq: nodeSeq,
        tokens: estimate(target.deriveEventMessage(target.events[nodeSeq]))
      }));
      return {
        logRevision: 0,
        baseline: { kind: "none", tokens: 0 },
        surfaceDeltaTokens: 0,
        totalTokens: nodes.reduce((total, node) => total + node.tokens, 0),
        surfaceTokens: nodes.reduce((total, node) => total + node.tokens, 0),
        nodes
      };
    }
  };

  const ctx = new Context();
  ctx.provide("tokenMeter", meter);
  const engine = new InstantCompactionEngine(ctx, { auto: false });

  const nodes = session.surface.nodes;
  const firstSeq = nodes[0];
  const endSeq = nodes[nodes.length - 3]; // keep the last two nodes; every cut is balanced here
  const tokensBefore = meter.measure(session).totalTokens;

  // Must resolve. Today the engine hands the compiler a cap far above the
  // span, the compiler compresses nothing, the framing cost lands on top, and
  // the shrink gate rejects the checkpoint instead of committing it.
  await compactSurfaceRegion(engine.regionDependencies(), session, firstSeq, endSeq, undefined, {
    owner: null,
    stability: "selected-span"
  }, undefined);

  assert.ok(meter.measure(session).totalTokens < tokensBefore);
});
