/**
 * Attribution of subagent reports in compiled regions.
 * @module dsh-compaction-instant/test/subagent-report
 *
 * WHY THIS EXISTS: a background subagent's result is delivered to the
 * orchestrator as a USER-role message. Compiled as plain user text it is
 * indistinguishable from something the human typed and from the
 * orchestrator's own narration. Since the compiled summary is re-read by the
 * model, that turns a worker's CLAIM into the agent's own CONCLUSION.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { fenceSubagentReport, subagentReportOf } from "../src/compiler.js";

const TICK = String.fromCharCode(96);

test("recognises the 'reported:' delivery shape", () => {
  const got = subagentReportOf(
    "Background subagent 6a89b0e0-fa87-4f7f-988f-1658a1bfd88d reported:\n\nPatch at /tmp/x.",
  );
  assert.notEqual(got, null);
  assert.equal(got.agentId, "6a89b0e0-fa87-4f7f-988f-1658a1bfd88d");
  assert.equal(got.kind, "reported");
  assert.equal(got.body, "Patch at /tmp/x.");
});

test("recognises the 'finished ... Its closing message:' delivery shape", () => {
  const got = subagentReportOf(
    "Background subagent 956d50df-2d61-4358-ad41-9ee3c94cb515 finished and will do no further work unless you send it more.Its closing message:\n\nVerdict: PASS",
  );
  assert.notEqual(got, null);
  assert.equal(got.agentId, "956d50df-2d61-4358-ad41-9ee3c94cb515");
  assert.equal(got.kind, "finished");
  assert.equal(got.body, "Verdict: PASS");
});

test("ordinary user text is NOT treated as a report", () => {
  // The failure direction matters: mislabelling the human as a subagent is
  // worse than missing a report, so anything unrecognised stays user text.
  assert.equal(subagentReportOf("Background subagent work is going well"), null);
  assert.equal(subagentReportOf("can you review the diff?"), null);
  assert.equal(subagentReportOf(""), null);
  assert.equal(subagentReportOf(undefined), null);
});

test("a report with no body is not a report", () => {
  assert.equal(subagentReportOf("Background subagent abcdef12-1111 reported:\n\n"), null);
});

test("fence is plain three backticks when the body has none", () => {
  const out = fenceSubagentReport({ agentId: "abcdef12", kind: "reported" }, "plain body");
  assert.ok(out.startsWith(TICK.repeat(3) + "subagent abcdef12 reported\n"));
  assert.ok(out.endsWith("\n" + TICK.repeat(3)));
  assert.ok(!out.startsWith(TICK.repeat(4)));
});

test("fence outgrows a fenced code block inside the report", () => {
  // THE CENTRAL HAZARD: reports routinely contain fenced code. A fixed
  // three-backtick wrapper would terminate at the report's own fence and
  // spill the rest into the surrounding document.
  const body = "before\n" + TICK.repeat(3) + "js\nconst a = 1;\n" + TICK.repeat(3) + "\nafter";
  const out = fenceSubagentReport({ agentId: "abcdef12", kind: "reported" }, body);
  assert.ok(out.startsWith(TICK.repeat(4) + "subagent"));
  assert.ok(out.endsWith("\n" + TICK.repeat(4)));
  // Body survives byte-exact, fences and all.
  assert.ok(out.includes(body));
});

test("fence outgrows even a four-backtick run", () => {
  const body = "x " + TICK.repeat(4) + " y";
  const out = fenceSubagentReport({ agentId: "abcdef12", kind: "finished" }, body);
  assert.ok(out.startsWith(TICK.repeat(5) + "subagent abcdef12 finished\n"));
});

test("the info string names the author, so attribution is machine-readable", () => {
  const out = fenceSubagentReport({ agentId: "956d50df", kind: "finished" }, "body");
  assert.ok(out.includes("subagent 956d50df finished"));
});
