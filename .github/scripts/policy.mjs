#!/usr/bin/env node
// Wording and attribution policy for this repository's public surface.
//
//   node .github/scripts/policy.mjs text <file>...      prose: pull request title and
//                                                        body, commit messages, release notes
//   node .github/scripts/policy.mjs diff <base> <head>  the paths and added lines of a
//                                                        commit range
//   node .github/scripts/policy.mjs tree                 every tracked text file
//
// What is refused:
//   - words from a restricted vocabulary, matched by token against SHA-256
//     prefixes so that the list itself is not part of this repository; the
//     word is reported by its hash, never printed
//   - a person's e-mail address, or a GitHub no-reply address that is not the
//     release identity's
//   - work-in-progress markers (TODO, FIXME, XXX, HACK, WIP)
//   - in prose: internal evaluation figures (precision, recall, F1, AUC, mAP,
//     accuracy) and references to retired repositories (-OLD)
//   - in code and docs: references to the project's previous GitHub home,
//     except in the files .github/policy.json lists (the ones that name what
//     is running or what signed an earlier release)
//
// Exit status 1 on any finding; findings are printed as GitHub annotations.
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";

const RELEASE_IDENTITY = "masseuse-ai[bot]";
const HASH_PREFIX_LENGTH = 16;
const RESTRICTED = new Set([
  "0033728f0fbc83a0", "03a4dba65e9bfab3", "0508a634445d4016", "066e6872d9312d71", "0f28c4960d96647e", "102626eace2d1293",
  "102cf10b5286bad9", "15cce715eca5ae54", "1871ee4724dbe67d", "1ca8148920f67922", "1d0864b81a8a857b", "2081cc0de301a35d",
  "20da8fd4fb86a13e", "22082517effcaaaa", "2484b821a44b8522", "25390976ce87338a", "26b41cefc2949db7", "27e030b863709772",
  "2944474a2c8b8c37", "2a6060b5e21aacf2", "2c32c156cf1a3222", "2ceabeb8a4a2e857", "31506a8448a761a4", "332a552909d05749",
  "38136d6e343860f6", "39a8076d0dc8d9a1", "3b6ffbe455366db0", "3ebc8e790959b8ef", "3f319f095c6630cd", "4140197ec7959b1c",
  "4ae1915834846378", "4c8027b3a281d10e", "4dc9418555a26aa6", "522788b65f01ccc0", "534e1374e3a9741d", "594810adcbee20bd",
  "5a36615e036838ff", "5a5f8061334fec4c", "5f9a6a761031bc6a", "61081efefc5da844", "6556c1a58444e10f", "665cb762e3bc03d0",
  "67beb8b817d4b5f1", "68ee7b6f81ea0b33", "6ac3c336e4094835", "6adc452d6b74df40", "6cd1442aa2721a94", "6ef9d9fd4a0456ee",
  "701d6943ae042bad", "737dd1bca21d67a7", "74305616b2bc16c0", "7902e27065cfa96f", "796e43a5a8cdb73b", "7f5a9ba14cb89dc2",
  "805c0cdff876faad", "81dba13489b1cb7a", "83ebccea9fff4079", "850c22e6c25989b4", "85266837dfc78c80", "8693f216c2503fbc",
  "89133bc00e9b3eb8", "8b272f292b3a0afb", "8c5c04391361cbf4", "8e5765deaa03d947", "90d8020dfbe31ad0", "94a2b576a9f577e9",
  "9557070a43b1b8de", "9880a3c66c8132b1", "98d44e13f455d916", "99e367bcad7a502f", "9ddcab3ac3610db0", "9f3a8f72976c6e75",
  "a02d538d50989a60", "a2a5fcb172d6cd44", "a368606514955567", "acf45488978d9ff4", "ad505b0be8a49b89", "ade74a96ae48fd4f",
  "b0a2a22c5338a2d2", "b0f1b0384c1e6b28", "b1706a6959a8187b", "b34e00e1626247d3", "b64608345c88e3e5", "b690cdbe59f14b3b",
  "b70acbfa700c079b", "b9a741c0601883de", "ba79229c14881e61", "bcb9209dc4fd8180", "be167ae3efd26cad", "c59758402ac47948",
  "c6603565c5159fbe", "c8bd849e3a5aebe0", "c927109d05e59dc0", "c94b2acca79ccea4", "cb5ae32cbd84a47c", "d11f3961726f0347",
  "d22961946894a784", "d2abdc5b50592638", "d2ce238bf4709dde", "d81480db9db2df87", "d84f691fc4981d9d", "d8b33bdb33e294cf",
  "da602cbc580a32bc", "daf8ee0b8faced43", "dbff2575956e5a07", "dcb50ee3659fb742", "dcc43fc8a425beda", "dd3d4a3f76c1d3dd",
  "de569a7f6852ba73", "dfb0ce07edf923f1", "e12b91523bb8e253", "e157956de2d36583", "e1b73076698c35dc", "e683679358a5aea3",
  "e9ca853fe88ecec4", "ebc98c1efce2b182", "ebec03d80c0d3bff", "eff26b18165ece9c", "f0b4965f5e0dd0a4", "f66ea2d273fe182c",
  "f6952d6eef555ddd", "fdca9cb0db1db597", "feb0e1a9ef5e4e0d", "ff5ae7e8d206066d",
]);

const PERSONAL_EMAIL = /[A-Za-z0-9._%+-]+@(?:gmail|googlemail|icloud|me|mac|outlook|hotmail|live|msn|yahoo|ymail|aol|proton|protonmail|pm|fastmail|hey)\.(?:com|me|ch|net|org)\b/i;
const NOREPLY = /(?:\d+\+)?([A-Za-z0-9-]+(?:\[bot\])?)@users\.noreply\.github\.com/g;
const MARKERS = /\b(?:TODO|FIXME|XXX|HACK|WIP)\b/;
const METRICS = [/\b(?:precision|recall|accuracy)\b/i, /\bF1\b/, /\bAUC\b/, /\bmAP\b/, /\bF-?score\b/i];
const RETIRED = /\b(?:masseuse-camlink|masseuse-video-tee)-OLD\b/i;
const PREVIOUS_HOME = /github\.com\/FemLed\/|ghcr\.io\/femled\/|FemLed\/masseuse-(?:camlink|video-tee)\b/i;

// This checker names the markers and rules it looks for, so it is not read.
const SKIP_PATH = /(^|\/)\.github\/scripts\/policy\.mjs$|(^|\/)(vendor|node_modules|bindings)\/|(^|\/)package-lock\.json$|\.lock$|\.(png|jpg|jpeg|gif|ico|icns|woff2?|ttf|otf|syso|car|mp4|webm|riv|lottie|bin|pb|onnx|gz|xz|zip|dmg|exe)$/;

const mode = process.argv[2];
const findings = [];
function finding(file, line, message) { findings.push({ file, line, message }); }

function config() {
  try {
    const c = JSON.parse(fs.readFileSync(path.join(process.cwd(), ".github", "policy.json"), "utf8"));
    return { previousHomeAllowed: (c.previousHomeAllowed || []).map(globToRegExp) };
  } catch { return { previousHomeAllowed: [] }; }
}
function globToRegExp(glob) {
  const re = glob.split("**").map((part) => part.split("*").map((s) => s.replace(/[.+^${}()|[\]\\?]/g, "\\$&")).join("[^/]*")).join(".*");
  return new RegExp(`^${re}$`);
}
const cfg = config();
const previousHomeAllowed = (file) => cfg.previousHomeAllowed.some((re) => re.test(file));

function tokenHashes(text) {
  const out = [];
  for (const tok of text.toLowerCase().split(/[^a-z0-9]+/)) {
    if (tok.length < 3) continue;
    const h = crypto.createHash("sha256").update(tok).digest("hex").slice(0, HASH_PREFIX_LENGTH);
    if (RESTRICTED.has(h)) out.push(h);
  }
  return out;
}
function checkLine(file, lineNo, line, { prose, code }) {
  for (const h of tokenHashes(line)) finding(file, lineNo, `a word from the restricted vocabulary (${h})`);
  if (PERSONAL_EMAIL.test(line)) finding(file, lineNo, "a personal e-mail address");
  for (const m of line.matchAll(NOREPLY)) if (m[1] !== RELEASE_IDENTITY) finding(file, lineNo, "a GitHub no-reply address that is not the release identity's");
  if (MARKERS.test(line)) finding(file, lineNo, "a work-in-progress marker (TODO, FIXME, XXX, HACK, WIP)");
  if (prose) {
    for (const re of METRICS) if (re.test(line)) { finding(file, lineNo, "an internal evaluation figure or its name (precision, recall, F1, AUC, mAP, accuracy)"); break; }
    if (RETIRED.test(line)) finding(file, lineNo, "a reference to a retired repository");
  }
  if (code && PREVIOUS_HOME.test(line) && !previousHomeAllowed(file)) finding(file, lineNo, "a reference to the project's previous GitHub home outside the files .github/policy.json allows");
}
function checkText(file, text, opts) {
  text.split(/\r?\n/).forEach((line, i) => checkLine(file, i + 1, line, opts));
}
function isBinary(buf) { return buf.subarray(0, 8192).includes(0); }
function git(args) {
  const r = spawnSync("git", args, { encoding: "buffer", maxBuffer: 512 * 1024 * 1024 });
  if (r.status !== 0) { console.error(`git ${args.join(" ")}: ${r.stderr.toString().trim()}`); process.exit(2); }
  return r.stdout;
}

if (mode === "text") {
  const files = process.argv.slice(3);
  if (files.length === 0) { console.error("text: no files"); process.exit(2); }
  for (const f of files) checkText(f, fs.readFileSync(f, "utf8"), { prose: true, code: false });
} else if (mode === "diff") {
  const [base, head] = process.argv.slice(3);
  if (!base || !head) { console.error("diff: base and head are required"); process.exit(2); }
  const names = git(["diff", "--no-renames", "--name-only", "-z", "--diff-filter=AM", base, head]).toString().split("\0").filter(Boolean);
  for (const name of names) {
    checkLine(name, 0, name, { prose: false, code: false }); // the path itself
    if (SKIP_PATH.test(name)) continue;
    const blob = git(["cat-file", "blob", `${head}:${name}`]);
    if (isBinary(blob)) continue;
    // Added lines only: what this change introduces.
    const patch = git(["diff", "--no-renames", "-U0", base, head, "--", name]).toString();
    let lineNo = 0;
    for (const line of patch.split("\n")) {
      const hunk = /^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@/.exec(line);
      if (hunk) { lineNo = Number(hunk[1]) - 1; continue; }
      if (line.startsWith("+") && !line.startsWith("+++")) { lineNo += 1; checkLine(name, lineNo, line.slice(1), { prose: /\.md$/i.test(name), code: true }); }
    }
  }
} else if (mode === "tree") {
  const names = git(["ls-files", "-z"]).toString().split("\0").filter(Boolean);
  for (const name of names) {
    checkLine(name, 0, name, { prose: false, code: false });
    if (SKIP_PATH.test(name)) continue;
    const buf = fs.readFileSync(name);
    if (buf.length > 4 * 1024 * 1024 || isBinary(buf)) continue;
    checkText(name, buf.toString("utf8"), { prose: false, code: true });
  }
} else {
  console.error("usage: policy.mjs text <file>... | diff <base> <head> | tree");
  process.exit(2);
}

if (findings.length === 0) {
  console.log(`policy: ${mode} ok`);
} else {
  for (const f of findings) console.log(`::error file=${f.file},line=${f.line}::${f.message}`);
  console.log(`policy: ${findings.length} finding(s)`);
  process.exit(1);
}
