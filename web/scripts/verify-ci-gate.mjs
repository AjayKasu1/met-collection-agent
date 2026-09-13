#!/usr/bin/env node
/**
 * verify-ci-gate.mjs
 *
 * Enforces that production Cloudflare Worker builds wait for GitHub Actions CI
 * quality checks to succeed on the commit being deployed.
 */

import { execSync } from "node:child_process";

const REQUIRED_CHECKS = [
  "Python 3.12 quality checks",
  "Node 20 web and Workers checks",
  "Production container builds",
];

const GITHUB_REPO = process.env.GITHUB_REPOSITORY || "AjayKasu1/met-collection-agent";
const MAX_ATTEMPTS = 40;
const POLL_INTERVAL_MS = 10000;

export function resolveCommitSha() {
  if (process.env.CF_PAGES_COMMIT_SHA) return process.env.CF_PAGES_COMMIT_SHA;
  if (process.env.WORKERS_CI_COMMIT_SHA) return process.env.WORKERS_CI_COMMIT_SHA;
  if (process.env.COMMIT_SHA) return process.env.COMMIT_SHA;
  try {
    return execSync("git rev-parse HEAD", { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
  } catch {
    return null;
  }
}

export function shouldEnforceGate() {
  if (process.env.VERIFY_CI_GATE === "true") return true;
  if (process.env.GITHUB_ACTIONS === "true") return false;
  if (process.env.CF_PAGES === "1" || process.env.WORKERS_CI === "1" || Boolean(process.env.CF_PAGES_COMMIT_SHA)) {
    return true;
  }
  if (process.env.NODE_ENV === "production" && resolveCommitSha()) return true;
  return false;
}

export async function verifyCheckRuns(commitSha) {
  console.log(`[CI-GATE] Verifying GitHub CI gate for commit ${commitSha} on ${GITHUB_REPO}...`);

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    try {
      const token = process.env.GH_TOKEN || process.env.GITHUB_TOKEN;
      const res = await fetch(`https://api.github.com/repos/${GITHUB_REPO}/commits/${commitSha}/check-runs`, {
        headers: {
          Accept: "application/vnd.github+json",
          "User-Agent": "met-agent-worker-ci-gate",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      });

      if (!res.ok) {
        const remaining = res.headers.get("x-ratelimit-remaining");
        const resetTime = res.headers.get("x-ratelimit-reset");
        console.warn(
          `[CI-GATE] GitHub API returned HTTP ${res.status} (remaining: ${remaining}, reset: ${resetTime}). Retrying in ${POLL_INTERVAL_MS / 1000}s...`,
        );
        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
        continue;
      }

      const data = await res.json();
      const checkRuns = Array.isArray(data.check_runs) ? data.check_runs : [];

      const statusMap = new Map();
      for (const check of checkRuns) {
        statusMap.set(check.name, check);
      }

      let allPassed = true;
      let anyFailed = false;
      const statusSummary = [];

      for (const reqName of REQUIRED_CHECKS) {
        const check = statusMap.get(reqName);
        if (!check) {
          allPassed = false;
          statusSummary.push(`${reqName}: not_started`);
          continue;
        }
        statusSummary.push(`${reqName}: ${check.status}/${check.conclusion}`);
        if (check.conclusion === "failure" || check.conclusion === "cancelled" || check.conclusion === "timed_out") {
          anyFailed = true;
          console.error(`[CI-GATE] Required check "${reqName}" failed with conclusion: ${check.conclusion}`);
        } else if (check.status !== "completed" || check.conclusion !== "success") {
          allPassed = false;
        }
      }

      if (anyFailed) {
        console.error(`[CI-GATE] Deployment BLOCKED: One or more required CI quality checks failed.`);
        process.exit(1);
      }

      if (allPassed) {
        console.log(`[CI-GATE] All required CI checks passed for commit ${commitSha}:`);
        for (const reqName of REQUIRED_CHECKS) {
          console.log(`  - ${reqName}: SUCCESS`);
        }
        console.log(`[CI-GATE] CI gate verified. Proceeding with Worker build.`);
        return;
      }

      console.log(`[CI-GATE] Attempt ${attempt}/${MAX_ATTEMPTS}: Waiting for CI checks (${statusSummary.join(", ")})...`);
    } catch (err) {
      console.warn(`[CI-GATE] Polling error: ${err.message}. Retrying...`);
    }

    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }

  console.error(`[CI-GATE] Deployment BLOCKED: Timed out waiting for GitHub CI checks to pass for commit ${commitSha}.`);
  process.exit(1);
}

async function main() {
  if (!shouldEnforceGate()) {
    console.log("[CI-GATE] Bypassing CI gate verification (non-production / GitHub Actions environment).");
    return;
  }

  const commitSha = resolveCommitSha();
  if (!commitSha) {
    console.error("[CI-GATE] Unable to determine commit SHA. Deployment blocked.");
    process.exit(1);
  }

  await verifyCheckRuns(commitSha);
}

if (process.argv[1] && import.meta.url.endsWith(process.argv[1])) {
  main().catch((err) => {
    console.error("[CI-GATE] Fatal error during gate check:", err);
    process.exit(1);
  });
}
