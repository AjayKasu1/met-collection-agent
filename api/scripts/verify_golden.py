"""Verify golden object identifiers against the prepared dataset and live museum API."""

from met_agent.ingestion.commands import run_command, verify

if __name__ == "__main__":
    raise SystemExit(run_command(verify))
