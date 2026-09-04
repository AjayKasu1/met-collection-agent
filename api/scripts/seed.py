"""Restore a verified index bundle without sending text to an embedding provider."""

from met_agent.ingestion.commands import run_command, seed

if __name__ == "__main__":
    raise SystemExit(run_command(seed))
