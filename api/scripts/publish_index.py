"""Export and optionally publish a verified index bundle from the repository root."""

from met_agent.ingestion.commands import publish_index, run_command

if __name__ == "__main__":
    raise SystemExit(run_command(publish_index))
