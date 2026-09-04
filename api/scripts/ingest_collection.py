"""Run the collection ingestion command from the repository root."""

from met_agent.ingestion.commands import ingest_collection, run_command

if __name__ == "__main__":
    raise SystemExit(run_command(ingest_collection))
