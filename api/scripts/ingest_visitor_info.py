"""Run robots-aware visitor-information ingestion from the repository root."""

from met_agent.ingestion.commands import ingest_visitor_info, run_command

if __name__ == "__main__":
    raise SystemExit(run_command(ingest_visitor_info))
