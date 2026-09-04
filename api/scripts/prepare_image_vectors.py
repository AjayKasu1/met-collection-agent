"""Prepare the published image-vector join without running an image model."""

from met_agent.ingestion.commands import prepare_image_vectors, run_command

if __name__ == "__main__":
    raise SystemExit(run_command(prepare_image_vectors))
