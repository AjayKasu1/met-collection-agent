"""Return a terminal contact suggestion without sending email or performing transactions."""

from met_agent.tools.models import Handoff, HandoffArguments


def handoff(arguments: HandoffArguments) -> Handoff:
    return Handoff(**arguments.model_dump())
