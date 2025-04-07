"""
A package that holds response schemas and models.
"""

__all__ = [
    "RESPONSES",
    "ErrorMessage",
    "SuccessMessage",
    "Agent",
    "Assembly",
    "JobResponse",
    "database_schema",
]

__author__ = "AI GBBS"

from .models import Assembly, Agent, JobResponse
from .responses import RESPONSES, ErrorMessage, SuccessMessage

database_schema = {
    "Agent Table": Agent.model_json_schema(),
    "Assembly Table": Assembly.model_json_schema(),
}
