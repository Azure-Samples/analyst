"""
This module defines the data models for the application using Pydantic.

These models represent the core entities of the augmented RAG system including agents,
assemblies, tools, and various data types (text, image, audio, video) used for
retrieval-augmented generation tasks.

Classes:
    Agent: Represents an AI agent with specific capabilities and configuration.
    Assembly: Represents a collection of agents working together for a specific objective.
    Tool: Represents a utility function available to agents.
    TextData: Represents textual data with metadata and embeddings.
    ImageData: Represents image data with metadata and embeddings.
    AudioData: Represents audio data with metadata and embeddings.
    VideoData: Represents video data with metadata and embeddings.
    JobResponse: Represents the response from a processing job.
"""

from typing import List, Literal
from pydantic import BaseModel, Field, field_validator


class Agent(BaseModel):
    """
    Represents an AI agent with its configuration and capabilities.

    Attributes:
        id (str): The unique identifier for the agent.
        name (str): The human-readable name of the agent.
        model_id (str): The identifier of the model used by this agent.
        metaprompt (str): The system prompt that defines the agent's behavior.
        objective (Literal["image", "text", "audio", "video"]): The data type this agent specializes in.
    """

    id: str = Field(..., description="Agent ID")
    name: str = Field(..., description="Agent Name")
    model_id: str = Field(..., description="Model ID")
    metaprompt: str = Field(..., description="Agent System Prompt")
    objective: Literal["code", "graphics", "analysis"] = Field(default='code', description="Agent Objective")

    @classmethod
    @field_validator("model_id")
    def model_must_be_small(cls, v):
        if len(v) > 32:
            raise ValueError("model ID shouldn't have more than 32 characters")
        return v

    @classmethod
    @field_validator("objective")
    def objective_must_be_small(cls, v):
        if len(v) > 32:
            raise ValueError("objective shouldn't have more than 32 characters")
        return v


class Assembly(BaseModel):
    """
    Represents a collection of agents working together toward a specific objective.

    Attributes:
        id (str): The unique identifier for the assembly.
        objective (str): The goal or task this assembly is designed to achieve.
        agents (List[Agent]): The collection of agents that form this assembly.
        roles (List[str]): The defined roles for agents within this assembly.
    """

    id: str = Field(..., description="Agent Assembly ID")
    objective: str = Field(..., description="The Agent Assembly Object to operate on")
    agents: List[Agent] = Field(..., description="Agents Assemblies")
    roles: List[str] = Field(..., description="Agent Roles ID")

    @classmethod
    @field_validator("roles")
    def roles_must_not_exceed_length(cls, v):
        for role in v:
            if len(role) > 360:
                raise ValueError("each role must have at most 360 characters")
        return v


class CSVData(BaseModel):
    """
    Represents textual data with associated metadata and vector embeddings.

    Attributes:
        source (str): Source location of the text data.
        value (str): The actual text content.
        objective (str): The purpose or context of this text.
        encoding (str): The text encoding format.
        tags (List[str]): Keywords or categories associated with this text.
        original_document (Optional[str]): Reference to source document if applicable.
        embeddings (Optional[List[float]]): Vector embeddings for semantic search.
    """
    source: str = Field(..., description="file name")
    content: str = Field(..., description="csv content")


class JobResponse(BaseModel):
    """
    Represents the response from a processing job executed by an assembly.

    Attributes:
        assembly_id (str): The identifier of the assembly that processed the job.
        prompt (str): The input prompt or query that initiated the job.
    """

    assembly_id: str = Field(..., description="Assembly ID")
    prompt: str = Field(..., description="Job Status")
    csv_data: CSVData = Field(..., description="CSV Data")
