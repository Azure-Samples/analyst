"""
Module: questions

This module implements the orchestration of multiple avatar agents (graders) in parallel 
using Semantic Kernel and Azure OpenAI services. It provides the core components needed 
to configure, execute, and aggregate responses from multiple grader agents. 

Components:
  - Mediator: An abstract interface for mediator implementations that handle notifications 
    from sub-avatar agents.
  - GraderBase: An abstract base class defining the interface for grader agents capable of 
    interacting with prompts asynchronously.
  - AnswerGrader: A concrete grader that uses a ChatCompletionAgent to process prompts and 
    generate responses. It loads its configuration via Jinja2 templates.
  - GraderFactory: A factory for building a shared Semantic Kernel and creating AnswerGrader 
    instances from an Assembly document.
  - AnswerOrchestrator: A high-level orchestrator that fetches an Assembly document from 
    Cosmos DB, creates grader instances, and executes their interactions concurrently or 
    sequentially.

Usage:
    1. Ensure environment variables (e.g. COSMOS_QNA_NAME, COSMOS_ENDPOINT, COSMOS_ASSEMBLY_TABLE, 
       AZURE_MODEL_KEY, AZURE_MODEL_URL) are defined in a .env file placed at the root (two levels 
       above this module).
    2. Prepare an Assembly document in Cosmos DB which contains grader (avatar) configurations.
    3. Instantiate an AnswerOrchestrator and call its run_interaction method with an assembly_id, 
       question, and answer to obtain aggregated responses from all configured grader agents.
       
Dependencies:
    - semantic_kernel: Provides ChatCompletionAgent, ChatHistory, KernelArguments, etc.
    - jinja2: Used for templating of prompt instructions.
    - azure.cosmos and azure.identity: For Cosmos DB access and authentication.
    - dotenv: For environment variable loading.
    - asyncio: For asynchronous execution.

"""

import os
import logging
import asyncio
from abc import ABC, abstractmethod

from typing import List, Optional, Literal, override

import jinja2
from dotenv import load_dotenv

import semantic_kernel as sk
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.contents import ChatHistory, ChatMessageContent, AuthorRole
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.exceptions import ServiceResponseException

from azure.cosmos import exceptions
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import DefaultAzureCredential

from app.schemas import Assembly, Agent
from .operators import Mediator

from app.plugins import StatisticalAnalysisPlugin, SeabornPlugin, CodeRunnerPlugin, ResearchPlugin

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_FILE = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_FILE)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


COSMOS_DB_NAME = os.getenv("COSMOS_QNA_NAME", "mydb")
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", "https://myendpoint.documents.azure.com:443/")
COSMOS_ASSEMBLY_TABLE = os.getenv("COSMOS_ASSEMBLY_TABLE", "assembly")
AZURE_MODEL_KEY = os.getenv("AZURE_MODEL_KEY", "")
AZURE_MODEL_URL = os.getenv("AZURE_MODEL_URL", "")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "prompts")
JINJA_ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES_DIR))


AVAILABLE_MODELS: dict[str, AzureChatCompletion] = {
    'default': AzureChatCompletion(
        service_id="default",
        api_key=AZURE_MODEL_KEY,
        deployment_name="gpt-4o",
        endpoint=AZURE_MODEL_URL
    ),
    'mini': AzureChatCompletion(
        service_id="mini",
        api_key=AZURE_MODEL_KEY,
        deployment_name="gpt-4o-mini",
        endpoint=AZURE_MODEL_URL
    ),
    'reasoning': AzureChatCompletion(
        service_id="reasoning",
        api_key=AZURE_MODEL_KEY,
        deployment_name="o3-mini",
        endpoint=AZURE_MODEL_URL,
        api_version="2024-12-01-preview",
    )
}


def sql_template(table: str, columns: dict[str, str], filters: dict[str, str]) -> str:
    prompt_template = JINJA_ENV.get_template("sql.jinja")
    rendered_prompt = prompt_template.render(
        table=table,
        columns=columns,
        filters=filters,
    )
    return rendered_prompt


class ToolerBase(ABC):
    """
    Abstract base for an avatar (grader) that can interact with a prompt.

    This class defines the core interface for graders, including an asynchronous
    interaction method.
    """
    def __init__(self, tooler: Agent, kernel: sk.Kernel) -> None:
        """
        Initialize an AnswerGrader.

        :param grader: The Grader configuration data (including id, name, metaprompt, model_id).
        :param kernel: A shared Semantic Kernel instance configured with services.
        """
        self._mediator: Optional[Mediator] = None
        self.kernel = kernel
        self.tooler = tooler
        self.agent: ChatCompletionAgent
        self.__prepare()

    @property
    def mediator(self) -> Optional[Mediator]:
        return self._mediator

    @mediator.setter
    def mediator(self, mediator: Mediator) -> None:
        """
        Set the mediator for the grader.

        :param mediator: The mediator instance to be set.
        """
        self._mediator = mediator

    def __prepare(self):
        """
        Prepare the grader for interaction by configuring the ChatCompletionAgent.

        Renders the instruction settings from a Jinja2 template and retrieves the settings based on
        the grader's model_id. Then creates the ChatCompletionAgent instance.
        """
        instruction_template = JINJA_ENV.get_template("instruction.jinja")
        rendered_settings = instruction_template.render(
            instructions=self.tooler.metaprompt,
        )
        settings = self.kernel.get_prompt_execution_settings_from_service_id(service_id=self.tooler.model_id)
        settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
        self.agent = ChatCompletionAgent(
            kernel=self.kernel,
            name=self.tooler.name,
            instructions=rendered_settings,
            arguments=KernelArguments(settings=settings),
        )

    async def add_tools(self, tools: list) -> None:  # pylint: disable=arguments-differ
        """
        Interact with a provided question and answer using the ChatCompletionAgent.

        Renders the prompt using a Jinja2 template, sends it to the agent, accumulates the responses,
        adds messages to the provided ChatHistory, and notifies the mediator upon completion.

        :param question: The question object for the interaction.
        :param answer: The answer object associated with the question.
        :param chat: A ChatHistory instance for logging the conversation.
        :return: The aggregated response generated by the agent.
        """
        for tool in tools:
            self.kernel.add_plugin(tool, tool.__class__.__name__)

    async def _execute(self, chat: ChatHistory, rendered_prompt: str) -> str:
        chat.add_message(ChatMessageContent(role=AuthorRole.USER, content=rendered_prompt))
        response = ""
        while True:
            async for message in self.agent.invoke(messages=chat.messages):  # type: ignore[assignment]
                response += message.content.content
                chat.add_message(ChatMessageContent(role=AuthorRole.ASSISTANT, content=message.content.content))

            if self.mediator:
                self.mediator.notify(
                    sender=self,
                    event="interaction_done",
                    data={
                        "tooler_id": self.tooler.id,
                        "tooler_name": self.tooler.name,
                    }
                )
            break
        return response

    @abstractmethod
    async def interact(self, prompt: str, chat: ChatHistory) -> str:
        """
        Asynchronously perform interaction with a prompt.

        Implementations must provide their own logic to process a prompt and generate
        a response.
        """


class CodeWriter(ToolerBase):
    """
    A grader that interacts with a prompt using a ChatCompletionAgent.

    Loads its personality and instructions from its profile, renders the prompt via a
    Jinja2 template, and generates a response. Notifies a mediator when interaction is complete.
    """

    @override
    @kernel_function(
        name="CodeAgent",
        description="implements reasoning logic for creating and executing code."
    )
    async def interact(self, prompt: str, chat: ChatHistory) -> str:  # pylint: disable=arguments-differ
        """
        Interact with a provided question and answer using the ChatCompletionAgent.

        Renders the prompt using a Jinja2 template, sends it to the agent, accumulates the responses,
        adds messages to the provided ChatHistory, and notifies the mediator upon completion.

        :param question: The question object for the interaction.
        :param answer: The answer object associated with the question.
        :param chat: A ChatHistory instance for logging the conversation.
        :return: The aggregated response generated by the agent.
        """
        await self.add_tools([CodeRunnerPlugin()])
        prompt_template = JINJA_ENV.get_template("code.jinja")
        rendered_prompt = prompt_template.render(prompt=prompt, tools=[tool.__class__.__name__ for tool in self.kernel.plugins])
        try:
            return await self._execute(chat, rendered_prompt)
        except ServiceResponseException as e:
            logger.error(f"Service response error: {e}")
            await asyncio.sleep(60)
            return await self.interact(prompt, chat)


class GraphWriter(ToolerBase):
    """
    A grader that interacts with a prompt using a ChatCompletionAgent.

    Loads its personality and instructions from its profile, renders the prompt via a
    Jinja2 template, and generates a response. Notifies a mediator when interaction is complete.
    """

    @override
    @kernel_function(
        name="DrawAgent",
        description="implements graphic creation for datasets and analyse data."
    )
    async def interact(self, prompt: str, chat: ChatHistory) -> str:  # pylint: disable=arguments-differ
        """
        Interact with a provided question and answer using the ChatCompletionAgent.

        Renders the prompt using a Jinja2 template, sends it to the agent, accumulates the responses,
        adds messages to the provided ChatHistory, and notifies the mediator upon completion.

        :param question: The question object for the interaction.
        :param answer: The answer object associated with the question.
        :param chat: A ChatHistory instance for logging the conversation.
        :return: The aggregated response generated by the agent.
        """
        await self.add_tools([SeabornPlugin()])
        prompt_template = JINJA_ENV.get_template("graphic.jinja")
        rendered_prompt = prompt_template.render(data=prompt)
        try:
            return await self._execute(chat, rendered_prompt)
        except ServiceResponseException as e:
            logger.error(f"Service response error: {e}")
            await asyncio.sleep(60)
            return await self.interact(prompt, chat)


class Statistian(ToolerBase):
    """
    A grader that interacts with a prompt using a ChatCompletionAgent.

    Loads its personality and instructions from its profile, renders the prompt via a
    Jinja2 template, and generates a response. Notifies a mediator when interaction is complete.
    """

    @override
    @kernel_function(
        name="StatsAgent",
        description="implements statistical analysis for datasets."
    )
    async def interact(self, prompt: str, chat: ChatHistory) -> str:  # pylint: disable=arguments-differ
        """
        Interact with a provided question and answer using the ChatCompletionAgent.

        Renders the prompt using a Jinja2 template, sends it to the agent, accumulates the responses,
        adds messages to the provided ChatHistory, and notifies the mediator upon completion.

        :param question: The question object for the interaction.
        :param answer: The answer object associated with the question.
        :param chat: A ChatHistory instance for logging the conversation.
        :return: The aggregated response generated by the agent.
        """
        await self.add_tools([StatisticalAnalysisPlugin(), ResearchPlugin()])
        prompt_template = JINJA_ENV.get_template("profile.jinja")
        rendered_prompt = prompt_template.render(audio=prompt)
        try:
            return await self._execute(chat, rendered_prompt)
        except ServiceResponseException as e:
            logger.error(f"Service response error: {e}")
            await asyncio.sleep(60)
            return await self.interact(prompt, chat)


class ToolerFactory:
    """
    Factory for building a shared Semantic Kernel and creating AnswerGrader instances.

    Uses the provided Assembly object to create graders for each agent defined in the assembly.
    """
    @staticmethod
    def __build_kernel() -> sk.Kernel:
        """
        Build and return a shared Semantic Kernel.

        Adds each service in AVAILABLE_MODELS to the kernel.

        :return: A configured Semantic Kernel instance.
        """
        kernel = sk.Kernel()
        for service_id, service in AVAILABLE_MODELS.items():
            try:
                kernel.add_service(service)
            except Exception as e:
                logger.error(f"Failed to add service {service_id}: {e}")
                pass
        return kernel

    def create_toolers(self, assembly: Assembly) -> List[ToolerBase]:
        """
        Create a list of AnswerGrader instances from an Assembly.

        Assumes that assembly.agents is a list of Grader configuration models.

        :param assembly: The Assembly containing agent configurations.
        :return: A list of instantiated AnswerGrader objects.
        """
        common_kernel = self.__build_kernel()
        toolers = []
        for agent in assembly.agents:
            match agent.objective.lower():
                case "code":
                    toolers.append(CodeWriter(agent, common_kernel))
                case "graphics":
                    toolers.append(GraphWriter(agent, common_kernel))
                case "analysis":
                    toolers.append(Statistian(agent, common_kernel))
                case _:
                    raise ValueError(f"Unsupported objective: {agent.objective}")
        return toolers


class ToolerOrchestrator:
    """
    High-level orchestrator that merges grader management and interaction execution.

    Fetches assemblies from Cosmos DB, creates grader instances via GraderFactory, and executes
    interactions in either parallel or sequential mode.
    """
    def __init__(self) -> None:
        self.graders: List[ToolerBase] = []

    async def _parallel_processing(self, prompt: str) -> List:
        """
        Execute the 'interact' method of all graders concurrently.

        :param question: The question to be processed.
        :param answer: The associated answer object.
        :return: A list of responses from all graders executed in parallel.
        """
        async def interact_with_tooler(tooler: ToolerBase, chat: ChatHistory) -> str:
            return await tooler.interact(prompt, chat)

        chat = ChatHistory()
        return await asyncio.gather(*(interact_with_tooler(tooler, chat) for tooler in self.toolers))

    async def _sequential_processing(self, prompt: str) -> List:
        """
        Execute the 'interact' method of all graders sequentially.

        :param question: The question to be processed.
        :param answer: The corresponding answer object.
        :return: A list of dictionaries mapping grader identifiers to their responses.
        """
        answers = []
        chat = ChatHistory()
        for index, tooler in enumerate(self.toolers):
            result = await tooler.interact(prompt, chat)
            answers.append({f"agent_{index}": result})
        return answers

    async def _llm_processing(self, prompt: str) -> List:
        """
        Execute the 'interact' method of all graders by using a new agent that has agents as tools.

        :param question: The question to be processed.
        :param answer: The corresponding answer object.
        :return: A list of dictionaries mapping grader identifiers to their responses.
        """
        kernel = sk.Kernel()
        service = AVAILABLE_MODELS.get("reasoning", None)
        if not service:
            raise ValueError("Reasoning service not found in available models.")
        kernel.add_service(service)
        kernel.add_plugins({tooler.tooler.name: tooler for tooler in self.toolers})
        instruction_template = JINJA_ENV.get_template("reason.jinja")
        rendered_settings = instruction_template.render(tools=[grader.tooler.name for grader in self.toolers])
        settings = AzureChatPromptExecutionSettings(
            service_id=service.service_id,
            max_completion_tokens=4000,
            reasoning_effort="high",
        )
        settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
        agent = ChatCompletionAgent(
            id=service.service_id,
            kernel=kernel,
            name="reasoning_agent",
            instructions=rendered_settings,
            arguments=KernelArguments(settings=settings)
        )
        chat = ChatHistory()
        chat.add_message(ChatMessageContent(role=AuthorRole.USER, content=prompt))
        result = []
        while True:
            async for message in agent.invoke(messages=chat.messages):  # type: ignore[assignment]
                result.append(message.content)
            break
        return [result]

    async def run_interaction(
            self,
            assembly: str | Assembly,
            prompt: str,
            strategy: Literal["parallel", "sequential", "llm"] = "parallel"
        ) -> str:
        """
        Orchestrate the grader interactions for a given assembly.

        This method performs the following steps:
          1) Builds a shared kernel.
          2) Creates AnswerGrader instances using GraderFactory.
          3) Fetches the Assembly document from Cosmos DB.
          4) Executes grader interactions using the specified strategy (parallel or sequential).
          5) Returns the aggregated responses.

        :param assembly_id: The ID of the Assembly in Cosmos DB.
        :param question: The question object to send to graders.
        :param answer: The answer object associated with the question.
        :param strategy: The processing strategy ("parallel" or "sequential").
        :return: The aggregated responses from all grader interactions.
        """
        factory = ToolerFactory()
        if isinstance(assembly, str):
            assembly = await self.fetch_assembly(assembly)
        self.toolers = factory.create_toolers(assembly)
        answers = await getattr(self, f"_{strategy}_processing")(prompt)
        return answers

    async def fetch_assembly(self, assembly_id: str) -> Assembly:
        """
        Fetch an Assembly document from Cosmos DB using its ID.

        Raises a ValueError if the database or assembly is not found.

        :param assembly_id: The ID of the assembly to fetch.
        :return: An Assembly object constructed from the retrieved document.
        """
        async with CosmosClient(COSMOS_ENDPOINT, DefaultAzureCredential()) as client:
            try:
                database = client.get_database_client(COSMOS_DB_NAME)
                await database.read()
            except exceptions.CosmosResourceNotFoundError as exc:
                raise ValueError(f"Database not found: {COSMOS_DB_NAME}") from exc

            container = database.get_container_client(COSMOS_ASSEMBLY_TABLE)
            try:
                item = await container.read_item(item=assembly_id, partition_key=assembly_id)
            except exceptions.CosmosResourceNotFoundError as exc:
                raise ValueError(f"Assembly not found: {assembly_id}") from exc
            return Assembly(**{"id": item["id"], "agents": item["avatars"], "topic_name": item["topic_name"]})
