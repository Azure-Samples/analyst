"""
The configuration for the mailing service.
"""

import logging
import os

from dotenv import find_dotenv, load_dotenv

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __app__, __version__
from app.agents import ToolerOrchestrator
from app.cosmos_crud import CosmosCRUD
from app.schemas import *


load_dotenv(find_dotenv())

BLOB_CONN = os.getenv("BLOB_CONNECTION_STRING", "")
MODEL_URL: str = os.environ.get("GPT4_URL", "")
MODEL_KEY: str = os.environ.get("GPT4_KEY", "")
MONITOR: str = os.environ.get("AZ_CONNECTION_LOG", "")
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", "")

COSMOS_ASSEMBLY_TABLE = os.getenv("COSMOS_ASSEMBLY_TABLE", "assembly")
COSMOS_TOOL_TABLE = os.getenv("COSMOS_TOOL_TABLE", "tool")
COSMOS_TEXTDATA_TABLE = os.getenv("COSMOS_TEXTDATA_TABLE", "textdata")
COSMOS_IMAGEDATA_TABLE = os.getenv("COSMOS_IMAGEDATA_TABLE", "imagedata")
COSMOS_AUDIODATA_TABLE = os.getenv("COSMOS_AUDIODATA_TABLE", "audiodata")
COSMOS_VIDEODATA_TABLE = os.getenv("COSMOS_VIDEODATA_TABLE", "videodata")


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


tags_metadata: list[dict] = [
    {
        "name": "Inference",
        "description": """
        Use agents to process multi-modal data for RAG.
        """,
    },
    {
        "name": "CRUD - Assemblies",
        "description": "CRUD endpoints for Assembly model.",
    },
    {
        "name": "CRUD - Tools",
        "description": "CRUD endpoints for Tool model.",
    },
    {
        "name": "CRUD - TextData",
        "description": "CRUD endpoints for TextData model.",
    },
    {
        "name": "CRUD - ImageData",
        "description": "CRUD endpoints for ImageData model.",
    },
    {
        "name": "CRUD - AudioData",
        "description": "CRUD endpoints for AudioData model.",
    },
    {
        "name": "CRUD - VideoData",
        "description": "CRUD endpoints for VideoData model.",
    },
]

description: str = """
    .
"""


app: FastAPI = FastAPI(
    title=__app__,
    version=__version__,
    description=description,
    openapi_tags=tags_metadata,
    openapi_url="/api/v1/openapi.json",
    responses=RESPONSES,  # type: ignore
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError  # pylint: disable=unused-argument
) -> JSONResponse:
    """
    validation_exception_handler Exception handler for validations.

    Args:
        request (Request): the request from the api
        exc (RequestValidationError): the validation raised by the process

    Returns:
        JSONResponse: A json encoded response with the validation errors.
    """

    response_body: ErrorMessage = ErrorMessage(
        success=False,
        type="Validation Error",
        title="Your request parameters didn't validate.",
        detail={"invalid-params": list(exc.errors())},
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(response_body),
    )


@app.exception_handler(ResponseValidationError)
async def response_exception_handler(
    request: Request, exc: ResponseValidationError  # pylint: disable=unused-argument
) -> JSONResponse:
    """
    response_exception_handler Exception handler for response validations.

    Args:
        request (Request): the request from the api
        exc (RequestValidationError): the validation raised by the process

    Returns:
        JSONResponse: A json encoded response with the validation errors.
    """

    response_body: ErrorMessage = ErrorMessage(
        success=False,
        type="Response Error",
        title="Found Errors on processing your requests.",
        detail={"invalid-params": list(exc.errors())},
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(response_body),
    )


@app.post("/evaluate", tags=["Agents", "Assembly"])
async def evaluate_judgment(job: JobResponse) -> JSONResponse:
    """
    Endpoint that evaluates a prompt using a Agent Assembly.
    """
    try:
        final_verdict = await ToolerOrchestrator().run_interaction(assembly=job.assembly_id, prompt=job.prompt)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

    response_body = SuccessMessage(
        title="Evaluation Complete",
        message="Judging completed successfully.",
        content={"assembly_id": job.assembly_id, "result": final_verdict},
    )

    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.get("/assemblies", tags=["CRUD - Assemblies"])
async def list_assemblies_endpoint() -> JSONResponse:
    crud = CosmosCRUD("COSMOS_ASSEMBLY_TABLE")
    items = await crud.list_items()
    response_body = SuccessMessage(
        title=f"{len(items) if items else 0} Assemblies Retrieved",
        message="Successfully retrieved assembly data.",
        content=items,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.post("/assemblies", tags=["CRUD - Assemblies"])
async def create_assembly(assembly: Assembly) -> JSONResponse:
    crud = CosmosCRUD("COSMOS_ASSEMBLY_TABLE")
    created = await crud.create_item(assembly.model_dump())
    response_body = SuccessMessage(
        title="Assembly Created",
        message="Assembly created successfully.",
        content=created,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.put("/assemblies/{assembly_id}", tags=["CRUD - Assemblies"])
async def update_assembly(assembly_id: str, assembly: Assembly) -> JSONResponse:
    crud = CosmosCRUD("COSMOS_ASSEMBLY_TABLE")
    try:
        existing = await crud.read_item(assembly_id)
    except Exception as exc:
        logger.error("Error reading assembly: %s", exc)
        raise HTTPException(status_code=404, detail="Assembly not found.") from exc
    updated = {**existing, **assembly.model_dump()}
    await crud.update_item(assembly_id, updated)
    response_body = SuccessMessage(
        title="Assembly Updated",
        message="Assembly updated successfully.",
        content=updated,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.delete("/assemblies/{assembly_id}", tags=["CRUD - Assemblies"])
async def delete_assembly(assembly_id: str) -> JSONResponse:
    crud = CosmosCRUD("COSMOS_ASSEMBLY_TABLE")
    try:
        await crud.delete_item(assembly_id)
    except Exception as exc:
        logger.error("Error deleting assembly: %s", exc)
        raise HTTPException(status_code=404, detail="Assembly not found.") from exc
    response_body = SuccessMessage(
        title="Assembly Deleted",
        message="Assembly deleted successfully.",
        content={"assembly_id": assembly_id},
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))
