# main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from routers.src.face import mediaPipe as mp
from routers.src.face import fatigue as ft
from routers.src.audio import whisper_small_en as ws
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import File, UploadFile
import logging
from logger.logger import logger

# Configure the body size limit - ADD THIS
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.requests import Request as StarletteRequest
import uvicorn

# Increase the max size to 100MB (or your desired limit)
MAX_SIZE = 100 * 1024 * 1024  # 100MB in bytes

# Override the default file size handling
StarletteUploadFile.spool_max_size = MAX_SIZE
StarletteRequest.body_max_size = MAX_SIZE

logging.basicConfig(
    level=logging.INFO,  # or DEBUG
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

class ErrorLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            if response.status_code == 413:
                logger.error(f"413 Payload Too Large error occurred for path: {request.url.path}")
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "Payload Too Large",
                        "message": f"The uploaded file exceeds the maximum allowed size of {MAX_SIZE/(1024*1024)}MB",
                        "path": str(request.url.path)
                    }
                )
            return response
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal Server Error", "message": str(e)}
            )

app = FastAPI(
    title="Modular FastAPI Service",
    version="1.0.0"
)

# Add error logging middleware
app.add_middleware(ErrorLoggingMiddleware)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=[
        "Content-Type", 
        "Authorization",
        "Accept",
        "Content-Disposition",
        "Content-Length",
        "X-Requested-With",
        "multipart/form-data",
        "boundary"
    ],
    expose_headers=["Content-Disposition"]
)

# Configure trusted hosts middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=[
        "104.214.171.210",    # Your VM IP
        "localhost",          # Local development
        "192.168.100.8",      # Local development
        "192.168.100.7",
        "192.168.1.6",          # Local development
    ]
)

# Register routers
app.include_router(mp.router)
app.include_router(ws.router)
app.include_router(ft.router)

if __name__ == "__main__":
    # If running directly with python, use these settings
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=5000,  # Changed from 8000 to 5000
        reload=True,
        limit_concurrency=20,
        limit_max_requests=100,
        timeout_keep_alive=120
    )