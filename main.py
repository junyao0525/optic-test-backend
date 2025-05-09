# main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from routers import mtcnn as ml
from routers import mediaPipe as mp
from routers import haar_cascades as hc
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

import logging
from logger.logger import logger
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
                        "message": "The uploaded file exceeds the maximum allowed size of 10MB",
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
    allow_origins=[
        "http://104.214.171.210",     # Your VM IP
        "http://104.214.171.210:8000", # Your VM IP with port
        "http://localhost:3000",      # Local development
        "http://localhost:8000",      # Local FastAPI
    ],
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
    ]
)

# Register routers

app.include_router(mp.router)
app.include_router(hc.router)
app.include_router(ml.router)

