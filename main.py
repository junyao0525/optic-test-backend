# main.py
from fastapi import FastAPI
from routers import mtcnn as ml
from routers import mediaPipe as mp
from routers import haar_cascades as hc
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Modular FastAPI Service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(mp.router)
app.include_router(hc.router)
app.include_router(ml.router)

