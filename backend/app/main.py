from fastapi import FastAPI

from app.api.routes import health, stylyze

app = FastAPI(
    title="stylyze API",
    description="Zero-shot arbitrary neural style transfer",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(stylyze.router, prefix="/api")
