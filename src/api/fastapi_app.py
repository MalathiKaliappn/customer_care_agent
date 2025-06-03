from fastapi import FastAPI
from src.api.upload_router import router as upload_router
from src.api.chat_router import router as chat_router
from src.api.evaluate_router import router as evaluate_router
from src.api.health import router as health_router

app = FastAPI(
    title="Customer Care Assistant API",
    version="1.0.0"
)

app.include_router(upload_router, prefix="/api/upload", tags=["Upload"])
app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
app.include_router(evaluate_router, prefix="/api/evaluate", tags=["Evaluation"])
app.include_router(health_router, prefix="/api/health", tags=["Health"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Customer Care Assistant!"}
