from fastapi import FastAPI
from app.api.medical_query_routes import router as medical_query_router
from app.utils.logger import get_logger

logger = get_logger(__name__)

def create_app():
    app = FastAPI()
    
    # Register Routers
    app.include_router(medical_query_router, prefix='/api')
    
    logger.debug("FastAPI application created and router registered")
    return app