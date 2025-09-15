from fastapi import FastAPI
from sqlalchemy import create_engine, text
from app.core.config import Settings, settings

app = FastAPI(title=settings.APP_NAME)
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)


@app.get("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            # Check pgvector exists
            conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
        return {"status": "ok", "db": "up", "pgvector": "enabled"}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/version")
def version():
    return {"app": settings.APP_NAME, "env": settings.ENV}
