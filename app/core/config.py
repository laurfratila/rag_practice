from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "RAG-Postgres API"
    ENV: str = "dev"

    POSTGRES_DB: str = "ragdb"
    POSTGRES_USER: str = "raguser"
    POSTGRES_PASSWORD: str = "ragpass"
    DB_HOST: str = "db"
    DB_PORT: int = 5432

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.POSTGRES_DB}"
        )

    class Config:
        env_file = ".env"


settings = Settings()
