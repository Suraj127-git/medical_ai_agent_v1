from pydantic import (
    BaseModel, 
    Field, 
    field_validator,
    AmqpDsn,
    PostgresDsn,
    RedisDsn,
    AnyUrl
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Dict, Any

class QdrantConfig(BaseModel):
    url: AnyUrl = Field(default="http://localhost:6333", description="Qdrant server URL")
    collection: str = Field(default="healthcare_collection", min_length=3)
    embedding_dim: int = Field(default=384, ge=128, le=2048)
    
    @field_validator('url')
    def validate_url(cls, v):
        if "localhost" in v.host and v.scheme != "http":
            raise ValueError("Localhost must use HTTP protocol")
        return v

class OllamaConfig(BaseModel):
    model: str = Field(default="llama3:latest", pattern=r"^[a-zA-Z0-9\-_:]+$")  # Use correct model name
    base_url: AnyUrl = Field(default="http://192.168.1.6:11434")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=50, le=4000)

class HuggingFaceConfig(BaseModel):
    model: str = Field(default="llama3:latest", pattern=r"^[a-zA-Z0-9\-_:]+$")
    api_key: str = Field(default="", min_length=3)

class TextProcessingConfig(BaseModel):
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=200)
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore"
    )

    qdrant: QdrantConfig = QdrantConfig()
    ollama: OllamaConfig = OllamaConfig()
    text_processing: TextProcessingConfig = TextProcessingConfig()
    data_path: str = Field(default="healthcare_data.txt", min_length=3)
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    @property
    def qdrant_client_config(self) -> Dict[str, Any]:
        return {
            "host": self.qdrant.url.host,
            "port": self.qdrant.url.port or 6333
        }
    
    @property
    def text_splitter_config(self) -> Dict[str, Any]:
        return self.text_processing.model_dump()

# Initialize configuration
config = AppConfig()