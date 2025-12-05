"""Configuration management for the background replacement system."""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class ProviderConfig(BaseModel):
    """Configuration for a single AI provider."""
    name: str
    enabled: bool = True
    timeout: int = Field(default=60, ge=1, le=300)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay: int = Field(default=2, ge=1, le=60)


class OpenAIConfig(ProviderConfig):
    """OpenAI ChatGPT provider configuration."""
    api_key: str
    model: str = "gpt-4o-mini"


class AlibabaConfig(ProviderConfig):
    """Alibaba Tongyi provider configuration."""
    api_key: str
    region: str = "cn-shanghai"


class StabilityConfig(ProviderConfig):
    """Stability AI provider configuration."""
    api_key: str
    endpoint: str = "https://api.stability.ai/v2beta"


class AppConfig(BaseModel):
    """Main application configuration."""
    
    # Provider configuration
    primary_provider: str = Field(default="tencent")
    enable_fallback: bool = Field(default=True)
    fallback_providers: List[str] = Field(default_factory=lambda: ["alibaba", "stability"])
    
    # Provider credentials
    # tencent: Optional[TencentConfig] = None
    alibaba: Optional[AlibabaConfig] = None
    stability: Optional[StabilityConfig] = None
    openai_chat: Optional[OpenAIConfig] = None
    # Google Cloud configuration
    gcp_project_id: Optional[str] = None
    google_credentials_path: Optional[str] = None
    google_sheets_credentials_path: Optional[str] = None
    google_drive_root_folder_id: Optional[str] = None
    google_drive_logs_folder_name: str = "Background_Generation_Logs"
    
    # Mask generation configuration
    removebg_api_key: Optional[str] = None  # Optional fallback ($0.01/image)
    
    # Logging configuration
    log_level: str = Field(default="INFO")
    log_provider_details: bool = Field(default=True)
    
    # Cloud Run configuration
    function_timeout: int = Field(default=540, ge=1, le=3600)
    memory_limit: str = "2Gi"
    cpu_limit: str = "2"
    
    @field_validator('primary_provider')
    @classmethod
    def validate_primary_provider(cls, v: str) -> str:
        """Validate primary provider name."""
        valid_providers = ["tencent", "alibaba", "stability","openai_chat"]
        if v.lower() not in valid_providers:
            raise ValueError(f"Primary provider must be one of {valid_providers}")
        return v.lower()
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @field_validator('fallback_providers')
    @classmethod
    def validate_fallback_providers(cls, v: List[str]) -> List[str]:
        """Validate fallback provider names."""
        valid_providers = ["tencent", "alibaba", "stability"]
        return [p.lower() for p in v if p.lower() in valid_providers]


def load_config_from_env() -> AppConfig:
    """Load configuration from environment variables."""
    logger.info("Loading configuration from environment variables")
    
    # Primary provider configuration
    primary_provider = os.getenv("INPAINT_PROVIDER", "tencent").lower()
    enable_fallback = os.getenv("ENABLE_FALLBACK", "true").lower() == "true"
    fallback_providers_str = os.getenv("FALLBACK_PROVIDERS", "alibaba,stability")
    fallback_providers = [p.strip() for p in fallback_providers_str.split(",") if p.strip()]
    
    # OpenAI ChatGPT configuration
    openai_chat_config = None
    if os.getenv("OPENAI_API_KEY"):
        openai_chat_config = OpenAIConfig(
            name="tencent",
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            timeout=int(os.getenv("INPAINT_TIMEOUT_SECONDS", "60")),
            max_retries=int(os.getenv("MAX_RETRY_ATTEMPTS", "3")),
            retry_delay=int(os.getenv("RETRY_DELAY_SECONDS", "2"))
        )
    # # Tencent configuration
    # tencent_config = None
    # if os.getenv("TENCENT_SECRET_ID") and os.getenv("TENCENT_SECRET_KEY"):
    #     tencent_config = TencentConfig(
    #         name="tencent",
    #         secret_id=os.getenv("TENCENT_SECRET_ID"),
    #         secret_key=os.getenv("TENCENT_SECRET_KEY"),
    #         region=os.getenv("TENCENT_REGION", "ap-shanghai"),
    #         timeout=int(os.getenv("INPAINT_TIMEOUT_SECONDS", "60")),
    #         max_retries=int(os.getenv("MAX_RETRY_ATTEMPTS", "3")),
    #         retry_delay=int(os.getenv("RETRY_DELAY_SECONDS", "2"))
    #     )
    
    # Alibaba configuration
    alibaba_config = None
    if os.getenv("ALIBABA_API_KEY"):
        alibaba_config = AlibabaConfig(
            name="alibaba",
            api_key=os.getenv("ALIBABA_API_KEY"),
            region=os.getenv("ALIBABA_REGION", "cn-shanghai"),
            timeout=int(os.getenv("INPAINT_TIMEOUT_SECONDS", "60")),
            max_retries=int(os.getenv("MAX_RETRY_ATTEMPTS", "3")),
            retry_delay=int(os.getenv("RETRY_DELAY_SECONDS", "2"))
        )
    
    # Stability AI configuration
    stability_config = None
    if os.getenv("STABILITY_API_KEY"):
        stability_config = StabilityConfig(
            name="stability",
            api_key=os.getenv("STABILITY_API_KEY"),
            endpoint=os.getenv("STABILITY_ENDPOINT", "https://api.stability.ai/v2beta"),
            timeout=int(os.getenv("INPAINT_TIMEOUT_SECONDS", "60")),
            max_retries=int(os.getenv("MAX_RETRY_ATTEMPTS", "3")),
            retry_delay=int(os.getenv("RETRY_DELAY_SECONDS", "2"))
        )
    
    # Google Cloud configuration
    gcp_project_id = os.getenv("GCP_PROJECT_ID")
    google_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    google_sheets_credentials_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
    google_drive_root_folder_id = os.getenv("GOOGLE_DRIVE_ROOT_FOLDER_ID")
    google_drive_logs_folder_name = os.getenv("GOOGLE_DRIVE_LOGS_FOLDER_NAME", "Background_Generation_Logs")
    
    # Mask generation configuration (Remove.bg is optional fallback)
    removebg_api_key = os.getenv("REMOVEBG_API_KEY")
    
    # Logging configuration
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_provider_details = os.getenv("LOG_PROVIDER_DETAILS", "true").lower() == "true"
    
    # Cloud Run configuration
    function_timeout = int(os.getenv("FUNCTION_TIMEOUT", "540"))
    memory_limit = os.getenv("MEMORY_LIMIT", "2Gi")
    cpu_limit = os.getenv("CPU_LIMIT", "2")
    
    config = AppConfig(
        primary_provider=primary_provider,
        enable_fallback=enable_fallback,
        fallback_providers=fallback_providers,
        alibaba=alibaba_config,
        stability=stability_config,
        openai_chat=openai_chat_config,
        gcp_project_id=gcp_project_id,
        google_credentials_path=google_credentials_path,
        google_sheets_credentials_path=google_sheets_credentials_path,
        google_drive_root_folder_id=google_drive_root_folder_id,
        google_drive_logs_folder_name=google_drive_logs_folder_name,
        removebg_api_key=removebg_api_key,
        log_level=log_level,
        log_provider_details=log_provider_details,
        function_timeout=function_timeout,
        memory_limit=memory_limit,
        cpu_limit=cpu_limit
    )
    
    logger.info(f"Configuration loaded: primary_provider={primary_provider}, fallback_enabled={enable_fallback}")
    return config


def load_config_from_json(config_path: Optional[str] = None) -> Optional[AppConfig]:
    """Load configuration from JSON file (optional fallback)."""
    if config_path is None:
        config_path = os.getenv("CONFIG_FILE_PATH", "config.json")
        
    config_file = Path(config_path)
    if not config_file.exists():
        logger.debug(f"Config file not found at {config_path}, skipping JSON config")
        return None
    
    try:
        logger.info(f"Loading configuration from JSON file: {config_path}")
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Expand environment variables in config values
        config_data = _expand_env_vars(config_data)
        
        # Parse provider configurations
        providers_config = config_data.get("providers", {})
        
        tencent_config = None
        if "tencent" in providers_config:
            tc = providers_config["tencent"]
            tencent_config = TencentConfig(
                name="tencent",
                secret_id=tc.get("secret_id", ""),
                secret_key=tc.get("secret_key", ""),
                region=tc.get("region", "ap-shanghai"),
                timeout=tc.get("timeout", 60),
                max_retries=tc.get("max_retries", 3),
                retry_delay=tc.get("retry_delay", 2)
            )
        
        alibaba_config = None
        if "alibaba" in providers_config:
            ac = providers_config["alibaba"]
            alibaba_config = AlibabaConfig(
                name="alibaba",
                api_key=ac.get("api_key", ""),
                region=ac.get("region", "cn-shanghai"),
                timeout=ac.get("timeout", 60),
                max_retries=ac.get("max_retries", 3),
                retry_delay=ac.get("retry_delay", 2)
            )
        
        stability_config = None
        if "stability" in providers_config:
            sc = providers_config["stability"]
            stability_config = StabilityConfig(
                name="stability",
                api_key=sc.get("api_key", ""),
                endpoint=sc.get("endpoint", "https://api.stability.ai/v2beta"),
                timeout=sc.get("timeout", 60),
                max_retries=sc.get("max_retries", 3),
                retry_delay=sc.get("retry_delay", 2)
            )
        
        inpaint_config = config_data.get("inpaint_provider", {})
        
        config = AppConfig(
            primary_provider=inpaint_config.get("primary", "tencent"),
            enable_fallback=inpaint_config.get("fallback_enabled", True),
            fallback_providers=inpaint_config.get("fallback_order", ["alibaba", "stability"]),
            tencent=tencent_config,
            alibaba=alibaba_config,
            stability=stability_config,
            gcp_project_id=config_data.get("gcp", {}).get("project_id"),
            google_credentials_path=config_data.get("gcp", {}).get("credentials_path"),
            google_drive_root_folder_id=config_data.get("google_drive", {}).get("root_folder_id"),
            google_drive_logs_folder_name=config_data.get("google_drive", {}).get("logs_folder_name", "Background_Generation_Logs"),
            removebg_api_key=config_data.get("mask_generation", {}).get("removebg_api_key"),
            log_level=config_data.get("logging", {}).get("level", "INFO"),
            log_provider_details=config_data.get("logging", {}).get("provider_details", True),
            function_timeout=config_data.get("cloud_run", {}).get("timeout", 540),
            memory_limit=config_data.get("cloud_run", {}).get("memory", "2Gi"),
            cpu_limit=config_data.get("cloud_run", {}).get("cpu", "2")
        )
        
        logger.info(f"Configuration loaded from JSON: primary_provider={config.primary_provider}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading config from JSON: {e}")
        return None


def _expand_env_vars(data: Any) -> Any:
    """Recursively expand environment variables in config values."""
    if isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
        env_var = data[2:-1]
        return os.getenv(env_var, data)
    return data


def get_config() -> AppConfig:
    """
    Get application configuration.
    Tries JSON config first, falls back to environment variables.
    """
    # Try JSON config first
    json_config = load_config_from_json()
    if json_config:
        return json_config
    
    # Fall back to environment variables
    return load_config_from_env()


def validate_config(config: AppConfig) -> bool:
    """
    Validate that required configuration is present.
    Returns True if valid, raises ValueError if invalid.
    """
    errors = []
    
    # Validate primary provider has credentials
    if config.primary_provider == "tencent" and not config.tencent:
        errors.append("Tencent provider selected but credentials not configured")
    elif config.primary_provider == "alibaba" and not config.alibaba:
        errors.append("Alibaba provider selected but credentials not configured")
    elif config.primary_provider == "stability" and not config.stability:
        errors.append("Stability AI provider selected but credentials not configured")
    
    # Validate Google Cloud credentials
    if not config.google_credentials_path and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        errors.append("Google Cloud credentials not configured (GOOGLE_APPLICATION_CREDENTIALS)")
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Configuration validation passed")
    return True

