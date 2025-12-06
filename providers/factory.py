"""Provider factory and fallback system."""

import logging
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image

from providers.base import (
    BaseInpaintProvider,
    ProviderError,
    ProviderAuthenticationError,
    ProviderAPIError
)
from providers.openai_chat import OpenAIChatProvider
from providers.tencent_hunyuan import TencentHunyuanProvider
from providers.metrics import ProviderMetrics, ProviderAttempt
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional providers (Phase 8) - conditional imports
try:
    from providers.alibaba_tongyi import AlibabaTongyiProvider
    ALIBABA_AVAILABLE = True
except ImportError:
    ALIBABA_AVAILABLE = False

try:
    from providers.stability_ai import StabilityAIProvider
    STABILITY_AVAILABLE = True
except ImportError:
    STABILITY_AVAILABLE = False

# Google Gemini provider
try:
    from providers.google_gemini import GoogleGeminiProvider
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Freepik provider
try:
    from providers.freepik import FreepikProvider
    FREEPIK_AVAILABLE = True
except ImportError:
    FREEPIK_AVAILABLE = False


class ProviderWithFallback:
    """
    Wrapper class that implements automatic fallback between providers.
    Tries primary provider first, then falls back to backup providers in order.
    """
    
    def __init__(
        self,
        primary_provider: BaseInpaintProvider,
        fallback_providers: List[BaseInpaintProvider],
        metrics: Optional[ProviderMetrics] = None
    ):
        """
        Initialize provider with fallback chain.
        
        Args:
            primary_provider: Primary provider to try first
            fallback_providers: List of fallback providers to try in order
            metrics: Optional metrics tracker
        """
        self.primary_provider = primary_provider
        self.fallback_providers = fallback_providers
        self.metrics = metrics or ProviderMetrics()
        self.all_providers = [primary_provider] + fallback_providers
    
    def get_provider_name(self) -> str:
        """Get display name for the provider chain."""
        fallback_names = ", ".join(p.get_provider_name() for p in self.fallback_providers)
        return f"{self.primary_provider.get_provider_name()} (fallback: {fallback_names})"
    
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        **kwargs
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Perform inpainting with automatic fallback.
        
        Args:
            image: Input image
            mask: Binary mask
            prompt: Text prompt
            **kwargs: Additional parameters
        
        Returns:
            Tuple of (inpainted_image, metadata_dict)
            Metadata includes which provider succeeded
        
        Raises:
            ProviderError: If all providers fail
        """
        errors = []
        
        # Try primary provider first
        providers_to_try = [self.primary_provider] + self.fallback_providers
        
        for provider in providers_to_try:
            provider_name = provider.get_provider_name()
            start_time = datetime.now()
            
            try:
                logger.info(f"Attempting inpainting with {provider_name}...")
                
                # Attempt inpainting
                result_image, metadata = provider.inpaint(image, mask, prompt, **kwargs)
                
                # Calculate duration
                duration = (datetime.now() - start_time).total_seconds()
                cost = metadata.get('cost', provider.get_cost_estimate())
                
                # Record successful attempt
                self.metrics.record_attempt(
                    provider_name=provider_name,
                    success=True,
                    duration=duration,
                    cost=cost
                )
                
                # Update metadata with provider info
                metadata['provider_used'] = provider_name
                metadata['fallback_used'] = provider != self.primary_provider
                metadata['providers_attempted'] = [
                    p.get_provider_name() for p in providers_to_try[:providers_to_try.index(provider) + 1]
                ]
                
                if provider != self.primary_provider:
                    logger.info(f"Fallback successful: {provider_name} completed the request")
                else:
                    logger.info(f"Primary provider {provider_name} completed successfully")
                
                return result_image, metadata
                
            except ProviderAuthenticationError as e:
                error_msg = f"Authentication failed for {provider_name}: {str(e)}"
                logger.error(error_msg)
                errors.append((provider_name, error_msg))
                
                duration = (datetime.now() - start_time).total_seconds()
                self.metrics.record_attempt(
                    provider_name=provider_name,
                    success=False,
                    duration=duration,
                    cost=0.0,
                    error_message=error_msg
                )
                
                # Authentication errors are usually fatal - don't retry
                continue
                
            except (ProviderAPIError, ProviderError) as e:
                error_msg = f"{provider_name} failed: {str(e)}"
                logger.warning(error_msg)
                errors.append((provider_name, error_msg))
                
                duration = (datetime.now() - start_time).total_seconds()
                cost = provider.get_cost_estimate()  # May have been charged even on failure
                
                self.metrics.record_attempt(
                    provider_name=provider_name,
                    success=False,
                    duration=duration,
                    cost=cost,
                    error_message=error_msg
                )
                
                # Continue to next provider
                continue
                
            except Exception as e:
                error_msg = f"Unexpected error with {provider_name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                errors.append((provider_name, error_msg))
                
                duration = (datetime.now() - start_time).total_seconds()
                self.metrics.record_attempt(
                    provider_name=provider_name,
                    success=False,
                    duration=duration,
                    cost=0.0,
                    error_message=error_msg
                )
                
                # Continue to next provider
                continue
        
        # All providers failed
        error_summary = "; ".join(f"{name}: {msg}" for name, msg in errors)
        logger.error(f"All providers failed. Errors: {error_summary}")
        
        raise ProviderError(
            f"All providers failed. Attempted: {', '.join(p.get_provider_name() for p in providers_to_try)}. "
            f"Errors: {error_summary}"
        )
    
    def get_cost_estimate(self, num_images: int = 1) -> float:
        """
        Get cost estimate from primary provider.
        
        Args:
            num_images: Number of images
        
        Returns:
            Cost estimate in USD
        """
        return self.primary_provider.get_cost_estimate(num_images)
    
    def get_metrics(self) -> ProviderMetrics:
        """Get metrics tracker."""
        return self.metrics


class InpaintingProviderFactory:
    """Factory for creating inpainting providers."""
    
    _provider_registry: Dict[str, type] = {
        'openai_chat': OpenAIChatProvider,
    }

    # Register optional providers if available
    if ALIBABA_AVAILABLE:
        _provider_registry['alibaba'] = AlibabaTongyiProvider

    if STABILITY_AVAILABLE:
        _provider_registry['stability'] = StabilityAIProvider
    
    if GEMINI_AVAILABLE:
        _provider_registry['google_gemini'] = GoogleGeminiProvider
        _provider_registry['gemini'] = GoogleGeminiProvider  # Alias
    
    if FREEPIK_AVAILABLE:
        _provider_registry['freepik'] = FreepikProvider
        
    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """
        Register a new provider class.
        
        Args:
            name: Provider identifier (e.g., 'tencent')
            provider_class: Provider class that extends BaseInpaintProvider
        """
        if not issubclass(provider_class, BaseInpaintProvider):
            raise ValueError(f"Provider class must extend BaseInpaintProvider")
        
        cls._provider_registry[name.lower()] = provider_class
        logger.info(f"Registered provider: {name}")

    @classmethod
    def create_provider(
        cls,
        provider_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseInpaintProvider:
        """
        Create a single provider instance.
        
        Args:
            provider_name: Provider identifier (e.g., 'tencent')
                            If None, reads from environment/config
            config: Configuration dictionary for the provider
        
        Returns:
            Provider instance
        
        Raises:
            ValueError: If provider name is invalid or not registered
        """
        if provider_name is None:
            # Try to read from config or environment
            import os
            provider_name = os.getenv('INPAINT_PROVIDER', 'tencent').lower()
        
        provider_name = provider_name.lower()
        
        if provider_name not in cls._provider_registry:
            available = ', '.join(cls._provider_registry.keys())
            raise ValueError(
                f"Provider '{provider_name}' not found. Available providers: {available}"
            )
        
        provider_class = cls._provider_registry[provider_name]
        provider_config = config or {}
        
        logger.info(f"Creating provider: {provider_name}")
        
        try:
            provider = provider_class(provider_config)
            return provider
        except Exception as e:
            logger.error(f"Failed to create provider {provider_name}: {str(e)}")
            raise

    @classmethod
    def create_with_fallback(
        cls,
        primary_provider_name: Optional[str] = None,
        fallback_provider_names: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[ProviderMetrics] = None
    ) -> ProviderWithFallback:
        """
        Create provider with automatic fallback chain.
        
        Args:
            primary_provider_name: Primary provider identifier
            fallback_provider_names: List of fallback provider identifiers
            config: Configuration dictionary (may contain provider-specific configs)
            metrics: Optional metrics tracker
        
        Returns:
            ProviderWithFallback instance
        
        Raises:
            ValueError: If providers cannot be created
        """
        import os
        
        # Get primary provider name
        if primary_provider_name is None:
            primary_provider_name = os.getenv('INPAINT_PROVIDER', 'tencent').lower()
        
        primary_provider_name = primary_provider_name.lower()
        
        # Get fallback providers
        if fallback_provider_names is None:
            fallback_str = os.getenv('FALLBACK_PROVIDERS', 'alibaba,stability')
            fallback_provider_names = [
                p.strip() for p in fallback_str.split(',') if p.strip()
            ]
        
        # Create primary provider
        primary_config = cls._extract_provider_config(config or {}, primary_provider_name)
        primary_provider = cls.create_provider(primary_provider_name, primary_config)
        
        # Create fallback providers
        fallback_providers = []
        for fallback_name in fallback_provider_names:
            fallback_name = fallback_name.lower()
            if fallback_name == primary_provider_name:
                logger.warning(f"Skipping fallback provider {fallback_name} (same as primary)")
                continue
            
            try:
                fallback_config = cls._extract_provider_config(config or {}, fallback_name)
                fallback_provider = cls.create_provider(fallback_name, fallback_config)
                fallback_providers.append(fallback_provider)
                logger.info(f"Added fallback provider: {fallback_name}")
            except Exception as e:
                logger.warning(
                    f"Failed to create fallback provider {fallback_name}: {str(e)}. "
                    f"Skipping this provider."
                )
                continue
        
        if not fallback_providers:
            logger.warning("No fallback providers available. Using primary provider only.")
        
        return ProviderWithFallback(
            primary_provider=primary_provider,
            fallback_providers=fallback_providers,
            metrics=metrics or ProviderMetrics()
        )

    @classmethod
    def _extract_provider_config(
        cls,
        config: Dict[str, Any],
        provider_name: str
    ) -> Dict[str, Any]:
        """
        Extract provider-specific configuration from main config.
        
        Args:
            config: Main configuration dictionary
            provider_name: Provider identifier
        
        Returns:
            Provider-specific configuration dictionary
        """
        provider_config = {}
        
        # Check for provider-specific config key
        if provider_name in config:
            provider_config.update(config[provider_name])
        
        # Also check for direct config keys
        if provider_name == 'tencent':
            provider_config.setdefault('secret_id', config.get('tencent_secret_id'))
            provider_config.setdefault('secret_key', config.get('tencent_secret_key'))
            provider_config.setdefault('region', config.get('tencent_region', 'ap-shanghai'))
        elif provider_name == 'alibaba':
            provider_config.setdefault('api_key', config.get('alibaba_api_key'))
            provider_config.setdefault('region', config.get('alibaba_region', 'cn-shanghai'))
        elif provider_name == 'stability':
            provider_config.setdefault('api_key', config.get('stability_api_key'))
            provider_config.setdefault('endpoint', config.get('stability_endpoint'))
        elif provider_name in ('google_gemini', 'gemini'):
            provider_config.setdefault('api_key', config.get('google_gemini_api_key'))
            provider_config.setdefault('model', config.get('gemini_model', 'gemini-2.5-flash'))
            provider_config.setdefault('image_model', config.get('gemini_image_model', 'gemini-2.5-flash-image'))
            provider_config.setdefault('imagen_model', config.get('imagen_model', 'imagegeneration@006'))
        elif provider_name == 'freepik':
            provider_config.setdefault('api_key', config.get('freepik_api_key'))
            provider_config.setdefault('base_url', config.get('freepik_base_url', 'https://api.freepik.com/v1'))
            provider_config.setdefault('max_search_results', config.get('freepik_max_results', 5))
        
        # Add common settings
        provider_config.setdefault('timeout', config.get('inpaint_timeout_seconds', 60))
        provider_config.setdefault('max_retries', config.get('max_retry_attempts', 3))
        provider_config.setdefault('retry_delay', config.get('retry_delay_seconds', 2))
        
        return provider_config

