"""Base abstract class for AI inpainting providers."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Base exception for provider-related errors."""
    pass


class ProviderAuthenticationError(ProviderError):
    """Raised when provider authentication fails."""
    pass


class ProviderAPIError(ProviderError):
    """Raised when provider API call fails."""
    pass


class ProviderTimeoutError(ProviderError):
    """Raised when provider request times out."""
    pass


class BaseInpaintProvider(ABC):
    """
    Abstract base class for all inpainting providers.
    All providers must implement this interface.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the provider.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config or {}
        self._authenticated = False
        self._provider_name = self.__class__.__name__
    
    @abstractmethod
    def authenticate(self) -> bool:
        """
        Authenticate with the provider API.
        
        Returns:
            True if authentication successful, False otherwise
        
        Raises:
            ProviderAuthenticationError: If authentication fails
        """
        pass
    
    @abstractmethod
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        **kwargs
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Perform inpainting on an image using the provided mask and prompt.
        
        Args:
            image: Input image (PIL Image)
            mask: Binary mask (white=area to replace, black=keep) (PIL Image)
            prompt: Text prompt describing desired background
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Tuple of (inpainted_image, metadata_dict)
            - inpainted_image: PIL Image with new background
            - metadata: Dictionary with processing metadata:
              - cost: float - Cost charged for this operation
              - processing_time: float - Time taken in seconds
              - model_used: str - Model/version used
              - any other provider-specific metadata
        
        Raises:
            ProviderAPIError: If API call fails
            ProviderTimeoutError: If request times out
            ProviderError: For other provider errors
        """
        pass
    
    @abstractmethod
    def get_cost_estimate(self, num_images: int = 1) -> float:
        """
        Calculate estimated cost for processing specified number of images.
        
        Args:
            num_images: Number of images to process
        
        Returns:
            Estimated cost in USD
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get the display name of the provider.
        
        Returns:
            Provider name string
        """
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status and health of the provider.
        
        Returns:
            Dictionary with status information:
            - authenticated: bool
            - provider_name: str
            - available: bool
            - any other provider-specific status
        """
        return {
            "authenticated": self._authenticated,
            "provider_name": self.get_provider_name(),
            "available": self._authenticated
        }
    
    def _ensure_authenticated(self) -> None:
        """
        Ensure provider is authenticated before making API calls.
        Attempts authentication if not already done.
        
        Raises:
            ProviderAuthenticationError: If authentication fails
        """
        if not self._authenticated:
            logger.info(f"Authenticating with {self.get_provider_name()}...")
            if not self.authenticate():
                raise ProviderAuthenticationError(
                    f"Failed to authenticate with {self.get_provider_name()}"
                )
            self._authenticated = True
            logger.info(f"Successfully authenticated with {self.get_provider_name()}")
    
    def _validate_image(self, image: Image.Image) -> None:
        """
        Validate input image format and size.
        
        Args:
            image: PIL Image to validate
        
        Raises:
            ValueError: If image is invalid
        """
        if not isinstance(image, Image.Image):
            raise ValueError("Image must be a PIL Image instance")
        
        width, height = image.size
        if width < 64 or height < 64:
            raise ValueError(f"Image too small: {width}x{height} (minimum 64x64)")
        
        if width > 4096 or height > 4096:
            raise ValueError(f"Image too large: {width}x{height} (maximum 4096x4096)")
    
    def _validate_mask(self, mask: Image.Image, image: Image.Image) -> None:
        """
        Validate mask format and dimensions match image.
        
        Args:
            mask: PIL Image mask to validate
            image: Original image for dimension comparison
        
        Raises:
            ValueError: If mask is invalid
        """
        if not isinstance(mask, Image.Image):
            raise ValueError("Mask must be a PIL Image instance")
        
        if mask.size != image.size:
            raise ValueError(
                f"Mask size {mask.size} does not match image size {image.size}"
            )
        
        # Convert to grayscale if needed
        if mask.mode != 'L':
            mask = mask.convert('L')
    
    def _prepare_image_for_api(self, image: Image.Image) -> bytes:
        """
        Convert PIL Image to bytes format suitable for API.
        
        Args:
            image: PIL Image to convert
        
        Returns:
            Image bytes in JPEG format
        """
        import io
        
        # Convert RGBA to RGB if needed
        if image.mode in ('RGBA', 'LA', 'P'):
            # Create white background
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                rgb_image.paste(image, mask=image.split()[3])  # Use alpha channel as mask
            else:
                rgb_image.paste(image)
            image = rgb_image
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        return buffer.getvalue()
    
    def _prepare_mask_for_api(self, mask: Image.Image) -> bytes:
        """
        Convert PIL Image mask to bytes format suitable for API.
        
        Args:
            mask: PIL Image mask to convert
        
        Returns:
            Mask bytes in PNG format
        """
        import io
        
        # Ensure mask is grayscale
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        # Save to bytes
        buffer = io.BytesIO()
        mask.save(buffer, format='PNG')
        return buffer.getvalue()



