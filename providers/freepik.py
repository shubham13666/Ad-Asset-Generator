"""Freepik inpainting provider implementation."""

import logging
from typing import Tuple, Dict, Any, Optional
import time
import io

from PIL import Image
import requests

from providers.base import (
    BaseInpaintProvider,
    ProviderError,
    ProviderAuthenticationError,
    ProviderAPIError,
)

logger = logging.getLogger(__name__)


class FreepikProvider(BaseInpaintProvider):
    """
    Freepik provider for background replacement.
    Uses Freepik API to search for suitable backgrounds and composite them.
    
    Cost: Varies based on Freepik subscription/API pricing
    Quality: High (uses professional stock images)
    Speed: 10-20 seconds
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        self.api_key = self.config.get("api_key") or self.config.get("FREEPIK_API_KEY")
        self.base_url = self.config.get("base_url", "https://api.freepik.com/v1")
        self.max_search_results = self.config.get("max_search_results", 5)
        
        if not self.api_key:
            raise ProviderError(
                "Freepik API key not configured. "
                "Set FREEPIK_API_KEY or provider.api_key"
            )
    
    def get_provider_name(self) -> str:
        return "Freepik"
    
    def authenticate(self) -> bool:
        """Authenticate with Freepik API."""
        try:
            # Test authentication by making a simple API call
            headers = {
                "X-Freepik-API-Key": self.api_key,
                "Accept": "application/json"
            }
            
            # Try to get user info or search (lightweight endpoint)
            test_url = f"{self.base_url}/resources/search"
            params = {
                "q": "test",
                "limit": 1
            }
            
            response = requests.get(
                test_url,
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 401:
                raise ProviderAuthenticationError(
                    "Freepik API authentication failed. Check your API key."
                )
            elif response.status_code == 403:
                raise ProviderAuthenticationError(
                    "Freepik API access forbidden. Check your API key permissions."
                )
            elif not response.ok:
                logger.warning(
                    f"Freepik API test returned status {response.status_code}. "
                    f"Continuing anyway..."
                )
            
            logger.info(f"Successfully authenticated with {self.get_provider_name()}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Freepik authentication request failed: {e}")
            raise ProviderAuthenticationError(
                f"Freepik authentication failed: {e}"
            ) from e
        except Exception as e:
            logger.error(f"Freepik authentication error: {e}")
            raise ProviderAuthenticationError(
                f"Freepik authentication failed: {e}"
            ) from e
    
    def get_cost_estimate(self, num_images: int = 1) -> float:
        """
        Freepik pricing varies based on subscription.
        Estimate: $0.10-0.50 per image (depending on license type)
        """
        # Conservative estimate
        return num_images * 0.20
    
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        **kwargs
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Perform inpainting using Freepik API.
        
        Steps:
        1. Search Freepik for suitable background images based on prompt
        2. Download the best matching background
        3. Composite the background with the original image using the mask
        """
        start_time = time.time()
        
        try:
            logger.info(f"Searching Freepik for background: {prompt[:80]}")
            
            # Step 1: Search for background images
            background_image = self._search_and_download_background(prompt, image.size)
            
            if not background_image:
                raise ProviderAPIError(
                    "Failed to find suitable background image on Freepik"
                )
            
            # Step 2: Composite the background with the original image
            logger.info("Compositing background with original image...")
            result_image = self._composite_background(image, mask, background_image)
            
            processing_time = time.time() - start_time
            
            metadata: Dict[str, Any] = {
                "provider": self.get_provider_name(),
                "model_used": "Freepik Stock Images",
                "original_prompt": prompt,
                "processing_time": processing_time,
                "cost": self.get_cost_estimate(1),
                "image_generated": True,
                "source": "freepik_stock"
            }
            
            logger.info(f"✅ Freepik inpainting completed in {processing_time:.2f}s!")
            return result_image, metadata
            
        except Exception as e:
            logger.error(f"Freepik error: {e}", exc_info=True)
            raise ProviderError(f"Freepik error: {e}") from e
    
    def _search_and_download_background(
        self,
        prompt: str,
        target_size: Tuple[int, int]
    ) -> Optional[Image.Image]:
        """
        Search Freepik for background images and download the best match.
        
        Args:
            prompt: Search query
            target_size: Desired image size (width, height)
        
        Returns:
            PIL Image of the background, or None if not found
        """
        try:
            headers = {
                "X-Freepik-API-Key": self.api_key,
                "Accept": "application/json"
            }
            
            # Search for images
            search_url = f"{self.base_url}/resources/search"
            params = {
                "q": prompt,
                "type": "photo",  # or "vector", "psd", etc.
                "orientation": "horizontal",  # or "vertical", "square"
                "limit": self.max_search_results,
                "order": "download"  # Sort by popularity
            }
            
            logger.info(f"Searching Freepik API: {prompt}")
            response = requests.get(
                search_url,
                headers=headers,
                params=params,
                timeout=30
            )
            
            if not response.ok:
                logger.error(
                    f"Freepik search failed: {response.status_code} - {response.text}"
                )
                raise ProviderAPIError(
                    f"Freepik search failed: {response.status_code}"
                )
            
            data = response.json()
            
            # Parse results
            resources = data.get("data", [])
            if not resources:
                logger.warning(f"No results found for query: {prompt}")
                return None
            
            # Get the first result (best match)
            best_match = resources[0]
            resource_id = best_match.get("id")
            download_url = best_match.get("urls", {}).get("preview") or best_match.get("url")
            
            if not download_url:
                logger.warning("No download URL found in search result")
                return None
            
            logger.info(f"Downloading background image: {resource_id}")
            
            # Download the image
            img_response = requests.get(download_url, timeout=30)
            if not img_response.ok:
                raise ProviderAPIError(f"Failed to download image: {img_response.status_code}")
            
            # Load as PIL Image
            background = Image.open(io.BytesIO(img_response.content))
            background = background.convert('RGB')
            
            # Resize to match target size
            background = background.resize(target_size, Image.Resampling.LANCZOS)
            
            logger.info(f"✅ Downloaded and resized background: {background.size}")
            return background
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Freepik API request failed: {e}")
            raise ProviderAPIError(f"Freepik API request failed: {e}") from e
        except Exception as e:
            logger.error(f"Error searching/downloading background: {e}")
            raise ProviderAPIError(f"Failed to get background: {e}") from e
    
    def _composite_background(
        self,
        image: Image.Image,
        mask: Image.Image,
        background: Image.Image
    ) -> Image.Image:
        """
        Composite the background with the original image using the mask.
        
        Args:
            image: Original image with foreground
            mask: Binary mask (white=background to replace, black=foreground to keep)
            background: New background image
        
        Returns:
            Composited image
        """
        try:
            # Ensure images are RGB
            if image.mode != 'RGB':
                if image.mode == 'RGBA':
                    # Composite on white background first
                    rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                    rgb_image.paste(image, mask=image.split()[3])
                    image = rgb_image
                else:
                    image = image.convert('RGB')
            
            # Ensure background matches image size
            if background.size != image.size:
                background = background.resize(image.size, Image.Resampling.LANCZOS)
            
            # Ensure mask is grayscale
            if mask.mode != 'L':
                mask = mask.convert('L')
            
            # Create result image
            result = image.copy()
            
            # Apply mask: where mask is white (255), use background; where black (0), keep original
            # Invert mask for compositing (white in mask = replace with background)
            mask_inverted = mask.point(lambda p: 255 - p)
            
            # Composite: paste background using inverted mask
            result.paste(background, mask=mask_inverted)
            
            logger.info("✅ Background composited successfully")
            return result
            
        except Exception as e:
            logger.error(f"Compositing error: {e}")
            raise ProviderAPIError(f"Failed to composite background: {e}") from e

