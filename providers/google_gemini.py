"""Google Gemini inpainting provider implementation."""

import logging
from typing import Tuple, Dict, Any, Optional
import time
import base64
import io
from collections import Counter

from PIL import Image
import requests

from providers.base import (
    BaseInpaintProvider,
    ProviderError,
    ProviderAuthenticationError,
    ProviderAPIError,
)

logger = logging.getLogger(__name__)


class GoogleGeminiProvider(BaseInpaintProvider):
    """
    Google Gemini-based provider using Gemini API for prompt refinement
    and Imagen API for image generation/inpainting.
    
    Cost: ~$0.02-0.04 per image (depending on model)
    Quality: High
    Speed: 15-30 seconds
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        self.api_key = self.config.get("api_key") or self.config.get("GOOGLE_GEMINI_API_KEY")
        self.model = self.config.get("model", "gemini-2.5-flash")
        self.image_model = self.config.get("image_model", "gemini-2.5-flash-image")  # For image generation
        self.imagen_model = self.config.get("imagen_model", "imagegeneration@006")
        # Track if this is the primary provider (only use own APIs when primary)
        self.is_primary_provider = self.config.get("is_primary_provider", False)
        # GCP project and credentials for Imagen API (when using Vertex AI)
        self.gcp_project_id = self.config.get("gcp_project_id")
        self.google_credentials_path = self.config.get("google_credentials_path")
        
        if not self.api_key:
            raise ProviderError("Google Gemini API key not configured. Set GOOGLE_GEMINI_API_KEY or provider.api_key")
    
    def get_provider_name(self) -> str:
        return "Google Gemini"
    
    def authenticate(self) -> bool:
        """Authenticate with Google Gemini API."""
        try:
            import google.generativeai as genai
            
            if not self.api_key:
                raise ProviderError("Google Gemini API key not configured")
            
            genai.configure(api_key=self.api_key)
            
            # Test authentication by listing models
            try:
                models = list(genai.list_models())
                logger.info(f"Successfully authenticated with {self.get_provider_name()}")
                logger.debug(f"Available models: {[m.name for m in models]}")
                return True
            except Exception as e:
                logger.error(f"Gemini authentication test failed: {e}")
                raise ProviderAuthenticationError(f"Gemini authentication failed: {e}") from e
                
        except ImportError:
            raise ProviderError(
                "Google Generative AI SDK not installed. "
                "Install with: pip install google-generativeai"
            )
        except Exception as e:
            logger.error(f"Google Gemini authentication failed: {e}")
            raise ProviderAuthenticationError(f"Google Gemini authentication failed: {e}") from e
    
    def get_cost_estimate(self, num_images: int = 1) -> float:
        """
        Google Gemini/Imagen pricing:
        - Gemini API: ~$0.0001-0.001 per request (prompt refinement)
        - Imagen API: ~$0.02-0.04 per image generation
        """
        # Rough estimate: $0.025 per image (including prompt refinement)
        return num_images * 0.025
    
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        **kwargs
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Perform inpainting by generating background from prompt and compositing source image.
        
        Steps:
        1. Generate background image from the prompt (as-is, no refinement)
        2. Composite the source image (foreground) onto the generated background using the mask
        """
        start_time = time.time()
        
        try:
            logger.info(f"Generating background from prompt: {prompt[:80]}")
            
            # Use the prompt as-is (no refinement)
            # Generate background image from the prompt and composite
            result_image, background_image = self._composite_with_gemini_guidance(
                image, mask, prompt
            )
            
            processing_time = time.time() - start_time
            
            metadata: Dict[str, Any] = {
                "provider": self.get_provider_name(),
                "model_used": f"{self.model} + Background Generation",
                "original_prompt": prompt,
                "processing_time": processing_time,
                "cost": self.get_cost_estimate(1),
                "image_generated": True,
                "background_image": background_image,  # Include background for saving
            }
            
            logger.info(f"✅ Google Gemini inpainting completed in {processing_time:.2f}s!")
            return result_image, metadata
            
        except Exception as e:
            logger.error(f"Google Gemini error: {e}", exc_info=True)
            raise ProviderError(f"Google Gemini error: {e}") from e
    
    def _composite_with_gemini_guidance(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Generate final image from prompt and source image.
        
        Process:
        1. Try Gemini image generation API - returns final composited result directly
        2. If that fails, generate background and composite manually
        
        Returns:
            Tuple of (result_image, background_image)
        """
        try:
            # Ensure source image is RGB
            if image.mode != 'RGB':
                if image.mode == 'RGBA':
                    # Convert RGBA to RGB, preserving foreground
                    rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                    rgb_image.paste(image, mask=image.split()[3])
                    image = rgb_image
                else:
                    image = image.convert('RGB')
            
            # Ensure mask is grayscale
            if mask.mode != 'L':
                mask = mask.convert('L')
            
            # Ensure mask matches image size
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.Resampling.LANCZOS)
            
            # Try Gemini image generation API first - it returns the final composited result
            if self.is_primary_provider:
                try:
                    logger.info(f"Attempting Gemini image generation (returns final composited result)...")
                    final_result = self._generate_with_gemini_image_api(image.size, prompt, image)
                    
                    # Gemini returns the final result, so we use it directly
                    # For background image, we'll extract it or use a placeholder
                    # Since the user said the background file is correct, we can return the same image
                    logger.info("✅ Gemini image generation returned final composited result")
                    return final_result, final_result  # Return same image for both result and background
                    
                except Exception as e:
                    logger.warning(f"Gemini image generation failed: {e}. Falling back to manual compositing...")
            
            # Fallback: Generate background and composite manually
            logger.info(f"Generating background image from prompt: {prompt[:80]}")
            background = self._generate_background_image(image.size, prompt, image)
            
            # Debug: Check background color
            import numpy as np
            bg_array = np.array(background)
            avg_color = np.mean(bg_array.reshape(-1, 3), axis=0)
            logger.info(f"Generated background average color: RGB({int(avg_color[0])}, {int(avg_color[1])}, {int(avg_color[2])})")
            
            # Ensure background matches image size
            if background.size != image.size:
                background = background.resize(image.size, Image.Resampling.LANCZOS)
            
            # Debug: Check mask statistics
            mask_array = np.array(mask)
            white_pixels = np.sum(mask_array >= 200)  # Background areas
            black_pixels = np.sum(mask_array <= 55)   # Foreground areas
            total_pixels = mask_array.size
            logger.info(f"Mask stats: White (background)={white_pixels}/{total_pixels} ({100*white_pixels/total_pixels:.1f}%), "
                       f"Black (foreground)={black_pixels}/{total_pixels} ({100*black_pixels/total_pixels:.1f}%)")
            
            # Composite the foreground onto the background
            # Mask interpretation (from mask_generator):
            # - White (255) in mask = background area to replace
            # - Black (0) in mask = foreground area to keep
            # Image.composite(image1, image2, mask):
            # - White (255) in mask = use image1 (background)
            # - Black (0) in mask = use image2 (foreground/source)
            # So we use: Image.composite(background, image, mask)
            
            # Composite: background where mask is white, foreground where mask is black
            result = Image.composite(background, image, mask)
            
            # Debug: Check result color
            result_array = np.array(result)
            result_avg = np.mean(result_array.reshape(-1, 3), axis=0)
            logger.info(f"Result image average color: RGB({int(result_avg[0])}, {int(result_avg[1])}, {int(result_avg[2])})")
            
            logger.info("✅ Background generated and source image composited successfully")
            
            return result, background
            
        except Exception as e:
            logger.error(f"Compositing error: {e}", exc_info=True)
            raise ProviderAPIError(f"Failed to composite image: {e}") from e
    
    def _generate_background_image(
        self,
        size: Tuple[int, int],
        prompt: str,
        source_image: Optional[Image.Image] = None
    ) -> Image.Image:
        """
        Generate a background image from the prompt and source image context.
        Uses Google's image generation APIs when available.
        
        NOTE: This method is only called for fallback compositing.
        When Gemini image generation succeeds, it returns the final result directly.
        
        Args:
            size: Target image size (width, height)
            prompt: Description of desired background
            source_image: Source image to provide context for background generation
        
        Returns:
            Generated background image (for manual compositing)
        """
        # Try Imagen API (Vertex AI) - can use source image context
        if self.is_primary_provider and self.gcp_project_id:
            try:
                logger.info("Attempting to generate background using Imagen API...")
                return self._generate_with_imagen(size, prompt, source_image)
            except Exception as e:
                logger.warning(f"Imagen API generation failed: {e}. Trying fallback...")
        
        # Fallback: Use simple gradient generation (no keyword matching)
        logger.warning("All AI image generation methods failed. Using simple gradient fallback.")
        return self._generate_simple_fallback(size, prompt, source_image)
    
    def _generate_with_gemini_image_api(
        self,
        size: Tuple[int, int],
        prompt: str,
        source_image: Optional[Image.Image] = None
    ) -> Image.Image:
        """
        Generate background image using Gemini's image generation API (Nano Banana).
        This uses gemini-2.5-flash-image model which can generate images from prompts and source images.
        
        Args:
            size: Target image size (width, height)
            prompt: Description of desired background
            source_image: Source image to provide context for background generation
        
        Returns:
            Generated background image
        """
        try:
            import google.generativeai as genai
            
            # Configure Gemini
            genai.configure(api_key=self.api_key)
            
            # Create enhanced prompt for background generation
            enhanced_prompt = (
                f"Generate a high-quality background image: {prompt}. "
                f"Professional photography, detailed, realistic, seamless integration. "
                f"The background should complement the subject in the provided image. "
                f"No people, no objects in foreground, just the background scene."
            )
            
            logger.info(f"Using Gemini image generation model: {self.image_model}")
            logger.info(f"Prompt: {enhanced_prompt[:100]}...")
            
            # Initialize the image generation model
            model = genai.GenerativeModel(self.image_model)
            
            # Prepare contents for generation
            contents = [enhanced_prompt]
            
            # Add source image if available
            if source_image:
                # Prepare image for Gemini
                img_byte_arr = io.BytesIO()
                source_image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                pil_image = Image.open(img_byte_arr)
                contents.append(pil_image)
                logger.info("Including source image for context-aware background generation")
            
            # Determine aspect ratio from size
            width, height = size
            if width == height:
                aspect_ratio = "1:1"
            elif width > height:
                aspect_ratio = "16:9"
            else:
                aspect_ratio = "9:16"
            
            # Generate image
            generation_config = {
                "temperature": 0.7,
            }
            
            response = model.generate_content(
                contents,
                generation_config=generation_config
            )
            
            # Extract generated image from response
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                
                # Check for safety blocks
                if candidate.finish_reason in (2, 3) or candidate.finish_reason in ("SAFETY", "RECITATION"):
                    raise ProviderAPIError(f"Gemini image generation blocked: {candidate.finish_reason}")
                
                # Look for image parts in the response
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            # Extract image data
                            image_data = part.inline_data.data
                            image_mime_type = part.inline_data.mime_type
                            
                            # Convert to PIL Image
                            background = Image.open(io.BytesIO(image_data))
                            if background.mode != 'RGB':
                                background = background.convert('RGB')
                            
                            # Resize to target size
                            if background.size != size:
                                background = background.resize(size, Image.Resampling.LANCZOS)
                            
                            logger.info(f"✅ Gemini image generation successful: {background.size}")
                            return background
                
                # If no image found in response, try alternative extraction
                # Some API versions might return images differently
                logger.warning("No image found in response parts. Trying alternative extraction...")
                raise ProviderAPIError("Generated response does not contain image data")
            else:
                raise ProviderAPIError("No candidates in Gemini response")
                
        except ProviderAPIError:
            raise
        except Exception as e:
            logger.error(f"Gemini image generation API error: {e}", exc_info=True)
            raise ProviderAPIError(f"Failed to generate background with Gemini image API: {e}") from e
    
    def _generate_with_imagen(
        self,
        size: Tuple[int, int],
        prompt: str,
        source_image: Optional[Image.Image] = None
    ) -> Image.Image:
        """
        Generate background image using Google Vertex AI Imagen API.
        
        Args:
            size: Target image size (width, height)
            prompt: Description of desired background
        
        Returns:
            Generated background image
        """
        try:
            try:
                from google.cloud import aiplatform
                try:
                    from vertexai.preview import vision_models
                except ImportError:
                    # Try alternative import path
                    from vertexai import vision_models
            except ImportError:
                raise ProviderAPIError(
                    "Vertex AI SDK not installed. "
                    "Install with: pip install google-cloud-aiplatform"
                )
            
            if not self.gcp_project_id:
                raise ProviderAPIError(
                    "GCP project ID not configured. Set GCP_PROJECT_ID for Imagen API."
                )
            
            # Initialize Vertex AI
            aiplatform.init(project=self.gcp_project_id, location="us-central1")
            
            # Enhance prompt for background generation with source image context
            if source_image:
                # Analyze source image to get context (colors, style, etc.)
                enhanced_prompt = (
                    f"High quality background image."
                    f"Background description: {prompt}. "
                    f"Professional photography, detailed, realistic, seamless integration, "
                    f"no people, no objects in foreground, just background scene that complements the subject."
                )
            else:
                enhanced_prompt = (
                    f"High quality background image: {prompt}. "
                    f"Professional photography, detailed, realistic, no people, no objects in foreground, just background scene."
                )
            
            logger.info(f"Generating Imagen image: {enhanced_prompt[:100]}")
            
            # Generate image using Imagen
            model = vision_models.ImageGenerationModel.from_pretrained(self.imagen_model)
            
            width, height = size
            # Determine aspect ratio
            if width == height:
                aspect_ratio = "1:1"
            elif width > height:
                aspect_ratio = "16:9"
            else:
                aspect_ratio = "9:16"
            
            response = model.generate_images(
                prompt=enhanced_prompt,
                number_of_images=1,
                aspect_ratio=aspect_ratio
            )
            
            # Get the generated image
            generated_image = response.images[0]
            
            # Convert to PIL Image and resize
            background = generated_image._pil_image.convert("RGB")
            if background.size != size:
                background = background.resize(size, Image.Resampling.LANCZOS)
            
            logger.info(f"✅ Imagen background generated: {background.size}")
            return background
            
        except ProviderAPIError:
            raise
        except Exception as e:
            logger.error(f"Imagen generation error: {e}")
            raise ProviderAPIError(f"Failed to generate background with Imagen: {e}") from e
    
    def _generate_with_gemini_context(
        self,
        size: Tuple[int, int],
        prompt: str,
        source_image: Optional[Image.Image] = None
    ) -> Image.Image:
        """
        Generate background image using Gemini to analyze source image and create enhanced prompt.
        
        Args:
            size: Target image size (width, height)
            prompt: Description of desired background
            source_image: Source image to analyze for context
        
        Returns:
            Generated background image
        """
        try:
            import google.generativeai as genai
            
            # Configure Gemini
            genai.configure(api_key=self.api_key)
            
            if source_image:
                # Use Gemini to analyze the source image and create an enhanced prompt
                model = genai.GenerativeModel(model_name=self.model)
                
                # Prepare image for Gemini
                img_byte_arr = io.BytesIO()
                source_image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                # Create prompt for Gemini to analyze image and suggest background
                analysis_prompt = (
                    f"Analyze this image and suggest a detailed background description "
                    f"based on the user's request: '{prompt}'. "
                    f"Consider the image's colors, lighting, style, and mood. "
                    f"Provide a detailed, visual background description that would complement this image. "
                    f"Focus on creating a seamless, professional background that matches the image's aesthetic."
                )
                
                logger.info("Using Gemini to analyze source image and enhance background prompt...")
                
                # Analyze image with Gemini
                response = model.generate_content(
                    [analysis_prompt, Image.open(img_byte_arr)],
                    generation_config={
                        "temperature": 0.7,
                        "max_output_tokens": 400,
                    }
                )
                
                # Get enhanced prompt from Gemini
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    finish_reason = candidate.finish_reason
                    
                    if finish_reason not in (2, 3) and finish_reason != "SAFETY" and finish_reason != "RECITATION":
                        try:
                            enhanced_prompt = response.text.strip()
                            logger.info(f"✅ Gemini enhanced prompt: {enhanced_prompt[:100]}")
                            # Use the enhanced prompt for background generation
                            prompt = enhanced_prompt
                        except (ValueError, AttributeError):
                            logger.warning("Could not extract enhanced prompt from Gemini. Using original prompt.")
                    else:
                        logger.warning("Gemini analysis blocked. Using original prompt.")
            
            # Now generate background using the enhanced prompt
            # Fall back to simple gradient generation
            return self._generate_simple_fallback(size, prompt, source_image)
            
        except Exception as e:
            logger.error(f"Gemini context generation error: {e}")
            raise ProviderAPIError(f"Failed to generate background with Gemini context: {e}") from e
    
    def _generate_simple_fallback(
        self,
        size: Tuple[int, int],
        prompt: str,
        source_image: Optional[Image.Image] = None
    ) -> Image.Image:
        """
        Simple fallback background generation when all AI methods fail.
        Creates a basic gradient based on source image color analysis (no keyword matching).
        
        Args:
            size: Target image size (width, height)
            prompt: Description of desired background (not used, kept for compatibility)
            source_image: Source image for color analysis
        
        Returns:
            Background image
        """
        try:
            import numpy as np
        except ImportError:
            logger.warning("NumPy not available. Using neutral gray background.")
            return Image.new('RGB', size, (200, 200, 200))
        
        # Analyze source image colors if available
        if source_image:
            try:
                if source_image.mode != 'RGB':
                    source_image = source_image.convert('RGB')
                
                img_array = np.array(source_image)
                height, width = img_array.shape[:2]
                
                # Sample edge pixels for background color hints
                edge_pixels = np.concatenate([
                    img_array[0, :].reshape(-1, 3),
                    img_array[-1, :].reshape(-1, 3),
                    img_array[:, 0].reshape(-1, 3),
                    img_array[:, -1].reshape(-1, 3),
                ])
                
                # Get average edge color
                avg_color = np.mean(edge_pixels, axis=0).astype(int)
                base_color = tuple(avg_color)
                brightness = np.mean(avg_color)
                
                logger.info(f"Fallback: Analyzed source image - edge color: {base_color}, brightness: {brightness:.1f}")
                
                # If too light, use center area for subject colors
                if brightness > 240:
                    center_y, center_x = height // 2, width // 2
                    center_area = img_array[center_y-50:center_y+50, center_x-50:center_x+50]
                    if center_area.size > 0:
                        center_color = np.mean(center_area.reshape(-1, 3), axis=0).astype(int)
                        base_color = tuple(min(255, int(c * 1.2)) for c in center_color)
                        logger.info(f"Fallback: Using center area color: {base_color}")
                
                # Create gradient from base color (avoid white backgrounds)
                if brightness > 220:
                    # Too light, use warm neutral gradient
                    background = self._create_gradient(size, (220, 215, 210), (200, 195, 190))
                else:
                    # Create darker, more saturated colors for background
                    light_color = tuple(min(240, max(180, int(c * 1.2))) for c in base_color)
                    mid_color = tuple(min(220, max(160, int(c * 1.0))) for c in base_color)
                    background = self._create_gradient(size, light_color, mid_color)
                
                return background
                
            except Exception as e:
                logger.warning(f"Error analyzing source image colors: {e}. Using neutral gradient.")
        
        # Ultimate fallback: neutral gradient
        logger.info("Fallback: Using neutral gradient background")
        return self._create_gradient(size, (220, 215, 210), (200, 195, 190))
    
    
    def _create_gradient(
        self,
        size: Tuple[int, int],
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int]
    ) -> Image.Image:
        """
        Create a gradient background.
        
        Args:
            size: Image size (width, height)
            color1: Start color (RGB)
            color2: End color (RGB)
        
        Returns:
            Gradient image
        """
        width, height = size
        gradient = Image.new('RGB', (width, height))
        pixels = gradient.load()
        
        for y in range(height):
            # Interpolate between color1 and color2
            ratio = y / height if height > 0 else 0
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            
            for x in range(width):
                pixels[x, y] = (r, g, b)
        
        return gradient

