"""Background mask generation using Rembg (FREE), Remove.bg fallback, and OpenCV fallback."""

import logging
import os
from typing import Optional, Tuple
from io import BytesIO
import requests

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Try to import Rembg, but make it optional
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logger.warning("Rembg not available. Install with: pip install rembg onnxruntime")

# Try to import OpenCV, but make it optional
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV not available. Install with: pip install opencv-python")


class MaskGeneratorError(Exception):
    """Base exception for mask generation errors."""
    pass


class RembgMaskGenerator:
    """Mask generator using Rembg (FREE, local)."""
    
    def __init__(self):
        """Initialize Rembg mask generator."""
        if not REMBG_AVAILABLE:
            raise ImportError("Rembg not available. Install with: pip install rembg onnxruntime")
        
        logger.info("Rembg mask generator initialized (FREE, local)")
    
    def generate_mask(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Generate background mask using Rembg.
        
        Args:
            image: Input image (PIL Image)
        
        Returns:
            Binary mask image (white=background, black=foreground), or None if error
        """
        try:
            # Save image to bytes
            image_bytes = BytesIO()
            if image.mode in ('RGBA', 'LA', 'P'):
                # Convert to RGB
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'RGBA':
                    rgb_image.paste(image, mask=image.split()[3])
                else:
                    rgb_image.paste(image)
                rgb_image.save(image_bytes, format='PNG')
            else:
                image.save(image_bytes, format='PNG')
            
            image_bytes.seek(0)
            
            # Use Rembg to remove background
            logger.info("Generating mask using Rembg (FREE, local)...")
            output_bytes = rembg_remove(image_bytes.getvalue())
            
            # Rembg returns image with transparent background
            # Convert to mask by checking alpha channel
            foreground_image = Image.open(BytesIO(output_bytes))
            
            # Resize if needed
            if foreground_image.size != image.size:
                foreground_image = foreground_image.resize(image.size, Image.LANCZOS)
            
            # Create mask from alpha channel or by comparing with original
            if foreground_image.mode == 'RGBA':
                # Extract alpha channel as mask (invert: white=background, black=foreground)
                alpha = foreground_image.split()[3]
                # Invert: background (transparent) becomes white, foreground becomes black
                mask_array = np.array(alpha)
                mask_array = 255 - mask_array  # Invert
                mask = Image.fromarray(mask_array, mode='L')
            else:
                # Create mask by comparing original and foreground
                mask = self._create_mask_from_comparison(image, foreground_image)
            
            logger.info("Successfully generated mask using Rembg")
            return mask
            
        except Exception as e:
            logger.error(f"Error generating mask with Rembg: {str(e)}", exc_info=True)
            return None
    
    def _create_mask_from_comparison(
        self,
        original: Image.Image,
        foreground: Image.Image
    ) -> Image.Image:
        """
        Create mask by comparing original and foreground images.
        Areas that differ significantly are considered background.
        
        Args:
            original: Original image
            foreground: Foreground image (background removed)
        
        Returns:
            Binary mask (white=background, black=foreground)
        """
        # Convert to RGB if needed
        if original.mode != 'RGB':
            original = original.convert('RGB')
        if foreground.mode != 'RGB':
            foreground = foreground.convert('RGB')
        
        # Convert to numpy arrays
        orig_array = np.array(original)
        fg_array = np.array(foreground)
        
        # Calculate difference
        diff = np.abs(orig_array.astype(float) - fg_array.astype(float))
        
        # Sum across color channels
        diff_sum = np.sum(diff, axis=2)
        
        # Threshold to create binary mask
        threshold = 30  # Adjust based on needs
        mask_array = (diff_sum > threshold).astype(np.uint8) * 255
        
        # Create mask image (white=background to replace, black=foreground to keep)
        mask = Image.fromarray(mask_array, mode='L')
        
        return mask


class RemoveBgMaskGenerator:
    """Mask generator using Remove.bg API ($0.01/image)."""
    
    REMOVEBG_API_URL = "https://api.remove.bg/v1.0/removebg"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Remove.bg mask generator.
        
        Args:
            api_key: Remove.bg API key. If None, reads from REMOVEBG_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('REMOVEBG_API_KEY')
        
        if not self.api_key:
            logger.warning("Remove.bg API key not configured")
        
        self.cost_per_image = 0.01  # $0.01 per image
    
    def generate_mask(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Generate background mask using Remove.bg API.
        
        Args:
            image: Input image (PIL Image)
        
        Returns:
            Binary mask image (white=background, black=foreground), or None if error
        """
        if not self.api_key:
            logger.warning("Remove.bg API key not configured, cannot generate mask")
            return None
        
        try:
            # Prepare image for API
            image_bytes = BytesIO()
            if image.mode in ('RGBA', 'LA', 'P'):
                # Convert to RGB
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'RGBA':
                    rgb_image.paste(image, mask=image.split()[3])
                else:
                    rgb_image.paste(image)
                rgb_image.save(image_bytes, format='PNG')
            else:
                image.save(image_bytes, format='PNG')
            
            image_bytes.seek(0)
            
            # Call Remove.bg API
            headers = {
                'X-Api-Key': self.api_key
            }
            
            files = {
                'image_file': ('image.png', image_bytes, 'image/png')
            }
            
            data = {
                'size': 'auto'
            }
            
            logger.info(f"Calling Remove.bg API for background removal ($0.01/image)...")
            response = requests.post(
                self.REMOVEBG_API_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                # Remove.bg returns the foreground image (without background)
                foreground_bytes = BytesIO(response.content)
                foreground_image = Image.open(foreground_bytes)
                
                # Resize if needed
                if foreground_image.size != image.size:
                    foreground_image = foreground_image.resize(image.size, Image.LANCZOS)
                
                # Create mask from alpha channel or by comparing with original
                if foreground_image.mode == 'RGBA':
                    alpha = foreground_image.split()[3]
                    mask_array = np.array(alpha)
                    mask_array = 255 - mask_array  # Invert: white=background, black=foreground
                    mask = Image.fromarray(mask_array, mode='L')
                else:
                    mask = self._create_mask_from_comparison(image, foreground_image)
                
                logger.info(f"Successfully generated mask using Remove.bg (cost: ${self.cost_per_image:.3f})")
                return mask
            else:
                error_msg = f"Remove.bg API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return None
                
        except requests.exceptions.Timeout:
            logger.error("Remove.bg API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Remove.bg API request failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error generating mask with Remove.bg: {str(e)}", exc_info=True)
            return None
    
    def _create_mask_from_comparison(
        self,
        original: Image.Image,
        foreground: Image.Image
    ) -> Image.Image:
        """Create mask by comparing original and foreground images."""
        if original.mode != 'RGB':
            original = original.convert('RGB')
        if foreground.mode != 'RGB':
            foreground = foreground.convert('RGB')
        
        orig_array = np.array(original)
        fg_array = np.array(foreground)
        
        diff = np.abs(orig_array.astype(float) - fg_array.astype(float))
        diff_sum = np.sum(diff, axis=2)
        
        threshold = 30
        mask_array = (diff_sum > threshold).astype(np.uint8) * 255
        
        mask = Image.fromarray(mask_array, mode='L')
        return mask
    
    def get_cost_estimate(self, num_images: int = 1) -> float:
        """Get cost estimate: $0.01 per image."""
        return self.cost_per_image * num_images


class OpenCVMaskGenerator:
    """Mask generator using OpenCV semantic segmentation (FREE, local)."""
    
    def __init__(self):
        """Initialize OpenCV mask generator."""
        if not OPENCV_AVAILABLE:
            raise ImportError("OpenCV not available. Install with: pip install opencv-python")
        
        logger.info("OpenCV mask generator initialized (FREE, local)")
    
    def generate_mask(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Generate background mask using OpenCV.
        
        Uses a simple approach based on edge detection and morphological operations.
        For better results, consider using deep learning models.
        
        Args:
            image: Input image (PIL Image)
        
        Returns:
            Binary mask image (white=background, black=foreground), or None if error
        """
        try:
            # Convert PIL to OpenCV format
            if image.mode == 'RGBA':
                # Convert RGBA to RGB
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])
                image = rgb_image
            
            # Convert to numpy array
            img_array = np.array(image.convert('RGB'))
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use adaptive threshold to detect edges
            thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )
            
            # Morphological operations to clean up mask
            kernel = np.ones((5, 5), np.uint8)
            
            # Closing to fill holes
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Opening to remove noise
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Invert mask (white=background, black=foreground)
            mask = cv2.bitwise_not(opened)
            
            # Convert back to PIL Image
            mask_image = Image.fromarray(mask, mode='L')
            
            logger.info("Successfully generated mask using OpenCV")
            return mask_image
            
        except Exception as e:
            logger.error(f"Error generating mask with OpenCV: {str(e)}", exc_info=True)
            return None


class MaskGenerator:
    """
    Unified mask generator with cascading fallback:
    1. Primary: Rembg (FREE, local)
    2. Fallback 1: Remove.bg ($0.01/image)
    3. Fallback 2: OpenCV (FREE, local)
    """
    
    def __init__(
        self,
        removebg_api_key: Optional[str] = None,
        use_removebg_fallback: bool = True,
        use_opencv_fallback: bool = True
    ):
        """
        Initialize mask generator.
        
        Args:
            removebg_api_key: Remove.bg API key (optional, for fallback)
            use_removebg_fallback: Whether to use Remove.bg as fallback
            use_opencv_fallback: Whether to use OpenCV as fallback
        """
        self.rembg_generator = None
        self.removebg_generator = None
        self.opencv_generator = None
        
        # Initialize Rembg (primary, FREE)
        if REMBG_AVAILABLE:
            try:
                self.rembg_generator = RembgMaskGenerator()
                logger.info("Rembg mask generator initialized (FREE, primary)")
            except Exception as e:
                logger.warning(f"Could not initialize Rembg generator: {str(e)}")
        else:
            logger.warning("Rembg not available. Install with: pip install rembg onnxruntime")
        
        # Initialize Remove.bg (fallback 1, $0.01/image)
        if use_removebg_fallback:
            removebg_key = removebg_api_key or os.getenv('REMOVEBG_API_KEY')
            if removebg_key:
                self.removebg_generator = RemoveBgMaskGenerator(removebg_key)
                logger.info("Remove.bg mask generator initialized as fallback ($0.01/image)")
            else:
                logger.info("Remove.bg API key not configured (optional fallback)")
        
        # Initialize OpenCV (fallback 2, FREE)
        if use_opencv_fallback and OPENCV_AVAILABLE:
            try:
                self.opencv_generator = OpenCVMaskGenerator()
                logger.info("OpenCV mask generator initialized as fallback (FREE)")
            except Exception as e:
                logger.warning(f"Could not initialize OpenCV generator: {str(e)}")
    
    def generate_mask(self, image: Image.Image) -> Tuple[Optional[Image.Image], str]:
        """
        Generate background mask with automatic fallback.
        
        Args:
            image: Input image (PIL Image)
        
        Returns:
            Tuple of (mask_image, method_used)
            - mask_image: Binary mask (white=background, black=foreground), or None if all methods fail
            - method_used: 'rembg', 'removebg', 'opencv', or 'failed'
        """
        # Try 1: Rembg (FREE, local)
        if self.rembg_generator:
            mask = self.rembg_generator.generate_mask(image)
            if mask:
                return mask, 'rembg'
            logger.warning("Rembg mask generation failed, trying Remove.bg fallback...")
        
        # Try 2: Remove.bg (API, $0.01/image)
        if self.removebg_generator:
            mask = self.removebg_generator.generate_mask(image)
            if mask:
                return mask, 'removebg'
            logger.warning("Remove.bg mask generation failed, trying OpenCV fallback...")
        
        # Try 3: OpenCV (FREE, local)
        if self.opencv_generator:
            mask = self.opencv_generator.generate_mask(image)
            if mask:
                return mask, 'opencv'
            logger.warning("OpenCV mask generation failed")
        
        logger.error("All mask generation methods failed")
        return None, 'failed'
    
    def is_available(self) -> bool:
        """Check if any mask generation method is available."""
        return (
            self.rembg_generator is not None or
            self.removebg_generator is not None or
            self.opencv_generator is not None
        )
    
    def get_cost_estimate(self, num_images: int = 1, method: str = 'rembg') -> float:
        """
        Get cost estimate for mask generation.
        
        Args:
            num_images: Number of images
            method: Method to use ('rembg', 'removebg', 'opencv')
        
        Returns:
            Cost estimate in USD
        """
        if method == 'rembg' or method == 'opencv':
            return 0.0  # FREE
        elif method == 'removebg':
            return 0.01 * num_images
        return 0.0
