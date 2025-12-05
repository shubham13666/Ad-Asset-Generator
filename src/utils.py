"""Utility functions for logging and common operations."""

import logging
import sys
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import google.cloud.logging as cloud_logging
from PIL import Image


def setup_logging(
    log_level: str = "INFO",
    use_cloud_logging: bool = False,
    project_id: Optional[str] = None
) -> logging.Logger:
    """
    Set up structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_cloud_logging: Whether to use Google Cloud Logging
        project_id: GCP project ID for cloud logging
    
    Returns:
        Configured logger instance
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler with formatted output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # Google Cloud Logging integration (if enabled)
    if use_cloud_logging and project_id:
        try:
            client = cloud_logging.Client(project=project_id)
            cloud_handler = client.get_default_handler()
            cloud_handler.setLevel(numeric_level)
            logger.addHandler(cloud_handler)
            logger.info("Google Cloud Logging enabled")
        except Exception as e:
            logger.warning(f"Failed to set up Google Cloud Logging: {e}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def format_timestamp(dt: Optional[datetime] = None, timezone: str = "IST") -> str:
    """
    Format datetime as string with timezone.
    
    Args:
        dt: Datetime object (defaults to now)
        timezone: Timezone string to append
    
    Returns:
        Formatted timestamp string
    """
    if dt is None:
        dt = datetime.now()
    
    return f"{dt.strftime('%Y-%m-%d %H:%M:%S')} {timezone}"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted duration string (e.g., "1m 23.45s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    
    minutes = int(seconds // 60)
    secs = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {secs:.2f}s"
    
    hours = int(minutes // 60)
    mins = minutes % 60
    
    return f"{hours}h {mins}m {secs:.2f}s"


def safe_int(value: Optional[str], default: int = 0) -> int:
    """
    Safely convert string to integer with default fallback.
    
    Args:
        value: String value to convert
        default: Default value if conversion fails
    
    Returns:
        Integer value or default
    """
    if value is None:
        return default
    
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value: Optional[str], default: float = 0.0) -> float:
    """
    Safely convert string to float with default fallback.
    
    Args:
        value: String value to convert
        default: Default value if conversion fails
    
    Returns:
        Float value or default
    """
    if value is None:
        return default
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def truncate_string(value: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to maximum length with suffix.
    
    Args:
        value: String to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
    
    Returns:
        Truncated string
    """
    if len(value) <= max_length:
        return value
    
    return value[:max_length - len(suffix)] + suffix


def get_image_metadata(image_path_or_image) -> Dict[str, Any]:
    """
    Extract image metadata (dimensions, format, size).
    
    Args:
        image_path_or_image: Path to image file or PIL Image object
    
    Returns:
        Dictionary with image metadata:
        - width: int
        - height: int
        - dimensions: str (e.g., "1920x1080")
        - format: str (e.g., "JPEG")
        - mode: str (e.g., "RGB")
        - size_bytes: int (file size in bytes, if available)
    """
    if isinstance(image_path_or_image, str):
        # Path provided
        with Image.open(image_path_or_image) as img:
            width, height = img.size
            format_str = img.format or "UNKNOWN"
            mode = img.mode
    else:
        # PIL Image provided
        img = image_path_or_image
        width, height = img.size
        format_str = getattr(img, 'format', 'UNKNOWN') or "UNKNOWN"
        mode = img.mode
    
    return {
        "width": width,
        "height": height,
        "dimensions": f"{width}x{height}",
        "format": format_str,
        "mode": mode
    }


def preserve_aspect_ratio(
    original_width: int,
    original_height: int,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None
) -> Tuple[int, int]:
    """
    Calculate dimensions that preserve aspect ratio while fitting within constraints.
    
    Args:
        original_width: Original image width
        original_height: Original image height
        max_width: Maximum width constraint
        max_height: Maximum height constraint
    
    Returns:
        Tuple of (new_width, new_height)
    """
    if not max_width and not max_height:
        return original_width, original_height
    
    aspect_ratio = original_width / original_height
    
    if max_width and max_height:
        # Fit within both constraints
        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        ratio = min(width_ratio, height_ratio)
    elif max_width:
        ratio = max_width / original_width
    elif max_height:
        ratio = max_height / original_height
    else:
        ratio = 1.0
    
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    
    return new_width, new_height


def convert_image_format(
    image: Image.Image,
    target_format: str = 'JPEG',
    quality: int = 95
) -> Image.Image:
    """
    Convert image to target format.
    
    Args:
        image: PIL Image to convert
        target_format: Target format ('JPEG', 'PNG', etc.)
        quality: Quality setting for JPEG (1-100)
    
    Returns:
        Converted PIL Image
    """
    if image.format == target_format:
        return image
    
    # Convert RGBA to RGB for JPEG
    if target_format == 'JPEG' and image.mode in ('RGBA', 'LA', 'P'):
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'RGBA':
            rgb_image.paste(image, mask=image.split()[3])
        else:
            rgb_image.paste(image)
        image = rgb_image
    
    return image


def validate_image_file(file_path: str) -> bool:
    """
    Validate that a file is a valid image.
    
    Args:
        file_path: Path to image file
    
    Returns:
        True if valid image, False otherwise
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Invalid image file {file_path}: {str(e)}")
        return False

