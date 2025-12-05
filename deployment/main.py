"""Google Cloud Function entry point for background replacement system."""

import os
import json
import logging
import traceback
from flask import Request

from src.pipeline import BackgroundReplacementPipeline
from src.config import get_config, validate_config
from src.utils import setup_logging

# Set up logging
log_level = os.getenv('LOG_LEVEL', 'INFO')
use_cloud_logging = os.getenv('USE_CLOUD_LOGGING', 'false').lower() == 'true'
project_id = os.getenv('GCP_PROJECT_ID')

setup_logging(
    log_level=log_level,
    use_cloud_logging=use_cloud_logging,
    project_id=project_id
)

logger = logging.getLogger(__name__)


def background_replacement_handler(request: Request) -> tuple:
    """
    Cloud Function HTTP entry point.
    
    Args:
        request: Flask request object from Cloud Scheduler
    
    Returns:
        Tuple of (response_body, status_code, headers)
    """
    try:
        logger.info("=" * 60)
        logger.info("Background Replacement Function Invoked")
        logger.info("=" * 60)
        
        # Log request details
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        
        # Get configuration
        logger.info("Loading configuration...")
        config = get_config()
        
        # Validate configuration
        try:
            validate_config(config)
        except ValueError as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return {
                'status': 'error',
                'message': f'Configuration error: {str(e)}'
            }, 500, {'Content-Type': 'application/json'}
        
        # Initialize and execute pipeline
        logger.info("Initializing pipeline...")
        pipeline = BackgroundReplacementPipeline(config)
        
        logger.info("Executing pipeline...")
        result = pipeline.execute()
        
        # Prepare response
        status_code = 200 if result['status'] == 'completed' else 500
        
        response_body = {
            'status': result['status'],
            'message': result.get('message', 'Pipeline execution completed'),
            'work_items': result.get('work_items', 0),
            'processed': result.get('processed', 0),
            'failed': result.get('failed', 0),
            'skipped': result.get('skipped', 0),
            'duration_seconds': result.get('duration', 0)
        }
        
        # Add metrics summary if available
        if 'metrics' in result:
            response_body['metrics_summary'] = {
                'total_cost': result['metrics'].get('total_cost', '$0.00'),
                'total_images': result['metrics'].get('total_images', 0),
                'processed': result['metrics'].get('processed', 0),
                'failed': result['metrics'].get('failed', 0)
            }
        
        logger.info(f"Pipeline execution completed: {json.dumps(response_body, indent=2)}")
        
        return response_body, status_code, {'Content-Type': 'application/json'}
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        error_trace = traceback.format_exc()
        
        logger.error(error_msg)
        logger.error(error_trace)
        
        return {
            'status': 'error',
            'message': error_msg,
            'error_type': type(e).__name__
        }, 500, {'Content-Type': 'application/json'}


# Cloud Functions entry point (for HTTP-triggered functions)
def main(request):
    """
    Main entry point for Google Cloud Function.
    
    Args:
        request: Flask request object
    
    Returns:
        Response tuple
    """
    return background_replacement_handler(request)



