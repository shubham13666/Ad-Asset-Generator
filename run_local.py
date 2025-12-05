#!/usr/bin/env python3
"""Local test script for background replacement system.

This script allows you to run the background replacement pipeline locally
for testing before deploying to Google Cloud Run.

Usage:
    python run_local.py
"""

import sys
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment variables from .env file")
else:
    print(f"⚠️  .env file not found at {env_path}")
    print(f"   Using system environment variables instead")

from src.pipeline import BackgroundReplacementPipeline
from src.config import get_config, validate_config
from src.utils import setup_logging

# Set up logging before importing other modules
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


def main():
    """Run the pipeline locally."""
    
    logger.info("=" * 60)
    logger.info("Starting Local Background Replacement Pipeline")
    logger.info("=" * 60)
    
    try:
        # Load configuration from .env file or environment variables
        logger.info("Loading configuration...")
        config = get_config()
        
        # Display configuration summary
        logger.info("\nConfiguration Summary:")
        logger.info(f"  Primary Provider: {config.primary_provider}")
        logger.info(f"  Fallback Enabled: {config.enable_fallback}")
        if config.enable_fallback:
            logger.info(f"  Fallback Providers: {', '.join(config.fallback_providers)}")
        logger.info(f"  Google Drive Root Folder ID: {config.google_drive_root_folder_id or 'Not set'}")
        logger.info(f"  Log Level: {config.log_level}")
        
        # Validate configuration
        logger.info("\nValidating configuration...")
        try:
            validate_config(config)
            logger.info("✅ Configuration validation passed")
        except ValueError as e:
            logger.error(f"❌ Configuration validation failed:")
            logger.error(f"   {str(e)}")
            logger.error("\nPlease check your .env file or environment variables.")
            return 1
        
        # Initialize pipeline
        logger.info("\nInitializing pipeline...")
        logger.info("  - Setting up Google Drive client...")
        logger.info("  - Setting up Google Sheets logger...")
        logger.info("  - Setting up mask generator (Rembg)...")
        logger.info("  - Setting up inpainting provider...")
        
        pipeline = BackgroundReplacementPipeline(config)
        logger.info("✅ Pipeline initialized successfully")
        
        # Execute pipeline
        logger.info("\n" + "=" * 60)
        logger.info("Executing Pipeline")
        logger.info("=" * 60)
        logger.info("")
        
        result = pipeline.execute()
        
        # Print results
        logger.info("")
        logger.info("=" * 60)
        logger.info("Pipeline Execution Results")
        logger.info("=" * 60)
        
        if result['status'] == 'completed':
            logger.info("✅ Status: COMPLETED")
            logger.info(f"   Work Items Found: {result.get('work_items', 0)}")
            logger.info(f"   Images Processed: {result.get('processed', 0)}")
            logger.info(f"   Images Failed: {result.get('failed', 0)}")
            logger.info(f"   Images Skipped: {result.get('skipped', 0)}")
            logger.info(f"   Duration: {result.get('duration', 0):.2f} seconds")
            
            # Display metrics if available
            if 'metrics' in result:
                metrics = result['metrics']
                logger.info("\nCost Summary:")
                logger.info(f"   Total Cost: {metrics.get('total_cost', '$0.00')}")
                
                if 'provider_stats' in metrics:
                    logger.info("\nProvider Statistics:")
                    for provider, stats in metrics['provider_stats'].items():
                        logger.info(f"   {provider}:")
                        logger.info(f"      Attempts: {stats.get('attempts', 0)}")
                        logger.info(f"      Successes: {stats.get('successes', 0)}")
                        logger.info(f"      Failures: {stats.get('failures', 0)}")
                        logger.info(f"      Success Rate: {stats.get('success_rate', '0%')}")
                        logger.info(f"      Total Cost: {stats.get('total_cost', '$0.00')}")
            
            logger.info("\n" + "=" * 60)
            logger.info("✅ Pipeline execution completed successfully!")
            logger.info("=" * 60)
            return 0
            
        elif result['status'] == 'error':
            logger.error("❌ Status: ERROR")
            logger.error(f"   Error: {result.get('error', 'Unknown error')}")
            logger.error("=" * 60)
            return 1
        
        else:
            logger.warning(f"⚠️  Status: {result.get('status', 'unknown')}")
            logger.warning(f"   Message: {result.get('message', 'No message')}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Pipeline execution interrupted by user")
        return 130
        
    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error("❌ Unexpected Error")
        logger.error("=" * 60)
        logger.error(f"Error Type: {type(e).__name__}")
        logger.error(f"Error Message: {str(e)}")
        logger.error("\nFull traceback:")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("=" * 60)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

