"""Main processing pipeline for background replacement."""

import logging
import re
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.config import get_config, AppConfig, validate_config
from src.google_drive_client import GoogleDriveClient
from src.google_sheets_logger import GoogleSheetsLogger
from src.mask_generator import MaskGenerator
from src.utils import (
    setup_logging,
    format_timestamp,
    get_image_metadata,
    truncate_string
)
from providers.factory import InpaintingProviderFactory
from providers.metrics import ProviderMetrics
from providers.base import ProviderError

logger = logging.getLogger(__name__)


class ProcessingWorkItem:
    """Represents a work item for processing."""
    
    def __init__(
        self,
        project_id: str,
        project_name: str,
        category: str,
        source_folder_id: str,
        output_folder_id: str,
        prompt: str,
        spreadsheet_id: str
    ):
        """Initialize work item."""
        self.project_id = project_id
        self.project_name = project_name
        self.category = category
        self.source_folder_id = source_folder_id
        self.output_folder_id = output_folder_id
        self.prompt = prompt
        self.spreadsheet_id = spreadsheet_id


class BackgroundReplacementPipeline:
    """Main pipeline for background replacement processing."""
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize pipeline.
        
        Args:
            config: Application configuration. If None, loads from environment/config.
        """
        self.config = config or get_config()
        
        # Validate configuration
        validate_config(self.config)
        
        # Initialize components
        self.drive_client = GoogleDriveClient(self.config.google_credentials_path, self.config.google_drive_root_folder_id)
        self.sheets_logger = GoogleSheetsLogger(self.config.google_sheets_credentials_path)
        self.mask_generator = MaskGenerator(
            removebg_api_key=self.config.removebg_api_key,
            use_removebg_fallback=True,
            use_opencv_fallback=True
        )
        
        # Initialize provider with fallback
        provider_config = self._build_provider_config()
        self.inpainter = InpaintingProviderFactory.create_with_fallback(
            primary_provider_name=self.config.primary_provider,
            fallback_provider_names=self.config.fallback_providers if self.config.enable_fallback else [],
            config=provider_config
        )
        
        # Metrics tracker
        self.metrics = self.inpainter.get_metrics()
        
        logger.info(f"Pipeline initialized with provider: {self.config.primary_provider}")
    
    def _build_provider_config(self) -> Dict[str, Any]:
        """Build provider configuration from app config."""
        config = {}
        
        if self.config.openai_chat:
            config['openai_chat'] = {
                 "api_key": self.config.openai_chat.api_key,
                 "model": self.config.openai_chat.model,  
                'timeout': self.config.openai_chat.timeout,
                'max_retries': self.config.openai_chat.max_retries,
                'retry_delay': self.config.openai_chat.retry_delay
            }
        
        if self.config.alibaba:
            config['alibaba'] = {
                'api_key': self.config.alibaba.api_key,
                'region': self.config.alibaba.region,
                'timeout': self.config.alibaba.timeout,
                'max_retries': self.config.alibaba.max_retries,
                'retry_delay': self.config.alibaba.retry_delay
            }
        
        # if self.config.stability:
        #     config['stability'] = {
        #         'api_key': self.config.stability.api_key,
        #         'endpoint': self.config.stability.endpoint,
        #         'timeout': self.config.stability.timeout,
        #         'max_retries': self.config.stability.max_retries,
        #         'retry_delay': self.config.stability.retry_delay
        #     }
        
        return config
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the complete processing pipeline.
        
        Returns:
            Dictionary with execution summary
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Starting Background Replacement Pipeline")
        logger.info("=" * 60)
        
        try:
            # Phase 0: Provider initialization is done in __init__
            logger.info("Phase 0: Provider initialized")
            
            # Phase 1: Discovery
            logger.info("Phase 1: Discovering source folders...")
            work_items = self._discover_work_items()
            
            if not work_items:
                logger.info("No work items found. Exiting.")
                return {
                    'status': 'completed',
                    'message': 'No work items to process',
                    'duration': (datetime.now() - start_time).total_seconds()
                }
            
            logger.info(f"Found {len(work_items)} work items to process")
            
            # Phase 2: Validation & Setup
            logger.info("Phase 2: Validating and setting up work items...")
            ready_items = self._validate_and_setup(work_items)
            
            if not ready_items:
                logger.warning("No ready work items after validation. Exiting.")
                return {
                    'status': 'completed',
                    'message': 'No ready work items',
                    'duration': (datetime.now() - start_time).total_seconds()
                }
            
            logger.info(f"Ready to process {len(ready_items)} work items")
            
            # Phase 3: Processing
            logger.info("Phase 3: Processing images...")
            results = self._process_images(ready_items)
            
            # Phase 4: Error handling & logging
            logger.info("Phase 4: Finalizing logs...")
            self._finalize_logs(ready_items, results, start_time)
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Pipeline completed in {duration:.2f} seconds")
            
            return {
                'status': 'completed',
                'work_items': len(work_items),
                'processed': results['processed'],
                'failed': results['failed'],
                'skipped': results['skipped'],
                'duration': duration,
                'metrics': self.metrics.get_summary()
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'duration': (datetime.now() - start_time).total_seconds()
            }
    
    def _discover_work_items(self) -> List[Dict[str, Any]]:
        """
        Phase 1: Discover source image folders.
        
        Returns:
            List of discovered work item dictionaries
        """
        work_items = []
        root_folder_id = self.config.google_drive_root_folder_id
        
        if not root_folder_id:
            logger.error("Google Drive root folder ID not configured")
            return work_items
        
        # Find all projects (top-level folders)
        projects = self.drive_client.list_folders(root_folder_id)
        logger.info(f"Found {len(projects)} projects")
        for project in projects:
            project_id = project['id']
            project_name = project['name']
            
            logger.info(f"Scanning project: {project_name}")
            
            # Find source image folders in this project
            source_folders = self.drive_client.find_source_image_folders(project_id)
            
            for source_folder in source_folders:
                category_match = re.search(r'source_images_(.+)', source_folder['name'], re.IGNORECASE)
                category = category_match.group(1) if category_match else 'unknown'
                
                work_items.append({
                    'project_id': project_id,
                    'project_name': project_name,
                    'category': category,
                    'source_folder_id': source_folder['id'],
                    'source_folder_name': source_folder['name']
                })
        
        return work_items
    
    def _validate_and_setup(self, work_items: List[Dict[str, Any]]) -> List[ProcessingWorkItem]:
        """
        Phase 2: Validate work items and set up output folders and sheets.
        
        Args:
            work_items: List of work item dictionaries
        
        Returns:
            List of validated ProcessingWorkItem objects
        """
        ready_items = []
        
        for item in work_items:
            try:
                source_folder_id = item['source_folder_id']
                
                # Check for prompt.txt
                prompt = self.drive_client.read_prompt_file(source_folder_id)
                if not prompt:
                    logger.warning(
                        f"No prompt.txt found in {item['source_folder_name']}. Skipping."
                    )
                    continue
                
                # Find or create output folder
                output_folder_id = self.drive_client.find_or_create_output_folder(
                    source_folder_id,
                    item['source_folder_name']
                )
                
                if not output_folder_id:
                    logger.warning(
                        f"Could not create output folder for {item['source_folder_name']}. Skipping."
                    )
                    continue
                
                # Get or create Google Sheet
                spreadsheet_id = self.sheets_logger.get_or_create_sheet(
                    item['project_name'],
                    parent_folder_id=item['project_id']
                )
                
                if not spreadsheet_id:
                    logger.warning(
                        f"Could not create sheet for {item['project_name']}. Skipping."
                    )
                    continue
                
                ready_items.append(ProcessingWorkItem(
                    project_id=item['project_id'],
                    project_name=item['project_name'],
                    category=item['category'],
                    source_folder_id=source_folder_id,
                    output_folder_id=output_folder_id,
                    prompt=prompt,
                    spreadsheet_id=spreadsheet_id
                ))
                
                logger.info(
                    f"Ready: {item['project_name']}/{item['category']} "
                    f"(prompt: {truncate_string(prompt, 50)})"
                )
                
            except Exception as e:
                logger.error(
                    f"Error validating work item {item.get('source_folder_name', 'unknown')}: {str(e)}"
                )
                continue
        
        return ready_items
    
    def _process_images(self, work_items: List[ProcessingWorkItem]) -> Dict[str, int]:
        """
        Phase 3: Process all images with provider fallback.
        
        Args:
            work_items: List of ready work items
        
        Returns:
            Dictionary with processing statistics
        """
        results = {
            'processed': 0,
            'failed': 0,
            'skipped': 0
        }
        
        for work_item in work_items:
            logger.info(f"Processing category: {work_item.category}")
            
            # Get all image files
            image_files = self.drive_client.get_image_files(work_item.source_folder_id)
            
            for image_file in image_files:
                try:
                    result = self._process_single_image(image_file, work_item)
                    
                    if result == 'success':
                        results['processed'] += 1
                        self.metrics.record_image_processed()
                    elif result == 'skipped':
                        results['skipped'] += 1
                        self.metrics.record_image_skipped()
                    else:
                        results['failed'] += 1
                        self.metrics.record_image_failed()
                        
                except Exception as e:
                    logger.error(
                        f"Unexpected error processing {image_file['name']}: {str(e)}",
                        exc_info=True
                    )
                    results['failed'] += 1
                    self.metrics.record_image_failed()
                    continue
        
        return results
    
    def _process_single_image(
        self,
        image_file: Dict[str, Any],
        work_item: ProcessingWorkItem
    ) -> str:
        """
        Process a single image.
        
        Args:
            image_file: Image file dictionary from Google Drive
            work_item: Processing work item
        
        Returns:
            'success', 'failed', or 'skipped'
        """
        source_image_url = image_file['url']
        source_image_name = image_file['name']
        
        # Check if already processed
        existing = self.sheets_logger.is_image_processed(
            work_item.spreadsheet_id,
            source_image_url,
            status_filter='success'
        )
        
        if existing:
            logger.info(f"Image already processed (skipping): {source_image_name}")
            return 'skipped'
        
        processing_start = time.time()
        
        try:
            # Download image
            logger.info(f"Downloading image: {source_image_name}")
            image_bytes = self.drive_client.download_file(image_file['id'])
            
            if not image_bytes:
                raise Exception("Failed to download image")
            
            # Load as PIL Image
            image = Image.open(BytesIO(image_bytes))
            
            # Get image metadata
            metadata = get_image_metadata(image)
            
            # Generate mask
            logger.info(f"Generating mask for: {source_image_name}")
            mask, mask_method = self.mask_generator.generate_mask(image)
            
            if not mask:
                raise Exception(f"Failed to generate mask (method: {mask_method})")
            
            # Perform inpainting
            logger.info(f"Inpainting: {source_image_name}")
            inpainted_image, inpaint_metadata = self.inpainter.inpaint(
                image,
                mask,
                work_item.prompt
            )
            
            # Prepare output filename
            output_filename = self._generate_output_filename(source_image_name)
            
            # Save inpainted image to bytes
            output_bytes = BytesIO()
            inpainted_image.save(output_bytes, format='JPEG', quality=95)
            output_bytes.seek(0)
            
            # Upload to output folder
            logger.info(f"Uploading output: {output_filename}")
            output_file = self.drive_client.upload_file(
                output_bytes.getvalue(),
                output_filename,
                work_item.output_folder_id,
                mime_type='image/jpeg'
            )
            
            if not output_file:
                raise Exception("Failed to upload output image")
            
            processing_time = time.time() - processing_start
            
            # Log to Google Sheet
            self.sheets_logger.log_processing_result(
                spreadsheet_id=work_item.spreadsheet_id,
                category=work_item.category,
                source_image_name=source_image_name,
                source_image_url=source_image_url,
                output_image_name=output_filename,
                output_image_url=output_file['url'],
                prompt=work_item.prompt,
                dimensions=metadata['dimensions'],
                provider_used=inpaint_metadata.get('provider_used', 'Unknown'),
                cost=inpaint_metadata.get('cost', 0.0),
                status='success',
                mask_quality=mask_method,
                error_message='',
                processing_time=processing_time,
                api_used=inpaint_metadata.get('provider_used', 'Unknown')
            )
            
            logger.info(
                f"Successfully processed: {source_image_name} "
                f"({processing_time:.2f}s, ${inpaint_metadata.get('cost', 0.0):.4f})"
            )
            
            return 'success'
            
        except ProviderError as e:
            # Provider errors are logged by the provider system
            processing_time = time.time() - processing_start
            self._log_failure(
                image_file,
                work_item,
                str(e),
                processing_time
            )
            return 'failed'
            
        except Exception as e:
            processing_time = time.time() - processing_start
            logger.error(f"Error processing {source_image_name}: {str(e)}")
            self._log_failure(
                image_file,
                work_item,
                str(e),
                processing_time
            )
            return 'failed'
    
    def _log_failure(
        self,
        image_file: Dict[str, Any],
        work_item: ProcessingWorkItem,
        error_message: str,
        processing_time: float
    ) -> None:
        """Log a failed processing attempt."""
        metadata = get_image_metadata(local_image_path)
        
        self.sheets_logger.log_processing_result(
            spreadsheet_id=work_item.spreadsheet_id,
            category=work_item.category,
            source_image_name=image_file['name'],
            source_image_url=image_file['url'],
            output_image_name='',
            output_image_url='',
            prompt=work_item.prompt,
            dimensions=metadata.get('dimensions', 'unknown'),
            provider_used='None',
            cost=0.0,
            status='failed',
            mask_quality='failed',
            error_message=error_message,
            processing_time=processing_time,
            api_used='None'
        )
    
    def _generate_output_filename(self, source_filename: str) -> str:
        """Generate output filename from source filename."""
        path = Path(source_filename)
        stem = path.stem
        return f"{stem}_output.jpg"
    
    def _finalize_logs(
        self,
        work_items: List[ProcessingWorkItem],
        results: Dict[str, int],
        start_time: datetime
    ) -> None:
        """
        Phase 4: Write execution logs to Google Drive.
        
        Args:
            work_items: List of processed work items
            results: Processing results dictionary
            start_time: Pipeline start time
        """
        # Create log content
        log_lines = []
        log_lines.append("=" * 60)
        log_lines.append("BACKGROUND REPLACEMENT PROCESSING LOG")
        log_lines.append("=" * 60)
        log_lines.append(f"Start Time: {format_timestamp(start_time)}")
        log_lines.append(f"End Time: {format_timestamp()}")
        log_lines.append("")
        
        log_lines.append("SUMMARY")
        log_lines.append("-" * 60)
        log_lines.append(f"Total Work Items: {len(work_items)}")
        log_lines.append(f"Images Processed: {results['processed']}")
        log_lines.append(f"Images Failed: {results['failed']}")
        log_lines.append(f"Images Skipped: {results['skipped']}")
        log_lines.append("")
        
        # Add provider statistics
        summary = self.metrics.format_summary_report()
        log_lines.append(summary)
        
        log_content = "\n".join(log_lines)
        
        # Write to Google Drive (if logs folder configured)
        # This would require additional implementation to find/create logs folder
        logger.info("Execution logs finalized")
        logger.info(summary)

