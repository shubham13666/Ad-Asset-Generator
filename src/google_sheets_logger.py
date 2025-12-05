"""Google Sheets logging and state management."""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.utils import format_timestamp, truncate_string

logger = logging.getLogger(__name__)


class GoogleSheetsLogger:
    """Google Sheets client for logging processing results."""
    
    # Sheet column headers (15 columns as per specification)
    SHEET_HEADERS = [
        'Timestamp',
        'Category',
        'Source Image Name',
        'Source Image URL',
        'Output Image Name',
        'Output Image URL',
        'Prompt Used',
        'Image Dimensions',
        'Provider Used',
        'Cost',
        'Status',
        'Mask Quality',
        'Error Message',
        'Processing Time (sec)',
        'API Used'
    ]
    
    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize Google Sheets logger.
        
        Args:
            credentials_path: Path to service account JSON file.
                            If None, uses GOOGLE_APPLICATION_CREDENTIALS env var.
        """
        self.credentials_path = credentials_path or os.getenv('GOOGLE_SHEETS_CREDENTIALS_PATH')
        
        if not self.credentials_path or not os.path.exists(self.credentials_path):
            raise ValueError(
                "Google credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS or provide credentials_path"
            )
        
        # Load credentials with Sheets scope
        self.credentials = service_account.Credentials.from_service_account_file(
            self.credentials_path,
            scopes=[
                'https://www.googleapis.com/auth/drive',
                'https://www.googleapis.com/auth/spreadsheets'
            ]
        )
        
        # Build services
        self.sheets_service = build('sheets', 'v4', credentials=self.credentials)
        self.drive_service = build('drive', 'v3', credentials=self.credentials)
        
        logger.info("Google Sheets logger initialized")
    
    def get_or_create_sheet(self, project_name: str, parent_folder_id: Optional[str] = None) -> Optional[str]:
        """
        Get or create Google Sheet for a project.
        
        Args:
            project_name: Project name
            parent_folder_id: Optional parent folder ID to create sheet in
        
        Returns:
            Spreadsheet ID, or None if error
        """
        sheet_name = f"{project_name}_Background_Log"
        
        # Check if sheet already exists
        existing_sheet_id = self._find_sheet_by_name(sheet_name, parent_folder_id)
        
        if existing_sheet_id:
            logger.info(f"Found existing sheet: {sheet_name} ({existing_sheet_id})")
            # Ensure headers are present
            self._ensure_headers(existing_sheet_id)
            return existing_sheet_id
        
        # Create new sheet
        logger.info(f"Creating new sheet: {sheet_name} :{parent_folder_id}")
        return self._create_sheet(sheet_name, parent_folder_id)
    
    def _find_sheet_by_name(
        self,
        sheet_name: str,
        parent_folder_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Find spreadsheet by name.
        
        Args:
            sheet_name: Sheet name
            parent_folder_id: Optional parent folder ID to search in
        
        Returns:
            Spreadsheet ID if found, None otherwise
        """
        try:
            query = f"name='{sheet_name}' and mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
            
            if parent_folder_id:
                query += f" and '{parent_folder_id}' in parents"
            
            results = self.drive_service.files().list(
                q=query,
                fields="files(id, name)"
            ).execute()
            
            items = results.get('files', [])
            
            if items:
                return items[0]['id']
            
            return None
            
        except HttpError as e:
            logger.error(f"Error finding sheet {sheet_name}: {str(e)}")
            return None
    
    def _create_sheet(
        self,
        sheet_name: str,
        parent_folder_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a new Google Sheet.
        
        Args:
            sheet_name: Sheet name
            parent_folder_id: Optional parent folder ID
        
        Returns:
            Spreadsheet ID, or None if error
        """
        try:
            # Create spreadsheet
            spreadsheet = {
                'properties': {
                    'title': sheet_name
                },
                'sheets': [{
                    'properties': {
                        'title': 'Processing Log',
                        'gridProperties': {
                            'rowCount': 1000,
                            'columnCount': len(self.SHEET_HEADERS)
                        }
                    }
                }]
            }
            
            spreadsheet = self.sheets_service.spreadsheets().create(
                body=spreadsheet,
                fields='spreadsheetId'
            ).execute()
            
            spreadsheet_id = spreadsheet.get('spreadsheetId')
            logger.info(f"Created spreadsheet {sheet_name} with ID {spreadsheet_id}")
            
            # Move to parent folder if specified
            if parent_folder_id:
                file = self.drive_service.files().get(
                    fileId=spreadsheet_id,
                    fields='parents'
                ).execute()
                
                previous_parents = ",".join(file.get('parents'))
                
                self.drive_service.files().update(
                    fileId=spreadsheet_id,
                    addParents=parent_folder_id,
                    removeParents=previous_parents,
                    fields='id, parents'
                ).execute()
            
            # Add headers
            self._ensure_headers(spreadsheet_id)
            
            return spreadsheet_id
            
        except HttpError as e:
            logger.error(f"Error creating sheet {sheet_name}: {str(e)}")
            return None
    
    def _ensure_headers(self, spreadsheet_id: str) -> None:
        """
        Ensure sheet has headers (add if missing).
        
        Args:
            spreadsheet_id: Spreadsheet ID
        """
        try:
            # Check if headers exist
            range_name = 'Processing Log!A1:O1'
            result = self.sheets_service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=range_name
            ).execute()
            
            values = result.get('values', [])
            
            # If no headers or headers don't match, add them
            if not values or values[0] != self.SHEET_HEADERS:
                body = {
                    'values': [self.SHEET_HEADERS]
                }
                
                self.sheets_service.spreadsheets().values().update(
                    spreadsheetId=spreadsheet_id,
                    range=range_name,
                    valueInputOption='RAW',
                    body=body
                ).execute()
                
                # Format header row
                requests = [{
                    'repeatCell': {
                        'range': {
                            'sheetId': 0,
                            'startRowIndex': 0,
                            'endRowIndex': 1
                        },
                        'cell': {
                            'userEnteredFormat': {
                                'backgroundColor': {'red': 0.2, 'green': 0.2, 'blue': 0.2},
                                'textFormat': {
                                    'foregroundColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0},
                                    'bold': True
                                }
                            }
                        },
                        'fields': 'userEnteredFormat(backgroundColor,textFormat)'
                    }
                }]
                
                self.sheets_service.spreadsheets().batchUpdate(
                    spreadsheetId=spreadsheet_id,
                    body={'requests': requests}
                ).execute()
                
                logger.info(f"Added headers to sheet {spreadsheet_id}")
                
        except HttpError as e:
            logger.error(f"Error ensuring headers: {str(e)}")
    
    def log_processing_result(
        self,
        spreadsheet_id: str,
        category: str,
        source_image_name: str,
        source_image_url: str,
        output_image_name: str,
        output_image_url: str,
        prompt: str,
        dimensions: str,
        provider_used: str,
        cost: float,
        status: str,
        mask_quality: str,
        error_message: str,
        processing_time: float,
        api_used: Optional[str] = None
    ) -> bool:
        """
        Log a processing result to the sheet.
        
        Args:
            spreadsheet_id: Spreadsheet ID
            category: Category name
            source_image_name: Source image filename
            source_image_url: Google Drive URL to source image
            output_image_name: Output image filename
            output_image_url: Google Drive URL to output image
            prompt: Prompt used
            dimensions: Image dimensions (e.g., "1920x1080")
            provider_used: Provider name that succeeded
            cost: Cost in USD
            status: Processing status ("success" or "failed")
            mask_quality: Mask quality method ("rembg", "removebg", "opencv", "auto", "manual")
            error_message: Error message if failed (empty if success)
            processing_time: Processing time in seconds
            api_used: API used (same as provider_used, kept for compatibility)
        
        Returns:
            True if logged successfully, False otherwise
        """
        try:
            timestamp = format_timestamp()
            api_used = api_used or provider_used
            
            # Truncate long values
            prompt = truncate_string(prompt, max_length=500)
            error_message = truncate_string(error_message, max_length=500)
            
            values = [[
                timestamp,
                category,
                source_image_name,
                source_image_url,
                output_image_name,
                output_image_url,
                prompt,
                dimensions,
                provider_used,
                cost,
                status,
                mask_quality,
                error_message,
                processing_time,
                api_used
            ]]
            
            body = {
                'values': values
            }
            
            result = self.sheets_service.spreadsheets().values().append(
                spreadsheetId=spreadsheet_id,
                range='Processing Log!A:O',
                valueInputOption='RAW',
                insertDataOption='INSERT_ROWS',
                body=body
            ).execute()
            
            logger.info(f"Logged processing result to sheet: {source_image_name} ({status})")
            return True
            
        except HttpError as e:
            logger.error(f"Error logging to sheet: {str(e)}")
            return False
    
    def is_image_processed(
        self,
        spreadsheet_id: str,
        source_image_url: str,
        status_filter: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Check if an image has already been processed.
        
        Args:
            spreadsheet_id: Spreadsheet ID
            source_image_url: Source image URL
            status_filter: Optional status filter ("success", "failed", etc.)
        
        Returns:
            Dictionary with row data if found, None otherwise
        """
        try:
            # Read all rows (skip header)
            range_name = 'Processing Log!A2:O'
            result = self.sheets_service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=range_name
            ).execute()
            
            values = result.get('values', [])
            
            for row in values:
                if len(row) < 4:
                    continue
                
                # Column 3 (index 3) is Source Image URL
                row_url = row[3] if len(row) > 3 else ""
                row_status = row[10] if len(row) > 10 else ""
                
                if row_url == source_image_url:
                    # Apply status filter if provided
                    if status_filter and row_status != status_filter:
                        continue
                    
                    # Return row data
                    return {
                        'timestamp': row[0] if len(row) > 0 else '',
                        'category': row[1] if len(row) > 1 else '',
                        'source_image_name': row[2] if len(row) > 2 else '',
                        'source_image_url': row_url,
                        'output_image_name': row[4] if len(row) > 4 else '',
                        'output_image_url': row[5] if len(row) > 5 else '',
                        'status': row_status,
                        'provider_used': row[8] if len(row) > 8 else '',
                        'cost': float(row[9]) if len(row) > 9 and row[9] else 0.0
                    }
            
            return None
            
        except HttpError as e:
            logger.error(f"Error checking if image processed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error parsing sheet data: {str(e)}")
            return None
    
    def get_sheet_url(self, spreadsheet_id: str) -> str:
        """
        Get shareable URL for the spreadsheet.
        
        Args:
            spreadsheet_id: Spreadsheet ID
        
        Returns:
            Shareable URL
        """
        return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"

