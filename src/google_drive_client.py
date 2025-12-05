"""Google Drive API client for folder discovery and file operations."""

import os
import logging
import re
import pickle
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from io import BytesIO

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
import time

logger = logging.getLogger(__name__)


class GoogleDriveClient:
    """Client for interacting with Google Drive API."""
    
    # MIME types for images
    IMAGE_MIME_TYPES = {
        'image/jpeg': ['.jpg', '.jpeg'],
        'image/png': ['.png'],
        'image/gif': ['.gif'],
        'image/webp': ['.webp']
    }
    
    # Supported image extensions
    IMAGE_EXTENSIONS = {ext for exts in IMAGE_MIME_TYPES.values() for ext in exts}
    
    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize Google Drive client.
        
        Args:
            credentials_path: Path to service account JSON file.
                            If None, uses GOOGLE_APPLICATION_CREDENTIALS env var.
        """
        self.credentials_path = credentials_path or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        if not self.credentials_path or not os.path.exists(self.credentials_path):
            raise ValueError(
                "Google credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS or provide credentials_path"
            )
        
        # Load credentials
        self.credentials = service_account.Credentials.from_service_account_file(
            self.credentials_path,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        
        # Build Drive service
        self.service = build('drive', 'v3', credentials=self.credentials)
        logger.info("Google Drive client initialized")

    SCOPES = ['https://www.googleapis.com/auth/drive']
    
    def __init__(self, credentials_path: str, root_folder_id: str):
        self.credentials_path = credentials_path
        self.root_folder_id = root_folder_id
        self.creds = None
        self._authenticate_oauth()
        self.service = build('drive', 'v3', credentials=self.creds)
    
    def _authenticate_oauth(self):
        """Authenticate using OAuth 2.0 (personal account)."""
        creds = None
        
        # Check for cached token
        # if os.path.exists('token.pickle'):
        #     with open('token.pickle', 'rb') as token:
        #         creds = pickle.load(token)
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path,
                    self.SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            # Save token for next time
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        self.creds = creds
    
    def list_folders(self, folder_id: str, name_filter: Optional[str] = None) -> List[Dict]:
        """
        List folders within a parent folder.
        
        Args:
            folder_id: Parent folder ID
            name_filter: Optional regex pattern to filter folder names
        
        Returns:
            List of folder dictionaries with id, name, and metadata
        """
        folders = []
        page_token = None
        
        try:
            while True:
                query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
                
                results = self.service.files().list(
                    q=query,
                    fields="nextPageToken, files(id, name, mimeType, createdTime, modifiedTime)",
                    pageToken=page_token,
                    pageSize=100
                ).execute()
                
                items = results.get('files', [])
                
                for item in items:
                    folder_name = item['name']
                    
                    # Apply name filter if provided
                    if name_filter:
                        if not re.search(name_filter, folder_name, re.IGNORECASE):
                            continue
                    
                    folders.append({
                        'id': item['id'],
                        'name': folder_name,
                        'mime_type': item['mimeType'],
                        'created_time': item.get('createdTime'),
                        'modified_time': item.get('modifiedTime')
                    })
                
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
            
            logger.info(f"Found {len(folders)} folders in {folder_id}")
            return folders
            
        except HttpError as e:
            logger.error(f"Error listing folders: {str(e)}")
            raise
    
    def search_folders_by_pattern(
        self,
        parent_id: str,
        pattern: str,
        recursive: bool = True
    ) -> List[Dict]:
        """
        Search for folders matching a pattern.
        
        Args:
            parent_id: Root folder ID to search from
            pattern: Regex pattern to match folder names
            recursive: Whether to search recursively
        
        Returns:
            List of matching folder dictionaries
        """
        matching_folders = []
        
        def search_recursive(folder_id: str, depth: int = 0) -> None:
            """Recursively search folders."""
            if depth > 10:  # Prevent infinite recursion
                logger.warning(f"Maximum recursion depth reached at folder {folder_id}")
                return
            
            folders = self.list_folders(folder_id)
            
            for folder in folders:
                folder_name = folder['name']
                
                if re.search(pattern, folder_name, re.IGNORECASE):
                    folder['parent_id'] = folder_id
                    matching_folders.append(folder)
                    logger.debug(f"Found matching folder: {folder_name} ({folder['id']})")
                
                # Continue recursive search if enabled
                if recursive:
                    search_recursive(folder['id'], depth + 1)
        
        search_recursive(parent_id)
        return matching_folders
    
    def find_source_image_folders(self, root_folder_id: str) -> List[Dict]:
        """
        Find all folders matching source_images_* pattern.
        
        Args:
            root_folder_id: Root folder ID (typically the Drive root or project folder)
        
        Returns:
            List of source image folder dictionaries
        """
        pattern = r'source_images_\w+'
        folders = self.search_folders_by_pattern(root_folder_id, pattern, recursive=True)
        
        logger.info(f"Found {len(folders)} source image folders")
        return folders
    
    def get_folder_files(
        self,
        folder_id: str,
        mime_type_filter: Optional[str] = None,
        extension_filter: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        List files in a folder.
        
        Args:
            folder_id: Folder ID
            mime_type_filter: Optional MIME type filter
            extension_filter: Optional list of file extensions to include
        
        Returns:
            List of file dictionaries
        """
        files = []
        page_token = None
        
        try:
            while True:
                query = f"'{folder_id}' in parents and trashed=false"
                
                if mime_type_filter:
                    query += f" and mimeType='{mime_type_filter}'"
                
                results = self.service.files().list(
                    q=query,
                    fields="nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime, webViewLink)",
                    pageToken=page_token,
                    pageSize=100
                ).execute()
                
                items = results.get('files', [])
                
                for item in items:
                    file_name = item['name']
                    file_ext = Path(file_name).suffix.lower()
                    
                    # Apply extension filter if provided
                    if extension_filter:
                        if file_ext not in extension_filter:
                            continue
                    
                    files.append({
                        'id': item['id'],
                        'name': file_name,
                        'mime_type': item['mimeType'],
                        'size': item.get('size', '0'),
                        'extension': file_ext,
                        'url': item.get('webViewLink', ''),
                        'created_time': item.get('createdTime'),
                        'modified_time': item.get('modifiedTime')
                    })
                
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
            
            logger.debug(f"Found {len(files)} files in folder {folder_id}")
            return files
            
        except HttpError as e:
            logger.error(f"Error listing files: {str(e)}")
            raise
    
    def get_image_files(self, folder_id: str) -> List[Dict]:
        """
        Get all image files from a folder.
        
        Args:
            folder_id: Folder ID
        
        Returns:
            List of image file dictionaries
        """
        image_files = []
        all_files = self.get_folder_files(folder_id)
        
        for file_info in all_files:
            # Check if it's an image by extension
            if file_info['extension'] in self.IMAGE_EXTENSIONS:
                image_files.append(file_info)
        
        logger.info(f"Found {len(image_files)} image files in folder {folder_id}")
        return image_files
    
    def read_text_file(self, file_id: str) -> Optional[str]:
        """
        Read content of a text file from Google Drive.
        
        Args:
            file_id: File ID
        
        Returns:
            File content as string, or None if error
        """
        try:
            request = self.service.files().get_media(fileId=file_id)
            file_content = BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            content = file_content.getvalue().decode('utf-8')
            logger.debug(f"Read text file {file_id}, {len(content)} bytes")
            return content
            
        except HttpError as e:
            logger.error(f"Error reading text file {file_id}: {str(e)}")
            return None
    
    def find_file_by_name(self, folder_id: str, filename: str) -> Optional[Dict]:
        """
        Find a file by name in a folder.
        
        Args:
            folder_id: Folder ID
            filename: Filename to search for
        
        Returns:
            File dictionary if found, None otherwise
        """
        try:
            query = f"'{folder_id}' in parents and name='{filename}' and trashed=false"
            
            results = self.service.files().list(
                q=query,
                fields="files(id, name, mimeType, size, webViewLink)"
            ).execute()
            
            items = results.get('files', [])
            
            if items:
                file_info = items[0]
                return {
                    'id': file_info['id'],
                    'name': file_info['name'],
                    'mime_type': file_info['mimeType'],
                    'size': file_info.get('size', '0'),
                    'url': file_info.get('webViewLink', '')
                }
            
            return None
            
        except HttpError as e:
            logger.error(f"Error finding file {filename}: {str(e)}")
            return None
    
    def read_prompt_file(self, folder_id: str) -> Optional[str]:
        """
        Read prompt.txt file from a folder.
        
        Args:
            folder_id: Folder ID
        
        Returns:
            Prompt content as string, or None if not found
        """
        prompt_file = self.find_file_by_name(folder_id, 'prompt.txt')
        
        if not prompt_file:
            logger.warning(f"prompt.txt not found in folder {folder_id}")
            return None
        
        prompt_content = self.read_text_file(prompt_file['id'])
        
        if prompt_content:
            prompt_content = prompt_content.strip()
            logger.info(f"Read prompt from folder {folder_id}: {len(prompt_content)} characters")
        
        return prompt_content
    
    def download_file(self, file_id: str, output_path: Optional[str] = None) -> Optional[bytes]:
        """
        Download a file from Google Drive.
        
        Args:
            file_id: File ID
            output_path: Optional local path to save file.
                        If None, returns file content as bytes.
        
        Returns:
            File content as bytes, or None if error
        """
        try:
            request = self.service.files().get_media(fileId=file_id)
            
            if output_path:
                # Save to file
                with open(output_path, 'wb') as f:
                    downloader = MediaIoBaseDownload(f, request)
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                
                logger.info(f"Downloaded file {file_id} to {output_path}")
                return None
            else:
                # Return as bytes
                file_content = BytesIO()
                downloader = MediaIoBaseDownload(file_content, request)
                
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                
                content = file_content.getvalue()
                logger.debug(f"Downloaded file {file_id}, {len(content)} bytes")
                return content
            
        except HttpError as e:
            logger.error(f"Error downloading file {file_id}: {str(e)}")
            return None
    
    def create_folder(self, name: str, parent_id: str) -> Optional[str]:
        """
        Create a folder in Google Drive.
        
        Args:
            name: Folder name
            parent_id: Parent folder ID
        
        Returns:
            Created folder ID, or None if error
        """
        try:
            file_metadata = {
                'name': name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_id]
            }
            
            folder = self.service.files().create(
                body=file_metadata,
                fields='id'
            ).execute()
            
            folder_id = folder.get('id')
            logger.info(f"Created folder '{name}' with ID {folder_id}")
            return folder_id
            
        except HttpError as e:
            logger.error(f"Error creating folder '{name}': {str(e)}")
            return None
    
    def find_or_create_output_folder(
        self,
        source_folder_id: str,
        source_folder_name: str
    ) -> Optional[str]:
        """
        Find or create output folder for a source folder.
        
        Args:
            source_folder_id: Source folder ID
            source_folder_name: Source folder name (e.g., 'source_images_category1')
        
        Returns:
            Output folder ID, or None if error
        """
        # Extract category from source folder name
        match = re.search(r'source_images_(.+)', source_folder_name, re.IGNORECASE)
        if not match:
            logger.warning(f"Could not extract category from folder name: {source_folder_name}")
            return None
        
        category = match.group(1)
        output_folder_name = f"Output_Folder_{category}"
        
        # Get parent folder ID
        try:
            source_folder = self.service.files().get(
                fileId=source_folder_id,
                fields='parents'
            ).execute()
            
            parent_id = source_folder['parents'][0] if source_folder.get('parents') else None
            
            if not parent_id:
                logger.error(f"Could not determine parent folder for {source_folder_id}")
                return None
            
            # Check if output folder already exists
            output_folder = self.find_file_by_name(parent_id, output_folder_name)
            
            if output_folder:
                logger.info(f"Output folder already exists: {output_folder_name} ({output_folder['id']})")
                return output_folder['id']
            
            # Create output folder
            logger.info(f"Creating output folder: {output_folder_name}")
            return self.create_folder(output_folder_name, parent_id)
            
        except HttpError as e:
            logger.error(f"Error finding/creating output folder: {str(e)}")
            return None
    
    def upload_file(
        self,
        file_content: bytes,
        filename: str,
        folder_id: str,
        mime_type: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Upload a file to Google Drive.
        
        Args:
            file_content: File content as bytes
            filename: Filename
            folder_id: Destination folder ID
            mime_type: Optional MIME type
        
        Returns:
            Uploaded file dictionary with id and url, or None if error
        """
        try:
            # Determine MIME type if not provided
            if not mime_type:
                ext = Path(filename).suffix.lower()
                for mime, exts in self.IMAGE_MIME_TYPES.items():
                    if ext in exts:
                        mime_type = mime
                        break
                
                if not mime_type:
                    mime_type = 'application/octet-stream'
            
            file_metadata = {
                'name': filename,
                'parents': [folder_id]
            }
            
            # Convert BytesIO to bytes, then upload
            if isinstance(file_content, BytesIO):
                file_bytes = file_content.getvalue()
            else:
                file_bytes = file_content

            # Use MediaIoBaseUpload instead for bytes/BytesIO
            from googleapiclient.http import MediaIoBaseUpload
            media = MediaIoBaseUpload(
                BytesIO(file_bytes),
                mimetype=mime_type,
                resumable=True
            )

            
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name, webViewLink, size'
            ).execute()
            
            file_info = {
                'id': file.get('id'),
                'name': file.get('name'),
                'url': file.get('webViewLink', ''),
                'size': file.get('size', '0')
            }
            
            logger.info(f"Uploaded file '{filename}' to folder {folder_id} ({len(file_content)} bytes)")
            return file_info
            
        except HttpError as e:
            logger.error(f"Error uploading file '{filename}': {str(e)}")
            return None
    
    def upload_text_file(
        self,
        content: str,
        filename: str,
        folder_id: str
    ) -> Optional[str]:
        """
        Upload a text file to Google Drive.
        
        Args:
            content: Text content
            filename: Filename
            folder_id: Destination folder ID
        
        Returns:
            Uploaded file ID, or None if error
        """
        file_bytes = content.encode('utf-8')
        file_info = self.upload_file(
            file_bytes,
            filename,
            folder_id,
            mime_type='text/plain'
        )
        
        return file_info['id'] if file_info else None



