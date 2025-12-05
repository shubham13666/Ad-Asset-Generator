"""Integration tests for Google Drive and Sheets integration."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from io import BytesIO

from src.google_drive_client import GoogleDriveClient
from src.google_sheets_logger import GoogleSheetsLogger


class TestGoogleDriveClient:
    """Test Google Drive client."""
    
    @pytest.fixture
    def mock_credentials(self):
        """Mock Google credentials."""
        with patch('src.google_drive_client.service_account.Credentials') as mock:
            yield mock
    
    @pytest.fixture
    def mock_service(self):
        """Mock Google Drive service."""
        return Mock()
    
    def test_initialization(self, mock_credentials):
        """Test client initialization."""
        with patch('src.google_drive_client.build') as mock_build:
            mock_build.return_value = Mock()
            
            client = GoogleDriveClient(credentials_path='/path/to/creds.json')
            
            assert client.credentials_path == '/path/to/creds.json'
            mock_build.assert_called_once()
    
    def test_list_folders(self, mock_service):
        """Test listing folders."""
        # Mock implementation would go here
        pass
    
    def test_find_source_image_folders(self):
        """Test finding source image folders."""
        # Mock implementation would go here
        pass


class TestGoogleSheetsLogger:
    """Test Google Sheets logger."""
    
    @pytest.fixture
    def mock_credentials(self):
        """Mock Google credentials."""
        with patch('src.google_sheets_logger.service_account.Credentials') as mock:
            yield mock
    
    def test_initialization(self, mock_credentials):
        """Test logger initialization."""
        with patch('src.google_sheets_logger.build') as mock_build:
            mock_build.return_value = Mock()
            
            logger = GoogleSheetsLogger(credentials_path='/path/to/creds.json')
            
            assert logger.credentials_path == '/path/to/creds.json'
            assert mock_build.call_count == 2  # Sheets and Drive services
    
    def test_sheet_headers(self):
        """Test sheet headers are correct."""
        assert len(GoogleSheetsLogger.SHEET_HEADERS) == 15
        assert GoogleSheetsLogger.SHEET_HEADERS[0] == 'Timestamp'
        assert GoogleSheetsLogger.SHEET_HEADERS[8] == 'Provider Used'
        assert GoogleSheetsLogger.SHEET_HEADERS[9] == 'Cost'



