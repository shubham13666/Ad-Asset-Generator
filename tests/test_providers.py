"""Unit tests for provider system."""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from PIL import Image
import io

from providers.base import (
    BaseInpaintProvider,
    ProviderError,
    ProviderAuthenticationError,
    ProviderAPIError
)
from providers.tencent_hunyuan import TencentHunyuanProvider
from providers.factory import InpaintingProviderFactory, ProviderWithFallback
from providers.metrics import ProviderMetrics, ProviderStats


class TestBaseProvider:
    """Test base provider abstract class."""
    
    def test_base_provider_is_abstract(self):
        """Test that BaseInpaintProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseInpaintProvider()
    
    def test_provider_error_hierarchy(self):
        """Test provider error exception hierarchy."""
        assert issubclass(ProviderAuthenticationError, ProviderError)
        assert issubclass(ProviderAPIError, ProviderError)


class TestTencentProvider:
    """Test Tencent Hunyuan provider."""
    
    @pytest.fixture
    def provider_config(self):
        """Fixture for provider configuration."""
        return {
            'secret_id': 'test_secret_id',
            'secret_key': 'test_secret_key',
            'region': 'ap-shanghai',
            'timeout': 60,
            'max_retries': 3,
            'retry_delay': 2
        }
    
    @pytest.fixture
    def test_image(self):
        """Fixture for test image."""
        return Image.new('RGB', (512, 512), color='red')
    
    @pytest.fixture
    def test_mask(self):
        """Fixture for test mask."""
        mask = Image.new('L', (512, 512), color=0)
        # Set some pixels to white (background to replace)
        pixels = mask.load()
        for x in range(256, 512):
            for y in range(256, 512):
                pixels[x, y] = 255
        return mask
    
    def test_provider_initialization(self, provider_config):
        """Test provider initialization."""
        provider = TencentHunyuanProvider(provider_config)
        assert provider.secret_id == 'test_secret_id'
        assert provider.secret_key == 'test_secret_key'
        assert provider.region == 'ap-shanghai'
        assert provider.get_provider_name() == "Tencent Hunyuan 3.0"
    
    def test_get_cost_estimate(self, provider_config):
        """Test cost estimate (should be FREE)."""
        provider = TencentHunyuanProvider(provider_config)
        assert provider.get_cost_estimate() == 0.0
        assert provider.get_cost_estimate(100) == 0.0
    
    @patch('providers.tencent_hunyuan.TENCENT_SDK_AVAILABLE', True)
    def test_validate_image(self, provider_config, test_image):
        """Test image validation."""
        provider = TencentHunyuanProvider(provider_config)
        
        # Valid image
        provider._validate_image(test_image)
        
        # Invalid: not PIL Image
        with pytest.raises(ValueError, match="Image must be a PIL Image"):
            provider._validate_image("not an image")
        
        # Invalid: too small
        small_image = Image.new('RGB', (32, 32))
        with pytest.raises(ValueError, match="too small"):
            provider._validate_image(small_image)
        
        # Invalid: too large
        large_image = Image.new('RGB', (5000, 5000))
        with pytest.raises(ValueError, match="too large"):
            provider._validate_image(large_image)
    
    def test_validate_mask(self, provider_config, test_image, test_mask):
        """Test mask validation."""
        provider = TencentHunyuanProvider(provider_config)
        
        # Valid mask
        provider._validate_mask(test_mask, test_image)
        
        # Invalid: wrong size
        wrong_mask = Image.new('L', (256, 256))
        with pytest.raises(ValueError, match="does not match"):
            provider._validate_mask(wrong_mask, test_image)
    
    def test_prepare_image_for_api(self, provider_config):
        """Test image preparation for API."""
        provider = TencentHunyuanProvider(provider_config)
        
        # RGB image
        rgb_image = Image.new('RGB', (100, 100), color='blue')
        image_bytes = provider._prepare_image_for_api(rgb_image)
        assert isinstance(image_bytes, bytes)
        assert len(image_bytes) > 0
        
        # RGBA image (should convert to RGB)
        rgba_image = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        image_bytes = provider._prepare_image_for_api(rgba_image)
        assert isinstance(image_bytes, bytes)
    
    def test_prepare_mask_for_api(self, provider_config, test_mask):
        """Test mask preparation for API."""
        provider = TencentHunyuanProvider(provider_config)
        
        mask_bytes = provider._prepare_mask_for_api(test_mask)
        assert isinstance(mask_bytes, bytes)
        assert len(mask_bytes) > 0


class TestProviderFactory:
    """Test provider factory."""
    
    def test_factory_registry(self):
        """Test provider registry."""
        assert 'tencent' in InpaintingProviderFactory._provider_registry
        assert InpaintingProviderFactory._provider_registry['tencent'] == TencentHunyuanProvider
    
    def test_create_provider_with_name(self):
        """Test creating provider with explicit name."""
        config = {
            'secret_id': 'test_id',
            'secret_key': 'test_key'
        }
        
        provider = InpaintingProviderFactory.create_provider('tencent', config)
        assert isinstance(provider, TencentHunyuanProvider)
        assert provider.secret_id == 'test_id'
    
    @patch.dict('os.environ', {'INPAINT_PROVIDER': 'tencent'})
    def test_create_provider_from_env(self):
        """Test creating provider from environment variable."""
        config = {
            'secret_id': 'test_id',
            'secret_key': 'test_key'
        }
        
        provider = InpaintingProviderFactory.create_provider(config=config)
        assert isinstance(provider, TencentHunyuanProvider)
    
    def test_create_provider_invalid_name(self):
        """Test creating provider with invalid name."""
        with pytest.raises(ValueError, match="not found"):
            InpaintingProviderFactory.create_provider('invalid_provider')
    
    def test_register_provider(self):
        """Test registering a new provider."""
        class TestProvider(BaseInpaintProvider):
            def authenticate(self):
                return True
            
            def inpaint(self, image, mask, prompt, **kwargs):
                return image, {}
            
            def get_cost_estimate(self, num_images=1):
                return 1.0
            
            def get_provider_name(self):
                return "Test Provider"
        
        InpaintingProviderFactory.register_provider('test', TestProvider)
        assert 'test' in InpaintingProviderFactory._provider_registry
        
        # Cleanup
        del InpaintingProviderFactory._provider_registry['test']


class TestProviderWithFallback:
    """Test provider fallback system."""
    
    @pytest.fixture
    def mock_primary_provider(self):
        """Fixture for mock primary provider."""
        provider = Mock(spec=BaseInpaintProvider)
        provider.get_provider_name.return_value = "Primary Provider"
        provider.get_cost_estimate.return_value = 0.0
        return provider
    
    @pytest.fixture
    def mock_fallback_provider(self):
        """Fixture for mock fallback provider."""
        provider = Mock(spec=BaseInpaintProvider)
        provider.get_provider_name.return_value = "Fallback Provider"
        provider.get_cost_estimate.return_value = 0.5
        return provider
    
    @pytest.fixture
    def test_image(self):
        """Fixture for test image."""
        return Image.new('RGB', (512, 512), color='red')
    
    @pytest.fixture
    def test_mask(self):
        """Fixture for test mask."""
        return Image.new('L', (512, 512), color=255)
    
    def test_fallback_success_on_primary(self, mock_primary_provider, mock_fallback_provider, test_image, test_mask):
        """Test successful inpainting with primary provider."""
        result_image = Image.new('RGB', (512, 512), color='blue')
        metadata = {'cost': 0.0, 'processing_time': 1.0}
        
        mock_primary_provider.inpaint.return_value = (result_image, metadata)
        
        fallback_wrapper = ProviderWithFallback(
            primary_provider=mock_primary_provider,
            fallback_providers=[mock_fallback_provider]
        )
        
        image, meta = fallback_wrapper.inpaint(test_image, test_mask, "test prompt")
        
        assert image == result_image
        assert meta['provider_used'] == "Primary Provider"
        assert meta['fallback_used'] is False
        mock_primary_provider.inpaint.assert_called_once()
        mock_fallback_provider.inpaint.assert_not_called()
    
    def test_fallback_on_primary_failure(self, mock_primary_provider, mock_fallback_provider, test_image, test_mask):
        """Test fallback to backup provider when primary fails."""
        result_image = Image.new('RGB', (512, 512), color='green')
        metadata = {'cost': 0.5, 'processing_time': 2.0}
        
        mock_primary_provider.inpaint.side_effect = ProviderAPIError("Primary failed")
        mock_fallback_provider.inpaint.return_value = (result_image, metadata)
        
        fallback_wrapper = ProviderWithFallback(
            primary_provider=mock_primary_provider,
            fallback_providers=[mock_fallback_provider]
        )
        
        image, meta = fallback_wrapper.inpaint(test_image, test_mask, "test prompt")
        
        assert image == result_image
        assert meta['provider_used'] == "Fallback Provider"
        assert meta['fallback_used'] is True
        mock_primary_provider.inpaint.assert_called_once()
        mock_fallback_provider.inpaint.assert_called_once()
    
    def test_fallback_all_providers_fail(self, mock_primary_provider, mock_fallback_provider, test_image, test_mask):
        """Test exception when all providers fail."""
        mock_primary_provider.inpaint.side_effect = ProviderAPIError("Primary failed")
        mock_fallback_provider.inpaint.side_effect = ProviderAPIError("Fallback failed")
        
        fallback_wrapper = ProviderWithFallback(
            primary_provider=mock_primary_provider,
            fallback_providers=[mock_fallback_provider]
        )
        
        with pytest.raises(ProviderError, match="All providers failed"):
            fallback_wrapper.inpaint(test_image, test_mask, "test prompt")


class TestProviderMetrics:
    """Test provider metrics tracking."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = ProviderMetrics()
        assert len(metrics._stats) == 0
    
    def test_record_attempt(self):
        """Test recording provider attempts."""
        metrics = ProviderMetrics()
        
        metrics.record_attempt('tencent', True, 2.5, 0.0)
        metrics.record_attempt('tencent', True, 3.0, 0.0)
        metrics.record_attempt('alibaba', False, 1.0, 0.0, "API error")
        
        stats = metrics.get_stats()
        assert 'tencent' in stats
        assert stats['tencent']['attempts'] == 2
        assert stats['tencent']['successes'] == 2
        assert stats['tencent']['failures'] == 0
        
        assert 'alibaba' in stats
        assert stats['alibaba']['attempts'] == 1
        assert stats['alibaba']['failures'] == 1
    
    def test_image_tracking(self):
        """Test image processing tracking."""
        metrics = ProviderMetrics()
        
        metrics.record_image_processed()
        metrics.record_image_processed()
        metrics.record_image_failed()
        metrics.record_image_skipped()
        
        summary = metrics.get_summary()
        assert summary['processed'] == 2
        assert summary['failed'] == 1
        assert summary['skipped'] == 1
        assert summary['total_images'] == 4
    
    def test_format_summary_report(self):
        """Test summary report formatting."""
        metrics = ProviderMetrics()
        
        metrics.record_attempt('tencent', True, 2.5, 0.0)
        metrics.record_image_processed()
        
        report = metrics.format_summary_report()
        assert 'PROCESSING SUMMARY' in report
        assert 'tencent' in report
        assert 'Total Images' in report



