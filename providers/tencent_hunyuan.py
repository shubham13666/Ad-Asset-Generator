"""Tencent Hunyuan 3.0 inpainting provider implementation."""

import time
import logging
from typing import Tuple, Dict, Any, Optional
from PIL import Image
import io
import json

from providers.base import (
    BaseInpaintProvider,
    ProviderError,
    ProviderAuthenticationError,
    ProviderAPIError,
    ProviderTimeoutError
)

logger = logging.getLogger(__name__)

try:
    from tencentcloud.common import credential
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile
    from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
    from tencentcloud.hunyuan.v20230901 import hunyuan_client, models
    TENCENT_SDK_AVAILABLE = True
except ImportError:
    TENCENT_SDK_AVAILABLE = False
    logger.warning("Tencent Cloud SDK not available. Install with: pip install tencentcloud-sdk-python")


class TencentHunyuanProvider(BaseInpaintProvider):
    """
    Tencent Hunyuan 3.0 inpainting provider.
    Cost: FREE
    Quality: High
    Speed: 20-25 seconds
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Tencent Hunyuan provider.
        
        Args:
            config: Configuration dictionary with:
                - secret_id: Tencent Cloud Secret ID
                - secret_key: Tencent Cloud Secret Key
                - region: Region (default: ap-shanghai)
                - timeout: Request timeout in seconds (default: 60)
                - max_retries: Maximum retry attempts (default: 3)
                - retry_delay: Delay between retries in seconds (default: 2)
        """
        super().__init__(config)
        self.secret_id = self.config.get('secret_id') or self.config.get('secretId')
        self.secret_key = self.config.get('secret_key') or self.config.get('secretKey')
        self.region = self.config.get('region', 'ap-shanghai')
        self.timeout = self.config.get('timeout', 60)
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 2)
        self._client = None
    
    def get_provider_name(self) -> str:
        """Get provider display name."""
        return "Tencent Hunyuan 3.0"
    
    def authenticate(self) -> bool:
        """
        Authenticate with Tencent Cloud API.
        
        Returns:
            True if authentication successful
        
        Raises:
            ProviderAuthenticationError: If authentication fails
        """
        if not TENCENT_SDK_AVAILABLE:
            raise ProviderAuthenticationError(
                "Tencent Cloud SDK not available. Install with: pip install tencentcloud-sdk-python"
            )
        
        if not self.secret_id or not self.secret_key:
            raise ProviderAuthenticationError(
                "Tencent credentials not configured. Set TENCENT_SECRET_ID and TENCENT_SECRET_KEY"
            )
        
        try:
            # Create credential object
            cred = credential.Credential(self.secret_id, self.secret_key)
            
            # Configure HTTP profile
            http_profile = HttpProfile()
            http_profile.endpoint = "hunyuan.tencentcloudapi.com"
            http_profile.reqTimeout = self.timeout
            
            # Configure client profile
            client_profile = ClientProfile()
            client_profile.httpProfile = http_profile
            
            # Create client (we'll use it for inpainting)
            self._client = hunyuan_client.HunyuanClient(cred, self.region, client_profile)
            
            logger.info(f"Successfully authenticated with {self.get_provider_name()}")
            return True
            
        except TencentCloudSDKException as e:
            error_msg = f"Tencent Cloud authentication failed: {str(e)}"
            logger.error(error_msg)
            raise ProviderAuthenticationError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during authentication: {str(e)}"
            logger.error(error_msg)
            raise ProviderAuthenticationError(error_msg) from e
    
    # def inpaint(
    #     self,
    #     image: Image.Image,
    #     mask: Image.Image,
    #     prompt: str,
    #     **kwargs
    # ) -> Tuple[Image.Image, Dict[str, Any]]:
    #     """
    #     Perform inpainting using Tencent Hunyuan API.
        
    #     Args:
    #         image: Input image (PIL Image)
    #         mask: Binary mask (white=background to replace, black=keep) (PIL Image)
    #         prompt: Text prompt describing desired background
    #         **kwargs: Additional parameters
        
    #     Returns:
    #         Tuple of (inpainted_image, metadata_dict)
        
    #     Raises:
    #         ProviderAPIError: If API call fails
    #         ProviderTimeoutError: If request times out
    #     """
    #     start_time = time.time()
        
    #     # Ensure authenticated
    #     self._ensure_authenticated()
        
    #     # Validate inputs
    #     self._validate_image(image)
    #     self._validate_mask(mask, image)
        
    #     if not prompt or not prompt.strip():
    #         raise ValueError("Prompt cannot be empty")
        
    #     # Prepare images for API
    #     image_bytes = self._prepare_image_for_api(image)
    #     mask_bytes = self._prepare_mask_for_api(mask)
        
    #     # Convert images to base64 for API
    #     import base64
    #     image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    #     mask_base64 = base64.b64encode(mask_bytes).decode('utf-8')
        
    #     # Create inpainting request
    #     # Note: Actual Tencent API endpoint and parameters may vary
    #     # This is a template implementation - adjust based on actual API documentation
    #     request = models.ImageInpaintRequest()
    #     request.ImageBase64 = image_base64
    #     request.MaskBase64 = mask_base64
    #     request.Prompt = prompt
        
    #     # Optional parameters from kwargs
    #     if 'negative_prompt' in kwargs:
    #         request.NegativePrompt = kwargs['negative_prompt']
        
    #     if 'strength' in kwargs:
    #         request.Strength = kwargs['strength']
        
    #     # Execute with retry logic
    #     last_exception = None
    #     for attempt in range(self.max_retries):
    #         try:
    #             logger.info(
    #                 f"Inpainting with {self.get_provider_name()} "
    #                 f"(attempt {attempt + 1}/{self.max_retries})"
    #             )
                
    #             # Make API call
    #             response = self._client.ImageInpaint(request)
                
    #             # Parse response
    #             if hasattr(response, 'ImageBase64') and response.ImageBase64:
    #                 # Decode base64 image
    #                 result_bytes = base64.b64decode(response.ImageBase64)
    #                 result_image = Image.open(io.BytesIO(result_bytes))
                    
    #                 # Calculate metrics
    #                 duration = time.time() - start_time
    #                 cost = 0.0  # FREE
                    
    #                 metadata = {
    #                     "cost": cost,
    #                     "processing_time": duration,
    #                     "model_used": "hunyuan-3.0",
    #                     "provider": self.get_provider_name(),
    #                     "attempts": attempt + 1
    #                 }
                    
    #                 logger.info(
    #                     f"Successfully inpainted image in {duration:.2f}s "
    #                     f"(cost: ${cost:.4f})"
    #                 )
                    
    #                 return result_image, metadata
    #             else:
    #                 raise ProviderAPIError("API response missing image data")
                    
    #         except TencentCloudSDKException as e:
    #             last_exception = e
    #             error_code = getattr(e, 'code', 'Unknown')
    #             error_msg = str(e)
                
    #             logger.warning(
    #                 f"Tencent API error (attempt {attempt + 1}/{self.max_retries}): "
    #                 f"{error_code} - {error_msg}"
    #             )
                
    #             # Check if it's a timeout error
    #             if 'timeout' in error_msg.lower() or error_code == 'RequestTimeout':
    #                 if attempt == self.max_retries - 1:
    #                     raise ProviderTimeoutError(
    #                         f"Request timeout after {self.max_retries} attempts"
    #                     ) from e
    #             # Check if it's a retryable error
    #             elif error_code not in ['InvalidParameter', 'InvalidRequest']:
    #                 if attempt < self.max_retries - 1:
    #                     time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
    #                     continue
                
    #             # Non-retryable error or max retries reached
    #             raise ProviderAPIError(
    #                 f"Tencent API error: {error_code} - {error_msg}"
    #             ) from e
                
    #         except Exception as e:
    #             last_exception = e
    #             logger.error(
    #                 f"Unexpected error during inpainting (attempt {attempt + 1}): {str(e)}"
    #             )
                
    #             if attempt < self.max_retries - 1:
    #                 time.sleep(self.retry_delay * (attempt + 1))
    #                 continue
                
    #             raise ProviderAPIError(f"Unexpected error: {str(e)}") from e
        
    #     # If we get here, all retries failed
    #     raise ProviderAPIError(
    #         f"Inpainting failed after {self.max_retries} attempts"
    #     ) from last_exception

    def _ensure_client(self):
        """Ensure Tencent client is initialized"""
        if not hasattr(self, '_client') or self._client is None:
            self.authenticate()  # This creates self._client
        return self._client


    def inpaint(self, image: Image.Image, mask: Image.Image, prompt: str, **kwargs) -> Tuple[Image.Image, dict]:
    # """Inpaint using Tencent Hunyuan Image Job (async)"""
        import base64
        import io
        import time
        from tencentcloud.hunyuan.v20230901 import models
        
        self._ensure_client()
        
        # Step 1: Prepare images
        image_bytes = self._prepare_image_for_api(image)
        mask_bytes = self._prepare_mask_for_api(mask)
        
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        mask_base64 = base64.b64encode(mask_bytes).decode('utf-8')
        
        logger.info(f"Submitting Hunyuan Image Job: {prompt[:50]}...")
        
        # Step 2: SUBMIT async job
        submit_req = models.SubmitHunyuanImageJobRequest()
        submit_req.Prompt = prompt
        # submit_req.ImageBase64 = image_base64
        # submit_req.MaskBase64 = mask_base64  # Your mask!
        # submit_req.Width = 1024
        # submit_req.Height = 1024
        # submit_req.Steps = 28  # Quality setting
        
        submit_resp = self._client.SubmitHunyuanImageJob(submit_req)
        job_id = submit_resp.JobId
        
        logger.info(f"✅ Job submitted: {job_id}")
        
        # Step 3: POLL for completion (max 5 min)
        for attempt in range(30):  # 30 * 10s = 5 minutes
            time.sleep(10)
            
            query_req = models.QueryHunyuanImageJobRequest()
            query_req.JobId = job_id
            
            query_resp = self._client.QueryHunyuanImageJob(query_req)
            
            status = query_resp.Status
            logger.info(f"Job {job_id}: {status} (attempt {attempt+1}/30)")
            
            if status == "SUCCESS":
                # Extract result image
                result_base64 = query_resp.ResultImage
                result_bytes = base64.b64decode(result_base64)
                result_image = Image.open(io.BytesIO(result_bytes)).convert("RGB")
                
                metadata = {
                    "model_used": "hunyuan-image-job",
                    "prompt": prompt,
                    "job_id": job_id,
                    "status": status
                }
                logger.info("✅ Hunyuan Image Job COMPLETED!")
                return result_image, metadata
                
            elif status == "FAILED":
                error_msg = getattr(query_resp, 'ErrorMessage', 'Unknown error')
                raise ProviderAPIError(f"Job failed: {error_msg}")
        
        raise ProviderTimeoutError("Job timeout after 5 minutes")


    
    def get_cost_estimate(self, num_images: int = 1) -> float:
        """
        Get cost estimate for processing images.
        Tencent Hunyuan is FREE.
        
        Args:
            num_images: Number of images
        
        Returns:
            Cost estimate in USD (always 0.0 for Tencent)
        """
        return 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get provider status."""
        status = super().get_status()
        status.update({
            "region": self.region,
            "sdk_available": TENCENT_SDK_AVAILABLE,
            "credentials_configured": bool(self.secret_id and self.secret_key)
        })
        return status



