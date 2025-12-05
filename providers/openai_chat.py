import logging
from typing import Tuple, Dict, Any, Optional

from PIL import Image
import io

from providers.base import (
    BaseInpaintProvider,
    ProviderError,
)

logger = logging.getLogger(__name__)


class OpenAIChatProvider(BaseInpaintProvider):
    """
    OpenAI ChatGPT-based provider using v1.0.0+ API.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        self.api_key = self.config.get("api_key") or self.config.get("OPENAI_API_KEY")
        self.model = self.config.get("model", "gpt-4o-mini")

        if not self.api_key:
            raise ProviderError("OpenAI API key not configured. Set OPENAI_API_KEY or provider.api_key")

    def get_provider_name(self) -> str:
        return "OpenAI DALL·E 3"

    def authenticate(self) -> bool:
        """Authenticate with OpenAI API."""
        try:
            from openai import OpenAI
            
            if not self.api_key:
                raise ProviderError("OpenAI API key not configured")
            
            client = OpenAI(api_key=self.api_key)
            logger.info(f"Successfully authenticated with {self.get_provider_name()}")
            return True
        except Exception as e:
            logger.error(f"OpenAI authentication failed: {e}")
            raise ProviderError(f"OpenAI authentication failed: {e}") from e

    def get_cost_estimate(self, num_images: int = 1) -> float:
        """
        OpenAI DALL·E 3 pricing:
        - 1024x1024: $0.020 per image
        - 1024x1792: $0.030 per image
        """
        return num_images * 0.020  # $0.020 per image

    def inpaint(
    self,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    **kwargs
) -> Tuple[Image.Image, Dict[str, Any]]:
   

        try:
            from openai import OpenAI
            import base64
            import io
            
            logger.info(f"Calling OpenAI DALL·E with prompt: {prompt[:80]}")

            client = OpenAI(api_key=self.api_key)

            # Step 1: Refine prompt with ChatGPT
            system_msg = (
                "You are an assistant that writes detailed background descriptions "
                "for image editing and inpainting."
            )

            user_msg = (
                f"User prompt for background replacement: {prompt}\n\n"
                "Rewrite this as a detailed, visual background description "
                "suitable for an image generation model."
            )

            chat_resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=300,
                temperature=0.8,
            )

            refined_prompt = chat_resp.choices[0].message.content.strip()
            logger.info(f"✅ Refined prompt: {refined_prompt[:80]}")

            # Step 2: Use DALL·E to actually edit the image
            # Convert PIL Image to PNG bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            img_byte_arr.name = "image.png"  # ✅ Set filename with .png extension

            mask_byte_arr = io.BytesIO()
            mask.save(mask_byte_arr, format='PNG')
            mask_byte_arr.seek(0)
            mask_byte_arr.name = "mask.png"  # ✅ Set filename with .png extension

            # Call DALL·E edit endpoint
            edit_response = client.images.edit(
                image=img_byte_arr,
                mask=mask_byte_arr,
                prompt=refined_prompt,
                n=1,
                size="1024x1024",
                model="dall-e-2"  # or "dall-e-2"
            )

            # Download the generated image
            import requests
            image_url = edit_response.data[0].url
            img_response = requests.get(image_url)
            result_image = Image.open(io.BytesIO(img_response.content)).convert("RGB")

            metadata: Dict[str, Any] = {
                "provider": self.get_provider_name(),
                "model_used": "DALL·E 3",
                "original_prompt": prompt,
                "refined_prompt": refined_prompt,
                "image_generated": True,
            }

            logger.info("✅ DALL·E image inpainting completed!")
            return result_image, metadata

        except Exception as e:
            logger.error(f"OpenAI DALL·E error: {e}", exc_info=True)
            raise ProviderError(f"OpenAI DALL·E error: {e}") from e

