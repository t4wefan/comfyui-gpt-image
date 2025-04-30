import os

from openai import OpenAI, APIError, AuthenticationError


class OpenAIClientWrapper:
    """
    Wraps the OpenAI client for easier initialization and error handling.
    """

    def __init__(self, api_key: str, base_url: str):
        """
        Initializes the OpenAI client.
        """
        if not api_key:
            raise ValueError("API Key cannot be empty.")
        if not base_url:
            raise ValueError("Base URL cannot be empty.")

        # Removing trailing slash if present
        if base_url.endswith("/"):
            base_url = base_url[:-1]

        try:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except NameError:
            raise ImportError(
                "OpenAI library is not installed. Please run 'pip install openai'."
            )
        except Exception as e:
            raise Exception(f"Failed to initialize OpenAI client: {e}") from e

    def generate_image(
        self, model_name: str, prompt: str, size: str, quality: str
    ):
        """
        Calls the OpenAI image generation API.
        """
        try:
            print(f"[GPT1ImageNode] Calling OpenAI images.generate...")
            print(
                f"[GPT1ImageNode] Payload (gen): model='{model_name}', size='{size}', quality='{quality}', Prompt: '{prompt[:60]}...'"
            )

            response = self.client.images.generate(
                model=model_name,
                prompt=prompt,
                n=1,
                size=size,
                quality=quality,
            )
            print(f"[GPT1ImageNode] Generation API call successful.")
            response.data[0].b64_json
            return response

        except AuthenticationError as e:
            print(f"[GPT1ImageNode] Authentication Error: {e}")
            raise Exception(
                f"OpenAI Authentication Error: Check your API Key. Details: {e}"
            ) from e
        except APIError as e:
            print(
                f"[GPT1ImageNode] OpenAI API Error: Status={e.status_code} Response={e.response}"
            )
            error_details = self._extract_error_details(e)
            raise Exception(
                f"OpenAI API Error (Status: {e.status_code}): {error_details}"
            ) from e
        except Exception as e:
            raise Exception(f"Image generation failed: {e}") from e

    def edit_image(
        self, model_name: str, prompt: str, image_files, mask_file: object
    ):
        """
        Calls the OpenAI image edit API.
        """
        try:
            print(f"[GPT1ImageNode] Calling OpenAI images.edit...")

            # Select the correct image argument based on single vs multiple images
            image_arg = (
                image_files[0][1]
                if len(image_files) == 1
                else [f[1] for f in image_files]
            )

            kwargs = {
                "model": model_name,
                "prompt": prompt,
                "image": image_arg,
                "n": 1
            }
            # Add mask only if it was processed successfully
            if mask_file:
                # The API expects 'mask' arg only when a single image is provided
                if len(image_files) == 1:
                    kwargs["mask"] = mask_file[1]
                    print("[GPT1ImageNode] Added mask to API call.")
                else:
                    print(
                        "[GPT1ImageNode] Warning: Mask ignored because multiple input images were provided (OpenAI API likely expects one image for mask editing)."
                    )

            response = self.client.images.edit(**kwargs)
            print(f"[GPT1ImageNode] Edit API call successful.")
            return response

        except AuthenticationError as e:
            print(f"[GPT1ImageNode] Authentication Error: {e}")
            raise Exception(
                f"OpenAI Authentication Error: Check your API Key. Details: {e}"
            ) from e
        except APIError as e:
            print(
                f"[GPT1ImageNode] OpenAI API Error: Status={e.status_code} Response={e.response}"
            )
            error_details = self._extract_error_details(e)
            raise Exception(
                f"OpenAI API Error (Status: {e.status_code}): {error_details}"
            ) from e
        except Exception as e:
            raise Exception(f"Image editing failed: {e}") from e

    def _extract_error_details(self, e):
        """Helper function to extract error details from APIError."""
        error_details = str(e)
        try:
            if e.response and hasattr(e.response, "text"):
                error_details = e.response.text[:500]  # Limit length
        except Exception:
            pass  # Ignore if we can't get response text
        return error_details
