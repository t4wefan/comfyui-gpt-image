import os
import traceback
import io

# --- ComfyUI Imports ---
try:
    from server import ComfyNodeABC, InputTypeDict, IO
except ImportError:
    # Define dummy classes/types if running outside ComfyUI for basic validation
    class ComfyNodeABC: pass
    InputTypeDict = dict
    class IO:
        STRING = "STRING"
        INT = "INT"
        FLOAT = "FLOAT"
        IMAGE = "IMAGE"
        MASK = "MASK"
        COMBO = "COMBO"

# Import modularized functionalities
from .openai_client import OpenAIClientWrapper
from .image_processing import process_image_tensor_to_png_bytes, process_api_response

class GPT1ImageNodeSyncOpenAILib(ComfyNodeABC):  # Renamed class for clarity
    """
    Generates or edits images synchronously using the 'gpt-image-1' model
    via an OpenAI-compatible API using the official OpenAI Python library.
    If input_images are provided, it uses the 'images.edit' endpoint.
    Otherwise, it uses the 'images.generate' endpoint.
    API Base URL and Key are configurable directly in the node inputs.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:  # type: ignore
        image_sizes = ["1024x1024", "1792x1024", "1024x1792"]
        return {
            "required": {
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "A photorealistic image...",
                    "tooltip": "Text prompt for generation or editing instructions",
                }),
                "api_key": (IO.STRING, {
                    "default": os.getenv("OPENAI_API_KEY", ""),  # Try getting from env var as default
                    "multiline": False,
                    "tooltip": "Your OpenAI (or compatible) API Key",
                }),
                "base_url": (IO.STRING, {
                    "default": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),  # Try getting from env var
                    "multiline": False,
                    "tooltip": "The base URL for the API endpoint (e.g., https://api.openai.com/v1)",
                }),
                "model_name": (IO.STRING, {  # Let user specify model if needed, default to gpt-image-1
                    "default": "gpt-image-1",
                    "multiline": False,
                    "tooltip": "The specific model name to use (e.g., gpt-image-1)",
                }),
            },
            "optional": {
                # Add optional image input for editing
                "input_images": (IO.IMAGE, {}),
                # Add optional mask input for editing (currently only supports single image + mask)
                "mask_image": (IO.MASK, {}),
                "quality": (IO.COMBO, {
                    "options": ["low","medium","high","auto"],
                    "default": "auto",
                    "tooltip": "[Generate only] Image quality (typically ignored for edits)",
                }),
                "size": (IO.COMBO, {
                    "options": image_sizes,
                    "default": "1024x1024",
                    "tooltip": "Target image dimensions (used for generation, potentially for edits)",
                }),
            },
            "hidden": {}
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "generate_or_edit_image"
    CATEGORY = "api node/openai_configured"

    def generate_or_edit_image(self, prompt: str, api_key: str, base_url: str, model_name: str,
                                 input_images=None, mask_image=None,
                                 quality: str = "standard", size: str = "1024x1024"):
        """
        Handles image generation or editing using the OpenAI Python library.
        """
        try:
            # --- Initialize OpenAI Client ---
            client_wrapper = OpenAIClientWrapper(api_key, base_url)

            # --- Determine Mode: Edit vs. Generate ---
            has_input_images = input_images is not None and input_images.nelement() > 0
            has_mask = mask_image is not None and mask_image.nelement() > 0

            if has_input_images:
                print(f"[GPT1ImageNode] Input images detected. Using Edit mode.")

                # --- Prepare Images for Edit API ---
                try:
                    image_byte_list = process_image_tensor_to_png_bytes(input_images)
                    if not image_byte_list:
                        raise ValueError("Input images provided but failed to process them into PNG bytes.")
                except Exception as img_err:
                    print(f"[GPT1ImageNode] Error processing input images: {img_err}")
                    traceback.print_exc()
                    raise Exception(f"Failed to process input images for editing: {img_err}") from img_err

                # Wrap bytes in io.BytesIO for the OpenAI library
                image_files = [(f'image_{idx}.png', io.BytesIO(img_bytes)) for idx, img_bytes in
                               enumerate(image_byte_list)]

                mask_file = None
                if has_mask:
                    # Note: Standard OpenAI 'edit' usually takes ONE input image with ONE mask.
                    # If multiple input images are provided with a mask, the behavior might be API-dependent
                    # or might error. Let's assume for now: if mask is given, use ONLY the FIRST input image.
                    if len(image_files) > 1:
                        print(
                            f"[GPT1ImageNode] Warning: Mask provided with {len(image_files)} input images. Using only the first image for editing with mask.")
                        image_files = [image_files[0]]  # Use only the first image

                    print(f"[GPT1ImageNode] Mask image detected. Processing it for editing.")
                    try:
                        mask_byte_list = process_image_tensor_to_png_bytes(mask_image)
                        if not mask_byte_list:
                            raise ValueError("Mask image provided but failed to process it into PNG bytes.")
                        # Edit endpoint expects a single mask file
                        mask_file = ('mask.png', io.BytesIO(mask_byte_list[0]))
                        print("[GPT1ImageNode] Mask prepared for API call.")
                    except Exception as mask_err:
                        print(f"[GPT1ImageNode] Error processing mask image: {mask_err}")
                        traceback.print_exc()
                        raise Exception(f"Failed to process mask image for editing: {mask_err}") from mask_err
                elif len(image_files) > 1:
                    print(f"[GPT1ImageNode] Sending {len(image_files)} images for editing without a mask.")

                # --- Call Edit API ---
                api_response_data = client_wrapper.edit_image(model_name, prompt, image_files, mask_file)

            else:
                # --- Call Generation API ---
                print(f"[GPT1ImageNode] No input images. Using Generation mode.")
                api_response_data = client_wrapper.generate_image(model_name, prompt, size, quality)
                
            # --- Process Response Image ---
            img_tensor = process_api_response(api_response_data)
            print(f"[GPT1ImageNode] Image received and processed successfully!")
            return (img_tensor,)

        except Exception as e:
            print(f"[GPT1ImageNode] An unexpected error occurred: {e}")
            traceback.print_exc()
            raise Exception(f"Processing failed: {e}") from e

# --- ComfyUI Registration ---
NODE_CLASS_MAPPINGS = {
    "GPT1ImageNode": GPT1ImageNodeSyncOpenAILib
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPT1ImageNode": "OpenAI Gen/Edit"
}