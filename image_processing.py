import io
import base64
import numpy as np
from PIL import Image
import torch

def process_image_tensor_to_png_bytes(image_tensor_batch):
    """Converts a batch of image tensors (or a single mask) to a list of PNG bytes."""
    if image_tensor_batch is None:
        return []

    # Handle both image (B, H, W, C) and mask (B, H, W) tensors
    is_mask = len(image_tensor_batch.shape) == 3
    if is_mask:
        # Add channel dim for mask processing, treat as grayscale
        image_tensor_batch = image_tensor_batch.unsqueeze(-1).repeat(1, 1, 1, 3)  # Repeat to RGB for PIL

    png_bytes_list = []
    # Input tensor shape: (Batch, Height, Width, Channels)
    for i in range(image_tensor_batch.shape[0]):
        img_tensor = image_tensor_batch[i]
        # Convert to NumPy array, denormalize (0-1 -> 0-255), change type
        # Clamp values just in case they are slightly outside [0, 1]
        img_np = np.clip(img_tensor.cpu().numpy() * 255, 0, 255).astype(np.uint8)

        # Create PIL Image
        if img_np.shape[-1] == 4:
            img_pil = Image.fromarray(img_np, 'RGBA')
        elif img_np.shape[-1] == 1 or is_mask:  # Handle grayscale/mask
            img_pil = Image.fromarray(img_np.squeeze(-1), 'L').convert(
                'RGBA')  # Convert mask to RGBA for OpenAI API
        else:  # Assume RGB
            img_pil = Image.fromarray(img_np, 'RGB').convert(
                'RGBA')  # Always convert to RGBA for consistency? OpenAI might prefer RGBA

        # Ensure mask is RGBA with alpha channel reflecting the mask
        if is_mask:
            # Create RGBA image where alpha is the mask
            # Use the grayscale mask value for Alpha, make RGB black? Or white?
            # Let's make RGB black and use the mask value for Alpha.
            # The API expects an RGBA image where transparent areas (alpha=0) indicate parts to edit.
            mask_alpha = img_pil.getchannel('A')  # Get original alpha from L conversion
            img_pil = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))  # Fully transparent black image
            img_pil.putalpha(mask_alpha)  # Apply the mask data to the alpha channel

        # Save to in-memory bytes buffer as PNG
        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG")
        buffer.seek(0)
        png_bytes_list.append(buffer.getvalue())
    return png_bytes_list

def process_api_response(api_response_data):
    """Processes the API response and returns the image tensor."""

    if not api_response_data.data[0].b64_json:
        raise Exception(f"Invalid API response: 'b64_json' missing. Response: {api_response_data}")

    result_image_b64 = api_response_data.data[0].b64_json
    if not result_image_b64:
        raise Exception("Failed to get image data from the API response.")

    img_bytes = base64.b64decode(result_image_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")  # Ensure RGBA consistent output format
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add batch dimension

    return img_tensor