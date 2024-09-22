import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch
import os

class FluxImageCaptionNode:
    NODE_NAME = "FluxImageCaption"
    DISPLAY_NAME = "Flux Image Caption"
    CATEGORY = "flux"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = "microsoft/Florence-2-base"
        self.processor = None
        self.model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_length": ("INT", {"default": 50, "min": 10, "max": 200, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "caption_image"

    def load_model(self):
        if self.model is None:
            print("Loading Florence-2 model...")
            
            def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
                if not str(filename).endswith("modeling_florence2.py"):
                    return get_imports(filename)
                imports = get_imports(filename)
                if "flash_attn" in imports:
                    imports.remove("flash_attn")
                return imports

            with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
                self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    attn_implementation="sdpa",
                    torch_dtype=torch.float16 if 'cuda' in str(self.device) else torch.float32,
                    trust_remote_code=True
                ).to(self.device).eval()
            
            print("Florence-2 model loaded successfully.")

    def caption_image(self, image, max_length):
        self.load_model()

        # Convert the image tensor to a numpy array
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
        else:
            image_np = np.array(image)

        # Print shape and dtype for debugging
        print(f"Image shape: {image_np.shape}, dtype: {image_np.dtype}")

        # Ensure the image is in the correct format (H, W, C)
        if len(image_np.shape) == 3:
            if image_np.shape[0] == 1 or image_np.shape[0] == 3:
                image_np = np.transpose(image_np, (1, 2, 0))
        elif len(image_np.shape) == 4:
            image_np = image_np.squeeze(0)  # Remove batch dimension if present
            if image_np.shape[0] == 1 or image_np.shape[0] == 3:
                image_np = np.transpose(image_np, (1, 2, 0))

        # If the image is grayscale, convert to RGB
        if len(image_np.shape) == 2 or (len(image_np.shape) == 3 and image_np.shape[-1] == 1):
            image_np = np.stack((image_np,)*3, axis=-1)

        # Ensure the image is in uint8 format and in the range [0, 255]
        if image_np.dtype != np.uint8:
            if np.max(image_np) <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)

        # Print shape and dtype after processing for debugging
        print(f"Processed image shape: {image_np.shape}, dtype: {image_np.dtype}")

        # Convert to PIL Image
        pil_image = Image.fromarray(image_np)
        
        # Process the image and generate caption
        inputs = self.processor(text="<MORE_DETAILED_CAPTION>", images=pil_image, return_tensors="pt")
        
        # Move inputs to the correct device and dtype
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if key == "input_ids" or key == "attention_mask":
                    inputs[key] = value.to(device=self.device, dtype=torch.long)
                else:
                    inputs[key] = value.to(device=self.device, dtype=self.model.dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task="<MORE_DETAILED_CAPTION>",
            image_size=(pil_image.width, pil_image.height)
        )
        caption = parsed_answer["<MORE_DETAILED_CAPTION>"]
        
        return (caption,)

# This function is needed to register the node in ComfyUI
NODE_CLASS_MAPPINGS = {
    "FluxImageCaptionNode": FluxImageCaptionNode
}

# This function is needed to add custom display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxImageCaptionNode": "Flux Image Caption"
}