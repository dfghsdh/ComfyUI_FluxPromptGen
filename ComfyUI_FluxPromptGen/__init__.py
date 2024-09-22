from .flux_prompt_generator_node import FluxPromptGeneratorNode
from .flux_image_caption_node import FluxImageCaptionNode

NODE_CLASS_MAPPINGS = {
    "FluxPromptGeneratorNode": FluxPromptGeneratorNode,
    "FluxImageCaptionNode": FluxImageCaptionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxPromptGeneratorNode": "Flux Prompt Generator",
    "FluxImageCaptionNode": "Flux Image Caption"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("ComfyUI_FluxPromptGen loaded successfully")
