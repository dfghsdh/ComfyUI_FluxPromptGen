import random
import json
import os
import re
from datetime import datetime
import time
import ollama

class FluxPromptGenerator:
    def __init__(self):
        self.rng = random.Random()
        self.load_json_files()

    def load_json_files(self):
        json_files = [
            "artform", "photo_type", "body_types", "default_tags", "roles", "hairstyles",
            "additional_details", "photography_styles", "device", "photographer", "artist",
            "digital_artform", "place", "lighting", "clothing", "composition", "pose", "background"
        ]
        self.data = {}
        for file_name in json_files:
            file_path = os.path.join("custom_nodes/ComfyUI_FluxPromptGen/data", f"{file_name}.json")
            try:
                with open(file_path, "r") as file:
                    data = json.load(file)
                    if not data:  # Check if the loaded data is empty
                        print(f"Warning: {file_path} is empty. Using default value.")
                        self.data[file_name] = ["default"]
                    else:
                        self.data[file_name] = data
            except FileNotFoundError:
                print(f"Warning: {file_path} not found. Using default value.")
                print(file_path)
                self.data[file_name] = ["default"]
            except json.JSONDecodeError:
                print(f"Warning: {file_path} is not a valid JSON file. Using default value.")
                self.data[file_name] = ["default"]

    def split_and_choose(self, input_str):
        choices = [choice.strip() for choice in input_str.split(",")]
        return self.rng.choices(choices, k=1)[0]

    def get_choice(self, input_str, default_choices):
        if input_str.lower() == "disabled":
            return ""
        elif input_str.lower() in ["random", "Any"]:
            if not default_choices:
                return "default"  # Return a default value if the list is empty
            return self.rng.choice(default_choices)
        else:
            return input_str

    def clean_consecutive_commas(self, input_string):
        cleaned_string = re.sub(r',\s*,', ',', input_string)
        return cleaned_string

    def process_string(self, replaced, seed):
        replaced = re.sub(r'\s*,\s*', ',', replaced)
        replaced = re.sub(r',+', ',', replaced)
        original = replaced
        
        first_break_clipl_index = replaced.find("BREAK_CLIPL")
        second_break_clipl_index = replaced.find("BREAK_CLIPL", first_break_clipl_index + len("BREAK_CLIPL"))
        
        if first_break_clipl_index != -1 and second_break_clipl_index != -1:
            clip_content_l = replaced[first_break_clipl_index + len("BREAK_CLIPL"):second_break_clipl_index]
            replaced = replaced[:first_break_clipl_index].strip(", ") + replaced[second_break_clipl_index + len("BREAK_CLIPL"):].strip(", ")
            clip_l = clip_content_l
        else:
            clip_l = ""
        
        first_break_clipg_index = replaced.find("BREAK_CLIPG")
        second_break_clipg_index = replaced.find("BREAK_CLIPG", first_break_clipg_index + len("BREAK_CLIPG"))
        
        if first_break_clipg_index != -1 and second_break_clipg_index != -1:
            clip_content_g = replaced[first_break_clipg_index + len("BREAK_CLIPG"):second_break_clipg_index]
            replaced = replaced[:first_break_clipg_index].strip(", ") + replaced[second_break_clipg_index + len("BREAK_CLIPG"):].strip(", ")
            clip_g = clip_content_g
        else:
            clip_g = ""
        
        t5xxl = replaced
        
        original = original.replace("BREAK_CLIPL", "").replace("BREAK_CLIPG", "")
        original = re.sub(r'\s*,\s*', ',', original)
        original = re.sub(r',+', ',', original)
        clip_l = re.sub(r'\s*,\s*', ',', clip_l)
        clip_l = re.sub(r',+', ',', clip_l)
        clip_g = re.sub(r'\s*,\s*', ',', clip_g)
        clip_g = re.sub(r',+', ',', clip_g)
        if clip_l.startswith(","):
            clip_l = clip_l[1:]
        if clip_g.startswith(","):
            clip_g = clip_g[1:]
        if original.startswith(","):
            original = original[1:]
        if t5xxl.startswith(","):
            t5xxl = t5xxl[1:]

        return original, seed, t5xxl, clip_l, clip_g

    def generate_prompt(self, seed, custom, subject, **kwargs):
        if seed is not None:
            self.rng = random.Random(seed)
        components = []
        if custom:
            components.append(custom)

        choices = {}
        for key, value in kwargs.items():
            if value.lower() in ["random", "Any"]:
                choices[key] = self.get_choice(value, self.data[key])
            else:
                choices[key] = value

        # ... (implement the rest of the prompt generation logic using the choices dictionary)

        prompt = " ".join(components)
        prompt = re.sub(" +", " ", prompt)
        replaced = prompt.replace("of as", "of")
        replaced = self.clean_consecutive_commas(replaced)

        return self.process_string(replaced, seed), choices

class FluxPromptGeneratorNode:
    def __init__(self):
        self.generator = FluxPromptGenerator()
        self.ollama_models = self.get_ollama_models()

    @classmethod
    def get_ollama_models(cls):
        try:
            models = ollama.list()
            return [model['name'] for model in models['models']]
        except Exception as e:
            print(f"Error fetching Ollama models: {e}")
            return ["llama2"]  # Default model if fetch fails

    @classmethod
    def INPUT_TYPES(s):
        generator = FluxPromptGenerator()  # Create a temporary instance
        ollama_models = s.get_ollama_models()
        
        def create_options(data_list):
            options = ["disabled", "random"]
            if data_list:
                options.extend(data_list)
            else:
                options.append("default")
            return options

        inputs = {
            "required": {
                "use_ollama": ("BOOLEAN", {"default": True}),
                "ollama_model": (ollama_models, {"default": ollama_models[0] if ollama_models else "llama2"}),
                "happy_talk": ("BOOLEAN", {"default": True}),
                "compress": ("BOOLEAN", {"default": True}),
                "compression_level": (["soft", "medium", "hard"], {"default": "hard"}),
                "poster": ("BOOLEAN", {"default": False}),
                "custom_base_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "custom": ("STRING", {"default": ""}),
                "subject": ("STRING", {"default": ""}),
                "OVERRIDE_STYLES": (["Keep Current", "Set All Random", "Set All Disabled"], {"default": "Keep Current"}),
            },
            "optional": {
                "append_custom_base_prompt": ("STRING", {"forceInput": True}),
            }
        }

        for key, value in generator.data.items():
            inputs["required"][key] = (create_options(value), {"default": "disabled"})

        return inputs

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("generated_prompt", "choices")
    FUNCTION = "generate"
    OUTPUT_NODE = False
    CATEGORY = "prompt"

    def generate(self, use_ollama, ollama_model, happy_talk, compress, compression_level, poster, custom_base_prompt, 
                 seed, custom, subject, OVERRIDE_STYLES, append_custom_base_prompt="", **kwargs):
        # Apply override styles
        if OVERRIDE_STYLES == "Set All Random":
            kwargs = {k: "random" for k in kwargs}
        elif OVERRIDE_STYLES == "Set All Disabled":
            kwargs = {k: "disabled" for k in kwargs}

        (initial_prompt, _, _, _, _), choices = self.generator.generate_prompt(
            seed, custom, subject, **kwargs
        )

        if use_ollama:
            final_prompt = self.generate_ollama(
                ollama_model, initial_prompt, happy_talk, compress, compression_level, poster, 
                custom_base_prompt, append_custom_base_prompt
            )
        else:
            final_prompt = initial_prompt

        choices_str = ", ".join([f"{k}: {v}" for k, v in choices.items() if v != "disabled"])
        return (final_prompt, choices_str)

    def generate_ollama(self, model, input_text, happy_talk, compress, compression_level, poster, 
                        custom_base_prompt="", append_custom_base_prompt=""):
        base_prompt = self.get_base_prompt(happy_talk, compress, compression_level, poster, 
                                           custom_base_prompt, append_custom_base_prompt)
        
        prompt = f"{base_prompt}\nDescription: {input_text}"
        
        print(f"Starting generation with Ollama {model}...")
        start_time = time.time()

        try:
            response = ollama.generate(model=model, prompt=prompt)
            output = response['response']
        except Exception as e:
            print(f"Error generating with Ollama: {e}")
            output = f"Error occurred while generating: {str(e)}"

        end_time = time.time()
        print(f"Generation completed in {end_time - start_time:.2f} seconds")

        return self.clean_output(output)

    def get_base_prompt(self, happy_talk, compress, compression_level, poster, 
                        custom_base_prompt, append_custom_base_prompt):
        default_happy_prompt = """Create a detailed visually descriptive caption of this description, which will be used as a prompt for a text to image AI system (caption only, no instructions like "create an image").Remove any mention of digital artwork or artwork style. Give detailed visual descriptions of the character(s), including ethnicity, skin tone, expression etc. Imagine using keywords for a still for someone who has aphantasia. Describe the image style, e.g. any photographic or art styles / techniques utilized. Make sure to fully describe all aspects of the cinematography, with abundant technical details and visual descriptions. If there is more than one image, combine the elements and characters from all of the images creatively into a single cohesive composition with a single background, inventing an interaction between the characters. Be creative in combining the characters into a single cohesive scene. Focus on two primary characters (or one) and describe an interesting interaction between them, such as a hug, a kiss, a fight, giving an object, an emotional reaction / interaction. If there is more than one background in the images, pick the most appropriate one. Your output is only the caption itself, no comments or extra formatting. The caption is in a single long paragraph. If you feel the images are inappropriate, invent a new scene / characters inspired by these. Additionally, incorporate a specific movie director's visual style and describe the lighting setup in detail, including the type, color, and placement of light sources to create the desired mood and atmosphere. Always frame the scene, including details about the film grain, color grading, and any artifacts or characteristics specific."""

        default_simple_prompt = """Create a brief, straightforward caption for this description, suitable for a text-to-image AI system. Focus on the main elements, key characters, and overall scene without elaborate details. Provide a clear and concise description in one or two sentences."""

        poster_prompt = """Analyze the provided description and extract key information to create a movie poster style description. Format the output as follows:
Title: A catchy, intriguing title that captures the essence of the scene, place the title in "".
Main character: Give a description of the main character.
Background: Describe the background in detail.
Supporting characters: Describe the supporting characters
Branding type: Describe the branding type
Tagline: Include a tagline that captures the essence of the movie.
Visual style: Ensure that the visual style fits the branding type and tagline.
You are allowed to make up film and branding names, and do them like 80's, 90's or modern movie posters."""

        if poster:
            base_prompt = poster_prompt
        elif custom_base_prompt.strip():
            base_prompt = custom_base_prompt
        else:
            base_prompt = default_happy_prompt if happy_talk else default_simple_prompt

        # Append the custom base prompt
        if append_custom_base_prompt.strip():
            base_prompt += f" {append_custom_base_prompt.strip()}"

        if compress and not poster:
            compression_chars = {
                "soft": 600 if happy_talk else 300,
                "medium": 400 if happy_talk else 200,
                "hard": 200 if happy_talk else 100
            }
            char_limit = compression_chars[compression_level]
            base_prompt += f" Compress the output to be concise while retaining key visual details. MAX OUTPUT SIZE no more than {char_limit} characters."

        return base_prompt

    def clean_output(self, output):
        try:
            # Clean up the output
            if ": " in output:
                output = output.split(": ", 1)[1].strip()
            elif output.lower().startswith("here"):
                sentences = output.split(". ")
                if len(sentences) > 1:
                    output = ". ".join(sentences[1:]).strip()
            return output
        except Exception as e:
            print(f"An error occurred while cleaning output: {e}")
            return f"Error occurred while processing the output: {str(e)}"

# This function is needed to register the node in ComfyUI
NODE_CLASS_MAPPINGS = {
    "FluxPromptGeneratorNode": FluxPromptGeneratorNode
}

# This function is needed to add custom display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxPromptGeneratorNode": "Flux Prompt Generator"
}
