import gradio as gr
import random
import requests
import json
import os
import re
from datetime import datetime
from huggingface_hub import InferenceClient
import subprocess
import torch
import devicetorch
from PIL import Image
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import random
import importlib.util
import sys
from groq import Groq
import time

#ADD YOUR OWN GROQ API KEY ON LINE 336

# Initialize Florence model
device = devicetorch.get(torch)

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    florence_model = AutoModelForCausalLM.from_pretrained(
        'microsoft/Florence-2-base',
        attn_implementation="sdpa",
        torch_dtype=torch.float16 if 'cuda' in str(device) else torch.float32,
        trust_remote_code=True
    ).to(device).eval()

florence_processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)



#device = "cuda" if torch.cuda.is_available() else "cpu"
#florence_model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)#.to(device).eval()
#florence_processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)

# Florence caption function
def florence_caption(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    inputs = florence_processor(text="<MORE_DETAILED_CAPTION>", images=image, return_tensors="pt").to(device)
    generated_ids = florence_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = florence_processor.post_process_generation(
        generated_text,
        task="<MORE_DETAILED_CAPTION>",
        image_size=(image.width, image.height)
    )
    return parsed_answer["<MORE_DETAILED_CAPTION>"]
    
# Load JSON files
def load_json_file(file_name):
    file_path = os.path.join("data", file_name)
    with open(file_path, "r") as file:
        return json.load(file)

ARTFORM = load_json_file("artform.json")
PHOTO_TYPE = load_json_file("photo_type.json")
BODY_TYPES = load_json_file("body_types.json")
DEFAULT_TAGS = load_json_file("default_tags.json")
ROLES = load_json_file("roles.json")
HAIRSTYLES = load_json_file("hairstyles.json")
ADDITIONAL_DETAILS = load_json_file("additional_details.json")
PHOTOGRAPHY_STYLES = load_json_file("photography_styles.json")
DEVICE = load_json_file("device.json")
PHOTOGRAPHER = load_json_file("photographer.json")
ARTIST = load_json_file("artist.json")
DIGITAL_ARTFORM = load_json_file("digital_artform.json")
PLACE = load_json_file("place.json")
LIGHTING = load_json_file("lighting.json")
CLOTHING = load_json_file("clothing.json")
COMPOSITION = load_json_file("composition.json")
POSE = load_json_file("pose.json")
BACKGROUND = load_json_file("background.json")

class PromptGenerator:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def split_and_choose(self, input_str):
        choices = [choice.strip() for choice in input_str.split(",")]
        return self.rng.choices(choices, k=1)[0]

    def get_choice(self, input_str, default_choices):
        if input_str.lower() == "disabled":
            return ""
        elif "," in input_str:
            return self.split_and_choose(input_str)
        elif input_str.lower() == "random":
            return self.rng.choices(default_choices, k=1)[0]
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

    def generate_prompt(self, seed, custom, subject, artform, photo_type, body_types, default_tags, roles, hairstyles,
                        additional_details, photography_styles, device, photographer, artist, digital_artform,
                        place, lighting, clothing, composition, pose, background, input_image, *args):
        print(f"Number of arguments received: {len(args)}")
        for i, arg in enumerate(args):
            print(f"Argument {i}: {arg}")
        kwargs = locals()
        del kwargs['self']
        
        seed = kwargs.get("seed", 0)
        if seed is not None:
            self.rng = random.Random(seed)
        components = []
        custom = kwargs.get("custom", "")
        if custom:
            components.append(custom)
        is_photographer = kwargs.get("artform", "").lower() == "photography" or (
            kwargs.get("artform", "").lower() == "random"
            and self.rng.choice([True, False])
        )

        subject = kwargs.get("subject", "")

        if is_photographer:
            selected_photo_style = self.get_choice(kwargs.get("photography_styles", ""), PHOTOGRAPHY_STYLES)
            if not selected_photo_style:
                selected_photo_style = "photography"
            components.append(selected_photo_style)
            if kwargs.get("photography_style", "") != "disabled" and kwargs.get("default_tags", "") != "disabled" or subject != "":
                components.append(" of")
        
        default_tags = kwargs.get("default_tags", "random")
        body_type = kwargs.get("body_types", "")
        if not subject:
            if default_tags == "random":
                if body_type != "disabled" and body_type != "random":
                    selected_subject = self.get_choice(kwargs.get("default_tags", ""), DEFAULT_TAGS).replace("a ", "").replace("an ", "")
                    components.append("a ")
                    components.append(body_type)
                    components.append(selected_subject)
                elif body_type == "disabled":
                    selected_subject = self.get_choice(kwargs.get("default_tags", ""), DEFAULT_TAGS)
                    components.append(selected_subject)
                else:
                    body_type = self.get_choice(body_type, BODY_TYPES)
                    components.append("a ")
                    components.append(body_type)
                    selected_subject = self.get_choice(kwargs.get("default_tags", ""), DEFAULT_TAGS).replace("a ", "").replace("an ", "")
                    components.append(selected_subject)
            elif default_tags == "disabled":
                pass
            else:
                components.append(default_tags)
        else:
            if body_type != "disabled" and body_type != "random":
                components.append("a ")
                components.append(body_type)
            elif body_type == "disabled":
                pass
            else:
                body_type = self.get_choice(body_type, BODY_TYPES)
                components.append("a ")
                components.append(body_type)
            components.append(subject)

        params = [
            ("roles", ROLES),
            ("hairstyles", HAIRSTYLES),
            ("additional_details", ADDITIONAL_DETAILS),
        ]
        for param in params:
            components.append(self.get_choice(kwargs.get(param[0], ""), param[1]))
        for i in reversed(range(len(components))):
            if components[i] in PLACE:
                components[i] += ","
                break
        if kwargs.get("clothing", "") != "disabled" and kwargs.get("clothing", "") != "random":
            components.append(", dressed in ")
            clothing = kwargs.get("clothing", "")
            components.append(clothing)
        elif kwargs.get("clothing", "") == "random":
            components.append(", dressed in ")
            clothing = self.get_choice(kwargs.get("clothing", ""), CLOTHING)
            components.append(clothing)

        if kwargs.get("composition", "") != "disabled" and kwargs.get("composition", "") != "random":
            components.append(",")
            composition = kwargs.get("composition", "")
            components.append(composition)
        elif kwargs.get("composition", "") == "random": 
            components.append(",")
            composition = self.get_choice(kwargs.get("composition", ""), COMPOSITION)
            components.append(composition)
        
        if kwargs.get("pose", "") != "disabled" and kwargs.get("pose", "") != "random":
            components.append(",")
            pose = kwargs.get("pose", "")
            components.append(pose)
        elif kwargs.get("pose", "") == "random":
            components.append(",")
            pose = self.get_choice(kwargs.get("pose", ""), POSE)
            components.append(pose)
        components.append("BREAK_CLIPG")
        if kwargs.get("background", "") != "disabled" and kwargs.get("background", "") != "random":
            components.append(",")
            background = kwargs.get("background", "")
            components.append(background)
        elif kwargs.get("background", "") == "random": 
            components.append(",")
            background = self.get_choice(kwargs.get("background", ""), BACKGROUND)
            components.append(background)

        if kwargs.get("place", "") != "disabled" and kwargs.get("place", "") != "random":
            components.append(",")
            place = kwargs.get("place", "")
            components.append(place)
        elif kwargs.get("place", "") == "random": 
            components.append(",")
            place = self.get_choice(kwargs.get("place", ""), PLACE)
            components.append(place + ",")

        lighting = kwargs.get("lighting", "").lower()
        if lighting == "random":
            selected_lighting = ", ".join(self.rng.sample(LIGHTING, self.rng.randint(2, 5)))
            components.append(",")
            components.append(selected_lighting)
        elif lighting == "disabled":
            pass
        else:
            components.append(", ")
            components.append(lighting)
        components.append("BREAK_CLIPG")
        components.append("BREAK_CLIPL")
        if is_photographer:
            if kwargs.get("photo_type", "") != "disabled":
                photo_type_choice = self.get_choice(kwargs.get("photo_type", ""), PHOTO_TYPE)
                if photo_type_choice and photo_type_choice != "random" and photo_type_choice != "disabled":
                    random_value = round(self.rng.uniform(1.1, 1.5), 1)
                    components.append(f", ({photo_type_choice}:{random_value}), ")

            params = [
                ("device", DEVICE),
                ("photographer", PHOTOGRAPHER),
            ]
            components.extend([self.get_choice(kwargs.get(param[0], ""), param[1]) for param in params])
            if kwargs.get("device", "") != "disabled":
                components[-2] = f", shot on {components[-2]}"
            if kwargs.get("photographer", "") != "disabled":
                components[-1] = f", photo by {components[-1]}"
        else:
            digital_artform_choice = self.get_choice(kwargs.get("digital_artform", ""), DIGITAL_ARTFORM)
            if digital_artform_choice:
                components.append(f"{digital_artform_choice}")
            if kwargs.get("artist", "") != "disabled":
                components.append(f"by {self.get_choice(kwargs.get('artist', ''), ARTIST)}")
        components.append("BREAK_CLIPL")

        prompt = " ".join(components)
        prompt = re.sub(" +", " ", prompt)
        replaced = prompt.replace("of as", "of")
        replaced = self.clean_consecutive_commas(replaced)

        return self.process_string(replaced, seed)
    
    def add_caption_to_prompt(self, prompt, caption):
        if caption:
            return f"{prompt}, {caption}"
        return prompt

class InferenceNode:
    def __init__(self):
    #ADD YOUR OWN GROQ API KEY HERE
        self.groq_client = Groq(api_key="YOUR-GROQ-API-KEY")
        self.groq_models = {
            "Mixtral 8x7B": "mixtral-8x7b-32768",
            "Llama 3.1 70B": "llama-3.1-70b-versatile",
            "Llama 3.1 8B": "llama-3.1-8b-instant"
        }
        self.ollama_url = "http://localhost:11434/api/generate"
        self.prompts_dir = "./prompts"
        os.makedirs(self.prompts_dir, exist_ok=True)

    def save_prompt(self, prompt):
        filename_text = "inference_" + prompt.split(',')[0].strip()
        filename_text = re.sub(r'[^\w\-_\. ]', '_', filename_text)
        filename_text = filename_text[:30]  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_text}_{timestamp}.txt"
        filename = os.path.join(self.prompts_dir, base_filename)
        
        with open(filename, "w") as file:
            file.write(prompt)
        
        print(f"Prompt saved to {filename}")

    def generate(self, model, input_text, happy_talk, compress, compression_level, poster, custom_base_prompt="", use_ollama=False):
        try:
            if use_ollama:
                return self.generate_ollama(model, input_text, happy_talk, compress, compression_level, poster, custom_base_prompt)
            else:
                return self.generate_groq(model, input_text, happy_talk, compress, compression_level, poster, custom_base_prompt)
        except Exception as e:
            print(f"An error occurred: {e}")
            return f"Error occurred while processing the request: {str(e)}"

    def generate_groq(self, model, input_text, happy_talk, compress, compression_level, poster, custom_base_prompt=""):
        model_id = self.groq_models.get(model, "llama-3.1-8b-instant")
        
        base_prompt = self.get_base_prompt(happy_talk, compress, compression_level, poster, custom_base_prompt)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Try your best to give best response possible to user."},
            {"role": "user", "content": f"{base_prompt}\nDescription: {input_text}"}
        ]
        
        print(f"Starting generation with Groq {model_id}...")
        start_time = time.time()

        chat_completion = self.groq_client.chat.completions.create(
            messages=messages,
            model=model_id,
            max_tokens=4000,
            temperature=0.7,
            top_p=0.95
        )

        end_time = time.time()
        print(f"Generation completed in {end_time - start_time:.2f} seconds")

        output = chat_completion.choices[0].message.content
        return self.clean_output(output)

    def generate_ollama(self, model, input_text, happy_talk, compress, compression_level, poster, custom_base_prompt=""):
        base_prompt = self.get_base_prompt(happy_talk, compress, compression_level, poster, custom_base_prompt)
        
        prompt = f"{base_prompt}\nDescription: {input_text}"
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        print(f"Starting generation with Ollama {model}...")
        start_time = time.time()

        response = requests.post(self.ollama_url, json=data)
        response.raise_for_status()

        end_time = time.time()
        print(f"Generation completed in {end_time - start_time:.2f} seconds")

        output = response.json()["response"]
        return self.clean_output(output)

    def get_base_prompt(self, happy_talk, compress, compression_level, poster, custom_base_prompt):
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

title = """<h1 align="center">FLUX Prompt Generator</h1>
<p><center>
<p align="center">Flux Prompt generator modified by <a href="https://www.patreon.com/aitrepreneur" target="_blank">[Aitrepreneur]</a>.</p>
<a href="https://x.com/gokayfem" target="_blank">[X gokaygokay]</a>
<a href="https://github.com/gokayfem" target="_blank">[Github gokayfem]</a>
<a href="https://github.com/dagthomas/comfyui_dagthomas" target="_blank">[comfyui_dagthomas]</a>
<p align="center">Create long prompts from images or simple words. Enhance your short prompts with prompt enhancer.</p>
</center></p>
"""

def create_interface():
    prompt_generator = PromptGenerator()
    inference_node = InferenceNode()
    
    with gr.Blocks(theme='bethecloud/storj_theme') as demo:
        
        gr.HTML(title)

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Accordion("Basic Settings"):
                    seed = gr.Slider(0, 30000, label='Seed', step=1, value=random.randint(0,30000))
                    custom = gr.Textbox(label="Custom Input Prompt (optional)")
                    subject = gr.Textbox(label="Subject (optional)")
                    
                    # Add the radio button for global option selection
                    global_option = gr.Radio(
                        ["Disabled", "Random", "No Figure Rand"],
                        label="Set all options to:",
                        value="Disabled"
                    )
                
                with gr.Accordion("Artform and Photo Type", open=False):
                    artform = gr.Dropdown(["disabled", "random"] + ARTFORM, label="Artform", value="disabled")
                    photo_type = gr.Dropdown(["disabled", "random"] + PHOTO_TYPE, label="Photo Type", value="disabled")
            
                with gr.Accordion("Character Details", open=False):
                    body_types = gr.Dropdown(["disabled", "random"] + BODY_TYPES, label="Body Types", value="disabled")
                    default_tags = gr.Dropdown(["disabled", "random"] + DEFAULT_TAGS, label="Default Tags", value="disabled")
                    roles = gr.Dropdown(["disabled", "random"] + ROLES, label="Roles", value="disabled")
                    hairstyles = gr.Dropdown(["disabled", "random"] + HAIRSTYLES, label="Hairstyles", value="disabled")
                    clothing = gr.Dropdown(["disabled", "random"] + CLOTHING, label="Clothing", value="disabled")
            
                with gr.Accordion("Scene Details", open=False):
                    place = gr.Dropdown(["disabled", "random"] + PLACE, label="Place", value="disabled")
                    lighting = gr.Dropdown(["disabled", "random"] + LIGHTING, label="Lighting", value="disabled")
                    composition = gr.Dropdown(["disabled", "random"] + COMPOSITION, label="Composition", value="disabled")
                    pose = gr.Dropdown(["disabled", "random"] + POSE, label="Pose", value="disabled")
                    background = gr.Dropdown(["disabled", "random"] + BACKGROUND, label="Background", value="disabled")
            
                with gr.Accordion("Style and Artist", open=False):
                    additional_details = gr.Dropdown(["disabled", "random"] + ADDITIONAL_DETAILS, label="Additional Details", value="disabled")
                    photography_styles = gr.Dropdown(["disabled", "random"] + PHOTOGRAPHY_STYLES, label="Photography Styles", value="disabled")
                    device = gr.Dropdown(["disabled", "random"] + DEVICE, label="Device", value="disabled")
                    photographer = gr.Dropdown(["disabled", "random"] + PHOTOGRAPHER, label="Photographer", value="disabled")
                    artist = gr.Dropdown(["disabled", "random"] + ARTIST, label="Artist", value="disabled")
                    digital_artform = gr.Dropdown(["disabled", "random"] + DIGITAL_ARTFORM, label="Digital Artform", value="disabled")
                
                generate_button = gr.Button("Generate Prompt")

            with gr.Column(scale=2):
                with gr.Accordion("Image and Caption", open=False):
                    input_image = gr.Image(label="Input Image (optional)")
                    caption_output = gr.Textbox(label="Generated Caption", lines=3)
                    create_caption_button = gr.Button("Create Caption")
                    add_caption_button = gr.Button("Add Caption to Prompt")

                with gr.Accordion("Prompt Generation", open=True):
                    output = gr.Textbox(label="Generated Prompt / Input Text", lines=4)
                    t5xxl_output = gr.Textbox(label="T5XXL Output", visible=True)
                    clip_l_output = gr.Textbox(label="CLIP L Output", visible=True)
                    clip_g_output = gr.Textbox(label="CLIP G Output", visible=True)
            
            with gr.Column(scale=2):
                with gr.Accordion("Prompt Generation with LLM", open=False):
                    use_ollama = gr.Checkbox(label="Use Ollama (local)", value=False)
                    model = gr.Dropdown(["Mixtral 8x7B", "Llama 3.1 70B", "Llama 3.1 8B"], label="Groq Model", value="Llama 3.1 8B")
                    ollama_model = gr.Textbox(label="Ollama Model Name", value="llama3", visible=False)
                    happy_talk = gr.Checkbox(label="Happy Talk", value=True)
                    compress = gr.Checkbox(label="Compress", value=True)
                    compression_level = gr.Radio(["soft", "medium", "hard"], label="Compression Level", value="hard")
                    poster = gr.Checkbox(label="Poster", value=False)
                    custom_base_prompt = gr.Textbox(label="Custom Base Prompt", lines=5)
                generate_text_button = gr.Button("Generate Prompt with LLM")
                text_output = gr.Textbox(label="Generated Text", lines=10)

        def create_caption(image):
            if image is not None:
                return florence_caption(image)
            return ""

        create_caption_button.click(
            create_caption,
            inputs=[input_image],
            outputs=[caption_output]
        )

        generate_button.click(
            prompt_generator.generate_prompt,
            inputs=[seed, custom, subject, artform, photo_type, body_types, default_tags, roles, hairstyles,
                    additional_details, photography_styles, device, photographer, artist, digital_artform,
                    place, lighting, clothing, composition, pose, background, input_image],
            outputs=[output, gr.Number(visible=False), t5xxl_output, clip_l_output, clip_g_output]
        )

        add_caption_button.click(
            prompt_generator.add_caption_to_prompt,
            inputs=[output, caption_output],
            outputs=[output]
        )
        
        def generate_text_with_model(use_ollama, model, ollama_model, input_text, happy_talk, compress, compression_level, poster, custom_base_prompt):
            print(f"Generating text with {'Ollama' if use_ollama else 'Groq'} model: {ollama_model if use_ollama else model}")
            output = inference_node.generate(
                ollama_model if use_ollama else model,
                input_text,
                happy_talk,
                compress,
                compression_level,
                poster,
                custom_base_prompt,
                use_ollama
            )
            print("Generation completed.")
            return output 

        generate_text_button.click(
            generate_text_with_model,
            inputs=[use_ollama, model, ollama_model, output, happy_talk, compress, compression_level, poster, custom_base_prompt],
            outputs=text_output
        )
        
        def toggle_model_input(use_ollama):
            return gr.update(visible=use_ollama), gr.update(visible=not use_ollama)

        use_ollama.change(
            toggle_model_input,
            inputs=[use_ollama],
            outputs=[ollama_model, model]
        )

        def update_all_options(choice):
            updates = {}
            if choice == "Disabled":
                for dropdown in [
                    artform, photo_type, body_types, default_tags, roles, hairstyles, clothing,
                    place, lighting, composition, pose, background, additional_details,
                    photography_styles, device, photographer, artist, digital_artform
                ]:
                    updates[dropdown] = gr.update(value="disabled")
            elif choice == "Random":
                for dropdown in [
                    artform, photo_type, body_types, default_tags, roles, hairstyles, clothing,
                    place, lighting, composition, pose, background, additional_details,
                    photography_styles, device, photographer, artist, digital_artform
                ]:
                    updates[dropdown] = gr.update(value="random")
            else:  # No Figure Random
                for dropdown in [photo_type, body_types, default_tags, roles, hairstyles, clothing, pose, additional_details]:
                    updates[dropdown] = gr.update(value="disabled")
                for dropdown in [artform, place, lighting, composition, background, photography_styles, device, photographer, artist, digital_artform]:
                    updates[dropdown] = gr.update(value="random")
            return updates
        
        global_option.change(
            update_all_options,
            inputs=[global_option],
            outputs=[
                artform, photo_type, body_types, default_tags, roles, hairstyles, clothing,
                place, lighting, composition, pose, background, additional_details,
                photography_styles, device, photographer, artist, digital_artform
            ]
        )

    return demo

if __name__ == "__main__":
    print("FLUX Prompt Generator Initialized! HAVE FUN :)")
    demo = create_interface()
    demo.launch()