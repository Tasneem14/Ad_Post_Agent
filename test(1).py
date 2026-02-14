# ======================================================
# content_orchestration.py
# ======================================================

import json
import copy
import os
from typing import Dict, Any, Optional
from jinja2 import Template
from google import genai
import replicate

# ======================================================
# Gemini Setup (NEW SDK – CORRECT)
# ======================================================

# Note: In a real scenario, the API key should be an environment variable.
# For this task, I'm keeping the provided key but it's better to use os.environ.get("GEMINI_API_KEY")
client = genai.Client(
    api_key="AIzaSyBOeFYIjQtbHs-sRoDmKH1DMpIZCCreFUg"
)

MODEL_NAME = "gemini-2.5-flash"

def generate_text_with_gemini(prompt: str) -> str:
    """
    Sends a text prompt to Gemini and returns generated text.
    """
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_output_tokens": 2000
            }
        )
        return response.text.strip()
    except Exception as e:
        return f"Error generating text with Gemini: {str(e)}"

# ======================================================
# Replicate Setup
# ======================================================

# Ensure REPLICATE_API_TOKEN is set in your environment
os.environ["REPLICATE_API_TOKEN"] = "your_token_here"

def generate_image_with_replicate(prompt: str, reference_image: Optional[str] = None) -> str:
    """
    Generates an image using Replicate (flux-schnell).
    If reference_image is provided, it can be used for models that support it.
    """
    print(f"Generating image with prompt: {prompt[:50]}...")
    
    input_params = {"prompt": prompt}
    
    # Note: flux-schnell doesn't directly take a reference image for style in the same way 
    # as some other models. For a general 'image generation' request, we'll use flux-schnell.
    
    try:
        output = replicate.run(
            "google/nano-banana",
            input=input_params
        )
        # output is usually a list of FileOutput objects or URLs
        if isinstance(output, list) and len(output) > 0:
            return str(output[0])
        return str(output)
    except Exception as e:
        return f"Error generating image: {str(e)}"

def generate_video_with_replicate(prompt: str, init_image: Optional[str] = None) -> str:
    """
    Generates a video using Replicate (ltx-video).
    If init_image is provided, it's used as the starting frame.
    """
    print(f"Generating video with prompt: {prompt[:50]}...")
    
    input_params = {
        "prompt": prompt,
        "aspect_ratio": "16:9",
        "model": "0.9.1"
    }
    
    file_handle = None
    if init_image:
        # Replicate can take a file handle or a URL
        if init_image.startswith("http"):
            input_params["image"] = init_image
        elif os.path.exists(init_image):
            file_handle = open(init_image, "rb")
            input_params["image"] = file_handle
        else:
            print(f"Warning: init_image path '{init_image}' not found. Proceeding without it.")

    try:
        output = replicate.run(
            "lightricks/ltx-video:8c47da66681d081eeb4d1261853087de23923a268a",
            input=input_params
        )
        # output is usually a list of URLs
        if isinstance(output, list) and len(output) > 0:
            return str(output[0])
        return str(output)
    except Exception as e:
        return f"Error generating video: {str(e)}"
    finally:
        if file_handle:
            file_handle.close()

# ======================================================
# CONFIG
# ======================================================

CONFIG_FILE = "platform_rules_config.json"

# ======================================================
# LOAD PLATFORM RULES
# ======================================================

def load_platform_rules() -> Dict[str, Any]:
    # Creating a dummy config if it doesn't exist for testing purposes
    if not os.path.exists(CONFIG_FILE):
        dummy_config = {
            "PLATFORM_RULES": {
                "X": {
                    "DEFAULT": {
                        "content_type": "post",
                        "objective": "engagement",
                        "tone": "professional",
                        "text_constraints": {"max_chars": 280},
                        "media_constraints": {"type": "optional"}
                    },
                    "PROMO": {
                        "objective": "conversion"
                    }
                }
            }
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(dummy_config, f)
            
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["PLATFORM_RULES"]

# ======================================================
# UTILS
# ======================================================

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result

# ======================================================
# BUILD GENERATION PLAN (FROM JSON)
# ======================================================

def build_generation_plan(platform: str, intent: str) -> Dict[str, Any]:
    rules = load_platform_rules()
    platform = platform.upper()

    if platform not in rules:
        raise ValueError(f"Platform {platform} not found")

    if intent not in rules[platform]:
        raise ValueError(f"Intent {intent} not found for platform {platform}")

    merged = deep_merge(
        rules[platform]["DEFAULT"],
        rules[platform][intent]
    )

    merged["_meta"] = {
        "platform": platform,
        "intent": intent
    }
    return merged

# ======================================================
# NORMALIZE PLAN (FINAL SHAPE)
# ======================================================

def normalize_plan(raw: Dict[str, Any], user_media_choice: Optional[str] = None) -> Dict[str, Any]:
    media = raw.get("media_constraints", {})
    text = raw.get("text_constraints", {})

    media_type = media.get("type")

    if media_type in ["optional", "image_or_short_video"]:
        allowed_types = ["image", "video"]
    elif media_type in ["video", "photo_carousel", "text_only"]:
        allowed_types = [media_type]
    else:
        allowed_types = []

    selected_type = (
        user_media_choice
        if user_media_choice in allowed_types
        else allowed_types[0] if allowed_types else None
    )

    return {
        "platform": raw.get("platform"),
        "content_type": raw.get("content_type"),
        "objective": raw.get("objective"),
        "tone": raw.get("tone"),

        "text_constraints": text,

        "media_constraints": {
            "allowed_types": allowed_types,
            "selected_type": selected_type,

            # ⬇️ NEW: preserve raw media constraints
            "raw": media,

            # normalized helpers
            "image": media.get("image", {}),
            "video": media.get("video", {}),
            "image_count": media.get("image_count"),
            "aspect_ratio": media.get("aspect_ratio"),
            "safe_zone_required": media.get("safe_zone_required"),
            "ugc_style": media.get("ugc_style"),
            "supports_audio": media.get("supports_audio"),
            "recommended_use_cases": media.get("recommended_use_cases"),
        },

        "cta_style": raw.get("cta_style"),
        "optimization_goal": raw.get("optimization_goal"),
        "variation_count": raw.get("variation_count"),

        "_meta": raw.get("_meta")
    }


# ======================================================
# PROMPT TEMPLATES (INSTRUCTIONS ONLY)
# ======================================================

TEXT_PROMPT_TEMPLATE = """
SYSTEM:
You are a senior social media copywriter who specializes in {{ platform }} {{ content_type }} content.

CONTEXT:
You are writing a {{ content_type }} post.
Your success is measured by {{ objective }}.
The audience scrolls fast, so clarity and impact matter.

YOUR MISSION:
Write a single post that communicates the idea clearly, hooks the reader immediately,
and feels native to {{ platform }}.

NON-NEGOTIABLE RULES:
{%- if text_constraints.min_chars and text_constraints.max_chars %}
- The final text must be between {{ text_constraints.min_chars }} and {{ text_constraints.max_chars }} characters.
{%- elif text_constraints.max_chars %}
- The final text must not exceed {{ text_constraints.max_chars }} characters.
{%- endif %}
{%- if text_constraints.max_words is defined and text_constraints.max_words is not none %}
- The final text must not exceed {{ text_constraints.max_words }} words.
{%- endif %}
{%- if text_constraints.max_emojis is defined and text_constraints.max_emojis is not none %}
- You may use at most {{ text_constraints.max_emojis }} emoji.
{%- endif %}
- The tone must be {{ tone }}.
{%- if text_constraints.hook_first_50_chars %}
- The first 50 characters must contain a strong hook.
{%- endif %}
{%- if text_constraints.allow_hashtags is not none %}
{%- if not text_constraints.allow_hashtags %}
- Hashtags are NOT allowed.
{%- else %}
- Hashtags are allowed if they fit naturally.
{%- endif %}
{%- endif %}
{%- if text_constraints.allow_mentions is not none %}
{%- if not text_constraints.allow_mentions %}
- Mentions are NOT allowed.
{%- else %}
- Mentions are allowed if relevant.
{%- endif %}
{%- endif %}

HOW TO WORK:
- Use ONLY the information provided in the USER PROVIDED DATA section.
- Do NOT invent facts, features, or claims.
- Write naturally, as a human copywriter would.

FINAL OUTPUT:
Return only the final post text.
No explanations. No formatting.
"""

MEDIA_PROMPT_TEMPLATE = """
SYSTEM:
You are a senior creative director producing media content for {{ platform }}.

CONTEXT:
This media asset supports a {{ content_type }} post.
The primary goal is {{ objective }}.
The content must feel native to {{ platform }} and optimized for fast-scrolling users.

MEDIA TYPE:
{{ media_type }}

TECHNICAL CONSTRAINTS (STRICT):
{%- if media_type == "image" %}
- Aspect ratio: {{ media_constraints.image.aspect_ratio }}
- Minimum resolution: {{ media_constraints.image.min_resolution }}
{%- if media_constraints.image.max_file_size_mb %}
- Maximum file size: {{ media_constraints.image.max_file_size_mb }} MB
{%- endif %}
{%- if media_constraints.image.text_overlay_allowed %}
- Text overlays are allowed.
{%- else %}
- Text overlays are NOT allowed.
{%- endif %}
{%- if media_constraints.image.branding_required %}
- Branding is required.
{%- else %}
- Branding is not required.
{%- endif %}
{%- if media_constraints.image.branding_position %}
- Branding position: {{ media_constraints.image.branding_position }}
{%- endif %}
{%- endif %}

{%- if media_type == "video" %}
{%- if media_constraints.video.max_duration_sec %}
- Max duration: {{ media_constraints.video.max_duration_sec }} seconds
{%- endif %}
{%- if media_constraints.video.aspect_ratio %}
- Aspect ratio: {{ media_constraints.video.aspect_ratio }}
{%- elif media_constraints.video.aspect_ratios %}
- Allowed aspect ratios: {{ media_constraints.video.aspect_ratios }}
{%- endif %}
{%- if media_constraints.video.captions_required %}
- Captions are required.
{%- else %}
- Captions are optional.
{%- endif %}
{%- if media_constraints.video.hook_first_sec %}
- The first {{ media_constraints.video.hook_first_sec }} seconds must contain a strong hook.
{%- endif %}
{%- if media_constraints.video.branding_first_sec %}
- Branding must appear within the first {{ media_constraints.video.branding_first_sec }} seconds.
{%- endif %}
{%- endif %}

{%- if media_type == "photo_carousel" %}
- This is a multi-image carousel.
{%- if media_constraints.image_count %}
- Number of images: between {{ media_constraints.image_count.min }} and {{ media_constraints.image_count.max }}.
{%- endif %}
{%- if media_constraints.aspect_ratio %}
- Aspect ratio: {{ media_constraints.aspect_ratio }}.
{%- endif %}
{%- if media_constraints.safe_zone_required %}
- Safe zones must be respected.
{%- endif %}
{%- if media_constraints.ugc_style %}
- The visual style should feel like authentic UGC.
{%- endif %}
{%- if media_constraints.recommended_use_cases %}
- Recommended use cases: {{ media_constraints.recommended_use_cases }}.
{%- endif %}
{%- endif %}

CREATIVE DIRECTION:
- The visual should clearly communicate the core idea.
- The message must be understandable without reading additional text.
- Avoid clutter, unnecessary elements, or visual noise.
{%- if media_type == "video" %}
- Prioritize clarity in the first few seconds.
{%- endif %}

USAGE RULES:
- Use ONLY the information provided in the USER PROVIDED DATA section.
- Do NOT introduce new concepts, claims, or visual elements not implied by the data.
- Follow all constraints exactly as listed above.

FINAL OUTPUT:
Return only a detailed visual description for generating the {{ media_type }}.
No explanations.
"""

# ======================================================
# BUILD PROMPTS
# ======================================================

def build_text_prompt(plan: Dict[str, Any]) -> str:
    return Template(TEXT_PROMPT_TEMPLATE).render(
        platform=plan["platform"],
        objective=plan["objective"],
        content_type=plan["content_type"],
        tone=plan["tone"],
        text_constraints=plan["text_constraints"]
    )

def build_media_prompt(plan: Dict[str, Any]) -> Optional[str]:
    media_type = plan["media_constraints"]["selected_type"]
    if not media_type:
        return None

    return Template(MEDIA_PROMPT_TEMPLATE).render(
        platform=plan["platform"],
        content_type=plan["content_type"],
        objective=plan["objective"],
        media_type=media_type,
        media_constraints=plan["media_constraints"]
    )

# ======================================================
# USER INPUTS → STRUCTURED NATURAL LANGUAGE
# ======================================================

def build_text_user_data_block(user_inputs: Dict[str, Any]) -> str:
    return f"""
====================
USER PROVIDED DATA
====================

CONTENT IDEA:
{user_inputs.get("content_idea")}

CONTEXT / DESCRIPTION:
{user_inputs.get("description")}

REFERENCE TEXT:
{user_inputs.get("reference_text")}

====================
RULES
====================
- All fields above are provided by the user
- Use them as the sole source of truth
- Do NOT invent missing information
"""


def build_media_user_data_block(
    user_inputs: Dict[str, Any],
    uploaded_files: Dict[str, Any]
) -> str:
    return f"""
====================
USER PROVIDED DATA
====================

CORE IDEA:
{user_inputs.get("content_idea")}

CONTEXT / DESCRIPTION:
{user_inputs.get("description")}

{f'''
VISUAL REFERENCE:
Use the attached image(s) as visual/style reference.
''' if uploaded_files.get("reference_image") else ""}

{f'''
INITIAL FRAME:
Use the attached image as the starting frame.
''' if uploaded_files.get("video_init_image") else ""}

====================
RULES
====================
- All fields above are provided by the user
- Use them as the sole source of truth
- Do NOT invent visual elements not implied by the data
"""


# ======================================================
# FINAL MODEL INPUT (WHAT YOU SEND TO MODELS)
# ======================================================

def build_final_model_input(
    plan: Dict[str, Any],
    user_inputs: Dict[str, Any],
    uploaded_files: Dict[str, Any]
) -> Dict[str, Any]:

    text_prompt = build_text_prompt(plan)
    media_prompt = build_media_prompt(plan)

    text_user_data = build_text_user_data_block(user_inputs)
    media_user_data = build_media_user_data_block(user_inputs, uploaded_files)

    # Construct the media payload according to the required schema
    media_constraints = plan.get("media_constraints", {})
    selected_type = media_constraints.get("selected_type")

    media_payload = {
        "media_model_input": (
            f"{media_prompt}\n\n{media_user_data}"
            if media_prompt else None
        ),
        "media_type": selected_type,
        "media_constraints": {
            "allowed_types": media_constraints.get("allowed_types", []),
            "selected_type": selected_type,
            "image": media_constraints.get("image"),
            "video": media_constraints.get("video"),
            "image_count": media_constraints.get("image_count"),
            "aspect_ratio": media_constraints.get("aspect_ratio"),
            "safe_zone_required": media_constraints.get("safe_zone_required", False),
            "ugc_style": media_constraints.get("ugc_style", False),
            "supports_audio": media_constraints.get("supports_audio", False),
            "recommended_use_cases": media_constraints.get("recommended_use_cases", [])
        },
        "reference_image": uploaded_files.get("reference_image"),
        "video_init_image": uploaded_files.get("video_init_image")
    }

    return {
        # ===== TEXT MODEL =====
        "text_model_input": f"{text_prompt}\n\n{text_user_data}",

        # ===== MEDIA PAYLOAD (NEW) =====
        "media_payload": media_payload,

        # ===== DEBUG / INSPECTION =====
        "generation_plan": plan
    }

# ======================================================
# DEMO
# ======================================================

if __name__ == "__main__":
    # 1. Setup Plan
    raw = build_generation_plan("X", "DEFAULT")
    plan = normalize_plan(raw, user_media_choice="image")

    user_inputs = {
        "content_idea": "Prompt engineering is becoming a career skill, not just a tech trick",
        "description": "Targeting students, juniors, and professionals looking to upskill in AI",
        "reference_text": "Encouraging, forward-looking, practical."
    }

    uploaded_files = {
        "reference_image": None, # Set to a path if you have one
        "video_init_image": None
    }

    final_input = build_final_model_input(plan, user_inputs, uploaded_files)

    print("\n===== GENERATION PLAN =====\n")
    print(json.dumps(plan, indent=2))

    # 2. Generate Text with Gemini
    print("\n===== Generating Text with Gemini =====\n")
    final_post_text = generate_text_with_gemini(final_input["text_model_input"])
    print(f"Post Text: {final_post_text}")

    # 3. Generate Media with Replicate
    media_type = final_input["media_payload"]["media_type"]
    
    # Use Gemini to generate a detailed visual prompt based on the post text and media constraints
    visual_prompt_request = f"Based on this social media post: '{final_post_text}', create a detailed visual prompt for an {media_type} generation. Focus on style, lighting, and composition. Return only the prompt."
    visual_prompt = generate_text_with_gemini(visual_prompt_request)
    print(f"\n===== Visual Prompt for Replicate =====\n{visual_prompt}")

    try:
        if media_type == "image":
            image_url = generate_image_with_replicate(
                visual_prompt, 
                reference_image=final_input["media_payload"]["reference_image"]
            )
            print(f"\n===== Generated Image URL =====\n{image_url}")
        elif media_type == "video":
            video_url = generate_video_with_replicate(
                visual_prompt, 
                init_image=final_input["media_payload"]["video_init_image"]
            )
            print(f"\n===== Generated Video URL =====\n{video_url}")
        elif media_type == "photo_carousel":
            print("Generating Carousel Images...")
            carousel_urls = []
            image_count_constraints = final_input["generation_plan"]["media_constraints"].get("image_count", {})
            min_images = image_count_constraints.get("min", 1) if isinstance(image_count_constraints, dict) else 1
            
            for i in range(min_images):
                url = generate_image_with_replicate(
                    prompt=f"{visual_prompt} - Slide {i+1}",
                    reference_image=final_input["media_payload"].get("reference_image")
                )
                carousel_urls.append(url)
                print(f"Slide {i+1} URL: {url}")
    except Exception as e:
        print(f"Generation failed: {e}")
