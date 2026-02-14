import json
import copy
import os
from typing import Dict, Any, Optional, List, TypedDict
from jinja2 import Template
from google import genai
import fal_client

# LangGraph imports
from langgraph.graph import StateGraph, END

# ======================================================
# AgentState Definition
# ======================================================
class AgentState(TypedDict):
    # Inputs
    platform: str
    intent: str
    user_inputs: Dict[str, Any]
    uploaded_files: Dict[str, Any]
    user_media_choice: Optional[str]
    
    # Intermediate Data
    generation_plan: Dict[str, Any]
    text_model_input: str
    media_payload: Dict[str, Any]
    visual_prompt: Optional[str] # Refined visual prompt from LLM
    
    # Outputs
    generated_text: Optional[str]
    generated_media_url: Optional[str]
    
    # Status & Errors
    status: str # e.g., "planning", "writing", "producing", "completed"
    errors: List[str]

# ======================================================
# Gemini Setup (from content_orchestrationfal.py)
# ======================================================
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY") # Use env var
)
MODEL_NAME = "gemini-2.5-flash"

def generate_text_with_gemini(prompt: str) -> str:
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
# fal.ai Setup (from content_orchestrationfal.py)
# ======================================================
# Ensure FAL_KEY is set in your environment
os.environ["FAL_KEY"] = os.environ.get("FAL_KEY", "YOUR_FAL_API_KEY")

def generate_image_with_fal(
    prompt: str, 
    constraints: Dict[str, Any],
    reference_image: Optional[str] = None
) -> str:
    print(f"Generating image with fal.ai (nano-banana) prompt: {prompt[:50]}...")
    
    arguments = {
        "prompt": prompt,
        "output_format": "png",
        "width": constraints.get("width", 1024),
        "height": constraints.get("height", 1024),
    }
    
    if reference_image:
        arguments["image_input"] = [reference_image]
    
    try:
        result = fal_client.subscribe(
            "fal-ai/nano-banana",
            arguments=arguments,
            with_logs=True
        )
        if 'images' in result and len(result['images']) > 0:
            return result['images'][0]['url']
        return str(result)
    except Exception as e:
        return f"Error generating image with fal.ai: {str(e)}"

def generate_video_with_fal(
    prompt: str, 
    constraints: Dict[str, Any],
    init_image: Optional[str] = None
) -> str:
    print(f"Generating video with fal.ai (veo3) prompt: {prompt[:50]}...")
    
    arguments = {
        "prompt": prompt,
        "duration": constraints.get("max_duration_sec", 8),
        "resolution": constraints.get("resolution", "1080p"),
        "aspect_ratio": constraints.get("aspect_ratio", "16:9"),
        "generate_audio": constraints.get("supports_audio", True)
    }
    
    if init_image:
        arguments["image"] = init_image

    try:
        result = fal_client.subscribe(
            "fal-ai/veo3",
            arguments=arguments,
            with_logs=True
        )
        if 'video' in result:
            return result['video']['url']
        return str(result)
    except Exception as e:
        return f"Error generating video with fal.ai: {str(e)}"

# ======================================================
# CONFIG & RULES (from content_orchestrationfal.py)
# ======================================================
CONFIG_FILE = "platform_rules_config.json" # Assuming this is in the same directory

def load_platform_rules() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_FILE):
        dummy_config = {"PLATFORM_RULES": {"X": {"DEFAULT": {"content_type": "post", "objective": "engagement", "tone": "professional", "text_constraints": {"max_chars": 280}, "media_constraints": {"type": "optional"}}}}}
        with open(CONFIG_FILE, "w") as f: json.dump(dummy_config, f)
    with open(CONFIG_FILE, "r", encoding="utf-8") as f: return json.load(f)["PLATFORM_RULES"]

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict): result[k] = deep_merge(result[k], v)
        else: result[k] = copy.deepcopy(v)
    return result

def build_generation_plan(platform: str, intent: str) -> Dict[str, Any]:
    rules = load_platform_rules()
    platform = platform.upper()
    if platform not in rules: raise ValueError(f"Platform {platform} not found")
    if intent not in rules[platform]: raise ValueError(f"Intent {intent} not found for platform {platform}")
    merged = deep_merge(rules[platform]["DEFAULT"], rules[platform][intent])
    merged["_meta"] = {"platform": platform, "intent": intent}
    return merged

def normalize_plan(raw: Dict[str, Any], user_media_choice: Optional[str] = None) -> Dict[str, Any]:
    media = raw.get("media_constraints", {})
    text = raw.get("text_constraints", {})
    raw_type = media.get("type")
    if raw_type in ["optional", "image_or_short_video"]: allowed_types = ["image", "video"]
    elif raw_type in ["video", "photo_carousel", "text_only"]: allowed_types = [raw_type]
    else: allowed_types = []
    selected_type = user_media_choice if user_media_choice in allowed_types else (allowed_types[0] if allowed_types else None)
    return {
        "platform": raw.get("platform"), "content_type": raw.get("content_type"), "objective": raw.get("objective"), "tone": raw.get("tone"),
        "text_constraints": text,
        "media_constraints": {
            "allowed_types": allowed_types, "selected_type": selected_type, "raw": media,
            "image": media.get("image", {}), "video": media.get("video", {}),
            "image_count": media.get("image_count"), "aspect_ratio": media.get("aspect_ratio"),
            "safe_zone_required": media.get("safe_zone_required", False), "ugc_style": media.get("ugc_style", False),
            "supports_audio": media.get("supports_audio", False), "recommended_use_cases": media.get("recommended_use_cases", []),
        },
        "cta_style": raw.get("cta_style"), "optimization_goal": raw.get("optimization_goal"), "variation_count": raw.get("variation_count"),
        "_meta": raw.get("_meta")
    }

# ======================================================
# PROMPT TEMPLATES (from content_orchestrationfal.py)
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
Return only the generated {{ media_type }}.
No explanations. No captions text unless explicitly required.
"""

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
    if not media_type or media_type == "text_only":
        return None
    return Template(MEDIA_PROMPT_TEMPLATE).render(
        platform=plan["platform"],
        content_type=plan["content_type"],
        objective=plan["objective"],
        media_type=media_type,
        media_constraints=plan["media_constraints"]
    )

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

{f"""
VISUAL REFERENCE:
Use the attached image(s) as visual/style reference.
""" if uploaded_files.get("reference_image") else ""}

{f"""
INITIAL FRAME:
Use the attached image as the starting frame.
""" if uploaded_files.get("video_init_image") else ""}

====================
RULES
====================
- All fields above are provided by the user
- Use them as the sole source of truth
- Do NOT invent visual elements not implied by the data
"""

def build_final_model_input(
    plan: Dict[str, Any],
    user_inputs: Dict[str, Any],
    uploaded_files: Dict[str, Any]
) -> Dict[str, Any]:
    text_prompt = build_text_prompt(plan)
    media_prompt = build_media_prompt(plan)
    text_user_data = build_text_user_data_block(user_inputs)
    media_user_data = build_media_user_data_block(user_inputs, uploaded_files)
    return {
        "text_model_input": f"{text_prompt}\n\n{text_user_data}",
        "media_model_input": (f"{media_prompt}\n\n{media_user_data}" if media_prompt else None),
        "media_payload": {
            "reference_image": uploaded_files.get("reference_image"),
            "video_init_image": uploaded_files.get("video_init_image")
        },
        "generation_plan": plan
    }

# ======================================================
# LangGraph Nodes
# ======================================================

def planner_node(state: AgentState) -> AgentState:
    print("--- Node: Planner ---")
    try:
        raw_plan = build_generation_plan(state["platform"], state["intent"])
        plan = normalize_plan(raw_plan, user_media_choice=state["user_media_choice"])
        return {"generation_plan": plan, "status": "plan_generated", "errors": []}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Planner failed: {str(e)}"], "status": "failed"}

def prompt_engineer_node(state: AgentState) -> AgentState:
    print("--- Node: Prompt Engineer ---")
    try:
        final_input = build_final_model_input(
            state["generation_plan"],
            state["user_inputs"],
            state["uploaded_files"]
        )
        return {
            "text_model_input": final_input["text_model_input"],
            "media_payload": final_input.get("media_payload", {}),
            "visual_prompt": final_input.get("media_model_input"), # Renamed for clarity
            "status": "prompts_engineered", 
            "errors": []
        }
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Prompt Engineer failed: {str(e)}"], "status": "failed"}

def copywriter_node(state: AgentState) -> AgentState:
    print("--- Node: Copywriter ---")
    try:
        if not state.get("text_model_input"):
            raise ValueError("Text model input missing.")
        
        post_text = generate_text_with_gemini(state["text_model_input"])
        return {"generated_text": post_text, "status": "text_generated", "errors": []}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Copywriter failed: {str(e)}"], "status": "failed"}

def visual_refiner_node(state: AgentState) -> AgentState:
    print("--- Node: Visual Refiner ---")
    try:
        media_type = state["generation_plan"]["media_constraints"].get("selected_type")
        if not media_type or media_type == "text_only":
            return {"visual_prompt": None, "status": "visual_refiner_skipped", "errors": []}

        if not state.get("generated_text"):
            raise ValueError("Generated text missing for media prompt refinement.")

        # Refine visual prompt using generated text
        visual_prompt_request = f"Based on this social media post: '{state['generated_text']}', create a detailed visual prompt for an {media_type} generation. Focus on style, lighting, and composition. Return only the prompt."
        visual_prompt = generate_text_with_gemini(visual_prompt_request)
        print(f"Refined Visual Prompt: {visual_prompt[:100]}...")
        return {"visual_prompt": visual_prompt, "status": "visual_prompt_refined", "errors": []}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Visual Refiner failed: {str(e)}"], "status": "failed"}

def media_producer_node(state: AgentState) -> AgentState:
    print("--- Node: Media Producer ---")
    try:
        media_type = state["generation_plan"]["media_constraints"].get("selected_type")
        if not media_type or media_type == "text_only":
            return {"generated_media_url": None, "status": "media_producer_skipped", "errors": []}

        if not state.get("visual_prompt"):
            raise ValueError("Visual prompt missing for media generation.")

        media_result = None
        constraints = state["generation_plan"]["media_constraints"]
        
        if media_type == "image":
            media_result = generate_image_with_fal(
                prompt=state["visual_prompt"],
                constraints=constraints.get("image", {}),
                reference_image=state["uploaded_files"].get("reference_image")
            )
        elif media_type == "video":
            media_result = generate_video_with_fal(
                prompt=state["visual_prompt"],
                constraints=constraints.get("video", {}),
                init_image=state["uploaded_files"].get("video_init_image")
            )
        elif media_type == "photo_carousel":
            print("Generating Carousel Images...")
            carousel_urls = []
            image_count_constraints = constraints.get("image_count", {})
            min_images = image_count_constraints.get("min", 1) if isinstance(image_count_constraints, dict) else 1
            for i in range(min_images):
                url = generate_image_with_fal(
                    prompt=f"{state["visual_prompt"]} - Slide {i+1}",
                    constraints=constraints.get("image", {}),
                    reference_image=state["uploaded_files"].get("reference_image")
                )
                carousel_urls.append(url)
            media_result = carousel_urls
            
        return {"generated_media_url": media_result, "status": "media_produced", "errors": []}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Media Producer failed: {str(e)}"], "status": "failed"}

# ======================================================
# Graph Definition
# ======================================================

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("planner", planner_node)
workflow.add_node("prompt_engineer", prompt_engineer_node)
workflow.add_node("copywriter", copywriter_node)
workflow.add_node("visual_refiner", visual_refiner_node)
workflow.add_node("media_producer", media_producer_node)

# Set entry point
workflow.set_entry_point("planner")

# Add edges
workflow.add_edge("planner", "prompt_engineer")
workflow.add_edge("prompt_engineer", "copywriter")
workflow.add_edge("copywriter", "visual_refiner")
workflow.add_edge("visual_refiner", "media_producer")
workflow.add_edge("media_producer", END)

# Compile the graph
app = workflow.compile()

# ======================================================
# Demo Run
# ======================================================
if __name__ == "__main__":
    # Ensure platform_rules_config.json exists for the demo
    if not os.path.exists(CONFIG_FILE):
        print(f"Warning: {CONFIG_FILE} not found. Creating a dummy one.")
        dummy_config = {"PLATFORM_RULES": {"X": {"DEFAULT": {"content_type": "post", "objective": "engagement", "tone": "professional", "text_constraints": {"max_chars": 280}, "media_constraints": {"type": "optional"}}}}}
        with open(CONFIG_FILE, "w") as f: json.dump(dummy_config, f)

    # Example User Inputs for X (Image)
    user_inputs_x = {
        "content_idea": "AI-powered social media content generation",
        "description": "A new tool for freelance marketers to create high-performing ads on X.",
        "reference_text": "Focus on efficiency, compliance, and ROI."
    }
    uploaded_files_x = {"reference_image": None, "video_init_image": None}

    print("\n--- Running LangGraph Agent for X (Image) ---")
    final_state_x = app.invoke({
        "platform": "X",
        "intent": "PAID_AD", 
        "user_inputs": user_inputs_x,
        "uploaded_files": uploaded_files_x,
        "user_media_choice": "image",
        "errors": []
    })
    print("\n--- Final State for X (Image) ---")
    print(f"Generated Text: {final_state_x.get("generated_text")}")
    print(f"Generated Media URL: {final_state_x.get("generated_media_url")}")
    print(f"Errors: {final_state_x.get("errors")}")

    # Example User Inputs for TikTok (Video)
    user_inputs_tiktok = {
        "content_idea": "Quick tutorial on AI prompt engineering",
        "description": "Short, engaging video for TikTok showing how to write effective prompts.",
        "reference_text": "Energetic, educational, fast-paced."
    }
    uploaded_files_tiktok = {"reference_image": None, "video_init_image": None}

    print("\n--- Running LangGraph Agent for TikTok (Video) ---")
    final_state_tiktok = app.invoke({
        "platform": "TikTok",
        "intent": "ORGANIC_PROMOTION", 
        "user_inputs": user_inputs_tiktok,
        "uploaded_files": uploaded_files_tiktok,
        "user_media_choice": "video",
        "errors": []
    })
    print("\n--- Final State for TikTok (Video) ---")
    print(f"Generated Text: {final_state_tiktok.get("generated_text")}")
    print(f"Generated Media URL: {final_state_tiktok.get("generated_media_url")}")
    print(f"Errors: {final_state_tiktok.get("errors")}")

   

