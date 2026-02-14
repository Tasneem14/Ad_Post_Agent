import json
from content_orchestration import build_generation_plan, normalize_plan, build_final_model_input

def test_dynamic_selection():
    print("Testing Dynamic Media Selection...")
    # X DEFAULT allows image or video (optional)
    raw = build_generation_plan("X", "DEFAULT")
    
    # Case 1: User chooses video
    plan_video = normalize_plan(raw, user_media_choice="video")
    assert plan_video["media_constraints"]["selected_type"] == "video"
    
    # Case 2: User chooses image
    plan_image = normalize_plan(raw, user_media_choice="image")
    assert plan_image["media_constraints"]["selected_type"] == "image"
    
    # Case 3: User choice is invalid for the platform
    plan_invalid = normalize_plan(raw, user_media_choice="photo_carousel")
    # Should fallback to default (optional -> image/video, usually first allowed)
    assert plan_invalid["media_constraints"]["selected_type"] in ["image", "video"]
    
    # Case 4: No user choice
    plan_none = normalize_plan(raw, user_media_choice=None)
    assert plan_none["media_constraints"]["selected_type"] in ["image", "video"]
    print("Dynamic Selection Test Passed!")

def test_payload_schema():
    print("\nTesting Media Payload Schema...")
    raw = build_generation_plan("TIKTOK", "PHOTO_CAROUSEL")
    plan = normalize_plan(raw, user_media_choice="photo_carousel")
    
    user_inputs = {"content_idea": "test", "description": "test", "reference_text": "test"}
    uploaded_files = {"reference_image": "ref.png", "video_init_image": None}
    
    final_input = build_final_model_input(plan, user_inputs, uploaded_files)
    payload = final_input["media_payload"]
    
    # Check required keys
    required_keys = ["media_model_input", "media_type", "media_constraints", "reference_image", "video_init_image"]
    for key in required_keys:
        assert key in payload, f"Missing key: {key}"
    
    # Check media_constraints sub-keys
    constraints = payload["media_constraints"]
    required_constraints = [
        "allowed_types", "selected_type", "image", "video", 
        "image_count", "aspect_ratio", "safe_zone_required", 
        "ugc_style", "supports_audio", "recommended_use_cases"
    ]
    for key in required_constraints:
        assert key in constraints, f"Missing constraint key: {key}"
        
    # Check user files
    assert payload["reference_image"] == "ref.png"
    assert payload["video_init_image"] is None
    
    print("Payload Schema Test Passed!")

if __name__ == "__main__":
    try:
        test_dynamic_selection()
        test_payload_schema()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
