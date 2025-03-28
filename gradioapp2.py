import re
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image, ImageDraw, ImageFont
import ffmpeg
import subprocess
import os
import tempfile
import time
import gc
import numpy as np
import json
import random
import traceback
from pathlib import Path
import shutil
from deep_sort_integration import DeepSORTTracker

def draw_bbox(image, bbox, label=None, effect_type="bbox", mosaic_size=15, zoom_ratio=1.5, area_adjust=0, show_bbox_outline=True):
    """
    Draw bounding box on an image with optional effects
    
    Args:
        image: PIL Image to draw on
        bbox: Bounding box coordinates [x1, y1, x2, y2] or list of bounding boxes
        label: Optional label text
        effect_type: Type of effect to apply - "bbox", "mosaic", or "zoom"
        mosaic_size: Size of mosaic blocks when using mosaic effect
        zoom_ratio: Zoom ratio when using zoom effect
        area_adjust: Pixels to adjust the bounding box size by (positive to expand, negative to contract)
        show_bbox_outline: Whether to show the bounding box outline and label (True for bbox, optional for effects)
        
    Returns:
        Modified PIL Image
    """
    # Make a copy of the image to avoid modifying the original
    image = image.copy()
    draw = ImageDraw.Draw(image)
    
    # Try to load a font that supports Korean characters
    try:
        # Try to find a system font that supports Korean
        # Common fonts that support Korean: Malgun Gothic, Noto Sans CJK, etc.
        font_path = None
        potential_fonts = [
            "C:/Windows/Fonts/malgun.ttf",  # Malgun Gothic on Windows
            "C:/Windows/Fonts/gulim.ttc",   # Gulim on Windows
            "C:/Windows/Fonts/batang.ttc",  # Batang on Windows
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Linux
            "/System/Library/Fonts/AppleSDGothicNeo.ttc"  # macOS
        ]
        
        for path in potential_fonts:
            if os.path.exists(path):
                font_path = path
                break
                
        font = ImageFont.truetype(font_path, size=14) if font_path else ImageFont.load_default()
    except Exception as e:
        print(f"Warning: Could not load font for Korean text: {e}")
        font = ImageFont.load_default()
    
    # Use only orange color for all bounding boxes for consistency
    color = "orange"
    
    if isinstance(bbox[0], list) or isinstance(bbox[0], tuple):
        # Multiple bounding boxes
        for i, box in enumerate(bbox):
            # Apply area adjustment
            x1, y1, x2, y2 = box
            if area_adjust != 0:
                x1 = max(0, x1 - area_adjust)
                y1 = max(0, y1 - area_adjust)
                x2 = min(image.width, x2 + area_adjust)
                y2 = min(image.height, y2 + area_adjust)
            
            # Apply the selected effect
            if effect_type == "mosaic":
                # Apply mosaic effect to the area
                apply_mosaic(image, x1, y1, x2, y2, mosaic_size)
            elif effect_type == "zoom":
                # Apply zoom effect to the area
                apply_zoom(image, x1, y1, x2, y2, zoom_ratio)
            
            # Draw bounding box outline and label if requested
            if show_bbox_outline:
                # Draw the bounding box outline
                draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
                
                # Add label text if provided
                if label:
                    # Use label as is without adding any index
                    box_label = label
                    
                    # Background for text
                    text_width, text_height = draw.textbbox((0, 0), box_label, font=font)[2:]
                    draw.rectangle((x1, y1 - text_height - 5, x1 + text_width, y1), fill=color)
                    # Draw text
                    draw.text((x1, y1 - text_height - 5), box_label, fill="black", font=font)
            elif effect_type == "bbox":
                # Always draw the bounding box and label for "bbox" effect, regardless of show_bbox_outline
                draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
                
                # Add label text if provided
                if label:
                    # Use label as is without adding any index
                    box_label = label
                    
                    # Background for text
                    text_width, text_height = draw.textbbox((0, 0), box_label, font=font)[2:]
                    draw.rectangle((x1, y1 - text_height - 5, x1 + text_width, y1), fill=color)
                    # Draw text
                    draw.text((x1, y1 - text_height - 5), box_label, fill="black", font=font)
    else:
        # Single bounding box
        # Apply area adjustment
        x1, y1, x2, y2 = bbox
        if area_adjust != 0:
            x1 = max(0, x1 - area_adjust)
            y1 = max(0, y1 - area_adjust)
            x2 = min(image.width, x2 + area_adjust)
            y2 = min(image.height, y2 + area_adjust)
        
        # Apply the selected effect
        if effect_type == "mosaic":
            # Apply mosaic effect to the area
            apply_mosaic(image, x1, y1, x2, y2, mosaic_size)
        elif effect_type == "zoom":
            # Apply zoom effect to the area
            apply_zoom(image, x1, y1, x2, y2, zoom_ratio)
        
        # Draw bounding box outline and label if requested
        if show_bbox_outline:
            # Draw the bounding box outline
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            
            # Add label text if provided
            if label:
                # Use label as is without adding any index
                box_label = label
                
                # Background for text
                text_width, text_height = draw.textbbox((0, 0), box_label, font=font)[2:]
                draw.rectangle((x1, y1 - text_height - 5, x1 + text_width, y1), fill=color)
                # Draw text
                draw.text((x1, y1 - text_height - 5), box_label, fill="black", font=font)
        elif effect_type == "bbox":
            # Always draw the bounding box and label for "bbox" effect, regardless of show_bbox_outline
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            
            # Add label text if provided
            if label:
                # Use label as is without adding any index
                box_label = label
                
                # Background for text
                text_width, text_height = draw.textbbox((0, 0), box_label, font=font)[2:]
                draw.rectangle((x1, y1 - text_height - 5, x1 + text_width, y1), fill=color)
                # Draw text
                draw.text((x1, y1 - text_height - 5), box_label, fill="black", font=font)
    
    return image

def extract_bbox_answer(content):
    """
    Extract bounding box coordinates from model output content.
    The function tries to extract coordinates in multiple formats and validates them.
    Enhanced to better detect multiple bounding boxes in various response formats.
    
    Args:
        content: String containing potential bounding box information
        
    Returns:
        List of bounding box coordinates
    """
    # List to store all found bounding boxes
    all_bboxes = []
    
    # Try to find JSON formatted bounding boxes first
    try:
        # Look for array of arrays pattern [[x,y,x,y],[x,y,x,y]]
        bbox_pattern = r'\[\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\](?:\s*,\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\])*\s*\]'
        match = re.search(bbox_pattern, content)
        
        if match:
            # Use json to parse the matched string safely
            bbox_json = match.group(0)
            bboxes = json.loads(bbox_json)
            # Validate structure
            if bboxes and all(len(box) == 4 for box in bboxes):
                all_bboxes.extend(bboxes)
                print(f"Found bounding boxes via JSON pattern: {bboxes}")
    except Exception as e:
        print(f"Error parsing multiple bboxes: {e}")
    
    # Try to find JSON objects with bbox fields
    try:
        # Look for pattern {"bbox": [x,y,x,y]}
        object_pattern = r'{\s*"bbox"\s*:\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]\s*}'
        for match in re.finditer(object_pattern, content):
            try:
                obj = json.loads(match.group(0))
                if 'bbox' in obj and len(obj['bbox']) == 4:
                    all_bboxes.append(obj['bbox'])
                    print(f"Found bounding box via JSON object: {obj['bbox']}")
            except Exception as e:
                print(f"Error parsing JSON object bbox: {e}")
    except Exception as e:
        print(f"Error in JSON object pattern matching: {e}")
    
    # Also try to find individual bounding boxes (even if we found some via JSON)
    try:
        # Match all individual bounding box patterns: [x,y,x,y]
        bbox_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
        bbox_matches = re.finditer(bbox_pattern, content)
        
        for bbox_match in bbox_matches:
            try:
                bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), 
                       int(bbox_match.group(3)), int(bbox_match.group(4))]
                # Validate coordinates (ensure x2 > x1 and y2 > y1)
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    all_bboxes.append(bbox)
                    print(f"Found individual bounding box: {bbox}")
            except Exception as e:
                print(f"Error parsing individual bbox: {e}")
    except Exception as e:
        print(f"Error in regex pattern matching: {e}")
    
    # Look for common patterns in the model's natural language responses
    # Find boxes described in text like "first box: [x, y, x, y]" or "bounding box for object 1: [x, y, x, y]"
    try:
        # Various patterns for object descriptions followed by coordinates
        patterns = [
            r'(?:first|1st|object\s*1|first\s*object)(?:[^[]*)\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]',
            r'(?:second|2nd|object\s*2|second\s*object)(?:[^[]*)\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]',
            r'(?:third|3rd|object\s*3|third\s*object)(?:[^[]*)\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]',
            r'(?:fourth|4th|object\s*4|fourth\s*object)(?:[^[]*)\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]',
            r'(?:fifth|5th|object\s*5|fifth\s*object)(?:[^[]*)\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]',
            r'bounding\s*box\s*(?:for|of)?\s*(?:the)?\s*(\w+)(?:[^[]*)\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]'
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                try:
                    if len(match.groups()) == 4:
                        bbox = [int(match.group(1)), int(match.group(2)), 
                               int(match.group(3)), int(match.group(4))]
                    elif len(match.groups()) == 5:  # For the last pattern with object name
                        bbox = [int(match.group(2)), int(match.group(3)), 
                               int(match.group(4)), int(match.group(5))]
                    else:
                        continue
                        
                    # Validate coordinates
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        all_bboxes.append(bbox)
                        print(f"Found bounding box from text description: {bbox}")
                except Exception as e:
                    print(f"Error parsing text description bbox: {e}")
    except Exception as e:
        print(f"Error in text description pattern matching: {e}")
    
    # Look for table-like format often used in model responses
    try:
        # Pattern for table rows like "Object 1: [x, y, x, y]"
        table_pattern = r'(?:object|item|bbox)\s*\d+\s*:?\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
        for match in re.finditer(table_pattern, content, re.IGNORECASE):
            try:
                bbox = [int(match.group(1)), int(match.group(2)), 
                       int(match.group(3)), int(match.group(4))]
                # Validate coordinates
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    all_bboxes.append(bbox)
                    print(f"Found bounding box from table format: {bbox}")
            except Exception as e:
                print(f"Error parsing table format bbox: {e}")
    except Exception as e:
        print(f"Error in table format pattern matching: {e}")
    
    # Deduplicate bounding boxes (in case they were found by multiple methods)
    unique_bboxes = []
    for bbox in all_bboxes:
        is_duplicate = False
        for existing_bbox in unique_bboxes:
            # Check if boxes are very similar (allowing for small differences due to formatting/parsing)
            if (abs(bbox[0] - existing_bbox[0]) < 5 and 
                abs(bbox[1] - existing_bbox[1]) < 5 and
                abs(bbox[2] - existing_bbox[2]) < 5 and
                abs(bbox[3] - existing_bbox[3]) < 5):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_bboxes.append(bbox)
    
    # If we found any valid bounding boxes, return them
    if unique_bboxes:
        print(f"Returning {len(unique_bboxes)} unique bounding boxes")
        return unique_bboxes
    
    # Fallback to a small invisible box if no valid boxes were found
    print("No valid bounding boxes found, returning default box")
    return [[0, 0, 1, 1]]

import spaces

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_video_info(video_path):
    """
    Get video information using ffprobe
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video information
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-select_streams", "v:0",
        "-of", "json",
        video_path
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Error getting video info: {result.stderr}")
    
    info = json.loads(result.stdout)
    stream = info['streams'][0]
    
    # Parse frame rate fraction (e.g., "30000/1001" -> 29.97)
    frame_rate_str = stream.get('r_frame_rate', '25/1')
    if '/' in frame_rate_str:
        num, den = map(int, frame_rate_str.split('/'))
        frame_rate = num / den
    else:
        frame_rate = float(frame_rate_str)
    
    return {
        'width': int(stream.get('width', 1280)),
        'height': int(stream.get('height', 720)),
        'frame_rate': frame_rate,
        'duration': float(stream.get('duration', 0))
    }

@spaces.GPU
def process_image_and_text(image, text, use_thinking=False, effect_type="bbox", mosaic_size=15, 
                          zoom_ratio=1.5, area_adjust=0, show_bbox_outline=True):
    """
    Process image and text input, return thinking process and bbox
    
    Args:
        image: PIL Image to process
        text: Text description of what to locate
        use_thinking: Whether to include the thinking process in the output
        effect_type: Type of effect to apply - "bbox", "mosaic", or "zoom"
        mosaic_size: Size of mosaic blocks when using mosaic effect
        zoom_ratio: Zoom ratio when using zoom effect
        area_adjust: Pixels to adjust the bounding box size by (positive to expand, negative to contract)
        show_bbox_outline: Whether to show the bounding box outline (True for bbox, optional for effects)
        
    Returns:
        Tuple of (thinking_process, processed_image, bbox_coordinates)
    """
    # Enhance prompt to explicitly request ALL matching objects
    question = f"Please find ALL instances and provide ALL the bounding box coordinates of ALL objects/regions that match this description: {text}. If multiple objects match, provide coordinates for EACH of them separately."
    
    # Create different templates based on whether thinking process is needed
    if use_thinking:
        # Template with thinking process included
        QUESTION_TEMPLATE = """{Question} 

1. Remember, first you must Only analyze the objects entered by the user in <think> </think> tags
2. Final answer in <answer> </answer> tags

Inside the answer tags, provide coordinates in this format:
- For a single object: [xmin, ymin, xmax, ymax]
- For multiple objects: [[x1min, y1min, x1max, y1max], [x2min, y2min, x2max, y2max], ...]
- You can also use a table format listing each object:
  Object 1: [x1min, y1min, x1max, y1max]
  Object 2: [x2min, y2min, x2max, y2max]
"""
    else:
        # Simplified template without thinking process
        QUESTION_TEMPLATE = """{Question} 

Provide coordinates in <answer> </answer> tags using this format:
- For a single object: [xmin, ymin, xmax, ymax]
- For multiple objects: [[x1min, y1min, x1max, y1max], [x2min, y2min, x2max, y2max], ...]
- You can also use a table format listing each object:
  Object 1: [x1min, y1min, x1max, y1max]
  Object 2: [x2min, y2min, x2max, y2max]
"""
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=question)},
            ],
        }
    ]
    
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text_prompt],
        images=image,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    )

    inputs = inputs.to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(inputs.input_ids[0]):] for out_ids in generated_ids
        ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True
    )[0]
    print("output_text: ", output_text)

    # Extract thinking process if requested
    if use_thinking:
        think_match = re.search(r'<think>(.*?)</think>', output_text, re.DOTALL)
        thinking_process = think_match.group(1).strip() if think_match else "No thinking process found"
    else:
        thinking_process = ""
    
    # Get bbox and draw
    bboxes = extract_bbox_answer(output_text)
    
    # Draw bbox on the image with the input text as label
    result_image = image.copy()
    result_image = draw_bbox(result_image, bboxes, label=text, 
                            effect_type=effect_type, mosaic_size=mosaic_size, zoom_ratio=zoom_ratio,
                            area_adjust=area_adjust, show_bbox_outline=show_bbox_outline)
    
    # Clear CUDA cache to free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return thinking_process, result_image, bboxes

def resize_image_if_needed(image, max_size=1024):
    """Resize image if it's too large to save memory"""
    width, height = image.size
    
    # If the image is already small enough, return it as is
    if width <= max_size and height <= max_size:
        return image
    
    # Calculate the new dimensions while maintaining aspect ratio
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    # Ensure dimensions are even (required for some video codecs like h264)
    new_width = new_width - (new_width % 2)
    new_width = new_width - (new_width % 2)
    new_height = new_height - (new_height % 2)
    
    # Resize the image
    return image.resize((new_width, new_height), Image.LANCZOS)

def extract_frames_from_video(video_path, fps=None, max_frames=None, resize_to=None, start_time=None, end_time=None):
    """
    Extract frames from a video using ffmpeg
    
    Args:
        video_path: Path to the video file
        fps: Frames per second to extract. If None, extracts all frames
        max_frames: Maximum number of frames to extract
        resize_to: Resize frames to this size (width, height) to save memory
        start_time: Start time in seconds to begin extraction (optional)
        end_time: End time in seconds to stop extraction (optional)
        
    Returns:
        List of PIL Image frames
    """
    video_info = get_video_info(video_path)
    width = video_info['width']
    height = video_info['height']
    
    if fps is None:
        # Use original video fps
        fps = video_info['frame_rate']
    
    # Setup ffmpeg process with resizing if needed
    ffmpeg_input = ffmpeg.input(video_path)
    
    # Apply resizing if specified
    if resize_to:
        # Ensure dimensions are even (required for h264)
        resize_to = (
            resize_to[0] - (resize_to[0] % 2),
            resize_to[1] - (resize_to[1] % 2)
        )
        ffmpeg_input = ffmpeg_input.filter('scale', resize_to[0], resize_to[1])
        width, height = resize_to
    
    # Extract frames
    out, _ = (
        ffmpeg_input
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', r=fps)
        .run(capture_stdout=True, quiet=True)
    )
    
    # Convert the raw bytes to numpy arrays then to PIL Images
    buffer_size = width * height * 3  # 3 for RGB channels
    frames = []
    
    for i in range(0, len(out), buffer_size):
        if i + buffer_size <= len(out):
            frame_bytes = out[i:i+buffer_size]
            frame_array = np.frombuffer(frame_bytes, np.uint8).reshape([height, width, 3])
            frame_image = Image.fromarray(frame_array)
            frames.append(frame_image)
            
            # Stop if we've reached the maximum number of frames
            if max_frames and len(frames) >= max_frames:
                break
    
    return frames

def process_video_and_text(video_path, text, output_path=None, test_mode=False, test_duration=3, 
                          start_time=None, end_time=None, max_resolution=800, fps_reduction_factor=2, 
                          save_to_results=True, use_thinking=False, effect_type="bbox", 
                          mosaic_size=15, zoom_ratio=1.5, area_adjust=0, show_bbox_outline=True,
                          processing_interval=20):
    """
    Process video frames with the model and save the result.
    Uses DeepSORT for multi-object tracking to maintain consistent object identities across frames.
    
    Args:
        video_path: Path to the input video
        text: Text description to locate in the video
        output_path: Path to save the output video. If None, a temporary file is created
        test_mode: If True, only process a specific duration for testing
        test_duration: Duration in seconds to process (for testing purposes)
        start_time: Start time in seconds to begin processing (optional)
        end_time: End time in seconds to stop processing (optional)
        max_resolution: Maximum width/height to resize frames to save memory
        fps_reduction_factor: Factor to reduce the fps by (to process fewer frames)
        save_to_results: If True, also save a copy to the results folder
        use_thinking: If True, include the thinking process in the output
        effect_type: Type of effect to apply - "bbox", "mosaic", or "zoom"
        mosaic_size: Size of mosaic blocks when using mosaic effect
        zoom_ratio: Zoom ratio when using zoom effect
        area_adjust: Pixels to adjust the bounding box size by (positive to expand, negative to contract)
        show_bbox_outline: Whether to show the bounding box outline (True for bbox, optional for effects)
        processing_interval: Number of frames between full VLM model processing (higher values = faster but less accurate)
        
    Returns:
        Path to the output video and thinking process
    """
    print("Starting video processing with DeepSORT multi-object tracking...")
    
    # Create temporary directory for frames if no output path is specified
    if output_path is None:
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "output_video.mp4")
    
    # Get video information
    video_info = get_video_info(video_path)
    original_width = video_info['width']
    original_height = video_info['height']
    original_fps = video_info['frame_rate']
    total_duration = video_info['duration']
    
    print(f"Video total duration: {total_duration:.2f} seconds")
    
    # Validate start_time and end_time if provided
    if start_time is not None:
        start_time = float(start_time)
        if start_time < 0:
            start_time = 0
        if start_time >= total_duration:
            start_time = 0
    
    if end_time is not None:
        end_time = float(end_time)
        if end_time <= 0 or end_time > total_duration:
            end_time = total_duration
    
    # If start_time > end_time, swap them
    if start_time is not None and end_time is not None and start_time > end_time:
        start_time, end_time = end_time, start_time
    
    # Calculate processing duration based on test_mode or time range
    if test_mode:
        if start_time is not None and end_time is not None:
            # In test mode, if we have a time range, limit test_duration to that range
            processing_duration = min(test_duration, end_time - start_time)
            # Adjust end_time to match test_duration if needed
            if processing_duration < (end_time - start_time):
                end_time = start_time + processing_duration
        else:
            # In test mode with no time range, just use test_duration
            processing_duration = test_duration
            start_time = 0
            end_time = min(test_duration, total_duration)
    else:
        # Not in test mode, process full time range if specified
        if start_time is not None and end_time is not None:
            processing_duration = end_time - start_time
        else:
            # No time range specified, process the entire video
            processing_duration = total_duration
            start_time = 0
            end_time = total_duration
    
    print(f"Processing video from {start_time:.2f}s to {end_time:.2f}s (duration: {processing_duration:.2f}s)")
    
    # Reduce fps to process fewer frames
    target_fps = original_fps / fps_reduction_factor
    
    # Calculate resize dimensions to save memory while maintaining aspect ratio
    if max(original_width, original_height) > max_resolution:
        if original_width > original_height:
            resize_width = max_resolution
            resize_height = int(original_height * (max_resolution / original_width))
        else:
            resize_height = max_resolution
            resize_width = int(original_width * (max_resolution / original_height))
        
        # Ensure dimensions are even (required for h264)
        resize_width = resize_width - (resize_width % 2)
        resize_height = resize_height - (resize_height % 2)
        
        resize_dims = (resize_width, resize_height)
    else:
        # If not resizing, still ensure dimensions are even
        resize_width = original_width - (original_width % 2)
        resize_height = original_height - (original_height % 2)
        
        if resize_width != original_width or resize_height != original_height:
            resize_dims = (resize_width, resize_height)
        else:
            resize_dims = None
    
    print(f"Original resolution: {original_width}x{original_height}, FPS: {original_fps}")
    if resize_dims:
        print(f"Processing at adjusted resolution: {resize_dims[0]}x{resize_dims[1]} (made even for codec compatibility)")
    print(f"Processing at reduced FPS: {target_fps:.2f} (from {original_fps:.2f})")
    
    # For testing, we'll process just a few seconds
    max_frames = int(target_fps * processing_duration)
    
    # Initialize DeepSORT tracker
    tracker = DeepSORTTracker(max_age=10)  # Increase max_age for longer track retention
    print("Initialized DeepSORT tracker for multi-object tracking")
    
    # Create temporary directory for processed frames
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract frames for the specified time range at reduced resolution if needed
        frames = extract_frames_from_video(
            video_path, 
            fps=target_fps,
            max_frames=max_frames,
            resize_to=resize_dims,
            start_time=start_time,
            end_time=end_time
        )
        
        # Process each frame
        processed_frames_paths = []
        thinking_process = ""
        last_bboxes = []  # Store last detected bboxes for frames where detection fails
        tracked_objects = {}  # Dictionary to store tracked objects {track_id: (bbox, class)}
        
        for i, frame in enumerate(frames):
            print(f"Processing frame {i+1}/{len(frames)}")
            
            # Explicitly clean up memory before processing each frame
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # For frame 0, always do full processing to detect objects
            # For other frames, only do full processing every N frames to save processing time
            if i == 0 or i % processing_interval == 0 or not tracked_objects:
                # Full processing with model
                frame_thinking, processed_frame, bboxes = process_image_and_text(
                    frame, text, use_thinking, effect_type, mosaic_size, zoom_ratio,
                    area_adjust=area_adjust, show_bbox_outline=show_bbox_outline
                )
                
                # If first frame or we got valid bboxes, update our tracker
                if i == 0:
                    thinking_process = frame_thinking
                
                if bboxes != [[0, 0, 1, 1]]:  # We have valid detections
                    last_bboxes = bboxes
                    
                    try:
                        # Format detections for DeepSORT
                        detections = []
                        for box in bboxes:
                            # Convert normalized coordinates to absolute for DeepSORT
                            x1, y1, x2, y2 = box
                            
                            # Check if coordinates are already in absolute pixels or normalized (0-1)
                            if max(x1, y1, x2, y2) <= 1.0:
                                # Convert normalized to absolute
                                x1 = int(x1 * frame.width)
                                y1 = int(y1 * frame.height)
                                x2 = int(x2 * frame.width)
                                y2 = int(y2 * frame.height)
                            
                            detections.append(([x1, y1, x2, y2], text))
                        
                        # Update tracker with new detections
                        tracked_results = tracker.update(np.array(frame), detections)
                        
                        # Update tracked objects dictionary
                        tracked_objects = {}
                        for box, class_name, track_id in tracked_results:
                            tracked_objects[track_id] = (box, class_name)
                    except Exception as e:
                        print(f"Error in DeepSORT tracking after detection: {e}")
                
                # Draw tracked objects
                processed_frame = frame.copy()
                if tracked_objects:
                    for track_id, (box, class_name) in tracked_objects.items():
                        # Box is already in normalized format from DeepSORTTracker.update
                        # Convert normalized (0-1) to absolute pixels for drawing
                        abs_box = [
                            int(box[0] * frame.width), 
                            int(box[1] * frame.height),
                            int(box[2] * frame.width), 
                            int(box[3] * frame.height)
                        ]
                        # Use only the class name for consistent appearance with VLM detection
                        box_label = f"{class_name}"
                        processed_frame = draw_bbox(processed_frame, abs_box, label=box_label,
                                                   effect_type=effect_type, mosaic_size=mosaic_size, zoom_ratio=zoom_ratio,
                                                   area_adjust=area_adjust, show_bbox_outline=show_bbox_outline)
                elif last_bboxes:  # Fallback if tracking failed but we have previous detections
                    processed_frame = draw_bbox(processed_frame, last_bboxes, label=text,
                                               effect_type=effect_type, mosaic_size=mosaic_size, zoom_ratio=zoom_ratio,
                                               area_adjust=area_adjust, show_bbox_outline=show_bbox_outline)
            else:
                # Use tracking only for intermediate frames (no expensive model inference)
                try:
                    # Update tracker with existing objects (no new detections)
                    tracked_results = tracker.update(np.array(frame), [])
                    
                    # Update tracked objects dictionary
                    tracked_objects = {}
                    for box, class_name, track_id in tracked_results:
                        tracked_objects[track_id] = (box, class_name)
                    
                    # Draw tracked objects
                    processed_frame = frame.copy()
                    for track_id, (box, class_name) in tracked_objects.items():
                        # Box is already in normalized format from DeepSORTTracker.update
                        # Convert normalized (0-1) to absolute pixels for drawing
                        abs_box = [
                            int(box[0] * frame.width), 
                            int(box[1] * frame.height),
                            int(box[2] * frame.width), 
                            int(box[3] * frame.height)
                        ]
                        # Use only the class name for consistent appearance with VLM detection
                        box_label = f"{class_name}"
                        processed_frame = draw_bbox(processed_frame, abs_box, label=box_label,
                                                   effect_type=effect_type, mosaic_size=mosaic_size, zoom_ratio=zoom_ratio,
                                                   area_adjust=area_adjust, show_bbox_outline=show_bbox_outline)
                        
                    # If no objects are being tracked, fallback to last detected bboxes
                    if not tracked_objects and last_bboxes:
                        processed_frame = frame.copy()
                        processed_frame = draw_bbox(processed_frame, last_bboxes, label=text,
                                                   effect_type=effect_type, mosaic_size=mosaic_size, zoom_ratio=zoom_ratio,
                                                   area_adjust=area_adjust, show_bbox_outline=show_bbox_outline)
                except Exception as e:
                    print(f"Error in DeepSORT tracking for intermediate frame: {e}")
                    # Fallback to last bboxes if tracking fails
                    processed_frame = frame.copy()
                    if last_bboxes:
                        processed_frame = draw_bbox(processed_frame, last_bboxes, label=text,
                                                   effect_type=effect_type, mosaic_size=mosaic_size, zoom_ratio=zoom_ratio,
                                                   area_adjust=area_adjust, show_bbox_outline=show_bbox_outline)
            
            # Save the processed frame
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
            processed_frame.save(frame_path)
            processed_frames_paths.append(frame_path)
            
            # Free up frame memory after processing
            del frame
            del processed_frame
            
            # Force garbage collection after each frame
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # If we have no processed frames, return early
        if not processed_frames_paths:
            return None, thinking_process
        
        # Final output path that was actually created (in case we fall back to GIF or image)
        final_output_path = None
        
        # Try creating an MP4 video first
        try:
            # Create video from processed frames with explicit codec settings
            frame_pattern = os.path.join(temp_dir, "frame_%04d.jpg")
            
            # Run ffmpeg with more detailed error output
            process = (
                ffmpeg
                .input(frame_pattern, pattern_type='sequence', framerate=target_fps)
                .output(output_path, vcodec='libx264', pix_fmt='yuv420p', r=target_fps, crf=23)
                .overwrite_output()
            )
            
            print("Running ffmpeg with command:")
            print(process.compile())
            
            # Run with better error capturing
            try:
                stdout, stderr = process.run(capture_stdout=True, capture_stderr=True)
                print("Video created successfully!")
                final_output_path = output_path
            except ffmpeg.Error as e:
                print(f"ffmpeg stdout:\n{e.stdout.decode('utf8')}")
                print(f"ffmpeg stderr:\n{e.stderr.decode('utf8')}")
                raise
        except Exception as e:
            print(f"Error creating MP4 video: {e}")
            
            # Fallback: Create a GIF if MP4 fails
            try:
                print("Trying to create a GIF instead...")
                gif_path = os.path.splitext(output_path)[0] + ".gif"
                
                # Create a GIF using PIL
                images = []
                for img_path in processed_frames_paths:
                    img = Image.open(img_path)
                    images.append(img)
                
                # Save as GIF
                if images:
                    # Calculate duration based on FPS (in milliseconds)
                    duration = int(1000 / target_fps)
                    images[0].save(
                        gif_path,
                        save_all=True,
                        append_images=images[1:],
                        optimize=False,
                        duration=duration,
                        loop=0
                    )
                    print(f"Created GIF at {gif_path}")
                    final_output_path = gif_path
                else:
                    raise Exception("No images to create GIF")
            except Exception as gif_error:
                print(f"Error creating GIF: {gif_error}")
                
                # Final fallback: Use the first frame if both video and GIF creation fail
                if processed_frames_paths:
                    final_output_path = processed_frames_paths[0]
                    print(f"Using first frame as fallback: {final_output_path}")
        
        # If save_to_results is True, copy the final result to the results folder
        if save_to_results and final_output_path:
            try:
                # Create results directory if it doesn't exist
                results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
                os.makedirs(results_dir, exist_ok=True)
                
                # Create a filename based on timestamp
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                
                # Add sanitized text to the filename (limit to 50 chars and replace problematic chars)
                sanitized_text = "".join(c if c.isalnum() else "_" for c in text)
                sanitized_text = sanitized_text[:50]
                
                # Create the result path
                result_path = os.path.join(results_dir, f"result_{sanitized_text}_{timestamp}{os.path.splitext(output_path)[1]}")
                
                # Copy the file
                shutil.copy2(final_output_path, result_path)
                print(f"Saved result to: {result_path}")
            except Exception as save_error:
                print(f"Error saving to results folder: {save_error}")
        
        return final_output_path, thinking_process if use_thinking else ""

def test_video_processing(video_path, text, test_mode=False, test_duration=3, 
                         start_time=None, end_time=None, save_to_results=True, 
                         use_thinking=False, effect_type="bbox", mosaic_size=15, zoom_ratio=1.5,
                         area_adjust=0, show_bbox_outline=True, processing_interval=20):
    """
    Test function to process a video for specified duration
    
    Args:
        video_path: Path to the input video
        text: Text description to locate in the video
        test_mode: If True, only process the test_duration
        test_duration: Duration in seconds to process
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
        save_to_results: If True, save a copy to the results folder
        use_thinking: If True, include the thinking process in the output
        effect_type: Type of effect to apply - "bbox", "mosaic", or "zoom"
        mosaic_size: Size of mosaic blocks when using mosaic effect
        zoom_ratio: Zoom ratio when using zoom effect
        area_adjust: Pixels to adjust the bounding box size by (positive to expand, negative to contract)
        show_bbox_outline: Whether to show the bounding box outline (True for bbox, optional for effects)
        processing_interval: Number of frames between full VLM model processing (higher values = faster but less accurate)
        
    Returns:
        Path to the output video, processing time, thinking process, and is_image
    """
    # Generate stable output filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    basename = os.path.basename(video_path)
    name, ext = os.path.splitext(basename)
    
    # Create results directory if not exists
    if save_to_results:
        os.makedirs("results", exist_ok=True)
    
    # Create stable output path in results folder if save_to_results=True
    # Otherwise just create a temporary output path
    stable_output_path = f"results/result_{text}_{timestamp}{ext}" if save_to_results else None
    
    # Measure processing time
    start_time_processing = time.time()
    
    # Process the video with memory optimization settings
    try:
        output_path, thinking_process = process_video_and_text(
            video_path, 
            text, 
            output_path=stable_output_path,
            test_mode=test_mode,
            test_duration=test_duration,
            start_time=start_time,
            end_time=end_time,
            max_resolution=800,  # Resize large frames to max 800px dimension
            fps_reduction_factor=2,  # Process half the frames
            save_to_results=False,  # Set to False to avoid saving the video twice
            use_thinking=use_thinking,
            effect_type=effect_type,
            mosaic_size=mosaic_size,
            zoom_ratio=zoom_ratio,
            area_adjust=area_adjust,
            show_bbox_outline=show_bbox_outline,
            processing_interval=processing_interval
        )
        
        end_time_processing = time.time()
        processing_time = end_time_processing - start_time_processing
        
        # Determine if output is image or video based on file extension
        is_image = output_path and (output_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
        
        return output_path, processing_time, thinking_process, is_image
    except Exception as e:
        end_time_processing = time.time()
        processing_time = end_time_processing - start_time_processing
        print(f"Error processing video: {str(e)}")
        traceback.print_exc()
        raise

def get_video_duration(video_path):
    """
    Get the duration of a video and format it as a string
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Formatted duration string and duration in seconds
    """
    try:
        video_info = get_video_info(video_path)
        duration_seconds = video_info['duration']
        
        # Format as HH:MM:SS
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        
        if hours > 0:
            formatted_duration = f"{hours:02d}:{minutes:02d}:{seconds:02d} ({duration_seconds:.2f}s)"
        else:
            formatted_duration = f"{minutes:02d}:{seconds:02d} ({duration_seconds:.2f}s)"
            
        return formatted_duration, duration_seconds
    except Exception as e:
        print(f"Error getting video duration: {e}")
        return "Unknown", 0

def apply_mosaic(image, x1, y1, x2, y2, block_size=15):
    """
    Apply a mosaic effect to a specified area of an image
    
    Args:
        image: PIL Image to modify
        x1, y1, x2, y2: Bounding box coordinates of the area to apply mosaic
        block_size: Size of the mosaic blocks
        
    Returns:
        None (modifies the image in-place)
    """
    # Make sure coordinates are integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Ensure coordinates are within image bounds
    width, height = image.size
    x1 = max(0, min(x1, width-1))
    x2 = max(0, min(x2, width-1))
    y1 = max(0, min(y1, height-1))
    y2 = max(0, min(y2, height-1))
    
    # Extract the region
    region = image.crop((x1, y1, x2, y2))
    
    # Calculate new dimensions that are divisible by block_size
    new_width = ((x2 - x1) // block_size) * block_size
    new_height = ((y2 - y1) // block_size) * block_size
    
    # If the new dimensions are too small, use the original dimensions
    if new_width < block_size or new_height < block_size:
        new_width = max(block_size, x2 - x1)
        new_height = max(block_size, y2 - y1)
    
    # Resize the region to reduced size
    small = region.resize((new_width // block_size, new_height // block_size), Image.NEAREST)
    
    # Scale it back up
    mosaic = small.resize((new_width, new_height), Image.NEAREST)
    
    # Paste back with the original size
    if new_width != (x2 - x1) or new_height != (y2 - y1):
        mosaic = mosaic.resize((x2 - x1, y2 - y1), Image.NEAREST)
    
    # Paste the mosaic region back into the original image
    image.paste(mosaic, (x1, y1))

def apply_zoom(image, x1, y1, x2, y2, zoom_ratio=1.5):
    """
    Apply a zoom effect to a specified area of an image by enlarging it without loss
    
    Args:
        image: PIL Image to modify
        x1, y1, x2, y2: Bounding box coordinates of the area to apply zoom
        zoom_ratio: How much to enlarge the area (e.g., 1.5 = 50% larger)
        
    Returns:
        None (modifies the image in-place)
    """
    # Make sure coordinates are integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Ensure coordinates are within image bounds
    width, height = image.size
    x1 = max(0, min(x1, width-1))
    x2 = max(0, min(x2, width-1))
    y1 = max(0, min(y1, height-1))
    y2 = max(0, min(y2, height-1))
    
    # Calculate region dimensions
    region_width = x2 - x1
    region_height = y2 - y1
    
    if region_width <= 0 or region_height <= 0:
        return  # Nothing to zoom
    
    # Extract the region
    region = image.crop((x1, y1, x2, y2))
    
    # Calculate the new dimensions for the enlarged region
    new_width = int(region_width * zoom_ratio)
    new_height = int(region_height * zoom_ratio)
    
    # Resize the region to be larger
    enlarged = region.resize((new_width, new_height), Image.LANCZOS)
    
    # Calculate where to place the enlarged region in the original image
    # Center it on the original region's center
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Calculate the top-left corner for pasting the enlarged image
    paste_x = center_x - new_width // 2
    paste_y = center_y - new_height // 2
    
    # Create a new canvas with the same size as the original image
    result = image.copy()
    
    # Paste the enlarged region onto the canvas
    # We need to handle the case where the enlarged region extends beyond the image boundaries
    # First determine what portion of the enlarged image will be visible
    visible_x1 = max(0, paste_x)
    visible_y1 = max(0, paste_y)
    visible_x2 = min(width, paste_x + new_width)
    visible_y2 = min(height, paste_y + new_height)
    
    # Calculate the corresponding region within the enlarged image
    enlarged_x1 = visible_x1 - paste_x if paste_x < 0 else 0
    enlarged_y1 = visible_y1 - paste_y if paste_y < 0 else 0
    enlarged_x2 = enlarged_x1 + (visible_x2 - visible_x1)
    enlarged_y2 = enlarged_y1 + (visible_y2 - visible_y1)
    
    # Extract the visible portion of the enlarged image
    visible_enlarged = enlarged.crop((enlarged_x1, enlarged_y1, enlarged_x2, enlarged_y2))
    
    # Paste this visible portion onto the original image at the correct position
    result.paste(visible_enlarged, (visible_x1, visible_y1))
    
    # Update the original image
    image.paste(result, (0, 0))

def gradio_interface(input_media, text, is_video=False, test_mode=False, test_duration=3, 
                    start_time=None, end_time=None, save_to_results=True, use_thinking=False,
                    effect_type="bbox", mosaic_size=15, zoom_ratio=1.5, area_adjust=0, 
                    show_bbox_outline=True, processing_interval=20):
    """
    Gradio interface for both image and video processing
    
    Args:
        input_media: Input image or video
        text: Description text to locate
        is_video: If True, process as video, otherwise as image
        test_mode: If True, only process the test_duration (for video)
        test_duration: Duration in seconds to process (for video)
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
        save_to_results: If True, save a copy to the results folder
        use_thinking: If True, include the thinking process in the output
        effect_type: Type of effect to apply - "bbox", "mosaic", or "zoom"
        mosaic_size: Size of mosaic blocks when using mosaic effect
        zoom_ratio: Zoom ratio when using zoom effect
        area_adjust: Pixels to adjust the bounding box size by (positive to expand, negative to contract)
        show_bbox_outline: Whether to show the bounding box outline (True for bbox, optional for effects)
        processing_interval: Number of frames between full VLM model processing (higher values = faster but less accurate)
        
    Returns:
        For image: thinking, status, output_image
        For video: thinking, output_video, status, fallback_image
    """
    if is_video:
        # Process as video
        # Save the uploaded video to a temporary file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.close()
        
        # Check if input_media is a string (file path) or an object with save method
        if isinstance(input_media, str):
            # Copy the file if it's a path
            shutil.copyfile(input_media, temp_video.name)
        else:
            # Save the uploaded video to the temporary file
            input_media.save(temp_video.name)
        
        # Process the video
        try:
            output_path, processing_time, thinking, is_image = test_video_processing(
                temp_video.name, text, test_mode, test_duration, start_time, end_time, 
                save_to_results, use_thinking, effect_type, mosaic_size, zoom_ratio,
                area_adjust=area_adjust, show_bbox_outline=show_bbox_outline,
                processing_interval=processing_interval
            )
            
            # Clean up temp file
            try:
                os.unlink(temp_video.name)
            except:
                pass
            
            if is_image:
                # Output is just an image from a short video
                try:
                    return thinking, None, f"Processed successfully in {processing_time:.2f} seconds. Video is too short, showing single frame instead.", Image.open(output_path)
                except Exception as e:
                    return thinking, None, f"Error loading fallback image: {e}", None
            else:
                # Output is a video
                return thinking, output_path, f"Processed successfully in {processing_time:.2f} seconds", None
        except Exception as e:
            # Clean up temp file
            try:
                os.unlink(temp_video.name)
            except:
                pass
            
            error_message = f"Error processing video: {str(e)}"
            print(error_message)
            traceback.print_exc()
            return "", None, error_message, None
    else:
        # Process as image
        try:
            thinking, result_image, _ = process_image_and_text(
                input_media, text, use_thinking, effect_type, mosaic_size, zoom_ratio,
                area_adjust=area_adjust, show_bbox_outline=show_bbox_outline
            )
            
            # Save image to results folder if requested
            if save_to_results:
                try:
                    # Create results directory if it doesn't exist
                    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
                    os.makedirs(results_dir, exist_ok=True)
                    
                    # Create a filename based on timestamp
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    
                    # Add sanitized text to the filename (limit to 50 chars and replace problematic chars)
                    sanitized_text = "".join(c if c.isalnum() else "_" for c in text)
                    sanitized_text = sanitized_text[:50]
                    
                    # Create the result path and save the image
                    result_path = os.path.join(results_dir, f"result_{sanitized_text}_{timestamp}.png")
                    result_image.save(result_path)
                    print(f"Saved result image to: {result_path}")
                except Exception as save_error:
                    print(f"Error saving image to results folder: {save_error}")
            
            return thinking, "Processed successfully", result_image
        except Exception as e:
            error_message = f"Error processing image: {str(e)}"
            print(error_message)
            traceback.print_exc()
            return "", error_message, None

if __name__ == "__main__":
    import gradio as gr
    
    # Load pretrained model for image processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained("SZhanZ/Qwen2.5VL-VLM-R1-REC-step500", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    model.to(device)
    processor = AutoProcessor.from_pretrained("SZhanZ/Qwen2.5VL-VLM-R1-REC-step500")
    
    with gr.Blocks(title="객체 감지 gradio") as demo:
        with gr.Tabs():
            with gr.Tab("Image Processing"):
                # Image processing tab content
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="pil", label="Input Image")
                        image_text = gr.Textbox(label="Description Text")
                        
                        # Add detailed function settings in an accordion
                        with gr.Accordion("Detailed Function Settings", open=False):
                            image_effect_type = gr.Radio(
                                ["bbox", "mosaic", "zoom"], 
                                label="Effect Type", 
                                value="bbox",
                                info="Select the effect to apply to detected areas"
                            )
                            with gr.Row(visible=False) as image_mosaic_row:
                                image_mosaic_size = gr.Slider(
                                    minimum=5, maximum=50, value=15, step=1,
                                    label="Mosaic Block Size",
                                    info="Size of mosaic blocks (larger = more pixelated)"
                                )
                            with gr.Row(visible=False) as image_zoom_row:
                                image_zoom_ratio = gr.Slider(
                                    minimum=1.1, maximum=3.0, value=1.5, step=0.1,
                                    label="Zoom Ratio",
                                    info="How much to zoom in on the detected area"
                                )
                            image_area_adjust = gr.Slider(
                                minimum=-50, maximum=50, value=0, step=1,
                                label="Area Adjustment",
                                info="Pixels to adjust the bounding box size by (positive to expand, negative to contract)"
                            )
                            image_show_bbox_outline = gr.Checkbox(label="Show Bounding Box Outline", value=True)
                        
                        image_auto_save = gr.Checkbox(label="Save result to 'results' folder", value=True)
                        image_use_thinking = gr.Checkbox(label="Include thinking process in output", value=False)
                        image_submit = gr.Button("Process Image")
                    
                    with gr.Column():
                        image_thinking = gr.Textbox(label="Thinking Process")
                        image_status = gr.Textbox(label="Status")
                        image_output = gr.Image(type="pil", label="Result Image")

            with gr.Tab("Video Processing"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Input Video")
                        video_text = gr.Textbox(label="Description Text")
                        video_duration_info = gr.Textbox(label="Video Duration", visible=True)
                        
                        # Time range selection
                        with gr.Row():
                            video_start_time = gr.Number(label="Start Time (seconds)", value=0, precision=2)
                            video_end_time = gr.Number(label="End Time (seconds)", value=None, precision=2)
                        
                        # Add detailed function settings in an accordion
                        with gr.Accordion("Detailed Function Settings", open=False):
                            video_effect_type = gr.Radio(
                                ["bbox", "mosaic", "zoom"], 
                                label="Effect Type", 
                                value="bbox",
                                info="Select the effect to apply to detected areas"
                            )
                            with gr.Row(visible=False) as video_mosaic_row:
                                video_mosaic_size = gr.Slider(
                                    minimum=5, maximum=50, value=15, step=1,
                                    label="Mosaic Block Size",
                                    info="Size of mosaic blocks (larger = more pixelated)"
                                )
                            with gr.Row(visible=False) as video_zoom_row:
                                video_zoom_ratio = gr.Slider(
                                    minimum=1.1, maximum=3.0, value=1.5, step=0.1,
                                    label="Zoom Ratio",
                                    info="How much to zoom in on the detected area"
                                )
                            video_area_adjust = gr.Slider(
                                minimum=-50, maximum=50, value=0, step=1,
                                label="Area Adjustment",
                                info="Pixels to adjust the bounding box size by (positive to expand, negative to contract)"
                            )
                            video_show_bbox_outline = gr.Checkbox(label="Show Bounding Box Outline", value=True)
                            video_processing_interval = gr.Slider(
                                minimum=1, maximum=60, value=20, step=1,
                                label="Processing Interval",
                                info="Number of frames between full model processing. Higher values = faster processing but potentially less accurate tracking."
                            )
                        
                        # Test mode checkbox
                        test_mode_checkbox = gr.Checkbox(label="Enable Test Mode", value=True)
                        
                        # Test duration in accordion that is visible when test mode is enabled
                        with gr.Accordion("Test Duration Settings", open=True, visible=True) as test_duration_accordion:
                            video_duration = gr.Slider(minimum=1, maximum=30, value=3, step=1, label="Test Duration (seconds)")
                        
                        auto_save = gr.Checkbox(label="Save result to 'results' folder", value=True)
                        video_use_thinking = gr.Checkbox(label="Include thinking process in output", value=False)
                        video_submit = gr.Button("Process Video")
                    
                    with gr.Column():
                        video_thinking = gr.Textbox(label="Thinking Process", lines=15)
                        video_status = gr.Textbox(label="Status", lines=5)
                        
                        # Create a tabbed interface for different result types
                        with gr.Tabs():
                            with gr.TabItem("Video Result"):
                                video_output = gr.Video(label="Result Video")
                            with gr.TabItem("Image Result"):
                                fallback_image = gr.Image(type="pil", label="Fallback Result Image")

        # Function to toggle mosaic and zoom settings visibility based on effect type
        def toggle_effect_settings(effect_type, for_image=True):
            if for_image:
                if effect_type == "mosaic":
                    return gr.update(visible=True), gr.update(visible=False)
                elif effect_type == "zoom":
                    return gr.update(visible=False), gr.update(visible=True)
                else:  # bbox
                    return gr.update(visible=False), gr.update(visible=False)
            else:
                if effect_type == "mosaic":
                    return gr.update(visible=True), gr.update(visible=False)
                elif effect_type == "zoom":
                    return gr.update(visible=False), gr.update(visible=True)
                else:  # bbox
                    return gr.update(visible=False), gr.update(visible=False)
        
        # Connect the image interface
        image_effect_type.change(
            fn=lambda x: toggle_effect_settings(x, True),
            inputs=[image_effect_type],
            outputs=[image_mosaic_row, image_zoom_row]
        )
        
        image_submit.click(
            fn=lambda img, txt, effect, mosaic, zoom, save, use_thinking, area, outline: gradio_interface(
                img, txt, is_video=False, save_to_results=save, use_thinking=use_thinking,
                effect_type=effect, mosaic_size=mosaic, zoom_ratio=zoom, area_adjust=area,
                show_bbox_outline=outline
            ),
            inputs=[image_input, image_text, image_effect_type, image_mosaic_size, 
                   image_zoom_ratio, image_auto_save, image_use_thinking, image_area_adjust,
                   image_show_bbox_outline],
            outputs=[image_thinking, image_status, image_output]
        )
        
        # Connect the video interface
        video_effect_type.change(
            fn=lambda x: toggle_effect_settings(x, False),
            inputs=[video_effect_type],
            outputs=[video_mosaic_row, video_zoom_row]
        )
        
        # Function to update duration info when video is uploaded
        def update_duration_info(video):
            if video is None:
                return "No video uploaded", gr.update(value=None), gr.update(value=None)
            
            # Save video to temp file for processing
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_video.close()
            
            # If video is a string (path), copy it, otherwise save the uploaded video
            if isinstance(video, str):
                import shutil
                shutil.copyfile(video, temp_video.name)
            else:
                video.save(temp_video.name)
            
            formatted_duration, duration_seconds = get_video_duration(temp_video.name)
            
            # Clean up temp file
            try:
                os.unlink(temp_video.name)
            except:
                pass
                
            return f"Video Duration: {formatted_duration}", gr.update(value=0), gr.update(value=duration_seconds)
        
        # Update duration info and end time when video is uploaded
        video_input.change(
            fn=update_duration_info,
            inputs=[video_input],
            outputs=[video_duration_info, video_start_time, video_end_time]
        )
        
        # Function to toggle test duration accordion visibility
        def toggle_test_duration_accordion(test_mode):
            return gr.update(visible=test_mode)
        
        # Toggle test duration accordion when test mode checkbox is changed
        test_mode_checkbox.change(
            fn=toggle_test_duration_accordion,
            inputs=[test_mode_checkbox],
            outputs=[test_duration_accordion]
        )
        
        # Connect the video interface
        video_submit.click(
            fn=lambda vid, txt, test, duration, start, end, effect, mosaic, zoom, save, use_thinking, area, outline, interval: gradio_interface(
                vid, txt, is_video=True, test_mode=test, test_duration=duration,
                start_time=start, end_time=end, save_to_results=save, use_thinking=use_thinking,
                effect_type=effect, mosaic_size=mosaic, zoom_ratio=zoom, area_adjust=area,
                show_bbox_outline=outline, processing_interval=interval
            ),
            inputs=[video_input, video_text, test_mode_checkbox, video_duration, 
                   video_start_time, video_end_time, video_effect_type, video_mosaic_size,
                   video_zoom_ratio, auto_save, video_use_thinking, video_area_adjust,
                   video_show_bbox_outline, video_processing_interval],
            outputs=[video_thinking, video_output, video_status, fallback_image]
        )
    
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)