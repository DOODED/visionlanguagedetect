import re
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image, ImageDraw, ImageFont
import ffmpeg
import numpy as np
import tempfile
import os
import time
import json
import subprocess
import gc
from pathlib import Path
import shutil
from deep_sort_integration import DeepSORTTracker

def draw_bbox(image, bbox, label=None):
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
    
    # Generate a unique color for each bounding box if multiple boxes
    colors = ["orange", "red", "green", "blue", "purple", "cyan", "magenta", "yellow"]
    
    if isinstance(bbox[0], list) or isinstance(bbox[0], tuple):
        # Multiple bounding boxes
        for i, box in enumerate(bbox):
            # Select color by index (cycling through the colors list)
            color = colors[i % len(colors)]
            
            x1, y1, x2, y2 = box
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            
            # Add label text if provided
            if label:
                # Add index to label for multiple objects
                box_label = label
                if len(bbox) > 1:
                    # If tracking IDs are in the label, don't add another index
                    if "ID:" not in label and "#" not in label:
                        box_label = f"{label} #{i+1}"
                
                # Background for text
                text_width, text_height = draw.textbbox((0, 0), box_label, font=font)[2:]
                draw.rectangle((x1, y1 - text_height - 5, x1 + text_width, y1), fill=color)
                # Draw text
                draw.text((x1, y1 - text_height - 5), box_label, fill="black", font=font)
    else:
        # Single bounding box
        x1, y1, x2, y2 = bbox
        draw.rectangle((x1, y1, x2, y2), outline="orange", width=3)
        
        # Add label text if provided
        if label:
            # Background for text
            text_width, text_height = draw.textbbox((0, 0), label, font=font)[2:]
            draw.rectangle((x1, y1 - text_height - 5, x1 + text_width, y1), fill="orange")
            # Draw text
            draw.text((x1, y1 - text_height - 5), label, fill="black", font=font)
    
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
def process_image_and_text(image, text, use_thinking=False):
    """Process image and text input, return thinking process and bbox"""
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
    result_image = draw_bbox(result_image, bboxes, label=text)
    
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
    new_height = new_height - (new_height % 2)
    
    # Resize the image
    return image.resize((new_width, new_height), Image.LANCZOS)

def extract_frames_from_video(video_path, fps=None, max_frames=None, resize_to=None):
    """
    Extract frames from a video using ffmpeg
    
    Args:
        video_path: Path to the video file
        fps: Frames per second to extract. If None, extracts all frames
        max_frames: Maximum number of frames to extract
        resize_to: Resize frames to this size (width, height) to save memory
        
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

def process_video_and_text(video_path, text, output_path=None, test_duration=3, max_resolution=800, fps_reduction_factor=2, save_to_results=True, use_thinking=False):
    """
    Process video frames with the model and save the result.
    Uses DeepSORT for multi-object tracking to maintain consistent object identities across frames.
    
    Args:
        video_path: Path to the input video
        text: Text description to locate in the video
        output_path: Path to save the output video. If None, a temporary file is created
        test_duration: Duration in seconds to process (for testing purposes)
        max_resolution: Maximum width/height to resize frames to save memory
        fps_reduction_factor: Factor to reduce the fps by (to process fewer frames)
        save_to_results: If True, also save a copy to the results folder
        use_thinking: If True, include the thinking process in the output
        
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
    max_frames = int(target_fps * test_duration)
    
    # Initialize DeepSORT tracker
    tracker = DeepSORTTracker(max_age=10)  # Increase max_age for longer track retention
    print("Initialized DeepSORT tracker for multi-object tracking")
    
    # Create temporary directory for processed frames
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract frames for the test duration at reduced resolution if needed
        frames = extract_frames_from_video(
            video_path, 
            fps=target_fps,
            max_frames=max_frames,
            resize_to=resize_dims
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
            processing_interval = 10  # Process fully every 10 frames
            if i == 0 or i % processing_interval == 0 or not tracked_objects:
                # Full processing with model
                frame_thinking, processed_frame, bboxes = process_image_and_text(frame, text, use_thinking)
                
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
                        # Include track ID in the label for visualization with Korean text support
                        box_label = f"{class_name} #{track_id}"
                        processed_frame = draw_bbox(processed_frame, abs_box, label=box_label)
                elif last_bboxes:  # Fallback if tracking failed but we have previous detections
                    processed_frame = draw_bbox(processed_frame, last_bboxes, label=text)
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
                        # Include track ID in the label for visualization with Korean text support
                        box_label = f"{class_name} #{track_id}"
                        processed_frame = draw_bbox(processed_frame, abs_box, label=box_label)
                        
                    # If no objects are being tracked, fallback to last detected bboxes
                    if not tracked_objects and last_bboxes:
                        processed_frame = frame.copy()
                        processed_frame = draw_bbox(processed_frame, last_bboxes, label=text)
                except Exception as e:
                    print(f"Error in DeepSORT tracking for intermediate frame: {e}")
                    # Fallback to last bboxes if tracking fails
                    processed_frame = frame.copy()
                    if last_bboxes:
                        processed_frame = draw_bbox(processed_frame, last_bboxes, label=text)
            
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

def test_video_processing(video_path, text, test_duration=3, save_to_results=True, use_thinking=False):
    """
    Test function to process a video for specified duration
    
    Args:
        video_path: Path to the input video
        text: Text description to locate in the video
        test_duration: Duration in seconds to process
        save_to_results: If True, save a copy to the results folder
        use_thinking: If True, include the thinking process in the output
        
    Returns:
        Path to the output video, processing time, and thinking process
    """
    start_time = time.time()
    
    # Set environment variable to expandable_segments to avoid memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Create a stable output path that won't be deleted with the tempdir
    # This helps ensure the file remains accessible to Gradio
    output_dir = os.path.join(tempfile.gettempdir(), "video_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique filename based on timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    stable_output_path = os.path.join(output_dir, f"output_video_{timestamp}.mp4")
    
    # Process the video with memory optimization settings
    try:
        output_path, thinking_process = process_video_and_text(
            video_path, 
            text, 
            output_path=stable_output_path,
            test_duration=test_duration,
            max_resolution=800,  # Resize large frames to max 800px dimension
            fps_reduction_factor=2,  # Process half the frames
            save_to_results=save_to_results,
            use_thinking=use_thinking
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Determine if output is image or video based on file extension
        is_image = output_path and (output_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
        
        return output_path, processing_time, thinking_process, is_image
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Error in test_video_processing: {e}")
        import traceback
        traceback.print_exc()
        
        # Return error information
        return None, processing_time, f"Error processing video: {str(e)}", False

def gradio_interface(input_media, text, is_video=False, test_duration=3, save_to_results=True, use_thinking=False):
    if is_video:
        # Save uploaded video to a temporary file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_video.close()
        
        # If input_media is a string (path), copy it, otherwise save the uploaded video
        if isinstance(input_media, str):
            import shutil
            shutil.copyfile(input_media, temp_video.name)
        else:
            input_media.save(temp_video.name)
        
        # Process the video
        try:
            output_path, processing_time, thinking, is_image = test_video_processing(
                temp_video.name, text, test_duration, save_to_results, use_thinking
            )
            
            status_msg = f"Processing time: {processing_time:.2f} seconds"
            
            # Add save location info if saving to results
            if save_to_results and output_path:
                results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
                status_msg += f"\nResult saved to: {results_dir}"
            
            # Debug output file details
            if output_path:
                status_msg += f"\nOutput file: {output_path}"
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    status_msg += f"\nFile size: {file_size/1024:.2f} KB"
                else:
                    status_msg += "\nFile does not exist!"
            
            # If output is an image (fallback), display it differently
            if is_image or output_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                status_msg += " (Created a single frame instead of video due to encoding issues)"
                img = Image.open(output_path)
                return thinking if use_thinking else "", None, status_msg, img
            elif output_path.lower().endswith('.gif'):
                # For GIF files, we need to handle it slightly differently
                status_msg += " (Created a GIF instead of MP4 due to encoding issues)"
                
                # We'll need to return the GIF path in the result for Gradio to display
                # But we need to make sure it's accessible to the Gradio UI
                accessible_path = output_path
                return thinking if use_thinking else "", accessible_path, status_msg, None
            else:
                # Make a copy of the video file to a web-accessible location if needed
                if not os.path.exists(output_path):
                    status_msg += "\nError: Output file not found!"
                    return thinking if use_thinking else "", None, status_msg, None
                
                # Return the output path for display in Gradio
                return thinking if use_thinking else "", output_path, status_msg, None
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            return thinking if use_thinking else "", None, f"Error: {str(e)}\n{error_trace}", None
    else:
        # Process as image
        try:
            thinking, result_image, _ = process_image_and_text(input_media, text, use_thinking)
            
            # If save_to_results is True, save the image to the results folder
            if save_to_results and result_image:
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
                    result_path = os.path.join(results_dir, f"result_{sanitized_text}_{timestamp}.jpg")
                    
                    # Save the image
                    result_image.save(result_path)
                    status_msg = f"Result saved to: {result_path}"
                except Exception as save_error:
                    status_msg = f"Error saving to results folder: {str(save_error)}"
            else:
                status_msg = ""
            
            return thinking if use_thinking else "", status_msg, result_image
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            return f"Error processing image: {str(e)}", error_trace, None

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
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="pil", label="Input Image")
                        image_text = gr.Textbox(label="Description Text")
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
                        video_duration = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Test Duration (seconds)")
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

        # Connect the image interface
        image_submit.click(
            fn=lambda img, txt, save, use_thinking: gradio_interface(img, txt, is_video=False, save_to_results=save, use_thinking=use_thinking),
            inputs=[image_input, image_text, image_auto_save, image_use_thinking],
            outputs=[image_thinking, image_status, image_output]
        )
        
        # Connect the video interface
        video_submit.click(
            fn=lambda vid, txt, dur, save, use_thinking: gradio_interface(vid, txt, is_video=True, test_duration=dur, save_to_results=save, use_thinking=use_thinking),
            inputs=[video_input, video_text, video_duration, auto_save, video_use_thinking],
            outputs=[video_thinking, video_output, video_status, fallback_image]
        )
    
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)