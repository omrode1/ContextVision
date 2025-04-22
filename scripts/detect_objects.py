from ultralytics import YOLO
import cv2
import os
from collections import defaultdict, deque
from ollama import Client

# Load model
model = YOLO("yolov8n.pt")  # Use the Nano version




def run_detection(image_path, grid_width=10, grid_height=10, min_confidence=0.5):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return []
    height, width = img.shape[:2]  # Get image dimensions

    # Run detection
    results = model(img)

    # Parse results
    detections = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < min_confidence:
                continue  # Skip low-confidence detections
            cls = int(box.cls[0])
            coords = box.xyxy[0].tolist()

            # Calculate centroid
            centroid_x = (coords[0] + coords[2]) / 2
            centroid_y = (coords[1] + coords[3]) / 2

            # Map to grid
            grid_x = int((centroid_x / width) * grid_width)
            grid_y = int((centroid_y / height) * grid_height)

            # Clamp values to grid boundaries
            grid_x = max(0, min(grid_x, grid_width-1))
            grid_y = max(0, min(grid_y, grid_height-1))

            detections.append({
                "class": model.names[cls],
                "confidence": conf,
                "bbox": coords,
                "centroid": (centroid_x, centroid_y),
                "grid_position": (grid_x, grid_y)

            })

    return detections

def get_position_description(grid_x, grid_y, grid_width, grid_height):
    """Convert grid coordinates to positional descriptions"""
    # Horizontal positioning
    if grid_x < grid_width // 3:
        horizontal = "left"
    elif grid_x < 2 * (grid_width // 3):
        horizontal = "center"
    else:
        horizontal = "right"

    # Vertical positioning
    if grid_y < grid_height // 3:
        vertical = "top"
    elif grid_y < 2 * (grid_height // 3):
        vertical = "middle"
    else:
        vertical = "bottom"

    # Combine for description (e.g., "top left", "center", "bottom right")
    if horizontal == "center" and vertical == "middle":
        return "center"
    elif horizontal == "center":
        return vertical
    elif vertical == "middle":
        return horizontal
    else:
        return f"{vertical} {horizontal}"


def describe_image(detections, grid_width, grid_height):
    print("Attempting to generate description...")
    client = Client(host='http://localhost:11434')

    # Format detections into natural language prompts
    object_descriptions = []
    for det in detections:
        position = get_position_description(
            det["grid_position"][0],
            det["grid_position"][1],
            grid_width,
            grid_height
        )
        count_str = f"{det.get('count', 1)} " if det.get('count', 1) > 1 else ""
        object_descriptions.append(f"{count_str}{det['class']} in the {position}")

    if not object_descriptions:
        print("No objects detected, skipping description generation.")
        return "No objects detected in the scene."

    # Create structured prompt
    prompt = f"""Visual description of scene composition:
    Objects: {', '.join(object_descriptions)}

    Instructions:
    1. Start with the most prominent object or group.
    2. Describe spatial layout using natural terms (e.g., 'top left', 'center').
    3. Note significant object groupings and empty spaces.
    4. Mention relative sizes or numbers if apparent (e.g., 'two chairs', 'large tv').
    5. Conclude with an overall impression of the scene's arrangement.

    Keep the description professional yet engaging, focusing on the spatial composition. Limit to 2-3 sentences."""
    print(f"Generated Prompt for Moondream:\n{prompt}")

    try:
        print("Sending request to Moondream...")
        response = client.chat(
            model='moondream', # Switch to the faster model
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            options={'temperature': 0.5}
        )
        print(f"Raw response from Moondream: {response}")
        description = response['message']['content'].strip()
        print(f"Generated Description: {description}")
        return description
    except Exception as e:
        print(f"--- ERROR during description generation ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {str(e)}")
        if "model not found" in str(e).lower():
            print("Ensure the model 'moondream' is available. Try: ollama pull moondream")
        elif "connection refused" in str(e).lower():
             print("Ensure the Ollama server is running. Try: ollama serve")
        return "Description currently unavailable due to an error."

def merge_similar_detections(detections, distance_threshold=50):
    merged = []
    used_indices = set()

    for i, det1 in enumerate(detections):
        if i in used_indices:
            continue

        current_group = [det1]
        det1['count'] = 1 # Initialize count
        used_indices.add(i)

        for j, det2 in enumerate(detections):
            if j <= i or j in used_indices:
                continue

            # Check class match and distance
            if det1['class'] == det2['class']:
                dx = abs(det1['centroid'][0] - det2['centroid'][0])
                dy = abs(det1['centroid'][1] - det2['centroid'][1])

                if dx < distance_threshold and dy < distance_threshold:
                    current_group.append(det2)
                    used_indices.add(j)

        # Process the group: average position, max confidence, sum count
        if len(current_group) > 1:
            sum_x = sum(d['centroid'][0] for d in current_group)
            sum_y = sum(d['centroid'][1] for d in current_group)
            avg_x = sum_x / len(current_group)
            avg_y = sum_y / len(current_group)
            max_conf = max(d['confidence'] for d in current_group)
            total_count = len(current_group)

            # Use the first detection as the base and update it
            merged_det = current_group[0].copy() # Important to copy
            merged_det['centroid'] = (avg_x, avg_y)
            merged_det['confidence'] = max_conf
            merged_det['count'] = total_count
             # Recalculate grid position based on average centroid (assuming fixed grid size for now)
            # Note: Need image width/height here if grid depends on it. Pass them if needed.
            # For simplicity, we'll keep the grid position of the first item for now.
            # A more accurate approach would require image dimensions here.
            merged.append(merged_det)
        else:
            # Single detection, add as is (with count 1)
             merged.append(det1) # Already has count = 1

    return merged


def realtime_analysis(camera_index=0, video_path=None, grid_size=10, frame_skip=5, display_scale=0.5, describe_interval=10):
    """
    Performs real-time object detection and description on camera stream or video file.

    Args:
        camera_index (int): Index of the camera to use (default: 0). Ignored if video_path is provided.
        video_path (str, optional): Path to the video file to process. Defaults to None (use camera).
        grid_size (int): Dimension of the grid for spatial analysis (e.g., 10 means 10x10 grid).
        frame_skip (int): Process every Nth frame to save computation.
        display_scale (float): Factor to scale the display window size.
        describe_interval (int): Generate description every Nth processed frame.
    """
    if video_path:
        print(f"Processing video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
    else:
        print(f"Starting real-time analysis from camera index: {camera_index}")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera index {camera_index}")
            return

    frame_count = 0
    description_queue = deque(maxlen=5) # Increased queue size for smoother text
    temp_frame_path = "temp_frame.jpg"
    processed_frame_count = 0 # Counter for frames sent to detection/description

    print(f"Configuration: describe_interval={describe_interval}, frame_skip={frame_skip}")
    print(f"Will attempt to generate description every {describe_interval} processed frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or cannot read frame.")
            break # Exit loop if no frame is captured

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue # Skip frame for display, but also for processing

        processed_frame_count += 1 # Increment processed frame counter
        print(f"Processing frame {processed_frame_count}/{describe_interval}")

        # Save temp frame for processing
        success = cv2.imwrite(temp_frame_path, frame)
        if not success:
            print("Error: Could not write temporary frame.")
            continue

        # Get frame dimensions *once* for potential use in merge/describe if needed
        # height, width = frame.shape[:2]

        # Process frame: Detect -> Merge -> Describe
        detections = run_detection(temp_frame_path, grid_size, grid_size)
        merged_detections = merge_similar_detections(detections)

        # --- Generate description only periodically ---
        if processed_frame_count % describe_interval == 0:
            print(f"INTERVAL REACHED! Frame count: {processed_frame_count}")
            if merged_detections:
                print(f"Attempting to describe {len(merged_detections)} merged detections")
                desc = describe_image(merged_detections, grid_size, grid_size)
                if desc:
                    description_queue.append(desc)
                    print(f"Added description to queue: {desc}")
            else:
                print("No merged detections to describe")

        # --- Display ---
        # Calculate dynamic display size
        orig_h, orig_w = frame.shape[:2]
        display_w = int(orig_w * display_scale)
        display_h = int(orig_h * display_scale)
        display_frame = cv2.resize(frame, (display_w, display_h))

        # Draw descriptions onto the frame
        text_y = 20 # Starting position for text
        font_scale = 0.5
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (0, 255, 0) # Green text
        bg_color = (0, 0, 0) # Black background for text

        # Draw background rectangles for text readability
        for i, d in enumerate(reversed(description_queue)): # Show newest first
            (text_width, text_height), baseline = cv2.getTextSize(d, font, font_scale, font_thickness)
            cv2.rectangle(display_frame, (5, text_y - text_height - baseline + 2), (5 + text_width, text_y + baseline), bg_color, -1) # Filled rectangle
            cv2.putText(display_frame, d, (5, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            text_y += text_height + baseline + 5 # Move down for next line
            if i >= 2: break # Limit displayed descriptions to 3

        cv2.imshow('Live Analysis', display_frame)

        # Exit condition
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break
        # Add pause functionality for video files
        if video_path and key == ord('p'):
            print("Paused. Press 'p' again to resume.")
            while True:
                 key_pause = cv2.waitKey(0) & 0xFF
                 if key_pause == ord('p'):
                     print("Resuming...")
                     break
                 elif key_pause == ord('q'):
                      print("Exiting from pause...")
                      cap.release()
                      cv2.destroyAllWindows()
                      if os.path.exists(temp_frame_path):
                          os.remove(temp_frame_path)
                      return # Exit function fully


    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    # Clean up the temporary frame file
    if os.path.exists(temp_frame_path):
        try:
            os.remove(temp_frame_path)
        except OSError as e:
            print(f"Error removing temporary file {temp_frame_path}: {e}")


# Test
if __name__ == "__main__":
    # --- Option 1: Realtime camera analysis ---
    # print("Starting real-time camera analysis...")
    # realtime_analysis(camera_index=0, frame_skip=10, display_scale=0.6)

    # --- Option 2: Video file analysis ---
    video_file_path = "/home/quantic/learning/What-m-i-seeing/glove_test.mp4" # <--- CHANGE THIS TO YOUR VIDEO FILE PATH
    if os.path.exists(video_file_path):
         print(f"Starting analysis for video file: {video_file_path}")
         # Using smaller describe_interval for testing moondream
         realtime_analysis(video_path=video_file_path, frame_skip=15, display_scale=0.7, describe_interval=5)
    else:
         print(f"Video file not found: {video_file_path}. Check the path.")
         # Fallback or default behavior if video not found?
         # print("Falling back to camera analysis...")
         # realtime_analysis(camera_index=0, frame_skip=10, display_scale=0.6)


    # --- Option 3: Single image analysis (comment out other options) ---
    # image_file = "/home/quantic/learning/What-m-i-seeing/PXL_20250421_042146663.jpg"
    # print(f"Analyzing single image: {image_file}")
    # detections = run_detection(image_file)
    # if detections:
    #     merged = merge_similar_detections(detections)
    #     description = describe_image(merged, grid_width=10, grid_height=10)
    #     print("\nImage Description:", description)
    #     print("\nMerged Detections:")
    #     for det in merged:
    #         print(det)
    # else:
    #     print("No detections found in the image.")
