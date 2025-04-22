# What-m-i-seeing

A real-time object detection and scene description system that analyzes video streams or files and provides natural language descriptions of what's being observed.

## Features

- Real-time object detection using YOLOv8
- Grid-based spatial mapping of detected objects
- Natural language scene descriptions using local LLM (Moondream)
- Support for both camera feeds and video files
- Merging of similar nearby detections
- Adjustable processing parameters for performance tuning

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- Ollama (for local LLM inference)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/What-m-i-seeing.git
cd What-m-i-seeing
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Install Ollama:
```bash
# For Linux
curl -fsSL https://ollama.com/install.sh | sh
```

4. Download required models:
```bash
# YOLOv8 nano model will be downloaded automatically
# Download Moondream model
ollama pull moondream
```

## Usage

### Basic Usage

Run with video file:
```bash
python scripts/detect_objects.py
```
Note: Edit the `video_file_path` variable in the script to point to your video file.

Run with camera:
```bash
# Uncomment option 1 in the main block and comment out option 2
python scripts/detect_objects.py
```

### Controls

- Press `q` to quit
- Press `p` to pause/resume (when processing video files)

### Configuration

Edit these parameters in the script to adjust performance:

- `frame_skip`: Process every Nth frame
- `describe_interval`: Generate description every Nth processed frame
- `grid_size`: Defines the resolution of spatial grid
- `display_scale`: Adjusts the display window size

## Customization

- Change YOLO model by editing: `model = YOLO("yolov8n.pt")`
- Use different LLM models: Change `model='moondream'` to any other Ollama model
- Adjust prompt by modifying the `prompt` variable in `describe_image()` 