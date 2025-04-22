# What-m-i-seeing

A real-time object detection and scene description system that analyzes video streams or files and provides natural language descriptions of what's being observed.



## Features

- Real-time object detection using YOLOv8
- Grid-based spatial mapping of detected objects
- Natural language scene descriptions using local LLM (Moondream)
- Support for both camera feeds and video files
- Merging of similar nearby detections
- Adjustable processing parameters for performance tuning
- Multiple model support (YOLOv8 variants and different LLMs)

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
# YOLOv8 models will be downloaded automatically
# Download the default LLM model (moondream)
ollama pull moondream

# For other LLM options:
ollama pull llava:13b    # More powerful but slower
ollama pull llava:7b     # Balance of speed and quality
ollama pull bakllava     # Alternative vision model
```

## Usage

### Basic Usage

Run with default settings (uses camera input):
```bash
python scripts/detect_objects.py
```

Run with a video file:
```bash
python scripts/detect_objects.py --video path/to/your/video.mp4
```

### Command Line Options

The script supports various command-line arguments:

```bash
python scripts/detect_objects.py --help
```

Key options:
```
--yolo {nano,small,medium,large,xlarge}
                      YOLO model size (default: nano)
--llm {moondream,llava13b,llava7b,bakllava}
                      LLM model for descriptions (default: moondream)
--video VIDEO         Path to video file (default: None, use camera)
--camera CAMERA       Camera index (default: 0)
--frame-skip FRAME_SKIP
                      Process every Nth frame (default: 15)
--describe-interval DESCRIBE_INTERVAL
                      Generate description every Nth processed frame (default: 10)
--display-scale DISPLAY_SCALE
                      Display window scale (default: 0.7)
--min-confidence MIN_CONFIDENCE
                      Minimum confidence for detections (default: 0.5)
--grid-size GRID_SIZE
                      Grid size for spatial analysis (default: 10)
```

### Examples

Use a more powerful YOLO model with default settings:
```bash
python scripts/detect_objects.py --yolo medium
```

Process a video with higher quality but slower LLM model:
```bash
python scripts/detect_objects.py --video myvideo.mp4 --llm llava13b
```

Optimize for performance with faster processing and less frequent descriptions:
```bash
python scripts/detect_objects.py --frame-skip 30 --describe-interval 20
```

### Controls

- Press `q` to quit
- Press `p` to pause/resume (when processing video files)

## Model Comparison

### YOLO Models
- `nano`: 3.2MB, fastest inference but less accurate
- `small`: 11.4MB, balanced speed and accuracy
- `medium`: 25.9MB, more accurate but slower
- `large`: 43.7MB, high accuracy, slower
- `xlarge`: 68.2MB, highest accuracy, slowest

### LLM Models
- `moondream`: 1.7GB, fast specialized vision model
- `llava7b`: 4.1GB, balanced model
- `bakllava`: 2.4GB, specialized vision model
- `llava13b`: 8.0GB, most powerful but slowest

## Customization

- Add new YOLO models by extending the `YOLO_MODELS` dictionary
- Add new LLM models by extending the `LLM_MODELS` dictionary
- Adjust prompt by modifying the `prompt` variable in `describe_image()` 