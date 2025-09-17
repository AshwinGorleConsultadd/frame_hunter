# Video Processing Script for Stair Repair Defect Detection

This script processes videos of building inspections to extract defect information from audio transcripts and generate structured data for repair documentation.

## Features

- **Video Loading**: Support for both local files and S3 buckets
- **Audio Extraction**: High-quality audio extraction using FFmpeg
- **Speech Recognition**: Accurate transcription using OpenAI Whisper
- **Intelligent Data Extraction**: LLM-powered extraction of defect information with context awareness
- **Structured Output**: Pydantic models for consistent data structure
- **Fallback Processing**: Rule-based extraction when LLM is unavailable

## Installation

### Prerequisites

1. **Python 3.8+**
2. **FFmpeg** - Install using your system package manager:
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

### Setup

1. **Clone or download the project files**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional, for LLM features):
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

5. **Test the installation**:
   ```bash
   python test_script.py
   ```

## Usage

### Basic Usage

```python
from main import VideoProcessor

# Initialize processor
processor = VideoProcessor(
    whisper_model_name="base.en",
    openai_api_key="your-openai-api-key"  # Optional
)

# Process local video
video_source = {
    "type": "local",
    "path": "/path/to/your/video.mp4"
}

defects = processor.process_video(video_source)
```

### S3 Usage

```python
# Process video from S3
video_source = {
    "type": "s3",
    "bucket": "your-bucket-name",
    "path": "path/to/video.mp4"
}

defects = processor.process_video(video_source)
```

### Command Line Usage

```bash
python main.py
```

## Configuration

Edit `config.py` to customize:

- Whisper model size (`WHISPER_MODEL`)
- Output directories
- AWS credentials
- Audio processing settings

## How It Works

### Step 1: Video Loading
- Supports local files and S3 buckets
- Automatic file validation and error handling

### Step 2: Audio Extraction
- Uses FFmpeg to extract high-quality audio
- Optimized for Whisper (16kHz, mono, PCM format)

### Step 3: Transcription
- OpenAI Whisper for accurate speech recognition
- Word-level timestamps for precise defect location

### Step 4: Data Extraction
- **LLM Method**: Uses OpenAI GPT with structured output parsing
- **Rule-based Fallback**: Regex patterns for basic extraction
- **Context Awareness**: Maintains building/apartment numbers across segments
- **Misspelling Handling**: Corrects common transcription errors

## Output Format

The script generates structured defect information:

```json
{
  "building_number": "2",
  "apartment_number": "218",
  "tread_number": "14",
  "priority": "2",
  "description": "top center cracks",
  "timestamp_start": 16.54,
  "timestamp_end": 17.03,
  "transcript_segment": "14 priority two top center cracks."
}
```

## File Structure

```
project/
├── main.py              # Main processing script
├── config.py            # Configuration settings
├── test_script.py       # Test and validation script
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── input/              # Input video files
├── output/             # Processing results
└── audio_folder/       # Extracted audio files
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**:
   - Install FFmpeg system-wide
   - Ensure it's in your PATH

2. **Whisper model download fails**:
   - Check internet connection
   - Try a smaller model (tiny, base)

3. **OpenAI API errors**:
   - Verify API key is correct
   - Check API quota and billing

4. **Memory issues**:
   - Use smaller Whisper model
   - Process shorter video segments

### Testing

Run the test script to verify everything works:

```bash
python test_script.py
```

This will:
- Check all dependencies
- Test audio extraction
- Test transcription
- Test full processing pipeline

## Future Enhancements

- **Step 4**: Screenshot capture at defect timestamps
- **CSV Export**: Generate Excel/CSV reports
- **Batch Processing**: Process multiple videos
- **Web Interface**: Browser-based processing
- **Real-time Processing**: Live video analysis

## License

This project is for internal company use. Please ensure compliance with your organization's policies regarding video processing and data handling.
