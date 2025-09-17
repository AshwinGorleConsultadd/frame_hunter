"""
Test script to verify the video processing functionality
"""

from pathlib import Path
from main import VideoProcessor
from config import (
    WHISPER_MODEL, OPENAI_API_KEY, OUTPUT_DIR, 
    TEST_VIDEO_LOCAL
)

def test_local_video_processing():
    """Test processing a local video file"""
    print("Testing local video processing...")
    
    # Check if test video exists
    if not TEST_VIDEO_LOCAL.exists():
        print(f"Test video not found at {TEST_VIDEO_LOCAL}")
        print("Please place a test video file in the input directory")
        return False
    
    try:
        # Initialize processor
        processor = VideoProcessor(
            whisper_model_name=WHISPER_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Process video
        video_source = {
            "type": "local",
            "path": str(TEST_VIDEO_LOCAL)
        }
        
        defects = processor.process_video(video_source, str(OUTPUT_DIR))
        
        print(f"✅ Successfully processed video and found {len(defects)} defects")
        
        # Print sample results
        if defects:
            print("\nSample defects found:")
            for i, defect in enumerate(defects[:3], 1):  # Show first 3
                print(f"  {i}. Building: {defect.building_number}, "
                      f"Apartment: {defect.apartment_number}, "
                      f"Tread: {defect.tread_number}, "
                      f"Priority: {defect.priority}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return False

def test_audio_extraction():
    """Test audio extraction functionality"""
    print("\nTesting audio extraction...")
    
    if not TEST_VIDEO_LOCAL.exists():
        print("Skipping audio extraction test - no test video found")
        return False
    
    try:
        processor = VideoProcessor()
        audio_path = str(OUTPUT_DIR / "test_audio.wav")
        
        processor.extract_audio_from_video(str(TEST_VIDEO_LOCAL), audio_path)
        
        if Path(audio_path).exists():
            print("✅ Audio extraction successful")
            return True
        else:
            print("❌ Audio file not created")
            return False
            
    except Exception as e:
        print(f"❌ Audio extraction failed: {e}")
        return False

def test_transcription():
    """Test transcription functionality"""
    print("\nTesting transcription...")
    
    if not TEST_VIDEO_LOCAL.exists():
        print("Skipping transcription test - no test video found")
        return False
    
    try:
        processor = VideoProcessor(whisper_model_name=WHISPER_MODEL)
        
        # Extract audio first
        audio_path = str(OUTPUT_DIR / "test_transcription_audio.wav")
        processor.extract_audio_from_video(str(TEST_VIDEO_LOCAL), audio_path)
        
        # Transcribe
        result = processor.transcribe_audio(audio_path)
        
        if result and 'segments' in result:
            print(f"✅ Transcription successful - {len(result['segments'])} segments")
            return True
        else:
            print("❌ Transcription failed - no segments found")
            return False
            
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'whisper',
        'ffmpeg',
        'pydantic'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'ffmpeg':
                import ffmpeg
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them using: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 Running Video Processing Tests")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        return
    
    # Run tests
    tests = [
        test_audio_extraction,
        test_transcription,
        test_local_video_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
