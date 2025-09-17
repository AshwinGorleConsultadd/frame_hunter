"""
Test script to verify LLM refinement improvements
"""

import os
import json
from main import VideoProcessor

def test_refinement():
    """Test the improved refinement function"""
    
    print("🧪 Testing LLM Refinement Improvements")
    print("=" * 50)
    
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set!")
        print("Please set your API key:")
        print("export OPENAI_API_KEY='sk-your-actual-api-key-here'")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    try:
        # Initialize processor
        processor = VideoProcessor(
            whisper_model_name="base.en",
            openai_api_key=api_key
        )
        
        if not processor.llm:
            print("❌ LLM not initialized")
            return False
        
        print("✅ LLM initialized successfully!")
        
        # Test with sample transcript data
        sample_transcript = {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "Alright, this is Ratchewing Creek on September 5th."},
                {"start": 5.0, "end": 8.0, "text": "I had apartment 111."},
                {"start": 8.0, "end": 12.0, "text": "This looks like..."},
                {"start": 12.0, "end": 15.0, "text": "I think I'll set some time."},
                {"start": 15.0, "end": 20.0, "text": "Tread number nine, priority two, bottom rear crack."},
                {"start": 20.0, "end": 25.0, "text": "Tread number four, priority two."},
                {"start": 25.0, "end": 30.0, "text": "Tread number 6 priority."},
                {"start": 30.0, "end": 35.0, "text": "1 top front crack."},
                {"start": 35.0, "end": 40.0, "text": "Tread number 9 priority."},
                {"start": 40.0, "end": 45.0, "text": "2 top front crack."},
                {"start": 45.0, "end": 50.0, "text": "I tried number 10 priority two top front rear crack."},
                {"start": 50.0, "end": 55.0, "text": "The department 123."}
            ]
        }
        
        print(f"📊 Testing with {len(sample_transcript['segments'])} segments")
        
        # Test refinement
        refined_chunks = processor.create_refined_transcript_chunks(sample_transcript)
        
        print(f"✅ Generated {len(refined_chunks)} refined chunks")
        
        # Display results
        for i, chunk in enumerate(refined_chunks, 1):
            print(f"Chunk {i}: [{chunk.start_time:.2f}s - {chunk.end_time:.2f}s] {chunk.description}")
        
        # Check if llm_output.txt was created
        if os.path.exists("output/llm_output.txt"):
            print("✅ LLM output saved to output/llm_output.txt")
        else:
            print("❌ LLM output file not found")
        
        return len(refined_chunks) > 1  # Should create multiple chunks
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_refinement()
    if success:
        print("\n🎉 Refinement test passed!")
    else:
        print("\n💥 Refinement test failed!")
