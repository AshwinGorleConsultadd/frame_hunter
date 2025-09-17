"""
Test script to verify JSON fixing functionality
"""

from main import VideoProcessor

def test_json_fixing():
    """Test the JSON fixing functionality"""
    
    print("🧪 Testing JSON Fixing Functionality")
    print("=" * 50)
    
    # Create a processor instance (without LLM for this test)
    processor = VideoProcessor(whisper_model_name="base.en")
    
    # Test malformed JSON that the LLM might return
    malformed_json = '''
    "description": "apartment 122",
    "start_time": 4.88,
    "end_time": 10.44
    "description": "tread number 11 priority one top front crack",
    "star...
    '''
    
    print("Testing malformed JSON:")
    print(malformed_json)
    print()
    
    # Test the fixing function
    chunks = processor._fix_and_parse_json(malformed_json)
    
    print(f"✅ Fixed and extracted {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}: [{chunk.start_time:.2f}s - {chunk.end_time:.2f}s] {chunk.description}")
    
    # Test another malformed case
    malformed_json2 = '''
    {
        "description": "tread number 9 priority 2 bottom rear crack",
        "start_time": 15.0,
        "end_time": 20.0
    },
    {
        "description": "tread number 4 priority 1 top front crack",
        "start_time": 25.0,
        "end_time": 30.0
    }
    '''
    
    print("\nTesting malformed JSON without array brackets:")
    print(malformed_json2)
    print()
    
    chunks2 = processor._fix_and_parse_json(malformed_json2)
    
    print(f"✅ Fixed and extracted {len(chunks2)} chunks:")
    for i, chunk in enumerate(chunks2, 1):
        print(f"  Chunk {i}: [{chunk.start_time:.2f}s - {chunk.end_time:.2f}s] {chunk.description}")
    
    return len(chunks) > 0 or len(chunks2) > 0

if __name__ == "__main__":
    success = test_json_fixing()
    if success:
        print("\n🎉 JSON fixing test passed!")
    else:
        print("\n💥 JSON fixing test failed!")
