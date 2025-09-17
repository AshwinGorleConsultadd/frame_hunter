"""
Test script to verify LLM fix
"""

import os
from main import VideoProcessor

def test_llm_connection():
    """Test LLM connection and PydanticOutputParser fix"""
    
    print("🧪 Testing LLM Connection and PydanticOutputParser Fix")
    print("=" * 60)
    
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
        
        if processor.llm:
            print("✅ LLM initialized successfully!")
            
            # Test a simple LLM call
            test_prompt = "Extract defect information from: 'Tread number 9, priority 2, top rear crack.'"
            response = processor.llm(test_prompt)
            print(f"✅ LLM response received: {response[:50]}...")
            
            return True
        else:
            print("❌ LLM not initialized")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_llm_connection()
