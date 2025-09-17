"""
Test script to verify defect description extraction
"""

import re
from main import RefinedTranscriptChunk, DefectInfo

def test_description_extraction():
    """Test the description extraction logic"""
    
    # Test cases
    test_cases = [
        "Tread number nine, priority one, top front crack.",
        "tread number eight priority one top rear crack",
        "Track, tread number 10, priority one, top rear crack, screenshot.",
        "Try number 11 priority one top front crack.",
        "tread number 18 priority one bottom rear crack tread number 17 priority one"
    ]
    
    print("🧪 Testing Defect Description Extraction")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: '{test_case}'")
        
        # Create a mock chunk
        chunk = RefinedTranscriptChunk(
            description=test_case,
            start_time=0.0,
            end_time=10.0
        )
        
        # Test the extraction logic
        description = chunk.description.lower()
        
        # Extract description - improved pattern matching
        defect_description = None
        
        # Look for specific defect patterns first
        defect_patterns = [
            r'(top|bottom|front|rear|center)\s+(rear|front|top|bottom|center)?\s*(crack|cracks|defect|defects)',
            r'(top|bottom|front|rear|center)\s+(crack|cracks|defect|defects)',
            r'(crack|cracks|defect|defects)\s+(top|bottom|front|rear|center)'
        ]
        
        for pattern in defect_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                defect_description = match.group(0).strip()
                break
        
        # Fallback: extract any crack/defect mention with better context
        if not defect_description:
            defect_keywords = ['crack', 'defect', 'damage', 'wear', 'broken']
            for keyword in defect_keywords:
                if keyword in description:
                    # Extract surrounding context (simplified)
                    words = description.split()
                    keyword_index = words.index(keyword) if keyword in words else -1
                    if keyword_index >= 0:
                        start_idx = max(0, keyword_index - 2)
                        end_idx = min(len(words), keyword_index + 3)
                        defect_description = ' '.join(words[start_idx:end_idx])
                        break
        
        print(f"Extracted Description: '{defect_description}'")
        
        if defect_description:
            print("✅ SUCCESS: Description extracted")
        else:
            print("❌ FAILED: No description extracted")

if __name__ == "__main__":
    test_description_extraction()
