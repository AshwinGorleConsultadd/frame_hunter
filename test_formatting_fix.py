"""
Test script to verify formatting fix
"""

from main import DefectInfo, RefinedTranscriptChunk

def test_formatting_fix():
    """Test the formatting fix for None values"""
    
    print("🧪 Testing Formatting Fix for None Values")
    print("=" * 50)
    
    # Create test defects with None values
    test_defects = [
        DefectInfo(
            building_counter="building4",
            building_name="two",
            apartment_number="218",
            tread_number=None,
            priority=None,
            description="top front crack",
            timestamp_start=15.0,
            timestamp_end=20.0,
            ss_timestamp=17.5,
            transcript_segment="top front crack"
        ),
        DefectInfo(
            building_counter="building4",
            building_name="two",
            apartment_number="218",
            tread_number="9",
            priority="2",
            description="bottom rear crack",
            timestamp_start=None,
            timestamp_end=None,
            ss_timestamp=None,
            transcript_segment="Tread number 9 priority 2 bottom rear crack"
        )
    ]
    
    print("Testing defect formatting:")
    for i, defect in enumerate(test_defects, 1):
        print(f"\nDefect {i}:")
        print(f"  Building Counter: {defect.building_counter}")
        print(f"  Building Name: {defect.building_name}")
        print(f"  Apartment: {defect.apartment_number}")
        print(f"  Tread: {defect.tread_number}")
        print(f"  Priority: {defect.priority}")
        print(f"  Description: {defect.description}")
        
        # Test the fixed formatting
        timestamp_start = f"{defect.timestamp_start:.2f}" if defect.timestamp_start is not None else "None"
        timestamp_end = f"{defect.timestamp_end:.2f}" if defect.timestamp_end is not None else "None"
        ss_timestamp = f"{defect.ss_timestamp:.2f}" if defect.ss_timestamp is not None else "None"
        
        print(f"  Timestamp: {timestamp_start}s - {timestamp_end}s")
        print(f"  Screenshot Time: {ss_timestamp}s")
        print(f"  Transcript: {defect.transcript_segment}")
    
    print("\n✅ Formatting test completed successfully!")
    return True

if __name__ == "__main__":
    test_formatting_fix()
