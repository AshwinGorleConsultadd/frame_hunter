"""
Test script for screenshot generator
"""

import os
import json
from screenshot_generator import generate_defect_report
from main import DefectInfo

def test_screenshot_generator():
    """Test the screenshot generator with sample data"""
    
    # Check if defects file exists
    defects_file = "output/extracted_defects.json"
    video_file = "input/part000.mp4"
    
    if not os.path.exists(defects_file):
        print(f"❌ Defects file not found: {defects_file}")
        print("Please run main.py first to generate defects.")
        return False
    
    if not os.path.exists(video_file):
        print(f"❌ Video file not found: {video_file}")
        print("Please ensure the video file exists.")
        return False
    
    try:
        # Load defects from JSON file
        with open(defects_file, 'r', encoding='utf-8') as f:
            defects_data = json.load(f)
        
        # Convert to DefectInfo objects
        defects = []
        for defect_data in defects_data:
            defect = DefectInfo(**defect_data)
            defects.append(defect)
        
        print(f"📊 Loaded {len(defects)} defects from {defects_file}")
        
        # Generate report
        print("🎬 Generating screenshots and CSV report...")
        csv_path = generate_defect_report(defects, video_file)
        
        print(f"✅ Report generated successfully!")
        print(f"📁 CSV file: {csv_path}")
        print(f"📸 Screenshots saved in: output/screenshots/")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_screenshot_generator()
