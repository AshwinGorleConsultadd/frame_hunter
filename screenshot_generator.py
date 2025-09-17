"""
Screenshot Generator Script

This script takes a list of defects and captures screenshots at specified timestamps,
then generates a CSV file with defect information and image links.
"""

import os
import csv
import cv2
import logging
from pathlib import Path
from typing import List, Optional
from main import DefectInfo

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ScreenshotGenerator:
    """Class for generating screenshots and CSV reports from defect information"""
    
    def __init__(self, video_path: str, output_dir: str = "output"):
        """
        Initialize the screenshot generator
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save screenshots and CSV
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.screenshots_dir = self.output_dir / "screenshots"
        
        # Create directories
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize video capture
        self.cap = None
        self._initialize_video_capture()
    
    def _initialize_video_capture(self):
        """Initialize OpenCV video capture"""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video file: {self.video_path}")
            
            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration = self.frame_count / self.fps
            
            logger.info(f"Video loaded: {self.duration:.2f}s duration, {self.fps:.2f} FPS")
            
        except Exception as e:
            logger.error(f"Failed to initialize video capture: {e}")
            raise
    
    def generate_screenshots_and_csv(self, defects: List[DefectInfo], csv_filename: str = "defects_report.csv") -> str:
        """
        Generate screenshots for all defects and create CSV report
        
        Args:
            defects: List of DefectInfo objects
            csv_filename: Name of the CSV file to generate
            
        Returns:
            Path to the generated CSV file
        """
        if not defects:
            logger.warning("No defects provided for screenshot generation")
            return None
        
        logger.info(f"Generating screenshots for {len(defects)} defects")
        
        # Generate screenshots
        screenshot_data = []
        for i, defect in enumerate(defects, 1):
            screenshot_info = self._capture_screenshot(defect, i)
            if screenshot_info:
                screenshot_data.append(screenshot_info)
        
        # Generate CSV report
        csv_path = self._generate_csv_report(screenshot_data, csv_filename)
        
        logger.info(f"Screenshot generation completed. Generated {len(screenshot_data)} screenshots")
        logger.info(f"CSV report saved to: {csv_path}")
        
        return str(csv_path)
    
    def _capture_screenshot(self, defect: DefectInfo, defect_number: int) -> Optional[dict]:
        """
        Capture screenshot for a single defect
        
        Args:
            defect: DefectInfo object
            defect_number: Sequential number for the defect
            
        Returns:
            Dictionary with screenshot information or None if failed
        """
        try:
            # Use ss_timestamp if available, otherwise calculate from start/end
            timestamp = defect.ss_timestamp
            if timestamp is None:
                if defect.timestamp_start is not None and defect.timestamp_end is not None:
                    timestamp = (defect.timestamp_start + defect.timestamp_end) / 2
                else:
                    logger.warning(f"No valid timestamp for defect {defect_number}")
                    return None
            
            # Ensure timestamp is within video duration
            if timestamp > self.duration:
                logger.warning(f"Timestamp {timestamp:.2f}s exceeds video duration {self.duration:.2f}s")
                timestamp = self.duration - 1
            
            # Calculate frame number
            frame_number = int(timestamp * self.fps)
            
            # Seek to the frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            
            if not ret:
                logger.error(f"Failed to read frame at timestamp {timestamp:.2f}s")
                return None
            
            # Generate filename
            filename = f"defect_{defect_number:03d}_{timestamp:.2f}s.jpg"
            screenshot_path = self.screenshots_dir / filename
            
            # Save screenshot
            cv2.imwrite(str(screenshot_path), frame)
            
            logger.info(f"Captured screenshot for defect {defect_number} at {timestamp:.2f}s: {filename}")
            
            return {
                'defect_number': defect_number,
                'building_counter': defect.building_counter,
                'building_name': defect.building_name,
                'apartment_number': defect.apartment_number,
                'tread_number': defect.tread_number,
                'priority': defect.priority,
                'description': defect.description,
                'timestamp': timestamp,
                'screenshot_path': str(screenshot_path),
                'transcript_segment': defect.transcript_segment
            }
            
        except Exception as e:
            logger.error(f"Failed to capture screenshot for defect {defect_number}: {e}")
            return None
    
    def _generate_csv_report(self, screenshot_data: List[dict], csv_filename: str) -> str:
        """
        Generate CSV report from screenshot data
        
        Args:
            screenshot_data: List of screenshot information dictionaries
            csv_filename: Name of the CSV file
            
        Returns:
            Path to the generated CSV file
        """
        csv_path = self.output_dir / csv_filename
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'defect_number',
                    'building_counter', 
                    'building_name',
                    'apartment_number',
                    'tread_number',
                    'priority',
                    'description',
                    'timestamp',
                    'screenshot_path',
                    'transcript_segment'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for data in screenshot_data:
                    writer.writerow(data)
            
            logger.info(f"CSV report generated with {len(screenshot_data)} entries")
            return str(csv_path)
            
        except Exception as e:
            logger.error(f"Failed to generate CSV report: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
            logger.info("Video capture released")


def generate_defect_report(defects: List[DefectInfo], video_path: str, output_dir: str = "output") -> str:
    """
    Convenience function to generate defect report with screenshots and CSV
    
    Args:
        defects: List of DefectInfo objects
        video_path: Path to the video file
        output_dir: Directory to save outputs
        
    Returns:
        Path to the generated CSV file
    """
    generator = ScreenshotGenerator(video_path, output_dir)
    
    try:
        csv_path = generator.generate_screenshots_and_csv(defects)
        return csv_path
    finally:
        generator.cleanup()


if __name__ == "__main__":
    """Test the screenshot generator"""
    import json
    
    # Load defects from JSON file for testing
    defects_file = "output/extracted_defects.json"
    video_file = "input/sample.mp4"
    
    if os.path.exists(defects_file):
        with open(defects_file, 'r', encoding='utf-8') as f:
            defects_data = json.load(f)
        
        # Convert to DefectInfo objects
        defects = []
        for defect_data in defects_data:
            defect = DefectInfo(**defect_data)
            defects.append(defect)
        
        # Generate report
        csv_path = generate_defect_report(defects, video_file)
        print(f"Report generated: {csv_path}")
    else:
        print(f"Defects file not found: {defects_file}")
        print("Run main.py first to generate defects.")
