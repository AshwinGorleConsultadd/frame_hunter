"""
Screenshot Taker Module
Takes screenshots from video at specific timestamps for defects
"""

import os
import cv2
import logging
from typing import List, Any, Dict
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def take_screenshots_for_defects(defects: List[Any], video_path: str, output_dir: str = "output") -> List[Dict[str, Any]]:
    """
    Take screenshots for each defect and return defects with image paths
    
    Args:
        defects: List of DefectInfo objects
        video_path: Path to the video file
        output_dir: Directory to save screenshots
        
    Returns:
        List of dictionaries containing defect data with image_path
    """
    try:
        logger.info("Starting screenshot capture for %d defects...", len(defects))
        
        # Create screenshots directory
        screenshots_dir = os.path.join(output_dir, "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Could not open video file: %s", video_path)
            return []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info("Video properties - FPS: %.2f, Duration: %.2fs", fps, duration)
        
        defects_with_image_path = []
        
        for i, defect in enumerate(defects):
            try:
                # Get screenshot timestamp
                screenshot_time = defect.ss_timestamp if hasattr(defect, 'ss_timestamp') and defect.ss_timestamp else None
                
                if screenshot_time is None:
                    logger.warning("No screenshot timestamp for defect %d, skipping", i + 1)
                    # Add defect without image path
                    defect_dict = _convert_defect_to_dict(defect)
                    defect_dict['image_path'] = None
                    defect_dict['image_filename'] = None
                    defects_with_image_path.append(defect_dict)
                    continue
                
                # Calculate frame number
                frame_number = int(screenshot_time * fps)
                
                # Ensure frame number is within bounds
                if frame_number >= total_frames:
                    frame_number = total_frames - 1
                elif frame_number < 0:
                    frame_number = 0
                
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                # Read frame
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning("Could not read frame at time %.2fs for defect %d", screenshot_time, i + 1)
                    # Add defect without image path
                    defect_dict = _convert_defect_to_dict(defect)
                    defect_dict['image_path'] = None
                    defect_dict['image_filename'] = None
                    defects_with_image_path.append(defect_dict)
                    continue
                
                # Generate filename
                filename = f"defect_{i+1:03d}_{screenshot_time:.2f}s.jpg"
                image_path = os.path.join(screenshots_dir, filename)
                
                # Save screenshot
                success = cv2.imwrite(image_path, frame)
                
                if success:
                    logger.info("Screenshot saved: %s", filename)
                    
                    # Add defect with image path
                    defect_dict = _convert_defect_to_dict(defect)
                    defect_dict['image_path'] = image_path
                    defect_dict['image_filename'] = filename
                    defects_with_image_path.append(defect_dict)
                else:
                    logger.warning("Failed to save screenshot for defect %d", i + 1)
                    # Add defect without image path
                    defect_dict = _convert_defect_to_dict(defect)
                    defect_dict['image_path'] = None
                    defect_dict['image_filename'] = None
                    defects_with_image_path.append(defect_dict)
                    
            except Exception as e:
                logger.error("Error processing defect %d: %s", i + 1, e)
                # Add defect without image path
                defect_dict = _convert_defect_to_dict(defect)
                defect_dict['image_path'] = None
                defect_dict['image_filename'] = None
                defects_with_image_path.append(defect_dict)
        
        # Release video capture
        cap.release()
        
        logger.info("Screenshot capture completed. Processed %d defects", len(defects_with_image_path))
        return defects_with_image_path
        
    except Exception as e:
        logger.error("Failed to take screenshots: %s", e)
        return []

def _convert_defect_to_dict(defect: Any) -> Dict[str, Any]:
    """
    Convert DefectInfo object to dictionary
    
    Args:
        defect: DefectInfo object
        
    Returns:
        Dictionary representation of the defect
    """
    try:
        # Try to use model_dump if it's a Pydantic model
        if hasattr(defect, 'model_dump'):
            return defect.model_dump()
        else:
            # Fallback to manual conversion
            return {
                'building_counter': getattr(defect, 'building_counter', None),
                'building_name': getattr(defect, 'building_name', None),
                'apartment_number': getattr(defect, 'apartment_number', None),
                'tread_number': getattr(defect, 'tread_number', None),
                'priority': getattr(defect, 'priority', None),
                'description': getattr(defect, 'description', None),
                'timestamp_start': getattr(defect, 'timestamp_start', None),
                'timestamp_end': getattr(defect, 'timestamp_end', None),
                'ss_timestamp': getattr(defect, 'ss_timestamp', None),
                'transcript_segment': getattr(defect, 'transcript_segment', None)
            }
    except Exception as e:
        logger.error("Error converting defect to dictionary: %s", e)
        return {}

def generate_screenshot_summary(defects_with_images: List[Dict[str, Any]], output_dir: str = "output") -> str:
    """
    Generate a summary of screenshot capture results
    
    Args:
        defects_with_images: List of defects with image paths
        output_dir: Output directory
        
    Returns:
        Path to summary file
    """
    try:
        summary_file = os.path.join(output_dir, "screenshot_summary.txt")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Screenshot Capture Summary\n")
            f.write("=" * 50 + "\n\n")
            
            total_defects = len(defects_with_images)
            defects_with_screenshots = sum(1 for d in defects_with_images if d.get('image_path'))
            defects_without_screenshots = total_defects - defects_with_screenshots
            
            f.write(f"Total Defects: {total_defects}\n")
            f.write(f"Defects with Screenshots: {defects_with_screenshots}\n")
            f.write(f"Defects without Screenshots: {defects_without_screenshots}\n\n")
            
            f.write("Defect Details:\n")
            f.write("-" * 30 + "\n")
            
            for i, defect in enumerate(defects_with_images):
                f.write(f"\nDefect {i+1}:\n")
                f.write(f"  Building: {defect.get('building_counter', 'N/A')}\n")
                f.write(f"  Apartment: {defect.get('apartment_number', 'N/A')}\n")
                f.write(f"  Tread: {defect.get('tread_number', 'N/A')}\n")
                f.write(f"  Priority: {defect.get('priority', 'N/A')}\n")
                f.write(f"  Timestamp: {defect.get('ss_timestamp', 'N/A')}\n")
                f.write(f"  Screenshot: {defect.get('image_filename', 'N/A')}\n")
                f.write(f"  Status: {'✅ Captured' if defect.get('image_path') else '❌ Failed'}\n")
        
        logger.info("Screenshot summary saved: %s", summary_file)
        return summary_file
        
    except Exception as e:
        logger.error("Failed to generate screenshot summary: %s", e)
        return ""

if __name__ == "__main__":
    # Test the module
    print("Screenshot Taker module loaded successfully!")
    print("Use take_screenshots_for_defects(defects, video_path, output_dir) to capture screenshots")
