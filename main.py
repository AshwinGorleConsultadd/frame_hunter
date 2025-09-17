"""
Video Processing Script for Stair Repair Defect Detection

This script processes videos of building inspections to extract defect information
from audio transcripts and generate structured data for repair documentation.
"""

import os
import json
import logging
import re
from typing import List, Optional, Dict, Any

# Audio processing imports
import whisper
import ffmpeg

# LLM and structured output imports
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# AWS imports (optional)
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DefectInfo(BaseModel):
    """Pydantic model for structured defect information"""
    building_counter: Optional[str] = Field(None, description="Building counter (building1, building2, etc.)")
    building_name: Optional[str] = Field(None, description="Building name if mentioned in video")
    apartment_number: Optional[str] = Field(None, description="Apartment number")
    tread_number: Optional[str] = Field(None, description="Tread number mentioned")
    priority: Optional[str] = Field(None, description="Priority level (1, 2, etc.) or (one, two, etc.)")
    description: Optional[str] = Field(None, description="Description of the defect, like (bottom rear crack, front rear crack, top front crack, top rear crack, etc.)")
    timestamp_start: Optional[float] = Field(None, description="Start timestamp in seconds")
    timestamp_end: Optional[float] = Field(None, description="End timestamp in seconds")
    transcript_segment: Optional[str] = Field(None, description="Original transcript segment")


class VideoProcessor:
    """Main class for processing repair videos and extracting defect information"""
    
    def __init__(self, whisper_model_name: str = "base.en", openai_api_key: Optional[str] = None):
        """
        Initialize the video processor
        
        Args:
            whisper_model_name: Whisper model to use for transcription
            openai_api_key: OpenAI API key for LLM processing
        """
        self.whisper_model_name = whisper_model_name
        self.whisper_model = None
        self.llm = None
        self.s3_client = None
        
        # Building tracking
        self.building_counter = 0
        self.current_building_name = None
        self.current_apartment_number = None
        
        # Initialize OpenAI LLM if API key provided
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            self.llm = OpenAI()
        
        # Initialize S3 client
        if boto3:
            try:
                self.s3_client = boto3.client('s3')
            except Exception as e:
                logger.warning("Could not initialize S3 client: %s", e)
        else:
            logger.warning("boto3 not available - S3 functionality disabled")
    
    def load_whisper_model(self):
        """Load the Whisper model for transcription"""
        try:
            logger.info("Loading Whisper model: %s", self.whisper_model_name)
            self.whisper_model = whisper.load_model(self.whisper_model_name)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error("Failed to load Whisper model: %s", e)
            raise
    
    def get_video_from_local(self, video_path: str) -> str:
        """
        Load video from local path
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to the video file
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info("Loading video from local path: %s", video_path)
        return video_path
    
    def get_video_from_s3(self, bucket_name: str, object_key: str, local_path: str) -> str:
        """
        Download video from S3 bucket
        
        Args:
            bucket_name: S3 bucket name
            object_key: S3 object key
            local_path: Local path to save the video
            
        Returns:
            Path to the downloaded video file
        """
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized")
        
        try:
            logger.info("Downloading video from S3: s3://%s/%s", bucket_name, object_key)
            self.s3_client.download_file(bucket_name, object_key, local_path)
            logger.info("Video downloaded to: %s", local_path)
            return local_path
        except ClientError as e:
            logger.error("Failed to download video from S3: %s", e)
            raise
    
    def extract_audio_from_video(self, video_path: str, output_audio_path: str) -> str:
        """
        Extract audio from video using FFmpeg
        
        Args:
            video_path: Path to the input video
            output_audio_path: Path to save the extracted audio
            
        Returns:
            Path to the extracted audio file
        """
        try:
            logger.info("Extracting audio from video: %s", video_path)
            
            # Use ffmpeg to extract audio
            (
                ffmpeg
                .input(video_path)
                .output(output_audio_path, acodec='pcm_s16le', ac=1, ar='16000')
                .overwrite_output()
                .run(quiet=True)
            )
            
            logger.info("Audio extracted to: %s", output_audio_path)
            return output_audio_path
            
        except ffmpeg.Error as e:
            logger.error("FFmpeg error: %s", e)
            raise
        except Exception as e:
            logger.error("Failed to extract audio: %s", e)
            raise
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcription result with segments and timestamps
        """
        if not self.whisper_model:
            self.load_whisper_model()
        
        try:
            logger.info("Transcribing audio: %s", audio_path)
            result = self.whisper_model.transcribe(audio_path, word_timestamps=True, language="en", condition_on_previous_text=False, no_speech_threshold=0.5 )
            logger.info("Transcription completed successfully")
            return result
        except Exception as e:
            logger.error("Failed to transcribe audio: %s", e)
            raise
    
    def create_llm_prompt(self) -> PromptTemplate:
        """Create the prompt template for LLM-based data extraction"""
        template = """
        You are an expert at extracting structured defect information from building inspection transcripts.
        
        Given the following transcript chunk, extract defect information. Focus on understanding the context and meaning.
        
        IMPORTANT CONTEXT UNDERSTANDING:
        - Transcript segments may be fragmented across multiple lines
        - A single defect might be split across multiple segments
        - Example: "Tread number 6 priority." + "1 top front crack." = Complete defect: "Tread number 6 priority 1 top front crack"
        - Look for patterns like: "Tread number X, priority Y"
        
        EXTRACTION RULES:
        1. Extract complete defect information by combining fragmented segments
        2. Look for tread numbers (may be written as "tread 9", "tread number 9", "tread nine", "tri 8")
        3. Look for priorities (may be written as "priority 1", "priority one", "priority 2", "priority two")
        4. Look for defect descriptions (top/bottom/front/rear + crack/defect etc)
        5. Use the timestamp from the segment that contains the main defect description
        6. Handle misspellings intelligently (thread -> tread, tred -> tread -> tri, etc.)
        
        BUILDING/APARTMENT CONTEXT:
        - If you see "apartment 111", "department 123", "building 2" - note these for context
        - These will be added automatically by the system
        Transcript chunk:
        {transcript_segments}
        
        {format_instructions}
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["transcript_segments"],
            partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=List[DefectInfo]).get_format_instructions()}
        )
    
    def extract_defects_with_llm(self, transcript_result: Dict[str, Any]) -> List[DefectInfo]:
        """
        Extract defect information using LLM with structured output
        
        Args:
            transcript_result: Whisper transcription result
            
        Returns:
            List of DefectInfo objects
        """
        if not self.llm:
            logger.warning("LLM not initialized, falling back to rule-based extraction")
            return self.extract_defects_rule_based(transcript_result)
        
        try:
            # Prepare transcript segments with timestamps
            segments_text = []
            for segment in transcript_result.get('segments', []):
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                text = segment.get('text', '').strip()
                
                if text:
                    segments_text.append(f"[{start_time:.2f}s - {end_time:.2f}s] {text}")
            
            transcript_segments = "\n".join(segments_text)
            
            # Create prompt and parse output
            parser = PydanticOutputParser(pydantic_object=List[DefectInfo])
            prompt = self.create_llm_prompt()
            
            formatted_prompt = prompt.format(transcript_segments=transcript_segments)
            response = self.llm(formatted_prompt)
            
            # Parse the response
            defects = parser.parse(response)
            
            # Add timestamp information from original segments
            self._enrich_defects_with_timestamps(defects, transcript_result)
            
            logger.info("Extracted %d defects using LLM", len(defects))
            return defects
            
        except Exception as e:
            logger.error("LLM extraction failed: %s", e)
            logger.info("Falling back to rule-based extraction")
            return self.extract_defects_rule_based(transcript_result)
    
    
    def extract_defects_rule_based(self, transcript_result: Dict[str, Any]) -> List[DefectInfo]:
        """
        Fallback rule-based extraction when LLM is not available
        
        Args:
            transcript_result: Whisper transcription result
            
        Returns:
            List of DefectInfo objects
        """
        defects = []
        
        # Reset building tracking for new video
        self.building_counter = 0
        self.current_building_name = None
        self.current_apartment_number = None
        
        for segment in transcript_result.get('segments', []):
            text = segment.get('text', '').strip().lower()
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            
            # Extract building information
            if any(keyword in text for keyword in ['building', 'department']):
                building_info = self._extract_building_info(text)
                if building_info:
                    self.building_counter += 1
                    self.current_building_name = building_info.get('name')
                    logger.info("Building detected: %s -> building%d", text.strip(), self.building_counter)
            
            # Extract apartment number
            if 'apartment' in text:
                apartment_match = self._extract_apartment_number(text)
                if apartment_match:
                    self.current_apartment_number = apartment_match
            
            # Extract defect information
            if any(keyword in text for keyword in ['tread', 'thread', 'tred', 'crack', 'defect']):
                defect_info = self._extract_defect_from_text(text, start_time, end_time)
                if defect_info:
                    defect_info.building_counter = f"building{self.building_counter}" if self.building_counter > 0 else None
                    defect_info.building_name = self.current_building_name
                    defect_info.apartment_number = self.current_apartment_number
                    defect_info.transcript_segment = segment.get('text', '')
                    defects.append(defect_info)
        
        logger.info("Extracted %d defects using rule-based method", len(defects))
        return defects
    
    def _extract_building_info(self, text: str) -> Optional[Dict[str, str]]:
        """Extract building information from text"""
        
        # Patterns for building numbers (including department)
        number_patterns = [
            r'building\s+(\d+)',
            r'building\s+number\s+(\d+)',
            r'department\s+(\d+)',
            r'department\s+(\d+)\s+building\s+(\d+)'  # "department 137 building one"
        ]
        
        # Patterns for building names
        name_patterns = [
            r'building\s+([a-zA-Z][a-zA-Z0-9\s]*)',
            r'building\s+name\s+([a-zA-Z][a-zA-Z0-9\s]*)'
        ]
        
        building_info = {}
        
        # Check for building numbers (including department)
        for pattern in number_patterns:
            match = re.search(pattern, text)
            if match:
                if 'department' in pattern and 'building' in pattern:
                    # Handle "department 137 building one" case
                    building_info['number'] = match.group(2)  # building number
                else:
                    building_info['number'] = match.group(1)
                break
        
        # Check for building names (only if no number found)
        if 'number' not in building_info:
            for pattern in name_patterns:
                match = re.search(pattern, text)
                if match:
                    name = match.group(1).strip()
                    # Only consider it a name if it's not just a number
                    if not name.isdigit():
                        building_info['name'] = name
                        break
        
        return building_info if building_info else None
    
    def _extract_apartment_number(self, text: str) -> Optional[str]:
        """Extract apartment number from text"""
        patterns = [
            r'apartment\s+(\d+)',
            r'apartment\s+number\s+(\d+)',
            r'department\s+(\d+)'  # Handle misspellings
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def _extract_defect_from_text(self, text: str, start_time: float, end_time: float) -> Optional[DefectInfo]:
        """Extract defect information from text segment"""
        
        # Look for tread patterns
        tread_patterns = [
            r'(?:tread|thread|tred)\s+(\d+)',
            r'tread\s+number\s+(\d+)'
        ]
        
        tread_number = None
        for pattern in tread_patterns:
            match = re.search(pattern, text)
            if match:
                tread_number = match.group(1)
                break
        
        # Look for priority patterns
        priority_patterns = [
            r'priority\s+(\d+)',
            r'priority\s+(one|two|three|four|five)'
        ]
        
        priority = None
        for pattern in priority_patterns:
            match = re.search(pattern, text)
            if match:
                priority = match.group(1)
                break
        
        # Look for defect descriptions
        defect_keywords = ['crack', 'defect', 'damage', 'wear', 'broken']
        description = None
        for keyword in defect_keywords:
            if keyword in text:
                # Extract surrounding context
                words = text.split()
                keyword_index = words.index(keyword) if keyword in words else -1
                if keyword_index >= 0:
                    start_idx = max(0, keyword_index - 2)
                    end_idx = min(len(words), keyword_index + 3)
                    description = ' '.join(words[start_idx:end_idx])
                    break
        
        if tread_number or priority or description:
            return DefectInfo(
                tread_number=tread_number,
                priority=priority,
                description=description,
                timestamp_start=start_time,
                timestamp_end=end_time
            )
        
        return None
    
    def _enrich_defects_with_timestamps(self, defects: List[DefectInfo], transcript_result: Dict[str, Any]):
        """Enrich defects with more accurate timestamp information"""
        # This method can be enhanced to provide more precise timestamps
        # based on word-level timestamps from Whisper
        # TODO: Implement word-level timestamp matching
    
    def _save_transcript_as_text(self, transcript_result: Dict[str, Any], output_file: str):
        """
        Save transcript as a readable text file for evaluation
        
        Args:
            transcript_result: Whisper transcription result
            output_file: Path to save the text file
        """
        try:
            logger.info("Saving transcript as text file: %s", output_file)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                # Write header
                f.write("VIDEO TRANSCRIPT\n")
                f.write("=" * 50 + "\n\n")
                
                # Write full text if available
                if 'text' in transcript_result:
                    f.write("FULL TRANSCRIPT:\n")
                    f.write("-" * 20 + "\n")
                    f.write(transcript_result['text'] + "\n\n")
                
                # Write segmented transcript with timestamps
                if 'segments' in transcript_result:
                    f.write("SEGMENTED TRANSCRIPT WITH TIMESTAMPS:\n")
                    f.write("-" * 40 + "\n")
                    
                    for i, segment in enumerate(transcript_result['segments'], 1):
                        start_time = segment.get('start', 0)
                        end_time = segment.get('end', 0)
                        text = segment.get('text', '').strip()
                        
                        # Format timestamp as MM:SS.mmm
                        start_formatted = f"{int(start_time//60):02d}:{start_time%60:06.3f}"
                        end_formatted = f"{int(end_time//60):02d}:{end_time%60:06.3f}"
                        
                        f.write(f"[{start_formatted} --> {end_formatted}] {text}\n")
                
                # Write summary
                f.write("\n" + "=" * 50 + "\n")
                f.write("TRANSCRIPT SUMMARY:\n")
                f.write(f"Total duration: {transcript_result.get('duration', 0):.2f} seconds\n")
                f.write(f"Number of segments: {len(transcript_result.get('segments', []))}\n")
                f.write(f"Language detected: {transcript_result.get('language', 'unknown')}\n")
            
            logger.info("Transcript text file saved successfully")
            
        except Exception as e:
            logger.error("Failed to save transcript text file: %s", e)
            raise
    
    def process_video(self, video_source: Dict[str, str], output_dir: str = "output") -> List[DefectInfo]:
        """
        Main method to process a video and extract defect information
        
        Args:
            video_source: Dictionary with source information
                - type: "local" or "s3"
                - path: local path (for local) or S3 key (for S3)
                - bucket: S3 bucket name (for S3)
            output_dir: Directory to save intermediate files
            
        Returns:
            List of extracted defect information
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Get video
        if video_source["type"] == "local":
            video_path = self.get_video_from_local(video_source["path"])
        elif video_source["type"] == "s3":
            local_video_path = os.path.join(output_dir, "downloaded_video.mp4")
            video_path = self.get_video_from_s3(
                video_source["bucket"], 
                video_source["path"], 
                local_video_path
            )
        else:
            raise ValueError("Invalid video source type. Use 'local' or 's3'")
        
        # Step 2: Extract audio
        audio_path = os.path.join(output_dir, "extracted_audio.wav")
        self.extract_audio_from_video(video_path, audio_path)
        
        # Step 3: Transcribe audio
        transcript_result = self.transcribe_audio(audio_path)
        
        # Save transcript for debugging
        transcript_file = os.path.join(output_dir, "transcript.json")
        with open(transcript_file, 'w', encoding='utf-8') as f:
            json.dump(transcript_result, f, indent=2)
        
        # Save transcript as readable text file for evaluation
        transcript_text_file = os.path.join(output_dir, "transcript.txt")
        self._save_transcript_as_text(transcript_result, transcript_text_file)
        
        # Step 4: Extract defects
        defects = self.extract_defects_with_llm(transcript_result)
        
        # Save defects for debugging
        defects_file = os.path.join(output_dir, "extracted_defects.json")
        with open(defects_file, 'w', encoding='utf-8') as f:
            json.dump([defect.dict() for defect in defects], f, indent=2)
        
        logger.info("Processing completed. Found %d defects.", len(defects))
        return defects


def main():
    """Main function to demonstrate usage"""
    # Initialize processor
    print("API Key: ", os.getenv("OPENAI_API_KEY"))
    processor = VideoProcessor(
        whisper_model_name="base.en",
        openai_api_key=os.getenv("OPENAI_API_KEY")  # Set this environment variable
    )
    
    # Example usage with local video
    video_source = {
        "type": "local",
        "path": "/Users/consultadd/Desktop/My project/consultadd_project/input/part000.mp4"
    }
    
    try:
        defects = processor.process_video(video_source)
        
        # Print results
        print(f"\nExtracted {len(defects)} defects:")
        for i, defect in enumerate(defects, 1):
            print(f"\nDefect {i}:")
            print(f"  Building Counter: {defect.building_counter}")
            print(f"  Building Name: {defect.building_name}")
            print(f"  Apartment: {defect.apartment_number}")
            print(f"  Tread: {defect.tread_number}")
            print(f"  Priority: {defect.priority}")
            print(f"  Description: {defect.description}")
            print(f"  Timestamp: {defect.timestamp_start:.2f}s - {defect.timestamp_end:.2f}s")
            
    except Exception as e:
        logger.error("Processing failed: %s", e)
        raise


if __name__ == "__main__":
    main()
