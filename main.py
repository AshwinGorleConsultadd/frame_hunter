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


class RefinedTranscriptChunk(BaseModel):
    """Pydantic model for refined transcript chunks"""
    description: str = Field(description="What is being talked about in this transcript chunk")
    start_time: float = Field(description="Start timestamp in seconds")
    end_time: float = Field(description="End timestamp in seconds")


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
    ss_timestamp: Optional[float] = Field(None, description="Estimated timestamp for taking screenshot")
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
            try:
                os.environ["OPENAI_API_KEY"] = openai_api_key
                self.llm = OpenAI()
                logger.info("✅ LLM initialized successfully")
                print("✅ LLM connected successfully!")
            except Exception as e:
                logger.error("Failed to initialize LLM: %s", e)
                self.llm = None
        else:
            self.llm = None
        
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
            # Check if audio file already exists
            if os.path.exists(output_audio_path):
                logger.info("Audio file already exists: %s", output_audio_path)
                logger.info("Skipping audio extraction - using existing file")
                return output_audio_path
            
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
    
    def create_refined_transcript_chunks(self, transcript_result: Dict[str, Any]) -> List[RefinedTranscriptChunk]:
        """
        Create refined transcript chunks using LLM with chunked processing for better accuracy
        
        Args:
            transcript_result: Whisper transcription result
            
        Returns:
            List of RefinedTranscriptChunk objects
        """
        if not self.llm:
            logger.warning("LLM not initialized, using simple chunking")
            return self._create_simple_chunks(transcript_result)
        
        try:
            # Get all segments
            segments = transcript_result.get('segments', [])
            all_refined_chunks = []
            
            # Process transcript in smaller chunks for better accuracy
            chunk_size = 10  # Process 10 segments at a time
            overlap_size = 2  # 2 segments overlap between chunks
            
            for i in range(0, len(segments), chunk_size - overlap_size):
                chunk_end = min(i + chunk_size, len(segments))
                chunk_segments = segments[i:chunk_end]
                
                # Process this chunk
                chunk_refined = self._process_refinement_chunk(chunk_segments, i)
                all_refined_chunks.extend(chunk_refined)
                
                logger.info("Processed refinement chunk %d/%d: %d segments -> %d refined chunks", 
                           i // (chunk_size - overlap_size) + 1, 
                           (len(segments) + chunk_size - overlap_size - 1) // (chunk_size - overlap_size),
                           len(chunk_segments), len(chunk_refined))
            
            # Remove duplicates from overlapping regions
            final_chunks = self._merge_overlapping_chunks(all_refined_chunks)
            
            logger.info("Created %d refined transcript chunks from %d original segments", 
                       len(final_chunks), len(segments))
            return final_chunks
            
        except Exception as e:
            logger.error("LLM transcript refinement failed: %s", e)
            logger.info("Falling back to simple chunking")
            return self._create_simple_chunks(transcript_result)
    
    def _process_refinement_chunk(self, chunk_segments: List[Dict[str, Any]], chunk_index: int) -> List[RefinedTranscriptChunk]:
        """Process a chunk of segments for refinement"""
        try:
            # Format chunk segments as text
            segments_text = []
            for segment in chunk_segments:
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                text = segment.get('text', '').strip()
                
                if text:
                    segments_text.append(f"[{start_time:.2f}s - {end_time:.2f}s] {text}")
            
            transcript_segments = "\n".join(segments_text)
            
            # Create LLM prompt for transcript refinement
            prompt = self._create_refinement_prompt()
            
            formatted_prompt = prompt.format(transcript_segments=transcript_segments)
            response = self.llm(formatted_prompt)
            
            # Save LLM output to file for debugging
            self._save_llm_output(response, chunk_index)
            
            # Parse the response manually since List[RefinedTranscriptChunk] is not a valid Pydantic model
            refined_chunks = self._parse_refinement_response(response)
            
            return refined_chunks
            
        except Exception as e:
            logger.error("Failed to process refinement chunk %d: %s", chunk_index // 15 + 1, e)
            return []
    
    def _parse_refinement_response(self, response: str) -> List[RefinedTranscriptChunk]:
        """Parse LLM response manually to extract RefinedTranscriptChunk objects"""
        try:
            import json
            import re
            
            logger.info(f"Parsing LLM response: {response[:200]}...")
            
            # Try to extract JSON from the response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                logger.info(f"Found JSON: {json_str[:200]}...")
                
                try:
                    data = json.loads(json_str)
                    logger.info(f"Parsed JSON data: {len(data)} items")
                    
                    chunks = []
                    for i, item in enumerate(data):
                        if isinstance(item, dict) and 'description' in item and 'start_time' in item and 'end_time' in item:
                            chunk = RefinedTranscriptChunk(
                                description=item['description'],
                                start_time=float(item['start_time']),
                                end_time=float(item['end_time'])
                            )
                            chunks.append(chunk)
                            logger.info(f"Chunk {i+1}: {item['description'][:50]}... ({item['start_time']}-{item['end_time']})")
                    
                    logger.info(f"Successfully parsed {len(chunks)} chunks from JSON")
                    return chunks
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    logger.error(f"Malformed JSON: {json_str}")
                    # Try to fix common JSON issues
                    return self._fix_and_parse_json(json_str)
            
            # Fallback: try to parse as simple text
            logger.info("JSON parsing failed, trying simple text parsing")
            return self._parse_simple_refinement_response(response)
            
        except Exception as e:
            logger.error("Failed to parse refinement response: %s", e)
            logger.error(f"Response was: {response}")
            return []
    
    def _fix_and_parse_json(self, json_str: str) -> List[RefinedTranscriptChunk]:
        """Try to fix common JSON issues and parse"""
        try:
            import json
            import re
            
            logger.info("Attempting to fix malformed JSON")
            
            # Common fixes for malformed JSON
            fixed_json = json_str
            
            # Fix missing array brackets
            if not fixed_json.strip().startswith('['):
                fixed_json = '[' + fixed_json
            if not fixed_json.strip().endswith(']'):
                fixed_json = fixed_json + ']'
            
            # Fix missing object brackets
            # Look for patterns like "description": "text", "start_time": 1.0, "end_time": 2.0
            pattern = r'"description":\s*"[^"]*",\s*"start_time":\s*[0-9.]+,\s*"end_time":\s*[0-9.]+'
            matches = re.findall(pattern, fixed_json)
            
            if matches:
                # Reconstruct proper JSON
                objects = []
                for match in matches:
                    # Extract values
                    desc_match = re.search(r'"description":\s*"([^"]*)"', match)
                    start_match = re.search(r'"start_time":\s*([0-9.]+)', match)
                    end_match = re.search(r'"end_time":\s*([0-9.]+)', match)
                    
                    if desc_match and start_match and end_match:
                        obj = {
                            "description": desc_match.group(1),
                            "start_time": float(start_match.group(1)),
                            "end_time": float(end_match.group(1))
                        }
                        objects.append(obj)
                
                if objects:
                    logger.info(f"Fixed JSON and extracted {len(objects)} objects")
                    chunks = []
                    for obj in objects:
                        chunk = RefinedTranscriptChunk(
                            description=obj['description'],
                            start_time=obj['start_time'],
                            end_time=obj['end_time']
                        )
                        chunks.append(chunk)
                    return chunks
            
            # Try to parse the fixed JSON
            data = json.loads(fixed_json)
            chunks = []
            for item in data:
                if isinstance(item, dict) and 'description' in item and 'start_time' in item and 'end_time' in item:
                    chunk = RefinedTranscriptChunk(
                        description=item['description'],
                        start_time=float(item['start_time']),
                        end_time=float(item['end_time'])
                    )
                    chunks.append(chunk)
            
            logger.info(f"Successfully fixed and parsed {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to fix JSON: {e}")
            return []
    
    def _parse_simple_refinement_response(self, response: str) -> List[RefinedTranscriptChunk]:
        """Fallback parsing for simple text responses"""
        chunks = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and ('[' in line and ']' in line):
                # Extract timestamp and description
                timestamp_match = re.search(r'\[([0-9.]+)s - ([0-9.]+)s\]', line)
                if timestamp_match:
                    start_time = float(timestamp_match.group(1))
                    end_time = float(timestamp_match.group(2))
                    description = line[timestamp_match.end():].strip()
                    
                    if description:
                        chunk = RefinedTranscriptChunk(
                            description=description,
                            start_time=start_time,
                            end_time=end_time
                        )
                        chunks.append(chunk)
        
        return chunks
    
    def _save_llm_output(self, response: str, chunk_index: int):
        """Save LLM output to file for debugging"""
        try:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            llm_output_file = output_dir / "llm_output.txt"
            
            with open(llm_output_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"CHUNK {chunk_index} - LLM OUTPUT\n")
                f.write(f"{'='*80}\n")
                f.write(response)
                f.write(f"\n{'='*80}\n")
            
            logger.info(f"Saved LLM output for chunk {chunk_index} to llm_output.txt")
            
        except Exception as e:
            logger.error(f"Failed to save LLM output: {e}")
    
    def _merge_overlapping_chunks(self, all_chunks: List[RefinedTranscriptChunk]) -> List[RefinedTranscriptChunk]:
        """Merge overlapping chunks from different processing runs"""
        if not all_chunks:
            return []
        
        # Sort chunks by start time
        sorted_chunks = sorted(all_chunks, key=lambda x: x.start_time)
        merged_chunks = []
        
        for chunk in sorted_chunks:
            if not merged_chunks:
                merged_chunks.append(chunk)
                continue
            
            last_chunk = merged_chunks[-1]
            
            # Check if chunks overlap or are very close (within 2 seconds)
            if chunk.start_time <= last_chunk.end_time + 2.0:
                # Check if they're describing the same thing (similar content)
                if self._are_chunks_related(last_chunk.description, chunk.description):
                    # Merge the chunks
                    merged_chunk = RefinedTranscriptChunk(
                        description=f"{last_chunk.description} {chunk.description}".strip(),
                        start_time=last_chunk.start_time,
                        end_time=max(last_chunk.end_time, chunk.end_time)
                    )
                    merged_chunks[-1] = merged_chunk
                else:
                    merged_chunks.append(chunk)
            else:
                merged_chunks.append(chunk)
        
        return merged_chunks
    
    def _are_chunks_related(self, desc1: str, desc2: str) -> bool:
        """Check if two chunk descriptions are related (part of same defect)"""
        desc1_lower = desc1.lower()
        desc2_lower = desc2.lower()
        
        # Check for common defect patterns
        defect_keywords = ['tread', 'priority', 'crack', 'defect', 'top', 'bottom', 'front', 'rear']
        
        # If both contain defect keywords, they might be related
        desc1_has_defect = any(keyword in desc1_lower for keyword in defect_keywords)
        desc2_has_defect = any(keyword in desc2_lower for keyword in defect_keywords)
        
        if desc1_has_defect and desc2_has_defect:
            # Check for specific patterns that indicate they should be merged
            if ('priority' in desc1_lower and 'priority' not in desc2_lower) or \
               ('priority' in desc2_lower and 'priority' not in desc1_lower):
                return True
            
            if ('tread' in desc1_lower and 'tread' in desc2_lower):
                return True
        
        return False
    
    def _create_refinement_prompt(self) -> PromptTemplate:
        """Create the prompt template for transcript refinement"""
        template = """
        You are an expert at restructuring building inspection transcripts into semantically meaningful chunks.
        
        CRITICAL: You MUST return valid JSON format. Do not truncate or cut off your response.
        
        Given the following randomly chunked transcript segments, restructure them into meaningful chunks.
        
        CRITICAL RULES:
        1. If a defect is split across multiple segments, combine them into one chunk
        2. If multiple defects are mentioned in one segment, split them into separate chunks
        3. Preserve all conversation content - do not remove anything
        4. Estimate timestamps intelligently when splitting/combining
        5. CREATE MULTIPLE CHUNKS - Don't combine everything into one chunk
        6. Each chunk should represent ONE complete thought or defect
        
        EXAMPLES:
        
        Input (split defect):
        [00:54.860 --> 01:01.860] Tread number 6 priority.
        [01:01.860 --> 01:06.860] 1 top front crack.
        
        Output (combined):
        description: "Tread number 6 priority 1 top front crack"
        start_time: 54.86
        end_time: 66.86
        
        Input (multiple defects in one segment):
        [00:00.000 --> 00:06.480] Tread number 9 priority 2 bottom rear crack. Tread number 4 priority 1 top front crack.
        
        Output (split):
        description: "Tread number 9 priority 2 bottom rear crack"
        start_time: 0.0
        end_time: 3.24
        
        description: "Tread number 4 priority 1 top front crack"
        start_time: 3.24
        end_time: 6.48
        
        DEFECT PATTERNS TO RECOGNIZE:
        - "Tread number X priority Y [location] [type] crack"
        - "Tread X priority Y [description]"
        - Any mention of tread numbers, priorities, cracks, defects
        
        CONVERSATION TO PRESERVE:
        - Building/apartment mentions
        - General conversation
        - Screenshot mentions
        - Any other content
        
        Transcript segments:
        {transcript_segments}
        
        CRITICAL JSON FORMAT REQUIREMENTS:
        - Start with [ and end with ]
        - Each chunk must be a complete object with { and }
        - Use proper JSON syntax with commas between objects
        - Do not truncate or cut off the response
        - Complete ALL objects - do not leave incomplete JSON
        - Ensure every object has all three fields: description, start_time, end_time
        
        CORRECT FORMAT EXAMPLE:
        [
            {
                "description": "apartment 122",
                "start_time": 4.88,
                "end_time": 10.44
            },
            {
                "description": "tread number 11 priority one top front crack",
                "start_time": 15.0,
                "end_time": 20.0
            }
        ]
        
        IMPORTANT: Return MULTIPLE chunks in proper JSON array format. Each chunk should be a separate object with description, start_time, and end_time fields. Do not combine everything into a single chunk.
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["transcript_segments"]
        )
    
    def _create_simple_chunks(self, transcript_result: Dict[str, Any]) -> List[RefinedTranscriptChunk]:
        """Fallback simple chunking when LLM is not available"""
        chunks = []
        
        for segment in transcript_result.get('segments', []):
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            text = segment.get('text', '').strip()
            
            if text:
                chunks.append(RefinedTranscriptChunk(
                    description=text,
                    start_time=start_time,
                    end_time=end_time
                ))
        
        logger.info("Created %d simple transcript chunks", len(chunks))
        return chunks
    
    def extract_defects_using_llm(self, refined_chunks: List[RefinedTranscriptChunk]) -> List[DefectInfo]:
        """
        Extract defects from refined transcript chunks using LLM
        
        Args:
            refined_chunks: List of refined transcript chunks
            
        Returns:
            List of DefectInfo objects
        """
        if not self.llm:
            logger.warning("LLM not initialized, using rule-based defect extraction")
            return self._extract_defects_rule_based_from_chunks(refined_chunks)
        
        try:
            # Step 1: Filter relevant chunks
            relevant_chunks = self._filter_relevant_chunks(refined_chunks)
            logger.info("Filtered %d relevant chunks from %d total chunks", len(relevant_chunks), len(refined_chunks))
            
            # Save relevant chunks for debugging
            relevant_chunks_file = os.path.join("output", "relevant_chunks.json")
            with open(relevant_chunks_file, 'w', encoding='utf-8') as f:
                json.dump([chunk.model_dump() for chunk in relevant_chunks], f, indent=2)
            logger.info("Saved %d relevant chunks to relevant_chunks.json", len(relevant_chunks))
            
            # Step 2: Extract context (buildings/apartments) from all chunks
            self._extract_context_from_chunks(refined_chunks)
            
            # Step 3: Process chunks in batches
            all_defects = []
            batch_size = 15
            
            for i in range(0, len(relevant_chunks), batch_size):
                batch = relevant_chunks[i:i + batch_size]
                batch_defects = self._process_defect_batch(batch, i // batch_size + 1)
                all_defects.extend(batch_defects)
                
                logger.info("Processed defect batch %d/%d: %d chunks -> %d defects", 
                           i // batch_size + 1, 
                           (len(relevant_chunks) + batch_size - 1) // batch_size,
                           len(batch), len(batch_defects))
            
            logger.info("Extracted %d total defects from %d relevant chunks", len(all_defects), len(relevant_chunks))
            return all_defects
            
        except Exception as e:
            logger.error("Defect extraction failed: %s", e)
            return []
    
    def _extract_defects_rule_based_from_chunks(self, refined_chunks: List[RefinedTranscriptChunk]) -> List[DefectInfo]:
        """Rule-based defect extraction from refined chunks (fallback when LLM not available)"""
        defects = []
        
        # Extract context first
        self._extract_context_from_chunks(refined_chunks)
        
        # Filter relevant chunks
        relevant_chunks = self._filter_relevant_chunks(refined_chunks)
        logger.info("Using rule-based extraction on %d relevant chunks", len(relevant_chunks))
        
        for chunk in relevant_chunks:
            description = chunk.description.lower()
            
            # Look for defect patterns
            if any(keyword in description for keyword in ['tread', 'track', 'try', 'tri', 'tred', 'thread']):
                defect_info = self._extract_defect_from_chunk(chunk)
                if defect_info:
                    defect_info.building_counter = f"building{self.building_counter}" if self.building_counter > 0 else None
                    defect_info.building_name = self.current_building_name
                    defect_info.apartment_number = self.current_apartment_number
                    defect_info.ss_timestamp = self._calculate_screenshot_timestamp(defect_info.timestamp_start, defect_info.timestamp_end)
                    defects.append(defect_info)
        
        logger.info("Extracted %d defects using rule-based method", len(defects))
        return defects
    
    def _extract_defect_from_chunk(self, chunk: RefinedTranscriptChunk) -> Optional[DefectInfo]:
        """Extract defect information from a single refined chunk"""
        description = chunk.description.lower()
        
        # Extract tread number
        tread_patterns = [
            r'(?:tread|track|try|tri|tred|thread)\s+(?:number\s+)?(\d+)',
            r'(?:tread|track|try|tri|tred|thread)\s+(\d+)',
            r'(\d+)\s+(?:tread|track|try|tri|tred|thread)'
        ]
        
        tread_number = None
        for pattern in tread_patterns:
            match = re.search(pattern, description)
            if match:
                tread_number = match.group(1)
                break
        
        # Extract priority
        priority_patterns = [
            r'priority\s+(\d+)',
            r'priority\s+(one|two|three|four|five)',
            r'(\d+)\s+priority',
            r'(one|two|three|four|five)\s+priority'
        ]
        
        priority = None
        for pattern in priority_patterns:
            match = re.search(pattern, description)
            if match:
                priority_val = match.group(1)
                # Convert word to number
                word_to_num = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5'}
                priority = word_to_num.get(priority_val, priority_val)
                break
        
        # Extract description - improved pattern matching for fallback
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
        
        # Extract defect if we have any relevant information
        if tread_number or priority or defect_description:
            defect_info = DefectInfo(
                tread_number=tread_number,
                priority=priority,
                description=defect_description,
                timestamp_start=chunk.start_time,
                timestamp_end=chunk.end_time,
                transcript_segment=chunk.description
            )
            # Calculate screenshot timestamp
            defect_info.ss_timestamp = self._calculate_screenshot_timestamp(chunk.start_time, chunk.end_time)
            return defect_info
        
        return None
    
    def _filter_relevant_chunks(self, refined_chunks: List[RefinedTranscriptChunk]) -> List[RefinedTranscriptChunk]:
        """Filter chunks that contain relevant defect information"""
        relevant_keywords = [
            'building', 'apartments', 'apartment', 'department',
            'track', 'try', 'tri', 'tred', 'treads', 'thread', 'tread',
            'priority', 'one', 'two', 'three', '1', '2', '3',
            'rear', 'front', 'top', 'bottom', 'center',
            'crack', 'cracks', 'defect', 'defects', 'damage', 'to'
        ]
        
        relevant_chunks = []
        for chunk in refined_chunks:
            description_lower = chunk.description.lower()
            
            # Check if chunk contains any relevant keywords
            if any(keyword in description_lower for keyword in relevant_keywords):
                relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    def _extract_context_from_chunks(self, refined_chunks: List[RefinedTranscriptChunk]):
        """Extract building/apartment context from all chunks"""
        # Reset context
        self.building_counter = 0
        self.current_building_name = None
        self.current_apartment_number = None
        
        for chunk in refined_chunks:
            description = chunk.description.lower()
            
            # Extract building information
            if any(keyword in description for keyword in ['building', 'department']):
                building_info = self._extract_building_info(description)
                if building_info:
                    self.building_counter += 1
                    self.current_building_name = building_info.get('name')
                    logger.info("Building detected: %s -> building%d", chunk.description.strip(), self.building_counter)
            
            # Extract apartment number
            if 'apartment' in description:
                apartment_match = self._extract_apartment_number(description)
                if apartment_match:
                    self.current_apartment_number = apartment_match
                    logger.info("Apartment detected: %s", apartment_match)
    
    def _process_defect_batch(self, batch_chunks: List[RefinedTranscriptChunk], batch_number: int) -> List[DefectInfo]:
        """Process a batch of chunks to extract defects"""
        try:
            # Format batch chunks for LLM
            chunks_text = []
            for chunk in batch_chunks:
                chunks_text.append(f"[{chunk.start_time:.2f}s - {chunk.end_time:.2f}s] {chunk.description}")
            
            batch_text = "\n".join(chunks_text)
            
            # Create LLM prompt for defect extraction
            prompt = self._create_defect_extraction_prompt()
            
            formatted_prompt = prompt.format(transcript_chunks=batch_text)
            response = self.llm(formatted_prompt)
            
            # Parse the response manually
            defects = self._parse_defect_response(response)
            
            # Add context and calculate screenshot timestamps
            for defect in defects:
                defect.building_counter = f"building{self.building_counter}" if self.building_counter > 0 else None
                defect.building_name = self.current_building_name
                defect.apartment_number = self.current_apartment_number
                defect.ss_timestamp = self._calculate_screenshot_timestamp(defect.timestamp_start, defect.timestamp_end)
            
            return defects
            
        except Exception as e:
            logger.error("Failed to process defect batch %d: %s", batch_number, e)
            return []
    
    def _parse_defect_response(self, response: str) -> List[DefectInfo]:
        """Parse LLM response manually to extract DefectInfo objects"""
        try:
            import json
            import re
            
            # Try to extract JSON from the response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                defects = []
                for item in data:
                    if isinstance(item, dict):
                        defect = DefectInfo(
                            building_counter=item.get('building_counter'),
                            building_name=item.get('building_name'),
                            apartment_number=item.get('apartment_number'),
                            tread_number=item.get('tread_number'),
                            priority=item.get('priority'),
                            description=item.get('description'),
                            timestamp_start=item.get('timestamp_start'),
                            timestamp_end=item.get('timestamp_end'),
                            transcript_segment=item.get('transcript_segment')
                        )
                        defects.append(defect)
                
                return defects
            
            # Fallback: try to parse as simple text
            return self._parse_simple_defect_response(response)
            
        except Exception as e:
            logger.error("Failed to parse defect response: %s", e)
            return []
    
    def _parse_simple_defect_response(self, response: str) -> List[DefectInfo]:
        """Fallback parsing for simple text responses"""
        defects = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and ('tread' in line.lower() or 'priority' in line.lower() or 'crack' in line.lower()):
                # Extract basic information using regex
                tread_match = re.search(r'tread\s+(?:number\s+)?(\d+)', line.lower())
                priority_match = re.search(r'priority\s+(one|two|three|four|five|\d+)', line.lower())
                description_match = re.search(r'(top|bottom|front|rear|center)\s+(rear|front|top|bottom|center)?\s*(crack|cracks)', line.lower())
                
                tread_number = tread_match.group(1) if tread_match else None
                priority = priority_match.group(1) if priority_match else None
                description = description_match.group(0) if description_match else None
                
                if tread_number or priority or description:
                    defect = DefectInfo(
                        tread_number=tread_number,
                        priority=priority,
                        description=description,
                        transcript_segment=line
                    )
                    defects.append(defect)
        
        return defects
    
    def _calculate_screenshot_timestamp(self, start_time: float, end_time: float) -> float:
        """Calculate optimal timestamp for taking screenshot"""
        if start_time is None or end_time is None:
            return None
        
        # Use middle of the timestamp range for screenshot
        return (start_time + end_time) / 2
    
    def _create_defect_extraction_prompt(self) -> PromptTemplate:
        """Create the prompt template for defect extraction"""
        template = """
        You are an expert at extracting structured defect information from building inspection transcript chunks.
        
        Given the following transcript chunks, extract defect information with exact timestamps.
        
        EXTRACTION RULES:
        1. Only extract chunks that contain defect information (tread numbers, priorities, cracks, etc.)
        2. Use EXACT timestamps as provided in the input
        3. Extract tread numbers (may be written as "tread 9", "tread number 9", "track 9", "try 9", etc.)
        4. Extract priorities (may be "priority 1", "priority one", "priority 2", "priority two", etc.)
        5. DEFECT DESCRIPTION EXTRACTION - THIS IS THE MOST IMPORTANT PART:
           - Look for defect descriptions like "top rear crack", "bottom front crack", "top center crack", etc.
           - The description should be clean and concise (e.g., "top rear crack" not "priority one top rear crack")
           - Extract ONLY the defect description, not the entire sentence
           - Common patterns: "top rear crack", "bottom front crack", "top center crack", "bottom rear crack", "front crack", "rear crack"
           - If you see "top front rear crack" or similar, extract as "top front rear crack"
           - CRITICAL: If the entire chunk is about a defect (like "tread number eight priority one top rear crack"), 
             extract the defect description part ("top rear crack") and ignore the tread/priority information
           - NEVER leave description as null if there's a crack/defect mentioned in the chunk
        6. Handle misspellings intelligently (thread -> tread, tred -> tread, etc.)
        7. ALWAYS extract the defect description - it should contain the location (top/bottom/front/rear) and type (crack/defect)
        
        TIMESTAMP REQUIREMENTS:
        - Use EXACT start_time and end_time from input chunks
        - Do not modify or estimate timestamps
        - timestamp_start and timestamp_end should match the input exactly
        
        PRIORITY EXTRACTION:
        - Convert word priorities to numbers: "one" -> "1", "two" -> "2", "three" -> "3"
        - Keep numeric priorities as strings: "1", "2", "3"
        
        EXAMPLES:
        
        Input:
        [552.96s - 561.88s] Track number 9 priority 2 top rear crack.
        
        Output:
        {{
            "tread_number": "9",
            "priority": "2", 
            "description": "top rear crack",
            "timestamp_start": 552.96,
            "timestamp_end": 561.88,
            "transcript_segment": "Track number 9 priority 2 top rear crack."
        }}
        
        Input:
        [24.54s - 37.30s] Track, tread number 10, priority one, top rear crack, screenshot.
        
        Output:
        {{
            "tread_number": "10",
            "priority": "1",
            "description": "top rear crack",
            "timestamp_start": 24.54,
            "timestamp_end": 37.30,
            "transcript_segment": "Track, tread number 10, priority one, top rear crack, screenshot."
        }}
        
        IMPORTANT: If you see a defect mentioned but cannot extract a clean description, 
        still include the defect but try to extract the best possible description from the context.
        
        Input:
        [265.12s - 277.52s] tread number eight priority one top rear crack
        
        Output:
        {{
            "tread_number": "8",
            "priority": "1",
            "description": "top rear crack",
            "timestamp_start": 265.12,
            "timestamp_end": 277.52,
            "transcript_segment": "tread number eight priority one top rear crack"
        }}
        
        Input:
        [1021.78s - 1025.46s] fourteen priority to top center cracks
        
        Output:
        {{
            "tread_number": "14",
            "priority": null,
            "description": "top center cracks", 
            "timestamp_start": 1021.78,
            "timestamp_end": 1025.46,
            "transcript_segment": "fourteen priority to top center cracks"
        }}
        
        Transcript chunks:
        {transcript_chunks}
        
        Please provide the extracted defects in JSON format as an array of objects with the required fields.
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["transcript_chunks"]
        )
    
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
        
        # Step 4: Create refined transcript chunks
        refined_chunks = self.create_refined_transcript_chunks(transcript_result)
        
        # Save refined transcript chunks
        refined_chunks_file = os.path.join(output_dir, "refined_transcript_chunks.json")
        with open(refined_chunks_file, 'w', encoding='utf-8') as f:
            json.dump([chunk.model_dump() for chunk in refined_chunks], f, indent=2)
        
        # Step 5: Extract defects using refined chunks
        defects = self.extract_defects_using_llm(refined_chunks)
        
        # Save defects for debugging
        defects_file = os.path.join(output_dir, "extracted_defects.json")
        with open(defects_file, 'w', encoding='utf-8') as f:
            json.dump([defect.model_dump() for defect in defects], f, indent=2)
        
        logger.info("Processing completed. Found %d defects.", len(defects))
        
        # Step 6: Generate screenshots and CSV report
        if defects:
            try:
                from screenshot_generator import generate_defect_report
                csv_path = generate_defect_report(defects, video_path, output_dir)
                logger.info(f"Defect report with screenshots generated: {csv_path}")
            except Exception as e:
                logger.error(f"Failed to generate screenshot report: {e}")
        
        return defects


def main():
    """Main function to demonstrate usage"""
    # Initialize processor
    api_key = os.getenv("OPENAI_API_KEY")
    print("API Key: ", api_key)
    
    if not api_key:
        print("⚠️  WARNING: OPENAI_API_KEY not set!")
        print("   The system will use simple chunking instead of LLM processing.")
        print("   For better results, set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print()
    
    processor = VideoProcessor(
        whisper_model_name="base.en",
        openai_api_key=api_key
    )
    
    # Example usage with local video
    video_source = {
        "type": "local",
        "path": "/Users/consultadd/Desktop/My project/consultadd_project/input/sample.mp4"
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
            timestamp_start = f"{defect.timestamp_start:.2f}" if defect.timestamp_start is not None else "None"
            timestamp_end = f"{defect.timestamp_end:.2f}" if defect.timestamp_end is not None else "None"
            ss_timestamp = f"{defect.ss_timestamp:.2f}" if defect.ss_timestamp is not None else "None"
            
            print(f"  Timestamp: {timestamp_start}s - {timestamp_end}s")
            print(f"  Screenshot Time: {ss_timestamp}s")
            print(f"  Transcript: {defect.transcript_segment}")
            
    except Exception as e:
        logger.error("Processing failed: %s", e)
        raise


if __name__ == "__main__":
    main()
