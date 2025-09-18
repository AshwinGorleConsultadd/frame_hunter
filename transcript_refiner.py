#!/usr/bin/env python3
"""
Thread Defect Processor - Clean Solution using OpenAI Function Calling

This script processes defect arrays to ensure each thread defect is in its own object.
Uses OpenAI's function calling for accurate and reliable processing.

Usage: Just paste your defects data into the defects variable below and run!
"""

import os
from openai import OpenAI
import json

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===========================================
# LOAD DEFECTS DATA FROM INPUT.JSON
# ===========================================
def load_defects_data():
    """Load defects data from input.json file."""
    try:
        with open('input.json', 'r') as f:
            defects_data = json.load(f)
        print(f"✅ Loaded {len(defects_data)} defects from input.json")
        return defects_data
    except FileNotFoundError:
        print("❌ Error: input.json not found!")
        print("Please make sure input.json is in the current directory.")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in input.json - {e}")
        return []
    except Exception as e:
        print(f"❌ Error loading input.json: {e}")
        return []

# ===========================================
# MANUAL DEFECTS DATA (Alternative to loading from file)
# ===========================================

# Function schema - only keep description, start_time, end_time
tools = [
    {
        "type": "function",
        "function": {
            "name": "normalize_defects",
            "description": "Ensure one defect per object, splitting or merging as needed",
            "parameters": {
                "type": "object",
                "properties": {
                    "defects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "start_time": {"type": "number"},
                                "end_time": {"type": "number"},
                                "tread_number": {"type": "integer"},
                            },
                            "required": ["description", "start_time", "end_time", "tread_number"]
                        }
                    }
                },
                "required": ["defects"]
            }
        }
    }
]

def process_defects_with_openai(defects_data):
    """Process defects using OpenAI function calling."""
    print(f"🔧 Processing {len(defects_data)} defects using OpenAI...")
    
    # Process in overlapping batches for better context handling
    batch_size = 18  # Process 18 defects at a time
    overlap_size = 1  # 1 object overlap between batches
    all_processed_defects = []
    
    i = 0
    batch_num = 1
    while i < len(defects_data):
        # Calculate batch end with overlap
        batch_end = min(i + batch_size, len(defects_data))
        batch = defects_data[i:batch_end]
        
        print(f"  Processing batch {batch_num} ({len(batch)} defects, starting from index {i})...")
        
        # Send batch to GPT for restructuring
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": """You are an assistant that cleans up defect transcripts so each object describes exactly one defect with accurate timestamps.

CRITICAL RULES:
1. Each object must contain exactly ONE thread defect
2. If a description mentions multiple thread defects, split them into separate objects
3. Calculate accurate timestamps by dividing the time range equally among multiple defects
4. Keep the original description format but make it clear and concise
5. Preserve all thread numbers, priorities, locations, and defect types
6. If a description has no thread defects, keep it as-is
7. Maintain the order of defects as much as possible

SCATTERED THREAD INFORMATION HANDLING:
- Look for thread information that is split across multiple consecutive objects
- Merge incomplete thread information into complete thread defects
- Examples of scattered information:
  * "Tread number 16." + "Priority 2." + "Bottom rear crack." → Merge into "Tread number 16, Priority 2, Bottom rear crack"
  * "Tread number 17." + "Priority one." + "Bottom rear crack." → Merge into "Tread number 17, Priority one, Bottom rear crack"
  * "Thread 5." + "Priority 1." + "Top front crack." → Merge into "Thread 5, Priority 1, Top front crack"

MERGING RULES:
- If you see a tread/thread number followed by priority and defect type in separate objects, merge them
- Combine the timestamps by taking the start_time of the first object and end_time of the last object
- Create a single complete description combining all the scattered information
- Only merge objects that are clearly part of the same thread defect

Examples:
- "tread number 3 priority 1 top rear crack tread number 4" → Split into 2 objects
- "Track, tread number 10, priority one, top rear crack, screenshot" → Keep as 1 object
- "Tread number 18. Priority one. Bottom rear crack. Tread number 17." → Split into 2 objects
- "This is apartment 122." → Keep as 1 object (no thread defects)
- "Tread number 16." + "Priority 2." + "Bottom rear crack." → Merge into 1 complete object"""
                },
                {
                    "role": "user", 
                    "content": f"""Clean and normalize these defects so each object contains only one complete defect. 

CRITICAL: Look for thread information scattered across consecutive objects and merge them into complete thread defects.
Pay special attention to cases where tread number, priority, and defect type are in separate consecutive objects.

Process ALL defects in the array:

{json.dumps(batch, indent=2)}"""
                }
            ],
            tools=tools,
            tool_choice="auto"
        )
        
        # Extract structured output
        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        
        # Handle overlap - avoid duplicates
        if batch_num == 1:
            all_processed_defects.extend(args["defects"])
        else:
            # Skip the first defect if it's an overlap
            all_processed_defects.extend(args["defects"][1:])
        
        print(f"    ✅ Processed {len(args['defects'])} defects from this batch")
        
        # Move to next batch with overlap
        i += batch_size - overlap_size
        batch_num += 1
    
    return all_processed_defects

def refine_transcript_chunks_symmentically(defects_data, api_key=None, save_to_file=None, verbose=True):
    """
    Process thread defects using OpenAI function calling.
    
    Args:
        defects_data (list): List of defect objects with description, start_time, end_time
        api_key (str, optional): OpenAI API key. If None, uses OPENAI_API_KEY environment variable
        save_to_file (str, optional): File path to save results. If None, doesn't save
        verbose (bool): Whether to print progress information
    
    Returns:
        list: Processed defects with normalized thread information
    """
    if verbose:
        print("🔧 Thread Defect Processor - OpenAI Function Calling")
        print("=" * 60)
    
    # Set API key
    if api_key:
        client.api_key = api_key
    elif not os.getenv("OPENAI_API_KEY"):
        if verbose:
            print("❌ Error: OPENAI_API_KEY environment variable not set!")
            print("Please set your OpenAI API key or pass it as parameter")
        return None
    
    if not defects_data:
        if verbose:
            print("❌ No data to process. Exiting.")
        return []
    
    if verbose:
        print(f"📊 Processing {len(defects_data)} defects...")
    
    # Process defects using OpenAI function calling
    try:
        processed_defects = process_defects_with_openai(defects_data)
        
        if verbose:
            print(f"\n✅ Processed {len(processed_defects)} defects")
        
        # Save to file if requested
        if save_to_file:
            with open(save_to_file, 'w') as f:
                json.dump(processed_defects, f, indent=2)
            if verbose:
                print(f"💾 Results saved to: {save_to_file}")
        
        if verbose:
            # Show summary
            thread_defects = [d for d in processed_defects if d.get('tread_number', 0) > 0]
            print(f"\n🎉 Summary: {len(defects_data)} input → {len(processed_defects)} output")
            print(f"📊 Found {len(thread_defects)} thread defects")
        
        return processed_defects
        
    except Exception as e:
        if verbose:
            print(f"❌ Error processing defects: {e}")
            print("Make sure your OpenAI API key is valid and has sufficient credits.")
        return None
