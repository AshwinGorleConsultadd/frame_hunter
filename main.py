# import os
# import subprocess
# import whisper

# # Input paths
# input_folder = "input"
# audio_output_folder = "audio_folder"
# video_file = "part000.mp4"
# video_path = os.path.join(input_folder, video_file)
# audio_path = os.path.join(audio_output_folder, "sample_audio.mp3")

# # Extract audio using ffmpeg
# subprocess.run([
#     "ffmpeg", "-i", video_path, "-vn", "-acodec", "mp3", audio_path, "-y"
# ])
# print(f"✅ Extracted audio: {audio_path}")

# # Load Whisper model (options: tiny, base, small, medium, large)
# model = whisper.load_model("base")  # "tiny" is fastest, "large" is most accurate

# # Transcribe audio
# result = model.transcribe(audio_path, verbose=True)

# # Print transcript with timestamps
# for segment in result["segments"]:
#     start = segment["start"]
#     end = segment["end"]
#     text = segment["text"]
#     print(f"[{start:.2f}s - {end:.2f}s] {text}")

import os
import subprocess
import whisper

# Input paths
input_folder = "input"
audio_output_folder = "audio_folder"
video_file = "part000.mp4"gti
video_path = os.path.join(input_folder, video_file)
audio_path = os.path.join(audio_output_folder, "sample_audio.mp3")

# Ensure audio output folder exists
os.makedirs(audio_output_folder, exist_ok=True)
os.makedirs(input_folder, exist_ok=True)

# Extract audio using ffmpeg
subprocess.run([
    "ffmpeg", "-i", video_path, "-vn", "-acodec", "mp3", audio_path, "-y"
])
print(f"✅ Extracted audio: {audio_path}")

# Load Whisper model (options: tiny, base, small, medium, large)
model = whisper.load_model("turbo")  # "tiny" is fastest, "large" is most accurate

# Transcribe audio
result = model.transcribe(audio_path, verbose=True, language="en")

# Output transcript file path
transcript_file = os.path.splitext(video_file)[0] + "_transcript.txt"
transcript_path = os.path.join(input_folder, transcript_file)

# Write transcript to file
with open(transcript_path, "w", encoding="utf-8") as f:
    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        line = f"[{start:.2f}s - {end:.2f}s] {text}\n"
        print(line.strip())   # print to console
        f.write(line)         # write to file

print(f"📝 Transcript saved to: {transcript_path}")
