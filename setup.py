"""
Setup script for the video processing project
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    print("🔍 Checking FFmpeg installation...")
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is installed")
            return True
        else:
            print("❌ FFmpeg not found")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg not found")
        return False

def install_ffmpeg_macos():
    """Install FFmpeg on macOS using Homebrew"""
    print("🍺 Installing FFmpeg using Homebrew...")
    return run_command("brew install ffmpeg", "FFmpeg installation")

def install_ffmpeg_ubuntu():
    """Install FFmpeg on Ubuntu/Debian"""
    print("📦 Installing FFmpeg using apt...")
    return run_command("sudo apt update && sudo apt install -y ffmpeg", "FFmpeg installation")

def setup_python_environment():
    """Set up Python virtual environment and install dependencies"""
    print("🐍 Setting up Python environment...")
    
    # Create virtual environment
    if not Path("venv").exists():
        if not run_command(f"{sys.executable} -m venv venv", "Virtual environment creation"):
            return False
    
    # Determine activation script
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate"
        pip_command = "venv\\Scripts\\pip"
    else:  # Unix-like
        activate_script = "source venv/bin/activate"
        pip_command = "venv/bin/pip"
    
    # Install dependencies
    if not run_command(f"{pip_command} install -r requirements.txt", "Python dependencies installation"):
        return False
    
    print("✅ Python environment setup completed")
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    
    directories = ["input", "output", "audio_folder"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up Video Processing Project")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Check FFmpeg
    ffmpeg_installed = check_ffmpeg()
    if not ffmpeg_installed:
        print("\n📋 FFmpeg installation required:")
        print("   macOS: brew install ffmpeg")
        print("   Ubuntu/Debian: sudo apt install ffmpeg")
        print("   Windows: Download from https://ffmpeg.org/download.html")
        
        # Try to install automatically on macOS
        if sys.platform == "darwin":
            try:
                subprocess.run(['which', 'brew'], check=True, capture_output=True)
                if install_ffmpeg_macos():
                    ffmpeg_installed = True
            except subprocess.CalledProcessError:
                print("   Homebrew not found. Please install FFmpeg manually.")
    
    # Setup Python environment
    if not setup_python_environment():
        print("❌ Python environment setup failed")
        return False
    
    # Create directories
    if not create_directories():
        print("❌ Directory creation failed")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Place your test video in the 'input' directory")
    print("2. Set OPENAI_API_KEY environment variable (optional)")
    print("3. Run: python test_script.py")
    print("4. Run: python main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
