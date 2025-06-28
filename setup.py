import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📋 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "data/pdfs",
        "data/texts",
        "models",
        "models/embedding",
        "models/llm"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created successfully!")

def main():
    """Main setup function"""
    print("🚀 RAG PDF Application Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        return False
    
    # Download models
    if not run_command("python install_models.py", "Downloading AI models"):
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo run the application:")
    print("streamlit run app.py")
    
    return True

if __name__ == "__main__":
    main()