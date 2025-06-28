#!/usr/bin/env python3
"""
Setup script for the distributed indexing system.
"""

import os
import sys
import shutil
from pathlib import Path


def create_env_file():
    """Create .env file from template if it doesn't exist."""
    env_file = Path(".env")
    example_env_file = Path("env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if example_env_file.exists():
        shutil.copy(example_env_file, env_file)
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file and set your GEMINI_API_KEY")
        return True
    else:
        print("❌ env.example file not found")
        return False


def create_directories():
    """Create necessary directories."""
    directories = [
        "logs",
        "data",
        "uploads",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import pydantic
        import google.generativeai
        import qdrant_client
        print("✅ All core dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False


def validate_environment():
    """Validate environment configuration."""
    # Validate environment before setup
    try:
        from config.config import validate_environment, print_configuration
        validate_environment()
        print_configuration()
    except ImportError as e:
        print(f"Warning: Could not import config validation: {e}")
    except Exception as e:
        print(f"Warning: Environment validation failed: {e}")


def main():
    """Main setup function."""
    print("🚀 Setting up Distributed Indexing System")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Create .env file
    print("\n🔧 Setting up environment...")
    if not create_env_file():
        sys.exit(1)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Validate environment
    print("\n✅ Setup completed!")
    print("\n📋 Next steps:")
    print("1. Edit .env file and set your GEMINI_API_KEY")
    print("2. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
    print("3. Run tests: python test_system.py")
    print("4. Start API: uvicorn core.api.main:app --reload")
    print("5. Run demo: python example_usage.py")
    
    print("\n🔗 Useful links:")
    print("- Gemini API Key: https://makersuite.google.com/app/apikey")
    print("- Qdrant Documentation: https://qdrant.tech/documentation/")
    print("- FastAPI Documentation: https://fastapi.tiangolo.com/")


if __name__ == "__main__":
    main() 