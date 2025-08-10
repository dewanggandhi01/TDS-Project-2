#!/usr/bin/env python3
"""
Setup script for Data Analyst Agent v3.0
Helps users configure their environment and test the LLM integration
"""

import os
import sys
import subprocess
import json

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; continue if unavailable
    pass

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    # Map pip package names to their importable module names
    packages = {
        'fastapi': 'fastapi',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'beautifulsoup4': 'bs4',              # pip -> module bs4
        'duckdb': 'duckdb',
        'requests': 'requests',
        'openai': 'openai',
    'python-dotenv': 'dotenv',            # load .env in scripts and app
        'jinja2': 'jinja2',                   # required by FastAPI templating
        'uvicorn': 'uvicorn',                 # used by run_server.py
        'python-multipart': 'multipart',      # required for UploadFile/Form parsing
    }

    missing = []

    for pip_name, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"✅ {pip_name} is installed")
        except ImportError:
            missing.append(pip_name)
            print(f"❌ {pip_name} is missing")

    if missing:
        print(f"\n📦 Installing missing packages: {', '.join(missing)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
            print("✅ All dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False

    return True

def check_openai_key():
    """Check if OpenAI API key is configured"""
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("\n🔧 To set up your OpenAI API key:")
        print("1. Get your API key from: https://platform.openai.com/api-keys")
        print("2. Set the environment variable:")
        print("   - Windows PowerShell: $env:OPENAI_API_KEY = 'your_key_here'")
        print("   - Windows CMD:        set OPENAI_API_KEY=your_key_here")
        print("   - Linux/Mac:          export OPENAI_API_KEY=your_key_here")
        print("3. Or create a .env file with: OPENAI_API_KEY=your_key_here")
        return False
    
    print("✅ OPENAI_API_KEY is configured")
    
    # Test the API key
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        _ = client.models.list()
        print("✅ OpenAI API key is valid")
        return True
    except Exception as e:
        print(f"❌ OpenAI API key test failed: {e}")
        return False

def is_server_running() -> bool:
    """Return True if the local server responds to /health."""
    try:
        import requests
        resp = requests.get("http://127.0.0.1:8000/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False

def test_server():
    """Test if the server can start"""
    print("\n🚀 Testing server startup...")
    
    try:
        # Try to import the app
        from app.main import app
        print("✅ FastAPI app imported successfully")
        
        # Test health endpoint (if server is running)
        try:
            import requests
            response = requests.get("http://127.0.0.1:8000/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print("✅ Server is running")
                print(f"   Service: {data.get('service')}")
                print(f"   LLM Available: {data.get('llm_available')}")
                return True
        except requests.exceptions.ConnectionError:
            print("ℹ️  Server is not running (this is normal)")
            print("   Start the server with: python run_server.py")
            return True
            
    except Exception as e:
        print(f"❌ Server test failed: {e}")
        return False

def run_tests():
    """Run the test suite"""
    print("\n🧪 Running tests...")
    
    try:
        result = subprocess.run([sys.executable, 'test_api.py'], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Tests passed")
            return True
        else:
            print("❌ Tests failed")
            print("Output:", result.stdout)
            print("Errors:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Tests timed out")
        return False
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🔧 Data Analyst Agent v3.0 Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check OpenAI API key
    llm_available = check_openai_key()
    
    # Test server
    if not test_server():
        return False
    
    # Run tests only if the server is already running to avoid false negatives
    if is_server_running():
        if not run_tests():
            return False
    else:
        print("\nℹ️  Skipping tests because the server is not running.")
        print("   Start the server in another terminal: python run_server.py")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Start the server: python run_server.py")
    print("2. Open the web interface: http://127.0.0.1:8000/")
    print("3. View API documentation: http://127.0.0.1:8000/docs")
    
    if llm_available:
        print("4. Try LLM-powered analysis with natural language queries!")
    else:
        print("4. Traditional analysis is available (LLM features require API key)")
    
    print("\n📚 For more information, see README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 