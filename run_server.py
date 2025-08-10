import os
import sys

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; continue if unavailable
    pass

# Import uvicorn with helpful error if missing
try:
    import uvicorn
except ImportError:
    print("❌ Missing dependency: uvicorn")
    print("   Install dependencies with: pip install -r requirements.txt")
    sys.exit(1)

from app.main import app

def check_llm_availability():
    """Check if LLM features are available and correctly configured."""
    key = os.getenv('OPENAI_API_KEY', '') or ''
    base_url = os.getenv('OPENAI_BASE_URL', '') or ''
    api_type = (os.getenv('OPENAI_API_TYPE', '') or '').lower()
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', '') or ''
    if not key:
        print("⚠️  Warning: OPENAI_API_KEY not set")
        print("   LLM features will be disabled")
        print("   Set OPENAI_API_KEY environment variable to enable LLM features")
        print("   Get your API key from: https://platform.openai.com/api-keys")
        return False
    if key.startswith('sk-'):
        print("✅ OpenAI key detected - LLM enabled")
        return True
    if api_type in {"azure", "azure_openai"} and azure_endpoint:
        print("✅ Azure OpenAI configuration detected - LLM enabled")
        return True
    if base_url:
        print("✅ Custom OpenAI gateway detected via OPENAI_BASE_URL - LLM enabled")
        return True
    print("⚠️  LLM configuration looks invalid (non sk- key without OPENAI_BASE_URL or Azure settings). Disabling LLM.")
    return False

def main():
    """Main function to start the server"""
    print("🚀 Starting Data Analyst Agent v3.0")
    print("=" * 50)
    
    # Check LLM availability
    llm_available = check_llm_availability()
    
    # Get configuration
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 8000))
    log_level = os.getenv('LOG_LEVEL', 'info')
    
    print(f"🌐 Server will start on: http://{host}:{port}")
    print(f"📊 Web Interface: http://{host}:{port}/")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔍 Health Check: http://{host}:{port}/health")
    
    if llm_available:
        print("🤖 LLM-powered analysis: Enabled")
    else:
        print("🧠 Traditional analysis: Enabled")
    
    print("\n" + "=" * 50)
    
    try:
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=True,
            log_level=log_level
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()