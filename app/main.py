from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.agent import process_task
from app.llm_agent import _quota_block_active
import logging
import os
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env if present (useful when starting via uvicorn directly)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; continue if unavailable
    pass

app = FastAPI(
    title="Data Analyst Agent API v2.0",
    description="An API that uses LLMs to source, prepare, analyze, and visualize any data",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static and templates directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")


def _llm_env_ok() -> bool:
    """Return True if LLM can be safely used given current env vars.

    Rules:
    - Standard OpenAI key that starts with 'sk-' is OK with default api.openai.com
    - Azure: require OPENAI_API_TYPE=azure (or azure_openai) and AZURE_OPENAI_ENDPOINT
    - Custom gateways: allow if OPENAI_BASE_URL is set
    """
    key = os.getenv('OPENAI_API_KEY', '') or ''
    if not key:
        return False
    base_url = os.getenv('OPENAI_BASE_URL', '') or ''
    api_type = (os.getenv('OPENAI_API_TYPE', '') or '').lower()
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', '') or ''
    if key.startswith('sk-'):
        return True
    if api_type in {"azure", "azure_openai"} and azure_endpoint:
        return True
    if base_url:
        return True
    return False

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/", response_class=HTMLResponse)
async def api_docs(request: Request):
    """Serve API documentation page"""
    return templates.TemplateResponse("api_docs.html", {"request": request})

@app.post("/api/")
async def analyze(file: UploadFile = File(...), use_llm: bool = Form(True)):
    """
    Analyze data based on the provided question file.

    Args:
        file: A text file containing the analysis question/task
        use_llm: Whether to use LLM for analysis (default: True)

    Returns:
        JSON response with analysis results
    """
    try:
        # Validate file type
        if not file.filename.endswith('.txt'):
            raise HTTPException(status_code=400, detail="File must be a .txt file")

        # Read file content
        content = (await file.read()).decode("utf-8")

        if not content.strip():
            raise HTTPException(status_code=400, detail="File is empty")

        # Respect user's choice but disable LLM if env is not correctly configured
        effective_use_llm = bool(use_llm) and _llm_env_ok()
        logger.info(f"Processing task: {content[:100]}...")
        logger.info(f"Using LLM: {effective_use_llm}")

        # Process the task
        result = process_task(content, use_llm=effective_use_llm)

        logger.info("Task processed successfully")

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /api/: {e}")
        return JSONResponse(
            content={"error": f"Internal server error: {str(e)}"},
            status_code=500
        )

@app.post("/api/analyze")
async def analyze_text(request: Request):
    """
    Analyze data based on text input (for web interface)
    """
    try:
        data = await request.json()
        task = data.get("task", "")
        url = data.get("url", "")
        requested_use_llm = data.get("use_llm", None)
        # Default to env capability when not explicitly provided; always AND with env check
        if requested_use_llm is None:
            use_llm = _llm_env_ok()
        else:
            use_llm = bool(requested_use_llm) and _llm_env_ok()
        
        if not task.strip():
            raise HTTPException(status_code=400, detail="Task is required")

        # Combine URL and task if provided
        full_task = f"Scrape data from {url}\n\n{task}" if url else task

        logger.info(f"Processing web task: {full_task[:100]}...")
        logger.info(f"Using LLM: {use_llm}")

        # Process the task
        result = process_task(full_task, use_llm=use_llm)

        logger.info("Web task processed successfully")

        return JSONResponse(content={"result": result, "success": True})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /api/analyze: {e}")
        return JSONResponse(
            content={"error": f"Internal server error: {str(e)}", "success": False},
            status_code=500
        )

@app.post("/api/llm/analyze")
async def llm_analyze(request: Request):
    """
    LLM-specific analysis endpoint
    """
    try:
        data = await request.json()
        task = data.get("task", "")
        url = data.get("url", "")
        
        if not task.strip():
            raise HTTPException(status_code=400, detail="Task is required")
        
        # Check if OpenAI API is properly configured
        if not _llm_env_ok():
            raise HTTPException(
                status_code=500, 
                detail=(
                    "LLM not configured correctly. Provide a valid OPENAI_API_KEY (sk-...) for OpenAI, "
                    "or set OPENAI_API_TYPE=azure with AZURE_OPENAI_ENDPOINT for Azure, or set OPENAI_BASE_URL for a gateway."
                )
            )
        
        # Combine URL and task if provided
        full_task = f"Scrape data from {url}\n\n{task}" if url else task
        
        logger.info(f"Processing LLM task: {full_task[:100]}...")
        
        # Process the task with LLM
        result = process_task(full_task, use_llm=True)
        
        logger.info("LLM task processed successfully")
        
        return JSONResponse(content={"result": result, "success": True, "method": "llm"})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /api/llm/analyze: {e}")
        return JSONResponse(
            content={"error": f"Internal server error: {str(e)}", "success": False},
            status_code=500
        )

@app.post("/api/traditional/analyze")
async def traditional_analyze(request: Request):
    """
    Traditional analysis endpoint (without LLM)
    """
    try:
        data = await request.json()
        task = data.get("task", "")
        url = data.get("url", "")
        
        if not task.strip():
            raise HTTPException(status_code=400, detail="Task is required")
        
        # Combine URL and task if provided
        full_task = f"Scrape data from {url}\n\n{task}" if url else task
        
        logger.info(f"Processing traditional task: {full_task[:100]}...")
        
        # Process the task without LLM
        result = process_task(full_task, use_llm=False)
        
        logger.info("Traditional task processed successfully")
        
        return JSONResponse(content={"result": result, "success": True, "method": "traditional"})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /api/traditional/analyze: {e}")
        return JSONResponse(
            content={"error": f"Internal server error: {str(e)}", "success": False},
            status_code=500
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Data Analyst Agent v2.0",
        "llm_available": _llm_env_ok() and not _quota_block_active(),
        "features": {
            "llm_analysis": _llm_env_ok() and not _quota_block_active(),
            "traditional_analysis": True,
            "web_scraping": True,
            "data_visualization": True
        }
    }

@app.get("/api/health")
async def api_health():
    """API health check endpoint"""
    llm_available = _llm_env_ok()
    return {
        "status": "healthy", 
        "version": "2.0.0", 
        "service": "Data Analyst Agent API",
        "llm_available": llm_available
    }

@app.get("/api/config")
async def get_config():
    """Get API configuration"""
    return {
    "llm_available": _llm_env_ok() and not _quota_block_active(),
        "supported_data_sources": ["Wikipedia", "CSV", "S3 Parquet"],
        "supported_analysis_types": ["statistical", "visualization", "correlation", "regression"],
        "max_file_size": "10MB",
        "timeout": "3 minutes"
    }