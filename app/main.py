from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.agent import process_task
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Data Analyst Agent API",
    description="An API that uses LLMs to source, prepare, analyze, and visualize any data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Data Analyst Agent API is running", "status": "healthy"}

@app.post("/api/")
async def analyze(file: UploadFile = File(...)):
    """
    Analyze data based on the provided question file.
    
    Args:
        file: A text file containing the analysis question/task
        
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
        
        logger.info(f"Processing task: {content[:100]}...")
        
        # Process the task
        result = process_task(content)
        
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Data Analyst Agent"}