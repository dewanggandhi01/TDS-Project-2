from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Data Analyst Agent API v2.0 - Lightweight",
    description="A lightweight API for Vercel deployment",
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

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Data Analyst Agent API v2.0",
        "status": "running",
        "endpoints": [
            "/health",
            "/api/status",
            "/docs"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0.0"}

@app.get("/api/status")
async def api_status():
    """API status endpoint"""
    return {
        "api": "Data Analyst Agent",
        "version": "2.0.0",
        "status": "operational",
        "environment": "production"
    }

@app.post("/api/analyze")
async def analyze_data():
    """Lightweight data analysis endpoint"""
    return {
        "message": "Analysis endpoint - lightweight version",
        "note": "Full analysis features require local deployment due to Vercel size limits"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
