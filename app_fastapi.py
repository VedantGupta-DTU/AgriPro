from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import base64
import numpy as np
from PIL import Image
import io
import torch
import threading
import time
import uvicorn
from typing import Dict, List, Optional, Any, Union
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Import our custom modules
from modules.data_handler import load_hyperspectral_data, create_rgb_visualization
from modules.iot_generator import generate_iot_data
from modules.model_handler import run_prediction, CropClassifier, PATCH_SIZE

app = FastAPI(title="Field Prime Viz API", 
              description="FastAPI backend for Field Prime Viz agricultural analytics",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables (to store data in memory for the session) ---
hypercube_data = None
ground_truth_data = None
trained_model = None
prediction_map_data = None
num_classes_global = int(os.getenv("NUM_CLASSES", "16"))  # Indian Pines has 16 classes (0-15, 0 is background)

# --- Configuration ---
DATA_FOLDER = os.getenv("DATA_FOLDER", 'data')
MODEL_PATH = os.path.join(os.getenv("MODELS_FOLDER", 'models'), os.getenv("MODEL_FILENAME", 'crop_classifier.pth'))  # PyTorch model path

# --- Load Model on Startup ---
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Field Prime Viz FastAPI application")
    await load_trained_model_on_startup()

async def load_trained_model_on_startup():
    global trained_model, num_classes_global
    # We need to know num_classes to initialize the model.
    # A robust way is to load data first to determine it.
    # For simplicity here, we'll use the environment variable or default to 16 (Indian Pines classes).
    
    try:
        if os.path.exists(MODEL_PATH):
            try:
                logger.info(f"Loading model from {MODEL_PATH} with {num_classes_global} classes")
                trained_model = CropClassifier(num_classes=num_classes_global)
                trained_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
                trained_model.eval()  # Set to evaluation mode
                logger.info(f"Successfully loaded trained PyTorch model from {MODEL_PATH}")
            except Exception as e:
                logger.error(f"Error loading PyTorch model: {e}")
                trained_model = None
        else:
            logger.warning(f"PyTorch model not found at {MODEL_PATH}. Some features will be unavailable.")
    except Exception as e:
        logger.error(f"Unexpected error during startup: {e}")
        # Continue running the app even if model loading fails

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def index():
    # For simplicity, we'll return a basic HTML response
    # In production, you'd want to serve your React app properly
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Field Prime Viz</title>
            <meta http-equiv="refresh" content="0;url=http://localhost:8080" />
        </head>
        <body>
            <p>Redirecting to frontend...</p>
        </body>
    </html>
    """

@app.get("/api/load_data")
async def api_load_data():
    global hypercube_data, ground_truth_data
    try:
        hypercube_data, ground_truth_data = load_hyperspectral_data(DATA_FOLDER)
        rgb_image_pil = create_rgb_visualization(hypercube_data)
        
        # Convert PIL Image to base64 string
        buffered = io.BytesIO()
        rgb_image_pil.save(buffered, format="PNG")
        rgb_image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {"success": True, "rgb_image_b64": f"data:image/png;base64,{rgb_image_b64}", "hypercube_shape": list(hypercube_data.shape)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

@app.get("/api/run_analysis")
async def api_run_analysis():
    global prediction_map_data
    if hypercube_data is None:
        raise HTTPException(status_code=400, detail="Please load hyperspectral data first.")
    if trained_model is None:
        raise HTTPException(status_code=400, detail="Trained PyTorch model not found. Please run train.py first.")

    try:
        # IoT Data
        iot_data = generate_iot_data(24)  # Simulate 24 hours of data

        # AI Prediction
        prediction_map_data, class_summary = run_prediction(trained_model, hypercube_data)
        
        # Convert prediction map to a flat list for easy transfer to JS
        prediction_map_flat = prediction_map_data.flatten().tolist()

        return {"success": True, "iot_data": iot_data, "prediction_map": prediction_map_flat, "class_summary": class_summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running analysis: {str(e)}")

@app.get("/api/get_spectral_signature")
async def api_get_spectral_signature(x: int, y: int):
    if hypercube_data is None:
        raise HTTPException(status_code=400, detail="Hyperspectral data not loaded.")

    try:
        # Extract the full spectral vector for the clicked pixel
        spectral_signature = hypercube_data[y, x, :].tolist()
        return {"success": True, "spectral_signature": spectral_signature}
    except IndexError:
        raise HTTPException(status_code=400, detail="Invalid pixel coordinates.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting spectral signature: {str(e)}")

class ReportRequest(BaseModel):
    format: str = "pdf"
    include_iot: bool = True
    include_analysis: bool = True
    include_spectral: bool = True

@app.post("/api/generate_report")
async def api_generate_report(request: ReportRequest):
    """Generate comprehensive agricultural report"""
    try:
        # Collect data based on request parameters
        report_data = {
            'timestamp': np.datetime_as_string(np.datetime64('now')),
            'report_type': 'Agricultural Analytics Report',
            'generated_by': 'Field Prime Viz System'
        }

        if request.include_iot and hypercube_data is not None:
            # Generate current IoT data
            iot_data = generate_iot_data(24)
            report_data['iot_data'] = iot_data
            report_data['iot_summary'] = {
                'avg_soil_moisture': float(np.mean([d['soil_moisture_pct'] for d in iot_data])),
                'avg_temperature': float(np.mean([d['temperature_c'] for d in iot_data])),
                'avg_humidity': float(np.mean([d['humidity_pct'] for d in iot_data]))
            }

        if request.include_analysis and prediction_map_data is not None:
            # Include analysis results
            unique, counts = np.unique(prediction_map_data, return_counts=True)
            class_distribution = dict(zip(unique.tolist(), counts.tolist()))
            report_data['analysis_results'] = {
                'prediction_map_shape': list(prediction_map_data.shape),
                'class_distribution': class_distribution,
                'total_pixels': int(len(prediction_map_data.flatten()))
            }

        if request.include_spectral and hypercube_data is not None:
            # Include spectral data summary
            report_data['spectral_summary'] = {
                'data_shape': list(hypercube_data.shape),
                'wavelengths': int(hypercube_data.shape[2] if len(hypercube_data.shape) > 2 else 0),
                'avg_spectral_signature': np.mean(hypercube_data, axis=(0, 1)).tolist()
            }

        if request.format == 'json':
            return {
                'success': True,
                'report': report_data,
                'format': 'json'
            }
        else:
            # For PDF format, create a simple text-based PDF content
            # In a real application, you'd use a proper PDF library like ReportLab
            pdf_content = f"""
AGRICULTURAL ANALYTICS REPORT
Generated: {report_data['timestamp']}
System: {report_data['generated_by']}

=== FIELD CONDITIONS SUMMARY ===
"""
            if 'iot_summary' in report_data:
                pdf_content += f"""
IoT Sensor Data Summary:
- Average Soil Moisture: {report_data['iot_summary']['avg_soil_moisture']:.1f}%
- Average Temperature: {report_data['iot_summary']['avg_temperature']:.1f}°C
- Average Humidity: {report_data['iot_summary']['avg_humidity']:.1f}%

"""

            if 'analysis_results' in report_data:
                pdf_content += f"""
AI Analysis Results:
- Total Pixels Analyzed: {report_data['analysis_results']['total_pixels']:,}
- Prediction Map Shape: {report_data['analysis_results']['prediction_map_shape']}
- Class Distribution: {report_data['analysis_results']['class_distribution']}

"""

            if 'spectral_summary' in report_data:
                pdf_content += f"""
Spectral Data Summary:
- Data Shape: {report_data['spectral_summary']['data_shape']}
- Wavelengths: {report_data['spectral_summary']['wavelengths']}
"""

            # In a real app, you'd convert this to a PDF and return it
            # For now, we'll just return the text content
            return {
                'success': True,
                'report': pdf_content,
                'format': 'pdf'
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

# --- Authentication Endpoints ---
class UserLogin(BaseModel):
    username: str
    password: str

class UserSignup(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None

@app.post("/api/auth/login")
async def login(user: UserLogin):
    # This is a simplified login endpoint
    # In a real application, you would validate credentials against a database
    if user.username and user.password:
        # Mock successful login
        return {
            "success": True,
            "user": {
                "username": user.username,
                "token": "mock-jwt-token-for-" + user.username
            }
        }
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/api/auth/signup")
async def signup(user: UserSignup):
    # This is a simplified signup endpoint
    # In a real application, you would store user data in a database
    if user.username and user.email and user.password:
        # Mock successful signup
        return {
            "success": True,
            "message": "User registered successfully",
            "user": {
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name
            }
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid user data")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "False").lower() == "true"
    
    logger.info(f"Starting FastAPI server on {host}:{port} (reload={reload})")
    uvicorn.run("app_fastapi:app", host=host, port=port, reload=reload)