from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import base64
import numpy as np
from PIL import Image
import io
import torch
import threading
import time

# Import our custom modules
from modules.data_handler import load_hyperspectral_data, create_rgb_visualization
from modules.iot_generator import generate_iot_data
from modules.model_handler import run_prediction, CropClassifier, PATCH_SIZE

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- Global Variables (to store data in memory for the session) ---
hypercube_data = None
ground_truth_data = None
trained_model = None
prediction_map_data = None
num_classes_global = 16 # Indian Pines has 16 classes (0-15, 0 is background)

# --- Configuration ---
DATA_FOLDER = 'data'
MODEL_PATH = os.path.join('models', 'crop_classifier.pth') # PyTorch model path

# --- Load Model on Startup ---
# @app.before_first_request # Deprecated in newer Flask versions
def _load_trained_model_on_startup():
    global trained_model, num_classes_global
    # We need to know num_classes to initialize the model.
    # A robust way is to load data first to determine it.
    # For simplicity here, we'll assume it's 16 (Indian Pines classes).
    # In a real app, you might save num_classes with the model or derive it.

    if os.path.exists(MODEL_PATH):
        try:
            trained_model = CropClassifier(num_classes=num_classes_global)
            trained_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            trained_model.eval() # Set to evaluation mode
            print(f"Successfully loaded trained PyTorch model from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
            trained_model = None
    else:
        print(f"PyTorch model not found at {MODEL_PATH}. Please run train.py first.")

# Call the function directly when the app starts
_load_trained_model_on_startup()

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/load_data')
def api_load_data():
    global hypercube_data, ground_truth_data
    try:
        hypercube_data, ground_truth_data = load_hyperspectral_data(DATA_FOLDER)
        rgb_image_pil = create_rgb_visualization(hypercube_data)
        
        # Convert PIL Image to base64 string
        buffered = io.BytesIO()
        rgb_image_pil.save(buffered, format="PNG")
        rgb_image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({'success': True, 'rgb_image_b64': f'data:image/png;base64,{rgb_image_b64}', 'hypercube_shape': hypercube_data.shape})
    except FileNotFoundError as e:
        return jsonify({'success': False, 'message': str(e)}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error loading data: {str(e)}'}), 500

@app.route('/api/run_analysis', methods=['GET'])
def api_run_analysis():
    global prediction_map_data
    if hypercube_data is None:
        return jsonify({'success': False, 'message': 'Please load hyperspectral data first.'}), 400
    if trained_model is None:
        return jsonify({'success': False, 'message': 'Trained PyTorch model not found. Please run train.py first.'}), 400

    try:
        # IoT Data
        iot_data = generate_iot_data(24) # Simulate 24 hours of data

        # AI Prediction
        prediction_map_data, class_summary = run_prediction(trained_model, hypercube_data)
        
        # Convert prediction map to a flat list for easy transfer to JS
        prediction_map_flat = prediction_map_data.flatten().tolist()

        return jsonify({'success': True, 'iot_data': iot_data, 'prediction_map': prediction_map_flat, 'class_summary': class_summary})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error running analysis: {str(e)}'}), 500

@app.route('/api/get_spectral_signature')
def api_get_spectral_signature():
    x = int(request.args.get('x'))
    y = int(request.args.get('y'))

    if hypercube_data is None:
        return jsonify({'success': False, 'message': 'Hyperspectral data not loaded.'}), 400

    try:
        # Extract the full spectral vector for the clicked pixel
        spectral_signature = hypercube_data[y, x, :].tolist()
        return jsonify({'success': True, 'spectral_signature': spectral_signature})
    except IndexError:
        return jsonify({'success': False, 'message': 'Invalid pixel coordinates.'}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error getting spectral signature: {str(e)}'}), 500

@app.route('/api/generate_report', methods=['POST'])
def api_generate_report():
    """Generate comprehensive agricultural report"""
    try:
        data = request.get_json() or {}
        report_format = data.get('format', 'pdf')
        include_iot = data.get('include_iot', True)
        include_analysis = data.get('include_analysis', True)
        include_spectral = data.get('include_spectral', True)

        # Collect data based on request parameters
        report_data = {
            'timestamp': np.datetime_as_string(np.datetime64('now')),
            'report_type': 'Agricultural Analytics Report',
            'generated_by': 'Field Prime Viz System'
        }

        if include_iot and hypercube_data is not None:
            # Generate current IoT data
            iot_data = generate_iot_data(24)
            report_data['iot_data'] = iot_data
            report_data['iot_summary'] = {
                'avg_soil_moisture': np.mean([d['soil_moisture_pct'] for d in iot_data]),
                'avg_temperature': np.mean([d['temperature_c'] for d in iot_data]),
                'avg_humidity': np.mean([d['humidity_pct'] for d in iot_data])
            }

        if include_analysis and prediction_map_data is not None:
            # Include analysis results
            unique, counts = np.unique(prediction_map_data, return_counts=True)
            class_distribution = dict(zip(unique.tolist(), counts.tolist()))
            report_data['analysis_results'] = {
                'prediction_map_shape': prediction_map_data.shape,
                'class_distribution': class_distribution,
                'total_pixels': len(prediction_map_data.flatten())
            }

        if include_spectral and hypercube_data is not None:
            # Include spectral data summary
            report_data['spectral_summary'] = {
                'data_shape': hypercube_data.shape,
                'wavelengths': hypercube_data.shape[2] if len(hypercube_data.shape) > 2 else 0,
                'avg_spectral_signature': np.mean(hypercube_data, axis=(0, 1)).tolist()
            }

        if report_format == 'json':
            return jsonify({
                'success': True,
                'report': report_data,
                'format': 'json'
            })
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
- Data Dimensions: {report_data['spectral_summary']['data_shape']}
- Number of Wavelengths: {report_data['spectral_summary']['wavelengths']}
- Average Spectral Signature Available: Yes

"""

            return jsonify({
                'success': True,
                'report_content': pdf_content,
                'format': 'pdf',
                'filename': f'agricultural_report_{report_data["timestamp"].replace(":", "-").replace("T", "_").split(".")[0]}.txt'
            })

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error generating report: {str(e)}'}), 500

@app.route('/api/export_data', methods=['POST'])
def api_export_data():
    """Export dashboard data in various formats"""
    try:
        data = request.get_json() or {}
        export_format = data.get('format', 'csv')
        data_types = data.get('data_types', ['iot'])
        date_range = data.get('date_range', None)

        # Collect requested data
        export_data = {}

        if 'iot' in data_types:
            # Generate IoT data
            iot_data = generate_iot_data(24)
            export_data['iot_data'] = iot_data

        if 'analysis' in data_types and prediction_map_data is not None:
            # Include analysis data
            unique, counts = np.unique(prediction_map_data, return_counts=True)
            export_data['analysis_data'] = {
                'prediction_map': prediction_map_data.tolist(),
                'class_distribution': dict(zip(unique.tolist(), counts.tolist())),
                'map_shape': prediction_map_data.shape
            }

        if 'spectral' in data_types and hypercube_data is not None:
            # Include spectral data (sample for export efficiency)
            export_data['spectral_data'] = {
                'data_shape': hypercube_data.shape,
                'sample_pixels': hypercube_data[0:10, 0:10, :].tolist()  # Sample data
            }

        # Format data based on requested format
        if export_format == 'csv':
            # Simple CSV format for IoT data
            if 'iot_data' in export_data:
                csv_content = "timestamp,soil_moisture_pct,temperature_c,humidity_pct\n"
                for i, data_point in enumerate(export_data['iot_data']):
                    csv_content += f"{data_point['timestamp']},{data_point['soil_moisture_pct']},{data_point['temperature_c']},{data_point['humidity_pct']}\n"
                
                return jsonify({
                    'success': True,
                    'data': csv_content,
                    'format': 'csv',
                    'filename': f'field_data_{np.datetime64("now").astype(str).replace(":", "-").split(".")[0]}.csv'
                })
            else:
                return jsonify({'success': False, 'message': 'No IoT data available for CSV export'}), 400

        elif export_format == 'json':
            return jsonify({
                'success': True,
                'data': export_data,
                'format': 'json',
                'filename': f'field_data_{np.datetime64("now").astype(str).replace(":", "-").split(".")[0]}.json'
            })

        else:
            return jsonify({'success': False, 'message': f'Unsupported export format: {export_format}'}), 400

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error exporting data: {str(e)}'}), 500

# --- Socket.IO Event Handlers ---

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f'Client connected: {request.sid}')
    print(f'Socket.IO connection established from {request.remote_addr}')
    emit('connect', {'status': 'connected', 'message': 'Connected to Field Prime Viz server'})
    print('Emitted connect confirmation to client')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f'Client disconnected: {request.sid}')

@socketio.on('request_initial_data')
def handle_request_initial_data():
    """Handle request for initial data"""
    print(f'Received request_initial_data from client: {request.sid}')
    try:
        global hypercube_data, ground_truth_data
        
        # Load data if not already loaded
        if hypercube_data is None or ground_truth_data is None:
            hypercube_data, ground_truth_data = load_hyperspectral_data(DATA_FOLDER)
        
        # Create RGB visualization
        rgb_image_pil = create_rgb_visualization(hypercube_data)
        
        # Convert PIL Image to base64 string
        buffered = io.BytesIO()
        rgb_image_pil.save(buffered, format="PNG")
        rgb_image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Emit initial data
        emit('initial_data', {
            'success': True,
            'rgb_image_b64': f'data:image/png;base64,{rgb_image_b64}',
            'hypercube_shape': hypercube_data.shape,
            'message': 'Initial data loaded successfully'
        })
        
    except FileNotFoundError as e:
        emit('connection_error', {'error': str(e)})
    except Exception as e:
        emit('connection_error', {'error': f'Error loading initial data: {str(e)}'})

@socketio.on('request_analysis')
def handle_request_analysis():
    """Handle request for AI analysis"""
    try:
        global prediction_map_data
        
        if hypercube_data is None:
            emit('analysis_error', {'error': 'Please load hyperspectral data first.'})
            return
            
        if trained_model is None:
            emit('analysis_error', {'error': 'Trained PyTorch model not found. Please run train.py first.'})
            return

        # Generate IoT data
        iot_data = generate_iot_data(24)

        # Run AI prediction
        prediction_map_data, class_summary = run_prediction(trained_model, hypercube_data)
        
        # Convert prediction map to a flat list for easy transfer to JS
        prediction_map_flat = prediction_map_data.flatten().tolist()

        # Emit analysis results
        emit('analysis_result', {
            'success': True,
            'iot_data': iot_data,
            'prediction_map': prediction_map_flat,
            'class_summary': class_summary,
            'message': 'Analysis completed successfully'
        })
        
        # Also emit IoT data update
        emit('iot_data_update', {
            'success': True,
            'iot_data': iot_data,
            'message': 'IoT data updated'
        })
        
    except Exception as e:
        emit('analysis_error', {'error': f'Error running analysis: {str(e)}'})

@socketio.on('request_iot_data')
def handle_request_iot_data():
    """Handle request for IoT data"""
    try:
        # Generate IoT data
        iot_data = generate_iot_data(24)
        
        # Emit IoT data update
        emit('iot_data_update', {
            'success': True,
            'iot_data': iot_data,
            'message': 'IoT data updated'
        })
        
    except Exception as e:
        emit('connection_error', {'error': f'Error generating IoT data: {str(e)}'})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='127.0.0.1', port=5000) # Run with socket.io support