# Field Prime Viz

## Smart India Hackathon 2025 - Precision Agriculture Platform

Field Prime Viz is a comprehensive precision agriculture platform developed for the Smart India Hackathon 2025. The application integrates advanced data visualization, machine learning, and IoT capabilities to provide farmers and agricultural professionals with actionable insights for improved crop management and yield optimization.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Services](#services)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)

## Overview

Field Prime Viz combines hyperspectral imaging analysis, IoT sensor data, and machine learning to deliver a complete agricultural monitoring and management solution. The platform enables users to visualize field health, analyze spectral data, manage farm fields, and make data-driven decisions to optimize agricultural operations.

## Features

- Real-time agricultural monitoring dashboard
- Hyperspectral image analysis
- 3D terrain visualization
- Field health mapping
- Crop classification using machine learning
- IoT sensor data integration
- Field management system
- Team collaboration tools
- Data source management

## Technology Stack

### Frontend
- React with TypeScript
- Vite for fast development and building
- Tailwind CSS for styling
- shadcn/ui component library
- Framer Motion for animations
- Recharts for data visualization
- React Three Fiber for 3D visualizations
- React Query for data fetching

### Backend
- Flask Python server
- PyTorch for machine learning models
- NumPy and SciPy for scientific computing
- Pandas for data manipulation
- Socket.IO for real-time data streaming

## Services

### 1. Dashboard

The Dashboard provides a comprehensive overview of agricultural data with real-time monitoring capabilities:

- **Key Metrics**: Displays critical agricultural indicators including crop health index, soil moisture levels, temperature trends, and yield predictions.
- **Real-time IoT Data**: Integrates sensor data for temperature, humidity, and soil moisture with automatic updates.
- **Field Health Visualization**: Shows crop health trends over time with interactive charts.
- **Alerts System**: Notifies users about critical conditions requiring attention.
- **Crop Distribution**: Visualizes the distribution of different crops across fields.

### 2. Health Map

The Health Map service provides detailed visualization of field health using hyperspectral imaging:

- **NDVI Visualization**: Displays Normalized Difference Vegetation Index maps for assessing plant health.
- **Multi-spectral Analysis**: Supports various vegetation indices for comprehensive health assessment.
- **Temporal Analysis**: Tracks changes in field health over time.
- **Field-specific Data**: Provides detailed health information for individual fields.
- **Anomaly Detection**: Identifies areas with potential issues requiring attention.

### 3. 3D Terrain Visualization

The Terrain Visualization service offers interactive 3D models of agricultural land:

- **3D Field Rendering**: Creates detailed three-dimensional models of fields using elevation data.
- **Layer Controls**: Allows toggling between different data layers (elevation, moisture, crop health).
- **Interactive Navigation**: Supports rotation, zoom, and pan for detailed exploration.
- **Viewpoint Selection**: Provides multiple camera perspectives (top-down, perspective, orthographic).
- **Data Overlay**: Visualizes sensor data and health metrics directly on the 3D terrain.

### 4. Spectral Analysis

The Spectral Analysis service enables advanced analysis of hyperspectral imaging data:

- **Spectral Signature Visualization**: Displays reflectance patterns across different wavelengths.
- **Multi-index Analysis**: Supports NDVI, GNDVI, EVI, and other vegetation indices.
- **Temporal Tracking**: Monitors changes in spectral signatures over time.
- **Band-specific Analysis**: Provides detailed information about specific spectral bands.
- **Anomaly Detection**: Identifies unusual spectral patterns that may indicate crop stress or disease.

### 5. Field Management

The Field Management service provides tools for organizing and monitoring agricultural fields:

- **Field Registry**: Maintains a database of all fields with key information (size, crop type, location).
- **Health Status Tracking**: Monitors and displays the current health status of each field.
- **Crop Planning**: Assists with planning crop rotations and planting schedules.
- **Field Addition**: Supports adding new fields with detailed information.
- **Inspection History**: Tracks and records field inspection dates and results.

### 6. Team Management

The Team Management service facilitates collaboration among agricultural professionals:

- **Member Management**: Supports adding, editing, and removing team members.
- **Role Assignment**: Defines different roles with specific permissions (Farm Manager, Field Technician, Data Analyst).
- **Activity Tracking**: Monitors team member activities and contributions.
- **Invitation System**: Allows inviting new members to join the platform.
- **Permission Management**: Controls access to different features based on user roles.

### 7. Data Sources

The Data Sources service manages connections to various data inputs:

- **Source Integration**: Connects to different data sources (satellite imagery, IoT sensors, weather services).
- **Data Visualization**: Displays data from different sources in a unified interface.
- **Connection Management**: Monitors the status of data source connections.
- **Historical Data Access**: Provides access to archived data from various sources.
- **Import/Export Tools**: Supports importing and exporting data in various formats.

### 8. Backend AI Services

The backend provides several AI and data processing services:

- **Hyperspectral Data Processing**: Loads and processes hyperspectral imaging data.
- **Crop Classification**: Uses a PyTorch-based neural network to classify crops from hyperspectral data.
- **IoT Data Simulation**: Generates realistic IoT sensor data for testing and demonstration.
- **Image Analysis**: Processes and analyzes uploaded field images.
- **Data Transformation**: Converts raw data into actionable insights and visualizations.

## Installation

### Prerequisites

- Node.js & npm - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)
- Python 3.8+ with pip

### Frontend Setup

```sh
# Clone the repository
git clone <YOUR_GIT_URL>

# Navigate to the project directory
cd field-prime-viz

# Install frontend dependencies
npm i

# Start the development server
npm run dev
```

### Backend Setup

```sh
# Install Python dependencies
pip install -r requirements.txt

# Run the Flask server
python app.py
```

## Usage

After starting both the frontend and backend servers:

1. Access the application at http://localhost:8081 (or the port specified in your Vite configuration)
2. Log in using the provided credentials or create a new account
3. Navigate through the different services using the sidebar navigation
4. Explore the dashboard for an overview of your agricultural data
5. Use the specialized tools for detailed analysis and management

## Development

### Project Structure

- `/src` - Frontend React application
  - `/components` - Reusable UI components
  - `/hooks` - Custom React hooks
  - `/lib` - Utility functions and libraries
  - `/pages` - Main application pages
- `/modules` - Backend Python modules
  - `data_handler.py` - Hyperspectral data processing
  - `model_handler.py` - Machine learning model implementation
  - `iot_generator.py` - IoT data simulation
- `/data` - Sample datasets and model files
- `/models` - Trained machine learning models

### Contributing

Contributions to Field Prime Viz are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
