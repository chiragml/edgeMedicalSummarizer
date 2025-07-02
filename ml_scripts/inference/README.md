# Inference Module

This module provides RESTful API endpoints for AI model inference, specifically designed for medical text generation and summarization. The module is built with Flask and Flask-RESTful, offering a scalable architecture that can be extended to support additional AI tasks beyond text generation.

## Features

- **Text Generation**: Currently supports medical text generation using various AI models
- **Chat Memory**: Maintains conversation history for contextual responses
- **Model Management**: Dynamic model loading and configuration
- **Request Validation**: Schema validation using Marshmallow
- **Health Monitoring**: Built-in health check and status endpoints
- **Extensible Architecture**: Modular design allows easy addition of new inference capabilities

## Current Capabilities

- **Text Generation**: Medical summaries, symptom explanations, and general medical text
- **Conversation Management**: Persistent chat history with memory management
- **Multiple Model Support**: Configurable model types (currently supports phi3_slm and others)

## API Endpoints

### 1. Status Check
Get current API status and model information
```bash
curl -X GET "http://localhost:5000/"
```

### 2. Health Check
Simple health monitoring endpoint
```bash
curl -X GET "http://localhost:5000/health"
```

### 3. Load Model
Load a specific model for inference
```bash
curl -X POST "http://localhost:5000/load" \
     -H "Content-Type: application/json" \
     -d '{"model_type": "phi3_slm", "memory_threshold": 15}'
```

**Parameters:**
- `model_type` (required): Type of model to load
- `model_path` (optional): Custom path to model files
- `memory_threshold` (optional): Memory management threshold (default: 20)
- `model_kwargs` (optional): Additional model configuration parameters

### 4. Generate Text
Generate text response with conversation context
```bash
curl -X POST "http://localhost:5000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What are the symptoms of diabetes?", "temperature": 0.7}'
```

**Parameters:**
- `prompt` (required): Input text for generation
- `temperature` (optional): Sampling temperature for generation
- `max_tokens` (optional): Maximum number of tokens to generate
- `model_kwargs` (optional): Additional generation parameters

### 5. Chat History
Retrieve current conversation history
```bash
curl -X GET "http://localhost:5000/history"
```

### 6. Reset Conversation
Clear chat history and start fresh conversation
```bash
curl -X POST "http://localhost:5000/reset"
```

## Usage Example

```python
from ml_scripts.inference import run_server, app

# Run the server
run_server(host="0.0.0.0", port=5000, debug=True)

# Or use with your own Flask app
from ml_scripts.inference import app
# app.run()
```

## Architecture

The module follows a modular design pattern:

```
inference/
├── __init__.py           # Main API endpoints and Flask app
├── text_generation.py   # Text generation pipeline
└── README.md            # This file
```

### Key Components

1. **Resource Classes**: Each endpoint is implemented as a Flask-RESTful Resource
2. **Schema Validation**: Marshmallow schemas ensure request data integrity
3. **Pipeline Architecture**: Modular text generation pipeline for easy extension
4. **Global State Management**: Singleton pattern for model instance management

## Extension Capabilities

The current architecture supports easy extension for additional AI tasks:

### Planned Extensions
- **Image Analysis**: Medical image processing and analysis
- **Speech Processing**: Voice-to-text and text-to-speech capabilities
- **Document Processing**: PDF and document parsing for medical records
- **Multi-modal Models**: Combined text, image, and audio processing

### Adding New Capabilities

To add new inference capabilities:

1. **Create New Pipeline**: Implement a new pipeline class similar to `TextGenerationPipeline`
2. **Add Resource Classes**: Create new Flask-RESTful Resource classes
3. **Define Schemas**: Add Marshmallow schemas for request validation
4. **Register Endpoints**: Add new endpoints to the API

Example structure for future extensions:
```python
# New pipeline
class ImageAnalysisPipeline:
    def analyze(self, image_data): ...

# New resource
class ImageAnalysisResource(Resource):
    def post(self): ...

# Register endpoint
api.add_resource(ImageAnalysisResource, '/analyze-image')
```

## Dependencies

- Flask
- Flask-RESTful
- Marshmallow
- Custom TextGenerationPipeline

## Configuration

The module supports various configuration options:

- **Host/Port**: Configurable server binding
- **Debug Mode**: Enable/disable debug logging
- **Memory Management**: Configurable memory thresholds
- **Model Parameters**: Flexible model configuration

## Error Handling

The API includes comprehensive error handling:

- **Validation Errors**: 400 status with detailed error messages
- **Model Loading Errors**: 500 status with error details
- **Generation Failures**: Graceful error responses
- **Missing Model**: Clear error messages when no model is loaded

## Future Roadmap

1. **Multi-Model Support**: Simultaneous loading of multiple specialized models
2. **Streaming Responses**: Real-time text generation with streaming
3. **Batch Processing**: Support for batch inference requests
4. **Model Caching**: Intelligent model caching and swapping
5. **Authentication**: Security features for production deployment
6. **Metrics**: Performance monitoring and usage analytics

## Getting Started

1. **Install Dependencies**: Ensure all required packages are installed
2. **Load a Model**: Use the `/load` endpoint to initialize a model
3. **Start Generating**: Use the `/generate` endpoint for text generation
4. **Monitor**: Use `/health` and `/` endpoints for monitoring

The inference module provides a solid foundation for medical AI applications while maintaining the flexibility to expand into additional AI capabilities as needed.