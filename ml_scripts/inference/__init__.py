"""
Text Generation Module

This module contains scripts for executing inference using AI models
to generate medical summaries and text outputs with Flask RESTful API endpoints.
"""
import logging
from typing import Dict, List, Optional, Any
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from marshmallow import Schema, fields, ValidationError

from .text_generation import TextGenerationPipeline

logger = logging.getLogger(__name__)

# Marshmallow schemas for request/response validation
class GenerateRequestSchema(Schema):
    prompt = fields.Str(required=True)
    temperature = fields.Float(missing=None)
    max_tokens = fields.Int(missing=None)
    model_kwargs = fields.Dict(missing={})

class LoadModelRequestSchema(Schema):
    model_type = fields.Str(required=True)
    model_path = fields.Str(missing=None)
    memory_threshold = fields.Int(missing=20)
    model_kwargs = fields.Dict(missing={})

# Global generator instance
_generator: Optional[TextGenerationPipeline] = None

# Flask app
app = Flask(__name__)
app.config['RESTFUL_JSON'] = {'ensure_ascii': False}
api = Api(app)
class StatusResource(Resource):
    """Get current API status and model information"""
    
    def get(self):
        """
        Get status
        
        curl -X GET "http://localhost:5000/"
        """
        global _generator
        
        if _generator is None:
            return {
                "status": "ready",
                "model_loaded": False,
                "model_type": None
            }
        
        return {
            "status": "ready",
            "model_loaded": _generator._model.check_health(),
            "model_type": _generator.model_type
        }

class LoadModelResource(Resource):
    """Load a model for text generation"""
    
    def post(self):
        """
        Load model
        
        curl -X POST "http://localhost:5000/load" \
             -H "Content-Type: application/json" \
             -d '{"model_type": "phi3_slm", "memory_threshold": 15}'
        """
        global _generator
        
        # Validate request
        schema = LoadModelRequestSchema()
        try:
            data = schema.load(request.get_json())
        except ValidationError as err:
            return {"error": "Validation error", "messages": err.messages}, 400
        
        try:
            _generator = TextGenerationPipeline(
                model_type=data['model_type'],
                model_path=data.get('model_path'),
                memory_threshold=data.get('memory_threshold', 20),
                **data.get('model_kwargs', {})
            )
            print(_generator.__dict__)
            if not _generator._is_loaded:
                return {"error": "Failed to initialize generator"}, 500
            logger.info(f"Model {data['model_type']} loaded successfully")
            
            return {
                "message": f"Model '{data['model_type']}' loaded successfully",
                "model_type": data['model_type'],
                "model_path": data.get('model_path')
            }
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return {"error": f"Failed to load model: {str(e)}"}, 500

class GenerateResource(Resource):
    """Generate text response with chat memory"""
    
    def post(self):
        """
        Generate text
        
        curl -X POST "http://localhost:5000/generate" \
             -H "Content-Type: application/json" \
             -d '{"prompt": "What are the symptoms of diabetes?", "temperature": 0.7}'
        """
        global _generator
        
        if _generator is None:
            return {
                "error": "No model loaded. Please load a model first using /load endpoint"
            }, 400
        
        # Validate request
        schema = GenerateRequestSchema()
        try:
            data = schema.load(request.get_json())
        except ValidationError as err:
            return {"error": "Validation error", "messages": err.messages}, 400
        
        try:
            # Prepare generation kwargs
            generation_kwargs = {}
            if data.get('temperature') is not None:
                generation_kwargs['temperature'] = data['temperature']
            if data.get('max_tokens') is not None:
                generation_kwargs['max_tokens'] = data['max_tokens']
            generation_kwargs.update(data.get('model_kwargs', {}))
            
            # Generate response
            response = _generator.generate(data['prompt'], **generation_kwargs)
            
            logger.info(f"Generated response for prompt: {data['prompt'][:50]}...")
            
            return {
                "response": response,
                "prompt": data['prompt'],
                "model_type": _generator.model_type
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {"error": f"Generation failed: {str(e)}"}, 500

class HistoryResource(Resource):
    """Get current chat history"""
    
    def get(self):
        """
        Get chat history
        
        curl -X GET "http://localhost:5000/history"
        """
        global _generator
        
        if _generator is None:
            return {
                "error": "No model loaded. Please load a model first using /load endpoint"
            }, 400
        
        try:
            history = _generator.get_history()
            
            return {
                "chat_history": history,
                "message_count": len(history)
            }
            
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return {"error": f"Failed to get history: {str(e)}"}, 500

class ResetResource(Resource):
    """Reset chat history"""
    
    def post(self):
        """
        Reset conversation
        
        curl -X POST "http://localhost:5000/reset"
        """
        global _generator
        
        if _generator is None:
            return {
                "error": "No model loaded. Please load a model first using /load endpoint"
            }, 400
        
        try:
            previous_count = len(_generator.chat_history)
            _generator.reset()
            
            logger.info("Chat history reset")
            
            return {
                "message": "Chat history reset successfully",
                "previous_message_count": previous_count
            }
            
        except Exception as e:
            logger.error(f"Failed to reset: {e}")
            return {"error": f"Failed to reset: {str(e)}"}, 500

class HealthResource(Resource):
    """Simple health check endpoint"""
    
    def get(self):
        if not _generator:
            return {"status": "unhealthy", "error": "No model loaded"}, 503
        
        try:
            return {'status': _generator._model.check_health()}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}, 503

class ModelInfoResource(Resource):
    """Get remote model information"""
    
    def get(self):
        """
        Get model info from remote server
        
        curl -X GET "http://localhost:5000/model/info"
        """
        global _generator
        
        if _generator is None:
            return {
                "error": "No model loaded. Please load a model first using /load endpoint"
            }, 400
        
        if not _generator._is_loaded:
            return {
                "error": "Model not loaded yet"
            }, 400
        
        try:
            model_info = _generator._model.get_model_info()
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": f"Failed to get model info: {str(e)}"}, 500

# Register resources with the API

api.add_resource(StatusResource, '/')
api.add_resource(LoadModelResource, '/load')
api.add_resource(GenerateResource, '/generate')
# api.add_resource(HistoryResource, '/history')
# api.add_resource(ResetResource, '/reset')
api.add_resource(HealthResource, '/health')
api.add_resource(ModelInfoResource, '/model/info')

# Convenience function to run the server
def run_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """
    Run the Flask server
    
    Args:
        host: Host to bind to
        port: Port to run on
        debug: Enable debug mode
    """
    app.run(host=host, port=port, debug=debug)

# Export public interface
__all__ = [
    'app',
    'api',
    'run_server',
    'TextGenerationPipeline'
]