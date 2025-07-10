"""
Phi4 Online Model Implementation
Makes API calls to remote server using environment variables
"""
import logging
import requests
import json
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Phi4OnlineModel:
    """
    Phi4 model that makes API calls to remote server
    """
    
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """
        Initialize Phi4 Online Model
        
        Args:
            model_path: Not used for online model
            **kwargs: Additional configuration
        """
        self.api_host = os.getenv('PHI4_API_HOST', None)
        self.api_port = int(os.getenv('PHI4_API_PORT', '5000'))
        self.api_url = f"http://{self.api_host}:{self.api_port}"
        self.timeout = kwargs.get('timeout', int(os.getenv('PHI4_API_TIMEOUT', '30')))
        
        # Default generation parameters
        self.generation_params = {
            'max_length': 100,  # Changed from max_tokens to max_length to match your server
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40
        }
        
        logger.info(f"Phi4OnlineModel initialized with API: {self.api_url}")
    
    def generate(self, prompt: str) -> str:
        """
        Generate text using remote API
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Generated text response
        """
        try:
            # Prepare API request
            payload = {
                'prompt': prompt,
                **self.generation_params
            }
            
            # Make API call
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            # Check response
            if response.status_code == 200:
                result = response.json()
                # Try different possible response keys from your server
                return result.get('generated_text', result.get('text', result.get('response', str(result))))
            else:
                logger.error(f"API call failed with status {response.status_code}: {response.text}")
                return f"Error: API call failed with status {response.status_code}"
                
        except requests.exceptions.Timeout:
            logger.error("API call timed out")
            return "Error: API call timed out"
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to API at {self.api_url}")
            return f"Error: Failed to connect to API at {self.api_url}"
        except Exception as e:
            logger.error(f"Unexpected error during API call: {e}")
            return f"Error: {str(e)}"
    
    def set_generation_params(self, **kwargs):
        """
        Update generation parameters
        
        Args:
            **kwargs: Parameters to update (temperature, max_length, etc.)
        """
        # Map max_tokens to max_length if provided
        if 'max_tokens' in kwargs:
            kwargs['max_length'] = kwargs.pop('max_tokens')
        
        self.generation_params.update(kwargs)
        logger.debug(f"Updated generation params: {self.generation_params}")
    
    def check_health(self) -> dict:
        """
        Check server health
        
        Returns:
            Health status from server
        """
        try:
            response = requests.get(
                f"{self.api_url}/health1",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "unhealthy", "error": f"Status code: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def get_model_info(self) -> dict:
        """
        Get model information from server
        
        Returns:
            Model information from server
        """
        try:
            response = requests.get(
                f"{self.api_url}/model/info",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get model info. Status code: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
