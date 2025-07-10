"""
Text Generation Inference Module

Simple text generation with chat memory.
"""
import logging
from typing import List, Dict, Any, Optional

from ..models import get_model, get_tokenizer, set_hyperparameters

logger = logging.getLogger(__name__)

class TextGenerationPipeline:
    """
    Simple text generation with chat memory.
    """
    
    def __init__(
        self,
        model_type: str,
        model_path: Optional[str] = None,
        memory_threshold: int = 20,
        **model_kwargs
    ):
        """
        Initialize the text generator.
        
        Args:
            model_type: Model name from models module
            model_path: Optional specific model path
            memory_threshold: Max chat messages to keep
            **model_kwargs: Arguments for model loading
        """
        self.model_type = model_type
        self.model_path = model_path
        self.memory_threshold = memory_threshold
        self.model_kwargs = model_kwargs
        
        # Chat history: list of {"role": "user/assistant", "content": "text"}
        self.chat_history: List[Dict[str, str]] = []
        
        # Model components (loaded lazily)
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        self._load_model()
        logger.info(f"TextGenerator initialized for model: {model_type}")
    
    def _load_model(self) -> None:
        """Load model if not already loaded"""
        logger.info(f"Loading model {self.model_type} from path {self.model_path}")
        if self._is_loaded:
            return

        try:
            if self.model_path:
                self._model = get_model(self.model_type, model_path=self.model_path, **self.model_kwargs)
                self._tokenizer = get_tokenizer(self.model_type, model_path=self.model_path, **self.model_kwargs)
            else:
                self._model = get_model(self.model_type, **self.model_kwargs)
                self._tokenizer = get_tokenizer(self.model_type, **self.model_kwargs)
            
            self._is_loaded = True
            logger.info(f"Model {self.model_type} loaded")
            print(f"Model loaded successfully: {self._model}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model {self.model_type}: {e}")
    
    def _manage_memory(self) -> None:
        """Keep chat history under threshold"""
        if len(self.chat_history) > self.memory_threshold:
            # Keep only recent messages
            self.chat_history = self.chat_history[-self.memory_threshold:]
            logger.debug(f"Chat memory trimmed to {len(self.chat_history)} messages")
    
    def _format_prompt(self, user_message: str) -> str:
        """Format chat history into prompt"""
        prompt_parts = []
        
        # Add chat history
        for msg in self.chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt_parts.append(f"{role}: {msg['content']}")
        
        # Add current message
        prompt_parts.append(f"User: {user_message}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def generate(self, prompt: str, **generation_kwargs) -> str:
        """
        Generate response with chat memory.
        
        Args:
            prompt: User input
            **generation_kwargs: Generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated response
        """
        # Load model if needed
        self._load_model()
        
        # Add user message to history
        self.chat_history.append({"role": "user", "content": prompt})
        
        try:
            # Set generation parameters if provided
            if generation_kwargs:
                set_hyperparameters(self.model_type, **generation_kwargs)
            
            # Format prompt with chat history
            formatted_prompt = self._format_prompt(prompt)
            
            # Generate response (assuming model has generate method)
            if hasattr(self._model, 'generate'):
                response = self._model.generate(formatted_prompt)
            else:
                response = f"Response to: {prompt}"  # Placeholder
                logger.warning("Model missing generate method")
            
            # Add response to history
            self.chat_history.append({"role": "assistant", "content": response})
            
            # Manage memory
            self._manage_memory()
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def reset(self) -> None:
        """Clear chat history"""
        self.chat_history = []
        logger.info("Chat history reset")
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get chat history"""
        return self.chat_history.copy()

__all__ = ['TextGenerationPipeline']