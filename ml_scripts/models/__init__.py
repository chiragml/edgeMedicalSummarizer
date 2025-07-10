"""
Models Package Interface

Easily switchable interface for different model implementations
"""
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Configuration: Change these imports and settings to switch models
# =================================================================

# Current active model configuration
ACTIVE_MODEL_TYPE = "phi4-online"

# Model imports - add new models here
if ACTIVE_MODEL_TYPE == "phi4-online":
    from .phi4_online.model import Phi4OnlineModel as ActiveModel
# elif ACTIVE_MODEL_TYPE == "local-llama":
#     from .local_llama.model import LocalLlamaModel as ActiveModel
# elif ACTIVE_MODEL_TYPE == "openai-gpt":
#     from .openai_gpt.model import OpenAIModel as ActiveModel
else:
    raise ImportError(f"Unknown model type: {ACTIVE_MODEL_TYPE}")

# =================================================================

def get_model(model_type: str, model_path: Optional[str] = None, **kwargs) -> Any:
    """
    Get model instance
    
    Args:
        model_type: Type of model (uses ACTIVE_MODEL_TYPE if matches)
        model_path: Optional model path
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Model instance
    """
    if model_type == ACTIVE_MODEL_TYPE:
        return ActiveModel(model_path=model_path, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Active type: {ACTIVE_MODEL_TYPE}")

def get_tokenizer(model_type: str, model_path: Optional[str] = None, **kwargs) -> None:
    """
    Get tokenizer instance
    
    Args:
        model_type: Type of tokenizer
        model_path: Optional tokenizer path
        **kwargs: Additional arguments for tokenizer initialization
        
    Returns:
        None (tokenizer not implemented yet)
    """
    logger.info(f"Tokenizer not implemented for {model_type}. Returning None.")
    return None

def set_hyperparameters(model_type: str, **kwargs) -> None:
    """
    Set hyperparameters for model
    
    Args:
        model_type: Type of model to configure
        **kwargs: Hyperparameters to set
    """
    if model_type == ACTIVE_MODEL_TYPE:
        logger.debug(f"set_hyperparameters called for {model_type} with {kwargs}")
        # Hyperparameters are handled by the model instance
    else:
        logger.warning(f"set_hyperparameters called for unsupported model type: {model_type}")

__all__ = ['get_model', 'get_tokenizer', 'set_hyperparameters']
