"""
Dynamic Model Discovery: Discover and register models dynamically (like HuggingGPT)
"""

from typing import Dict, List, Optional, Any
import requests
import json


class ModelDiscovery:
    """
    Discovers available models dynamically, similar to HuggingGPT's approach.
    Supports both local models and remote APIs (HuggingFace, etc.).
    """
    
    def __init__(
        self,
        huggingface_token: Optional[str] = None,
        local_model_endpoint: Optional[str] = None,
    ):
        """
        Initialize model discovery.
        
        Args:
            huggingface_token: HuggingFace API token
            local_model_endpoint: Local model server endpoint
        """
        self.huggingface_token = huggingface_token
        self.local_model_endpoint = local_model_endpoint
        self.discovered_models: Dict[str, List[Dict[str, Any]]] = {}
    
    def discover_huggingface_models(
        self,
        task_types: List[str],
        limit: int = 10,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Discover models from HuggingFace Hub for given task types.
        
        Args:
            task_types: List of task types to discover models for
            limit: Maximum models per task type
            
        Returns:
            Dictionary mapping task types to model lists
        """
        if not self.huggingface_token:
            return {}
        
        discovered = {}
        
        # HuggingFace task mapping
        task_mapping = {
            "text-generation": "text-generation",
            "summarization": "summarization",
            "translation": "translation",
            "question-answering": "question-answering",
            "text-classification": "text-classification",
            "image-classification": "image-classification",
            "object-detection": "object-detection",
            "text-to-image": "text-to-image",
            "image-to-text": "image-to-text",
            "automatic-speech-recognition": "automatic-speech-recognition",
        }
        
        for task_type in task_types:
            hf_task = task_mapping.get(task_type, task_type)
            
            try:
                # Query HuggingFace API
                url = f"https://huggingface.co/api/models"
                params = {
                    "filter": hf_task,
                    "sort": "downloads",
                    "direction": "-1",
                    "limit": limit,
                }
                headers = {
                    "Authorization": f"Bearer {self.huggingface_token}",
                }
                
                response = requests.get(url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    models = response.json()
                    discovered[task_type] = [
                        {
                            "name": model["id"],
                            "task": task_type,
                            "type": "huggingface",
                            "description": model.get("pipeline_tag", ""),
                            "cost_per_token": 0.001,  # Default estimate
                            "latency_ms": 200,  # Default estimate
                            "url": f"https://huggingface.co/{model['id']}",
                        }
                        for model in models[:limit]
                    ]
            except Exception as e:
                print(f"Error discovering HuggingFace models for {task_type}: {e}")
                discovered[task_type] = []
        
        self.discovered_models.update(discovered)
        return discovered
    
    def discover_local_models(self) -> List[Dict[str, Any]]:
        """
        Discover models from local model server.
        
        Returns:
            List of local model configurations
        """
        if not self.local_model_endpoint:
            return []
        
        try:
            response = requests.get(
                f"{self.local_model_endpoint}/models",
                timeout=5,
            )
            if response.status_code == 200:
                models = response.json()
                return [
                    {
                        "name": model.get("id", ""),
                        "task": model.get("task", ""),
                        "type": "local",
                        "cost_per_token": 0.0005,  # Local models are cheaper
                        "latency_ms": model.get("latency_ms", 100),
                        "available": True,
                    }
                    for model in models
                ]
        except Exception as e:
            print(f"Error discovering local models: {e}")
        
        return []
    
    def get_available_models(
        self,
        task_type: str,
        prefer_local: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get available models for a task type.
        
        Args:
            task_type: Type of task
            prefer_local: Whether to prefer local models
            
        Returns:
            List of available model configurations
        """
        available = []
        
        # Get local models
        local_models = self.discover_local_models()
        local_for_task = [m for m in local_models if m.get("task") == task_type]
        
        # Get HuggingFace models
        hf_models = self.discovered_models.get(task_type, [])
        
        # Combine and sort
        if prefer_local:
            available = local_for_task + hf_models
        else:
            available = hf_models + local_for_task
        
        return available
    
    def register_model(
        self,
        model_config: Dict[str, Any],
    ):
        """
        Manually register a model.
        
        Args:
            model_config: Model configuration dictionary
        """
        task_type = model_config.get("task", "general")
        if task_type not in self.discovered_models:
            self.discovered_models[task_type] = []
        self.discovered_models[task_type].append(model_config)

