"""
Base Worker: Abstract base class for all worker models
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time


@dataclass
class WorkerConfig:
    """Configuration for a worker model"""
    name: str
    model_type: str  # "llm", "vision", "audio", "code", "document", "data"
    cost_per_token: float
    latency_ms: float
    api_key: Optional[str] = None
    model_id: Optional[str] = None


@dataclass
class WorkerOutput:
    """Standardized output from a worker model"""
    content: Any
    metadata: Dict[str, Any]
    cost: float
    latency_ms: float
    success: bool
    error: Optional[str] = None


class BaseWorker(ABC):
    """
    Abstract base class for all worker models.
    Ensures standardized interface and I/O schema.
    """
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.name = config.name
        self.model_type = config.model_type
        
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> WorkerOutput:
        """
        Process input data and return standardized output.
        
        Args:
            input_data: Dictionary containing:
                - task: Task description
                - data: Input data (text, image, audio, etc.)
                - parameters: Optional task-specific parameters
                
        Returns:
            WorkerOutput with standardized format
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data format.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def estimate_cost(self, input_data: Dict[str, Any]) -> float:
        """
        Estimate cost for processing input.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Estimated cost
        """
        # Default: estimate based on input size
        if "data" in input_data:
            data = input_data["data"]
            if isinstance(data, str):
                num_tokens = len(data.split()) * 1.3  # Rough token estimate
            elif isinstance(data, dict):
                num_tokens = sum(len(str(v).split()) for v in data.values()) * 1.3
            else:
                num_tokens = 100  # Default estimate
        else:
            num_tokens = 100
        
        return num_tokens * self.config.cost_per_token
    
    def estimate_latency(self, input_data: Dict[str, Any]) -> float:
        """
        Estimate latency for processing input.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Estimated latency in milliseconds
        """
        return self.config.latency_ms
    
    def _create_output(
        self,
        content: Any,
        metadata: Dict[str, Any],
        cost: float,
        latency_ms: float,
        success: bool = True,
        error: Optional[str] = None,
    ) -> WorkerOutput:
        """Helper method to create standardized output"""
        return WorkerOutput(
            content=content,
            metadata=metadata,
            cost=cost,
            latency_ms=latency_ms,
            success=success,
            error=error,
        )

