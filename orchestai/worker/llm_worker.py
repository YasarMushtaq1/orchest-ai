"""
LLM Worker: Interface for Large Language Model workers
"""

import time
import os
from typing import Dict, Any, Optional
from pathlib import Path
from orchestai.worker.base_worker import BaseWorker, WorkerConfig, WorkerOutput

# Try to import OpenAI (optional)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    # Fallback: manual loading if python-dotenv not installed
    def load_env_file():
        env_file = Path(__file__).parent.parent.parent / ".env"
        if env_file.exists():
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()
    load_env_file()


class LLMWorker(BaseWorker):
    """
    Worker for Large Language Models (GPT-4, GPT-3.5, Llama, etc.)
    """
    
    def __init__(self, config: WorkerConfig):
        super().__init__(config)
        # In production, this would initialize the actual model/API client
        self.model_initialized = False
        
    def _initialize_model(self):
        """Initialize the LLM model"""
        if not self.model_initialized:
            if OPENAI_AVAILABLE:
                # Initialize OpenAI client if API key is available
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    openai.api_key = api_key
                    self.use_real_api = True
                else:
                    self.use_real_api = False
            else:
                self.use_real_api = False
            self.model_initialized = True
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate LLM input format"""
        required_keys = ["task", "data"]
        if not all(key in input_data for key in required_keys):
            return False
        
        # Allow both string and dict data (dict might contain text field)
        data = input_data["data"]
        if isinstance(data, dict):
            # Extract text from dict if present
            if "text" in data:
                return True
            # Or if it's a nested structure, accept it
            return True
        elif isinstance(data, str):
            return True
        
        return False
    
    def process(self, input_data: Dict[str, Any]) -> WorkerOutput:
        """
        Process text input using LLM.
        
        Args:
            input_data: Dictionary with:
                - task: Task description (e.g., "summarize", "translate", "generate")
                - data: Input text
                - parameters: Optional dict with max_tokens, temperature, etc.
                
        Returns:
            WorkerOutput with generated text
        """
        if not self.validate_input(input_data):
            return self._create_output(
                content=None,
                metadata={},
                cost=0.0,
                latency_ms=0.0,
                success=False,
                error="Invalid input format",
            )
        
        self._initialize_model()
        
        start_time = time.time()
        
        try:
            task = input_data["task"]
            data = input_data["data"]
            parameters = input_data.get("parameters", {})
            
            # Extract text from data (handle both string and dict)
            if isinstance(data, dict):
                text = data.get("text", str(data))
            else:
                text = str(data)
            
            # Call LLM (real API if available, otherwise mock)
            result = self._call_llm(task, text, parameters)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Use actual API cost if available, otherwise estimate
            if hasattr(self, 'last_api_cost'):
                cost = self.last_api_cost
            else:
                cost = self.estimate_cost(input_data)
            
            metadata = {
                "model": self.name,
                "task": task,
                "input_length": len(text),
                "output_length": len(result) if isinstance(result, str) else 0,
                "parameters": parameters,
            }
            
            return self._create_output(
                content=result,
                metadata=metadata,
                cost=cost,
                latency_ms=latency_ms,
                success=True,
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return self._create_output(
                content=None,
                metadata={},
                cost=0.0,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )
    
    def _call_llm(self, task: str, text: str, parameters: Dict[str, Any]) -> str:
        """
        Call LLM API (OpenAI if available, otherwise mock).
        """
        # Use real API if available and configured
        if hasattr(self, 'use_real_api') and self.use_real_api and OPENAI_AVAILABLE:
            try:
                # Map worker name to OpenAI model
                model_map = {
                    "gpt-4": "gpt-4",
                    "gpt-3.5-turbo": "gpt-3.5-turbo",
                    "gpt-4o": "gpt-4o",
                }
                model = model_map.get(self.name, "gpt-3.5-turbo")
                
                # Use new OpenAI API (v1.0+)
                try:
                    # Try new API format
                    from openai import OpenAI
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": f"You are a helpful assistant. {task}."},
                            {"role": "user", "content": text}
                        ],
                        max_tokens=parameters.get("max_tokens", 500),
                        temperature=parameters.get("temperature", 0.7),
                    )
                    
                    result = response.choices[0].message.content
                    
                    # Update cost based on actual usage
                    usage = response.usage
                    # Cost per 1K tokens (approximate)
                    input_cost = 0.0015 if "gpt-3.5" in model else 0.03
                    output_cost = 0.002 if "gpt-3.5" in model else 0.06
                    actual_cost = (usage.prompt_tokens * input_cost + usage.completion_tokens * output_cost) / 1000
                    # Store for later retrieval
                    self.last_api_cost = actual_cost
                    
                    return result
                except AttributeError:
                    # Fallback to old API format if needed
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": f"You are a helpful assistant. {task}."},
                            {"role": "user", "content": text}
                        ],
                        max_tokens=parameters.get("max_tokens", 500),
                        temperature=parameters.get("temperature", 0.7),
                    )
                    result = response.choices[0].message.content
                    usage = response.usage
                    actual_cost = (usage.prompt_tokens * 0.001 + usage.completion_tokens * 0.002) / 1000
                    self.last_api_cost = actual_cost
                    return result
            except Exception as e:
                # Fallback to mock if API call fails
                print(f"API call failed, using mock: {e}")
                return self._call_llm_mock(task, text, parameters)
        else:
            # Mock implementation
            return self._call_llm_mock(task, text, parameters)
    
    def _call_llm_mock(self, task: str, text: str, parameters: Dict[str, Any]) -> str:
        """Mock LLM implementation"""
        if "summarize" in task.lower():
            return f"Summary of: {text[:100]}..."
        elif "translate" in task.lower():
            return f"Translated: {text}"
        elif "generate" in task.lower():
            return f"Generated content based on: {text[:50]}..."
        else:
            return f"Processed: {text}"

