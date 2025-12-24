# API Integration Documentation

This document explains how OrchestAI integrates with external APIs, particularly the OpenAI API.

## Table of Contents

1. [OpenAI API Integration](#openai-api-integration)
2. [API Key Management](#api-key-management)
3. [Cost Tracking](#cost-tracking)
4. [Error Handling](#error-handling)
5. [Mock Mode](#mock-mode)
6. [Adding New APIs](#adding-new-apis)

---

## OpenAI API Integration

### Overview

OrchestAI integrates with the OpenAI API through the `LLMWorker` class. The integration supports:
- Real API calls when API key is available
- Automatic fallback to mock responses when API key is missing
- Cost tracking and latency measurement
- Error handling and retries

### Implementation

**Location:** `orchestai/worker/llm_worker.py`

**Key Method:**
```python
def _call_llm(self, task: str, text: str, parameters: Dict[str, Any]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.name,
                messages=[{"role": "user", "content": f"{task}: {text}"}],
                **parameters
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call failed, using mock: {e}")
            return self._mock_response(task, text)
    else:
        return self._mock_response(task, text)
```

### API Version

**Current Version:** OpenAI API v1.0+

**Important:** The code uses the new `openai.OpenAI()` client (v1.0+), not the deprecated `openai.ChatCompletion.create()`.

**Migration Note:**
- Old (v0.x): `openai.ChatCompletion.create(...)`
- New (v1.0+): `openai.OpenAI().chat.completions.create(...)`

---

## API Key Management

### Setting API Key

**Method 1: Environment Variable**
```bash
export OPENAI_API_KEY=sk-proj-...
```

**Method 2: .env File**
```bash
# .env file
OPENAI_API_KEY=sk-proj-...
```

**Loading .env File:**
```python
from dotenv import load_dotenv
load_dotenv()  # Loads .env file automatically
```

### Security Best Practices

1. **Never Commit API Keys**
   - Add `.env` to `.gitignore`
   - Use environment variables in production
   - Rotate keys regularly

2. **Use Different Keys for Development/Production**
   - Development: Lower rate limits, cheaper models
   - Production: Higher rate limits, better models

3. **Monitor API Usage**
   - Track costs in execution logs
   - Set up billing alerts
   - Review usage regularly

---

## Cost Tracking

### Cost Calculation

**Per-Token Cost:**
```python
# In LLMWorker
input_tokens = estimate_tokens(prompt)
output_tokens = estimate_tokens(response)
total_tokens = input_tokens + output_tokens
cost = total_tokens * self.config.cost_per_token
```

**Cost Configuration:**
```yaml
worker_models:
  - name: "gpt-4"
    cost_per_token: 0.03  # $0.03 per 1K tokens
  - name: "gpt-3.5-turbo"
    cost_per_token: 0.002  # $0.002 per 1K tokens
```

### Cost Tracking in Execution

**ExecutionResult:**
```python
result = orchestrator.execute(instruction)
print(f"Total cost: ${result.total_cost:.4f}")
```

**Per-Task Costs:**
```python
for task_id, metrics in result.task_metrics.items():
    print(f"Task {task_id}: ${metrics['cost']:.4f}")
```

### Cost Optimization

**RL Model Selector:**
- Learns to select cheaper models when appropriate
- Balances cost with quality
- Optimizes for total workflow cost

**Reward Structure:**
```python
reward = (
    10.0 * success -           # Success reward
    0.1 * total_cost +         # Cost penalty
    -0.01 * latency_ms         # Latency penalty
)
```

---

## Error Handling

### API Error Types

**1. Authentication Errors**
```python
# Missing or invalid API key
openai.AuthenticationError
```

**2. Rate Limit Errors**
```python
# Too many requests
openai.RateLimitError
```

**3. API Errors**
```python
# Server errors
openai.APIError
```

**4. Network Errors**
```python
# Connection issues
requests.exceptions.ConnectionError
```

### Error Handling Strategy

**In LLMWorker:**
```python
try:
    response = client.chat.completions.create(...)
    return response.choices[0].message.content
except Exception as e:
    print(f"API call failed, using mock: {e}")
    return self._mock_response(task, text)
```

**In OrchestrationSystem:**
```python
# Retry mechanism
if not worker_output.success and retry_count < self.max_retries:
    retry_count += 1
    time.sleep(self.retry_delay)
    # Retry task
```

### Retry Configuration

**System Configuration:**
```yaml
system:
  max_retries: 3
  retry_delay: 1.0  # seconds
```

---

## Mock Mode

### When Mock Mode is Used

1. **No API Key**: API key not set in environment
2. **API Failure**: API call fails (network, rate limit, etc.)
3. **Testing**: For testing without API costs

### Mock Response Format

**LLMWorker Mock:**
```python
def _mock_response(self, task: str, text: str) -> str:
    # Simple mock response
    return f"Processed: {text}"
```

**Mock Behavior:**
- Returns deterministic responses
- No actual API calls
- Useful for development and testing
- Zero cost

### Using Mock Mode

**For Testing:**
```python
# Don't set OPENAI_API_KEY
# System will automatically use mock responses
```

**For Development:**
```python
# Use mock mode to avoid API costs during development
# Switch to real API for production
```

---

## Adding New APIs

### Adding a New API Integration

**Step 1: Create Worker Class**
```python
class NewAPIWorker(BaseWorker):
    def __init__(self, config: WorkerConfig):
        super().__init__(config)
        # Initialize API client
    
    def _call_api(self, task: str, data: Any) -> str:
        # Make API call
        pass
    
    def process(self, input_data: Dict) -> WorkerOutput:
        # Process input and return output
        pass
```

**Step 2: Add to Worker Layer**
```python
# In worker_layer.py
if config.model_type == "new_api":
    worker = NewAPIWorker(config)
```

**Step 3: Update Configuration**
```yaml
worker_models:
  - name: "new-api-model"
    model_type: "new_api"
    cost_per_token: 0.001
    latency_ms: 200
```

### API Integration Checklist

- [ ] API client initialization
- [ ] Error handling
- [ ] Cost tracking
- [ ] Latency measurement
- [ ] Mock mode support
- [ ] Retry logic
- [ ] Configuration support

---

## Best Practices

### 1. API Key Management

- Use environment variables
- Never commit keys to git
- Rotate keys regularly
- Use different keys for dev/prod

### 2. Cost Management

- Track costs in execution logs
- Set up billing alerts
- Use cheaper models when possible
- Monitor usage regularly

### 3. Error Handling

- Implement retry logic
- Fallback to mock mode
- Log errors for debugging
- Handle rate limits gracefully

### 4. Performance

- Cache API responses when possible
- Batch requests when supported
- Use appropriate timeouts
- Monitor latency

---

## Troubleshooting

### Common Issues

**1. "API call failed, using mock"**
- Check API key is set correctly
- Verify API key is valid
- Check network connection
- Review API error message

**2. High Costs**
- Review cost configuration
- Check token usage
- Use cheaper models
- Optimize prompts

**3. Rate Limit Errors**
- Reduce request frequency
- Implement exponential backoff
- Use multiple API keys (if allowed)
- Upgrade API tier

**4. Slow Responses**
- Check network latency
- Use faster models
- Optimize prompts
- Consider caching

---

## Example Usage

### Basic Usage

```python
from orchestai.utils.setup import setup_system
from orchestai.utils.config_loader import load_config

# Load configuration
config = load_config("config.yaml")

# Setup system (loads API key from .env)
orchestrator = setup_system(config)

# Execute task (uses real API if key is set)
result = orchestrator.execute(
    instruction="Summarize this text",
    input_data={"text": "Long text here..."}
)

# Check results
print(f"Success: {result.success}")
print(f"Cost: ${result.total_cost:.4f}")
print(f"Output: {result.outputs[-1]}")
```

### Testing Without API

```python
# Don't set OPENAI_API_KEY
# System will use mock responses
result = orchestrator.execute(...)
# No API costs, deterministic responses
```

---

## Next Steps

- See `REAL_WORLD_TESTING_GUIDE.md` for real-world testing
- See `06_CONFIGURATION_GUIDE.md` for configuration
- See `05_TRAINING_PIPELINE.md` for training with real APIs

