# ConfigLoader Documentation

## Overview

The `ConfigLoader` class is a core component of the GROQ CLOUD API client that handles loading, validating, and accessing configuration parameters. It provides a robust mechanism for retrieving configuration settings from JSON files and ensuring that all required parameters are present and correctly formatted.

## Purpose

The primary purpose of the `ConfigLoader` class is to:

1. Load configuration settings from a JSON file
2. Validate the configuration structure and values
3. Provide access to model-specific configuration parameters
4. Handle configuration errors with detailed exception messages

## Configuration File Format

The configuration file must be a valid JSON file with the following structure:

```json
{
  "model_name": {
    "base_url": "https://api.groq.com/v1",
    "api_key": "your_api_key_here",
    "req_per_minute": 60,
    "req_per_day": 1000,
    "token_per_minute": 10000,
    "token_per_day": 100000
  },
  "another_model": {
    "base_url": "https://api.groq.com/v1",
    "api_key": "your_api_key_here",
    "req_per_minute": 60,
    "req_per_day": 1000,
    "token_per_minute": 10000,
    "token_per_day": 100000
  }
}
```

Each model configuration must include the following required fields:

| Field | Type | Description |
|-------|------|-------------|
| `base_url` | string | The base URL for the GROQ API |
| `api_key` | string | The API key for authenticating with the GROQ API |
| `req_per_minute` | integer | Maximum requests per minute (or -1 for unlimited) |
| `req_per_day` | integer | Maximum requests per day (or -1 for unlimited) |
| `token_per_minute` | integer | Maximum tokens per minute (or -1 for unlimited) |
| `token_per_day` | integer | Maximum tokens per day (or -1 for unlimited) |

## API Reference

### Constructor

```python
def __init__(self, config_path: str) -> None
```

Initializes a new instance of the `ConfigLoader` class with the specified configuration file path.

**Parameters:**
- `config_path` (str): Path to the configuration file.

**Raises:**
- `ConfigLoaderException`: If the configuration file path is invalid (empty or not a string).

**Example:**
```python
from CutomGroqChat.config_loader import ConfigLoader

# Initialize with a configuration file path
config_loader = ConfigLoader("config.json")
```

### Methods

#### `load_config`

```python
def load_config(self) -> Dict[str, Dict[str, Any]]
```

Loads the configuration from the specified file.

**Returns:**
- Dictionary containing the configuration parameters.

**Raises:**
- `ConfigLoaderException`: If there is an error loading the configuration, such as:
  - The configuration file does not exist
  - The configuration file contains invalid JSON
  - Any other error occurs during loading

**Example:**
```python
# Load the configuration
try:
    config = config_loader.load_config()
    print(f"Loaded configuration for {len(config)} models")
except ConfigLoaderException as e:
    print(f"Error loading configuration: {e.message}")
```

#### `get_model_config`

```python
def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]
```

Gets the configuration for a specific model.

**Parameters:**
- `model_name` (str): Name of the model.

**Returns:**
- Dictionary containing the model configuration, or None if not found.

**Example:**
```python
# Get configuration for a specific model
model_config = config_loader.get_model_config("llama-3-70b-chat")
if model_config:
    print(f"API Key: {model_config['api_key']}")
    print(f"Base URL: {model_config['base_url']}")
else:
    print("Model configuration not found")
```

### Internal Methods

#### `_validate_config`

```python
def _validate_config(self) -> None
```

Validates the loaded configuration to ensure all required keys are present and have valid values.

**Rate Limit Values:**
- `req_per_minute`
- `req_per_day`
- `token_per_minute`
- `token_per_day`

All rate limit values can be either positive integers or `-1` for unlimited.

**Raises:**
- `ConfigLoaderException`: If the configuration is invalid:
  - Empty configuration or no models defined
  - Missing required fields for a model
  - Invalid rate limit values (not positive integers or -1)

## Exception Handling

The `ConfigLoader` class uses the `ConfigLoaderException` class for error handling, which provides detailed information about the error:

```python
# Example of catching ConfigLoaderException
try:
    config_loader = ConfigLoader("config.json")
    config = config_loader.load_config()
except ConfigLoaderException as e:
    print(f"Error: {e.message}")
    print(f"Config key: {e.config_key}")
```

The `ConfigLoaderException` class includes:
- `message`: A descriptive error message
- `config_key`: The specific configuration key associated with the error

## Implementation Details

### File Loading Process

1. Check if the configuration file exists
2. Open and read the file
3. Parse the JSON content
4. Validate the configuration structure and values
5. Return the validated configuration dictionary

### Configuration Validation

The validation process checks:
1. That the configuration is not empty
2. That each model configuration has all required fields
3. That all rate limit values are valid (positive integers or -1)

### Error Identification

The class provides detailed error information to help identify the specific issue:
- For missing fields, it identifies which fields are missing
- For invalid rate limits, it specifies which field has an invalid value
- For file-related errors, it identifies the nature of the file problem

## Integration with GROQ API Client

The `ConfigLoader` class is designed to be integrated with the GROQ API client:

```python
from CutomGroqChat.config_loader import ConfigLoader
from CutomGroqChat.groq_client import GroqClient

# Initialize the configuration loader
config_loader = ConfigLoader("config.json")
config = config_loader.load_config()

# Get configuration for a specific model
model_config = config_loader.get_model_config("llama-3-70b-chat")

# Initialize the GROQ client with the model configuration
client = GroqClient(model_config)
```

## Best Practices

1. **Keep your API key secure**: Do not commit your configuration file with actual API keys to version control
2. **Set appropriate rate limits**: Configure rate limits based on your usage requirements
3. **Handle exceptions**: Always handle `ConfigLoaderException` to provide appropriate feedback
4. **Validate your configuration**: Ensure your configuration file follows the required format

## Limitations and Known Issues

- The implementation has a bug in the rate limit validation condition that could be fixed in a future release
- The class only supports JSON configuration files 