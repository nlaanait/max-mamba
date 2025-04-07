# Max Mamba

This is an implementation of Mamba2 architecture in MAX.

## Project Structure

```
src/max_mamba/
├── __init__.py          # Package initialization
├── config.py            # Configuration class for Mamba2 models
├── layers/              # Core model components
│   ├── __init__.py      # Layer exports
│   ├── cache.py         # Cache implementation for inference
│   ├── conv.py          # 1D Convolution layer
│   ├── mixer.py         # Main Mamba mixer block
│   └── rmsnorm.py       # RMS normalization layer
├── ops.py               # Custom operations (softplus, padding, other mamba kernels (e.g. parallel scan))
```

## Installation

1. Install [Magic](https://github.com/modularml/magic).
2. Clone this repository:

   ```bash
   git clone https://github.com/nlaanait/max-mamba.git
   cd max-mamba
   ```

3. Create and activate a Magic environment:
   ```bash
   magic install -a
   ```

## Running Tests

The project uses `pytest` for testing. To run all tests:

```bash
# Run all tests
magic shell -e 'test'
pytest tests/

# Run a specific test file
pytest tests/test_rmsnorm_gated.py
pytest tests/test_cache.py
pytest tests/test_conv.py
pytest tests/test_pad.py
```
