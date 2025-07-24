# TurboProp Model Guide

## Overview

TurboProp supports multiple embedding models optimized for different use cases. This guide helps you choose and configure the right model for your needs.

## Model Categories

### 1. Sentence Transformer Models (FastEmbed Backend)

These models use the FastEmbed library and are automatically downloaded and cached:

#### sentence-transformers/all-MiniLM-L6-v2 (Default)
- **Best for**: General-purpose code search, quick setup
- **Dimensions**: 384
- **Size**: ~23MB
- **Languages**: Primarily English, some multilingual capability
- **Performance**: Very fast, low memory usage
- **Use cases**: Getting started, CI/CD environments, resource-constrained systems

#### sentence-transformers/all-MiniLM-L12-v2
- **Best for**: Better accuracy when compute resources allow
- **Dimensions**: 384  
- **Size**: ~44MB
- **Performance**: Moderate speed, better accuracy than L6-v2

### 2. Specialized Code Models (GGUF Backend)

#### nomic-embed-code.Q5_K_S.gguf
- **Best for**: Specialized code search and understanding
- **Dimensions**: 768
- **Size**: ~2.5GB
- **Languages**: Python, Java, Ruby, PHP, JavaScript, Go, and more
- **Special features**:
  - Trained specifically on code datasets
  - Understanding of code semantics and structure
  - Quantized for efficiency
- **Performance**: Slower initial load, good inference speed
- **Use cases**: Code-heavy projects, technical documentation, API exploration

### 3. Multilingual Models (Hugging Face Backend)

#### Qwen/Qwen3-Embedding-0.6B
- **Best for**: Multilingual projects, instruction-based search
- **Dimensions**: 1024
- **Size**: ~600MB
- **Languages**: 100+ languages including programming languages
- **Special features**:
  - Instruction-based embeddings
  - State-of-the-art multilingual performance
  - Excellent code understanding
- **Performance**: Moderate load time, good inference speed
- **Use cases**: International projects, mixed-language codebases, advanced search scenarios

## Model Selection Decision Tree

```
Start Here: What's your primary use case?
│
├── Quick setup, English-only code
│   └── sentence-transformers/all-MiniLM-L6-v2
│
├── Code-specific search, performance matters
│   └── nomic-embed-code.Q5_K_S.gguf
│
├── Multilingual project or advanced features needed
│   └── Qwen/Qwen3-Embedding-0.6B
│
└── Maximum accuracy, resources not constrained
    └── Qwen/Qwen3-Embedding-0.6B with instructions
```

## Advanced Features

### Instruction-Based Embeddings (Qwen3 Only)

Qwen3 models support instructions to optimize embeddings for specific tasks:

```bash
# For code search
tp index --model "Qwen/Qwen3-Embedding-0.6B" \
  --instruction "Represent this code for semantic search"

# For documentation search  
tp index --model "Qwen/Qwen3-Embedding-0.6B" \
  --instruction "Represent this documentation for question answering"

# For API search
tp search "authentication" \
  --model "Qwen/Qwen3-Embedding-0.6B" \
  --instruction "Find API endpoints related to user authentication"
```

### Performance Optimization

#### Model Caching
All models are automatically cached after first download:
- FastEmbed models: Managed by FastEmbed library
- GGUF models: Cached in `~/.turboprop/models/`
- Hugging Face models: Cached with model files

#### Batch Processing
Configure batch sizes for optimal performance:
```yaml
# .turboprop.yml
embedding:
  batch_size: 32  # Adjust based on available memory
  
models:
  "nomic-embed-code.Q5_K_S.gguf":
    batch_size: 8  # Smaller batches for large models
```

#### Memory Management
```bash
# Check model memory requirements
tp model info "nomic-embed-code.Q5_K_S.gguf"

# Clear model cache if needed
tp model clear

# Clear specific model
tp model clear "nomic-embed-code.Q5_K_S.gguf"
```

## Troubleshooting

### Common Issues

#### Model Download Failures
```bash
# Check network connectivity
tp model download "model-name" --verbose

# Use alternative cache directory
export TURBOPROP_CACHE_DIR="/custom/path"
tp model download "model-name"
```

#### Memory Issues
```bash
# Check system resources before using large models
tp model info "nomic-embed-code.Q5_K_S.gguf"

# Use streaming for large repositories
tp index --repo . --model "model-name" --streaming
```

#### Performance Issues
```bash
# Run benchmark to compare models
tp benchmark --models "sentence-transformers/all-MiniLM-L6-v2,nomic-embed-code.Q5_K_S.gguf"

# Monitor resource usage
tp index --repo . --model "model-name" --verbose
```

## Migration Guide

### From Single Model to Multi-Model

If you're upgrading from a previous version:

1. **Existing indexes remain compatible** - no re-indexing required
2. **Default behavior unchanged** - same model used if not specified
3. **Gradual migration** - try new models on test projects first

### Re-indexing Considerations

You may want to re-index when:
- Switching to a specialized model for better accuracy
- Using instruction-based embeddings
- Changing from single-language to multilingual model

```bash
# Re-index with new model
tp index --repo . --model "Qwen/Qwen3-Embedding-0.6B" --force-rebuild
```

## See Also

- **[README](README.md)** - Getting started guide and basic usage
- **[Configuration Guide](CONFIGURATION.md)** - Model configuration options and settings
- **[API Reference](docs/API.md)** - Programmatic model management and usage
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Model-specific issues and solutions
- **[Installation Guide](INSTALLATION.md)** - Installing models and dependencies