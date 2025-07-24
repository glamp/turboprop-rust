# Migration Guide: Adding Model Support

## Overview

This guide helps existing TurboProp users migrate to the new multi-model support introduced in version X.X.

## What's Changed

### New Features
- Multiple embedding model support
- Specialized code models (GGUF)
- Multilingual models (Qwen3)
- Instruction-based embeddings
- Enhanced CLI with model management

### Backward Compatibility
- **Existing indexes work unchanged** - no re-indexing required
- **CLI commands unchanged** - same default behavior
- **Configuration files compatible** - new options are additive

## Migration Steps

### Step 1: Update TurboProp
```bash
# Update to latest version
cargo install turboprop --force

# Verify new model support
tp model list
```

### Step 2: Explore Available Models
```bash
# See all available models
tp model list

# Get detailed information
tp model info "nomic-embed-code.Q5_K_S.gguf"
tp model info "Qwen/Qwen3-Embedding-0.6B"
```

### Step 3: Test New Models (Optional)
```bash
# Create test index with new model
tp index --repo ./test-project --model "nomic-embed-code.Q5_K_S.gguf"

# Compare search results
tp search "authentication" --model "sentence-transformers/all-MiniLM-L6-v2"
tp search "authentication" --model "nomic-embed-code.Q5_K_S.gguf"
```

### Step 4: Update Configuration (Optional)
```yaml
# .turboprop.yml - new optional settings
default_model: "sentence-transformers/all-MiniLM-L6-v2"  # explicit default

models:
  "Qwen/Qwen3-Embedding-0.6B":
    instruction: "Represent this code for semantic search"
```

## When to Re-index

Re-indexing is **optional** but recommended when:

1. **Switching to specialized models**: For code-heavy projects, `nomic-embed-code.Q5_K_S.gguf` may provide better results
2. **Adding multilingual support**: For international projects, `Qwen/Qwen3-Embedding-0.6B` offers better multilingual understanding
3. **Using instruction-based search**: Qwen3 with instructions can improve search relevance

```bash
# Re-index with new model
tp index --repo . --model "new-model-name" --force-rebuild
```

## Performance Considerations

### Model Sizes and Download Times
- Sentence transformers: < 50MB, instant
- Qwen3: ~600MB, moderate download
- Nomic code: ~2.5GB, long initial download

### Runtime Performance
```bash
# Compare model performance
tp benchmark --models "sentence-transformers/all-MiniLM-L6-v2,nomic-embed-code.Q5_K_S.gguf"
```

## Configuration File Changes

### Before (v1.x)
```yaml
# .turboprop.yml
max_filesize: "2mb"
model: "sentence-transformers/all-MiniLM-L6-v2"
cache_dir: "~/.turboprop-cache"
worker_threads: 4
batch_size: 32
```

### After (v2.x) - Backward Compatible
```yaml
# .turboprop.yml - all old settings still work
max_filesize: "2mb"
model: "sentence-transformers/all-MiniLM-L6-v2"  # still works
cache_dir: "~/.turboprop-cache"
worker_threads: 4
batch_size: 32

# New optional settings
default_model: "sentence-transformers/all-MiniLM-L6-v2"  # preferred way
models:
  "Qwen/Qwen3-Embedding-0.6B":
    instruction: "Represent this code for semantic search"
    cache_dir: "~/.turboprop/qwen3-cache"
  
  "nomic-embed-code.Q5_K_S.gguf":
    batch_size: 8  # smaller batches for large models

# Resource management (new)
max_memory_usage: "8GB"
warn_large_models: true
```

## CLI Command Changes

### Unchanged Commands
All existing commands work exactly the same:
```bash
# These work identically to before
tp index --repo .
tp search "query" --repo .
tp index --repo . --model "sentence-transformers/all-MiniLM-L6-v2"
```

### New Commands
```bash
# Model management (new)
tp model list
tp model info "model-name"
tp model download "model-name"
tp model clear

# Benchmarking (new)
tp benchmark --models "model1,model2"

# Enhanced search with instructions (new)
tp search "query" --model "Qwen/Qwen3-Embedding-0.6B" --instruction "Find code related to X"
```

## Migration Scenarios

### Scenario 1: Stay with Current Setup
**Who**: Users happy with current performance
**Action**: No changes required
**Result**: Everything continues working as before

```bash
# No changes needed - existing workflows unchanged
tp index --repo .
tp search "query" --repo .
```

### Scenario 2: Improve Code Search Accuracy
**Who**: Users wanting better code understanding
**Action**: Switch to code-specialized model
**Result**: Better semantic understanding of code

```bash
# Test the specialized model
tp index --repo . --model "nomic-embed-code.Q5_K_S.gguf"
tp search "authentication logic" --model "nomic-embed-code.Q5_K_S.gguf"

# If satisfied, update config
echo 'default_model: "nomic-embed-code.Q5_K_S.gguf"' > .turboprop.yml
```

### Scenario 3: Add Multilingual Support
**Who**: Teams with international codebases
**Action**: Use multilingual model with instructions
**Result**: Better search across different languages

```bash
# Index with multilingual model
tp index --repo . --model "Qwen/Qwen3-Embedding-0.6B" \
  --instruction "Represent this code for semantic search"

# Update configuration
cat > .turboprop.yml << EOF
default_model: "Qwen/Qwen3-Embedding-0.6B"
models:
  "Qwen/Qwen3-Embedding-0.6B":
    instruction: "Represent this code for semantic search"
EOF
```

### Scenario 4: Gradual Migration
**Who**: Cautious users wanting to test incrementally
**Action**: Test new models on specific projects
**Result**: Validate improvements before full migration

```bash
# Create separate test index
mkdir test-migration
cd test-migration
tp index --repo ../main-project --model "nomic-embed-code.Q5_K_S.gguf"

# Compare results
tp search "error handling" --repo . > new-results.txt
cd ../main-project
tp search "error handling" --repo . > old-results.txt

# Compare and decide
diff old-results.txt ../test-migration/new-results.txt
```

## Troubleshooting Migration

### Issue: New commands not recognized
```bash
$ tp model list
error: unrecognized subcommand 'model'
```
**Solution**: Ensure you've installed the latest version
```bash
tp --version  # Should show new version
cargo install turboprop --force  # If version is old
```

### Issue: Model download failures
```bash
Error: Failed to download model
```
**Solution**: Check network and disk space
```bash
# Check available space
df -h ~/.turboprop

# Try with verbose output
tp model download "model-name" --verbose

# Use custom cache directory if needed
export TURBOPROP_CACHE_DIR="/custom/path"
tp model download "model-name"
```

### Issue: Performance degradation
```bash
# Old searches were faster
```
**Solution**: Check system resources and model requirements
```bash
# Check model memory requirements
tp model info "current-model"

# Consider using lighter model for resource-constrained environments
tp index --repo . --model "sentence-transformers/all-MiniLM-L6-v2"

# Optimize batch sizes
cat > .turboprop.yml << EOF
embedding:
  batch_size: 16  # Reduce if memory constrained
EOF
```

### Issue: Existing indexes incompatible
```bash
Error: Index format not supported
```
**Solution**: This shouldn't happen, but if it does:
```bash
# Backup existing index
cp -r .turboprop-index .turboprop-index.backup

# Re-index with same model (should be compatible)
tp index --repo . --force-rebuild
```

### Issue: Configuration conflicts
```bash
Error: Invalid configuration
```
**Solution**: Check for conflicting settings
```bash
# Validate configuration
tp index --repo . --dry-run

# Use minimal config initially
cat > .turboprop.yml << EOF
default_model: "sentence-transformers/all-MiniLM-L6-v2"
EOF
```

## Rollback Procedure

If you need to revert to the previous version:

```bash
# 1. Install previous version (if available)
cargo install turboprop --version "1.x.x"

# 2. Remove new configuration options
# Edit .turboprop.yml to remove model-specific sections

# 3. Existing indexes should still work
tp search "test query" --repo .
```

## Validation Checklist

After migration, validate that everything works:

- [ ] `tp --version` shows expected version
- [ ] `tp model list` shows available models
- [ ] Existing indexes still work: `tp search "test" --repo .`
- [ ] New model works: `tp model info "new-model"`
- [ ] Configuration loads: `tp index --repo . --dry-run`
- [ ] Performance is acceptable: `tp benchmark`

## Getting Help

If you encounter issues during migration:

1. **Check documentation**: Review README.md and MODELS.md
2. **Run diagnostics**: Use `--verbose` flag for detailed output
3. **Test minimal setup**: Start with basic configuration
4. **Compare versions**: Use `tp --version` to confirm update
5. **Check system resources**: Ensure adequate memory/disk space

For additional support, see the main README.md troubleshooting section.