# TurboProp Configuration Guide

TurboProp can be configured through command-line arguments, configuration files, and environment variables. This guide covers all available configuration options.

## Configuration Priority

Configuration is applied in the following order (highest to lowest priority):

1. **Command-line arguments** (highest priority)
2. **Configuration file** (`.turboprop.yml`)
3. **Environment variables**
4. **Default values** (lowest priority)

## Configuration File

Create a `.turboprop.yml` file in your repository root or home directory for persistent settings:

```yaml
# .turboprop.yml - Complete configuration example

# === Indexing Configuration ===
max_filesize: "2mb"              # Maximum file size to index
model: "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
cache_dir: "~/.turboprop-cache"  # Model cache directory
worker_threads: 4                # Number of processing threads
batch_size: 32                   # Embedding batch size
watch_debounce: 500              # File watch debounce time (ms)

# === Search Configuration ===
default_limit: 10                # Default number of search results
similarity_threshold: 0.3        # Default similarity threshold
default_output: "json"           # Default output format (json/text)

# === File Filtering ===
include_patterns:                # Files to include (glob patterns)
  - "**/*.rs"
  - "**/*.js"
  - "**/*.ts"
  - "**/*.py"
  - "**/*.go"
  - "**/*.java"
  - "**/*.c"
  - "**/*.cpp"
  - "**/*.h"
  - "**/*.hpp"

exclude_patterns:                # Files to exclude (glob patterns)
  - "**/target/**"
  - "**/node_modules/**"
  - "**/.git/**"
  - "**/build/**"
  - "**/dist/**"
  - "**/*.min.js"
  - "**/*.bundle.js"

# === Performance Tuning ===
chunk_size: 1000                 # Characters per text chunk
chunk_overlap: 100               # Overlap between chunks
max_concurrent_files: 50         # Max files processed concurrently
memory_limit: "1gb"              # Memory limit for operations

# === Logging and Debug ===
log_level: "info"               # error, warn, info, debug, trace
verbose: false                  # Enable verbose output
quiet: false                    # Suppress non-essential output

# === Storage Configuration ===
index_dir: ".turboprop"         # Index storage directory
compression_enabled: true       # Enable index compression
backup_count: 3                 # Number of index backups to keep
```

### Minimal Configuration

For most users, a minimal configuration is sufficient:

```yaml
# .turboprop.yml - Minimal configuration
max_filesize: "2mb"
model: "sentence-transformers/all-MiniLM-L6-v2"
worker_threads: 4
default_output: "text"
```

## Environment Variables

All configuration options can be set via environment variables using the `TURBOPROP_` prefix:

```bash
# General settings
export TURBOPROP_MAX_FILESIZE="5mb"
export TURBOPROP_MODEL="sentence-transformers/all-MiniLM-L12-v2"
export TURBOPROP_CACHE_DIR="~/.cache/turboprop"
export TURBOPROP_WORKER_THREADS=8
export TURBOPROP_BATCH_SIZE=64

# Search settings
export TURBOPROP_DEFAULT_LIMIT=20
export TURBOPROP_SIMILARITY_THRESHOLD=0.5
export TURBOPROP_DEFAULT_OUTPUT="json"

# Performance settings
export TURBOPROP_CHUNK_SIZE=1500
export TURBOPROP_CHUNK_OVERLAP=150
export TURBOPROP_MAX_CONCURRENT_FILES=100
export TURBOPROP_MEMORY_LIMIT="2gb"

# Logging
export TURBOPROP_LOG_LEVEL="debug"
export TURBOPROP_VERBOSE=true

# Rust-specific logging
export RUST_LOG="turboprop=debug,info"
```

Add these to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) to make them persistent.

## Command-Line Arguments

All configuration options can be overridden via command-line arguments:

### Index Command Options

```bash
tp index [OPTIONS] --repo <REPO>

OPTIONS:
    --repo <PATH>              Repository path to index [default: .]
    --max-filesize <SIZE>      Maximum file size (e.g., "2mb", "500kb", "1gb")
    --model <MODEL>            Embedding model name
    --cache-dir <DIR>          Cache directory for models
    --worker-threads <NUM>     Number of worker threads
    --batch-size <NUM>         Batch size for embeddings [default: 32]
    --chunk-size <NUM>         Characters per chunk [default: 1000]
    --chunk-overlap <NUM>      Overlap between chunks [default: 100]
    --watch                    Monitor file changes
    --watch-debounce <MS>      Watch debounce time in milliseconds [default: 500]
    --verbose                  Enable verbose output
    --quiet                    Suppress non-essential output
    --force                    Force reindex of all files
    --exclude <PATTERN>        Exclude pattern (can be used multiple times)
    --include <PATTERN>        Include pattern (can be used multiple times)
```

### Search Command Options

```bash
tp search <QUERY> [OPTIONS]

OPTIONS:
    --repo <PATH>              Repository path to search [default: .]
    --limit <NUM>              Maximum results to return [default: 10]
    --threshold <FLOAT>        Similarity threshold (0.0-1.0)
    --output <FORMAT>          Output format: json, text [default: json]
    --filetype <EXT>           Filter by file extension
    --sort-by <FIELD>          Sort by: score, file, date [default: score]
    --include-content          Include full content in results
    --exclude-pattern <PATTERN> Exclude files matching pattern
    --since <DATE>             Only search files modified since date
    --context-lines <NUM>      Lines of context around matches [default: 3]
```

## Configuration Options Reference

### Model Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | String | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face model name |
| `cache_dir` | Path | Platform default | Model cache directory |
| `model_revision` | String | `main` | Model revision/branch |
| `model_auth_token` | String | None | Hugging Face auth token |

**Available Models:**
- `sentence-transformers/all-MiniLM-L6-v2` (default, 384 dims, ~90MB)
- `sentence-transformers/all-MiniLM-L12-v2` (384 dims, ~130MB)
- `sentence-transformers/all-mpnet-base-v2` (768 dims, ~420MB, higher quality)
- `sentence-transformers/paraphrase-MiniLM-L6-v2` (384 dims, ~90MB)

### Performance Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `worker_threads` | Integer | CPU cores | Number of processing threads |
| `batch_size` | Integer | 32 | Embedding batch size |
| `chunk_size` | Integer | 1000 | Characters per text chunk |
| `chunk_overlap` | Integer | 100 | Overlap between chunks |
| `max_concurrent_files` | Integer | 50 | Max files processed concurrently |
| `memory_limit` | String | `1gb` | Memory limit for operations |
| `max_filesize` | String | `2mb` | Maximum file size to index |

### Search Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `default_limit` | Integer | 10 | Default number of search results |
| `similarity_threshold` | Float | 0.3 | Default similarity threshold (0.0-1.0) |
| `default_output` | String | `json` | Default output format |
| `context_lines` | Integer | 3 | Lines of context around matches |
| `max_results` | Integer | 1000 | Maximum possible results |

### File Filtering

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `include_patterns` | Array | Common code files | Glob patterns to include |
| `exclude_patterns` | Array | Build/cache dirs | Glob patterns to exclude |
| `follow_symlinks` | Boolean | false | Follow symbolic links |
| `respect_gitignore` | Boolean | true | Respect .gitignore rules |
| `hidden_files` | Boolean | false | Include hidden files |

### Storage Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `index_dir` | String | `.turboprop` | Index storage directory |
| `compression_enabled` | Boolean | true | Enable index compression |
| `backup_count` | Integer | 3 | Number of index backups |
| `cleanup_on_exit` | Boolean | false | Clean temporary files on exit |

### Watch Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `watch_debounce` | Integer | 500 | File change debounce time (ms) |
| `watch_recursive` | Boolean | true | Watch subdirectories |
| `watch_hidden` | Boolean | false | Watch hidden files |
| `auto_index_on_change` | Boolean | true | Auto-index changed files |

## Per-Project Configuration

You can have different configurations for different projects by placing `.turboprop.yml` in each project root:

```
project1/
├── .turboprop.yml          # Project-specific config
├── src/
└── ...

project2/
├── .turboprop.yml          # Different config
├── lib/
└── ...
```

## Global Configuration

Place a global configuration in your home directory:

```bash
# Linux/macOS
~/.turboprop.yml

# Windows
%USERPROFILE%\.turboprop.yml
```

Project-specific configs override global settings.

## Configuration Examples

### High-Performance Setup
```yaml
# For powerful machines with lots of memory
worker_threads: 16
batch_size: 128
max_concurrent_files: 200
memory_limit: "4gb"
model: "sentence-transformers/all-mpnet-base-v2"
chunk_size: 2000
```

### Low-Resource Setup
```yaml
# For machines with limited resources
worker_threads: 2
batch_size: 8
max_concurrent_files: 10
memory_limit: "512mb"
model: "sentence-transformers/all-MiniLM-L6-v2"
chunk_size: 500
```

### JavaScript/TypeScript Project
```yaml
include_patterns:
  - "**/*.js"
  - "**/*.ts"
  - "**/*.jsx"
  - "**/*.tsx"
  - "**/*.vue"

exclude_patterns:
  - "**/node_modules/**"
  - "**/dist/**"
  - "**/build/**"
  - "**/*.min.js"
  - "**/*.bundle.js"

max_filesize: "1mb"
```

### Python Project
```yaml
include_patterns:
  - "**/*.py"
  - "**/*.pyi"

exclude_patterns:
  - "**/__pycache__/**"
  - "**/venv/**"
  - "**/env/**"
  - "**/.pytest_cache/**"
  - "**/build/**"
  - "**/dist/**"

max_filesize: "5mb"
```

### Large Codebase
```yaml
# Configuration for large codebases (10,000+ files)
worker_threads: 12
batch_size: 64
max_concurrent_files: 100
memory_limit: "8gb"
chunk_size: 800
chunk_overlap: 80
compression_enabled: true
backup_count: 1
```

## Configuration Validation

TurboProp validates your configuration on startup. Common validation errors:

**Invalid file size format:**
```yaml
max_filesize: "2MB"  # ❌ Wrong (uppercase)
max_filesize: "2mb"  # ✅ Correct (lowercase)
```

**Invalid threshold range:**
```yaml
similarity_threshold: 1.5  # ❌ Wrong (> 1.0)
similarity_threshold: 0.8  # ✅ Correct (0.0-1.0)
```

**Invalid worker thread count:**
```yaml
worker_threads: 0   # ❌ Wrong (must be > 0)
worker_threads: 4   # ✅ Correct
```

## Troubleshooting Configuration

### Debug Configuration Loading
```bash
# Enable debug logging to see configuration loading
export RUST_LOG=turboprop=debug
tp index --repo . --verbose
```

### Validate Configuration
```bash
# TurboProp will report configuration errors on startup
tp --help  # Basic validation
tp index --repo . --dry-run  # Full validation (if supported)
```

### Common Issues

**Configuration file not found:**
- Check file name: `.turboprop.yml` (with leading dot)
- Check file location: project root or home directory
- Check file permissions: must be readable

**Model download fails:**
- Check `cache_dir` permissions
- Verify internet connectivity
- Try a different model

**Performance issues:**
- Reduce `batch_size` if out of memory
- Reduce `worker_threads` if CPU usage too high
- Increase `chunk_size` for better performance with large files

## Best Practices

1. **Start Simple**: Begin with minimal configuration and add options as needed
2. **Profile Performance**: Use different settings and measure indexing time
3. **Version Control**: Include `.turboprop.yml` in your repository
4. **Document Changes**: Comment your configuration choices
5. **Test Settings**: Verify configuration works with your specific codebase

## Advanced Configuration

### Custom Model Configuration
```yaml
# Using a custom model
model: "your-org/custom-code-model"
model_revision: "v1.2"
model_auth_token: "${HUGGINGFACE_TOKEN}"  # Environment variable
```

### Conditional Configuration
```yaml
# Different settings based on environment
development:
  log_level: "debug"
  verbose: true

production:
  log_level: "warn"
  quiet: true
```

### Integration with CI/CD
```yaml
# Optimized for CI environments
ci_mode: true
worker_threads: 2
batch_size: 16
cache_dir: "/tmp/turboprop-cache"
log_level: "warn"
```