# TurboProp

[![Crates.io](https://img.shields.io/crates/v/turboprop.svg)](https://crates.io/crates/turboprop)
[![Documentation](https://docs.rs/turboprop/badge.svg)](https://docs.rs/turboprop)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/your-org/turboprop-rust)

**TurboProp** (`tp`) is a fast semantic code search and indexing tool written in Rust. It uses machine learning embeddings to enable intelligent code search across your codebase, making it easy to find relevant code snippets based on natural language queries.

## Key Features

- **Semantic Search**: Find code by meaning, not just keywords
- **Git Integration**: Respects `.gitignore` and only indexes files under source control
- **Watch Mode**: Automatically updates the index when files change
- **File Type Filtering**: Search within specific file types
- **Multiple Output Formats**: JSON for tools, human-readable text for reading
- **Performance Optimized**: Handles codebases from 50 to 10,000+ files
- **Easy Configuration**: Optional `.turboprop.yml` configuration file

## Quick Start

### Installation

#### Via Cargo (Recommended)
```bash
cargo install turboprop
```

#### From Source
```bash
git clone https://github.com/your-org/turboprop-rust
cd turboprop-rust
cargo build --release
# Binary will be in target/release/tp
```

### Basic Usage

1. **Index your codebase**:
   ```bash
   tp index --repo . --max-filesize 2mb
   ```

2. **Search for code**:
   ```bash
   tp search "jwt authentication" --repo .
   ```

3. **Filter by file type**:
   ```bash
   tp search --filetype .js "jwt authentication" --repo .
   ```

4. **Get human-readable output**:
   ```bash
   tp search "jwt authentication" --repo . --output text
   ```

## Model Support

TurboProp now supports multiple embedding models to optimize for different use cases:

### Available Models

#### Sentence Transformer Models (FastEmbed)
- `sentence-transformers/all-MiniLM-L6-v2` (default)
  - Fast and lightweight, good for general use
  - 384 dimensions, ~23MB
  - Automatic download and caching

- `sentence-transformers/all-MiniLM-L12-v2`
  - Better accuracy with slightly more compute
  - 384 dimensions, ~44MB

#### Specialized Code Models
- `nomic-embed-code.Q5_K_S.gguf`
  - Specialized for code search and retrieval
  - 768 dimensions, ~2.5GB
  - Supports multiple programming languages
  - Quantized for efficient inference

#### Multilingual Models
- `Qwen/Qwen3-Embedding-0.6B`
  - State-of-the-art multilingual support (100+ languages)
  - 1024 dimensions, ~600MB
  - Supports instruction-based embeddings
  - Excellent for code and text retrieval

### Model Selection Guide

Choose your model based on your use case:

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| General code search | `sentence-transformers/all-MiniLM-L6-v2` | Fast, reliable, good balance |
| Specialized code search | `nomic-embed-code.Q5_K_S.gguf` | Optimized for code understanding |
| Multilingual projects | `Qwen/Qwen3-Embedding-0.6B` | Best multilingual support |
| Low resource environments | `sentence-transformers/all-MiniLM-L6-v2` | Smallest memory footprint |
| Maximum accuracy | `Qwen/Qwen3-Embedding-0.6B` | State-of-the-art performance |

### Usage Examples

#### Basic Model Selection
```bash
# List available models
tp model list

# Get model information
tp model info "Qwen/Qwen3-Embedding-0.6B"

# Download a model before use
tp model download "nomic-embed-code.Q5_K_S.gguf"
```

#### Indexing with Different Models
```bash
# Use default model
tp index --repo ./my-project

# Use specialized code model
tp index --repo ./my-project --model "nomic-embed-code.Q5_K_S.gguf"

# Use multilingual model with instruction
tp index --repo ./my-project \
  --model "Qwen/Qwen3-Embedding-0.6B" \
  --instruction "Represent this code for semantic search"
```

#### Searching with Model Consistency
```bash
# Search using the same model used for indexing
tp search "jwt authentication" --model "nomic-embed-code.Q5_K_S.gguf"

# Use instruction for context-aware search (Qwen3 only)
tp search "error handling" \
  --model "Qwen/Qwen3-Embedding-0.6B" \
  --instruction "Find code related to error handling and exceptions"
```

### Configuration File Support

Create `.turboprop.yml` in your project root:
```yaml
# Default model for all operations
default_model: "sentence-transformers/all-MiniLM-L6-v2"

# Model-specific configurations
models:
  "Qwen/Qwen3-Embedding-0.6B":
    instruction: "Represent this code for semantic search"
    cache_dir: "~/.turboprop/qwen3-cache"
  
  "nomic-embed-code.Q5_K_S.gguf":
    cache_dir: "~/.turboprop/nomic-cache"

# Performance settings
embedding:
  batch_size: 32
  cache_embeddings: true
  
# Resource limits
max_memory_usage: "8GB"
warn_large_models: true
```

## Complete Usage Guide

### Indexing Command

The `index` command creates a searchable index of your codebase:

```bash
tp index [OPTIONS] --repo <REPO>
```

#### Options:
- `--repo <PATH>`: Repository path to index (default: current directory)
- `--max-filesize <SIZE>`: Maximum file size to index (e.g., "2mb", "500kb", "1gb")
- `--watch`: Monitor file changes and update index automatically
- `--model <MODEL>`: Embedding model to use (default: "sentence-transformers/all-MiniLM-L6-v2")
- `--cache-dir <DIR>`: Cache directory for models and data
- `--worker-threads <N>`: Number of worker threads for processing
- `--batch-size <N>`: Batch size for embedding generation (default: 32)
- `--verbose`: Enable verbose output

#### Examples:

```bash
# Basic indexing
tp index --repo .

# Index with size limit and watch mode
tp index --repo . --max-filesize 2mb --watch

# Use custom model and cache directory
tp index --repo . --model "sentence-transformers/all-MiniLM-L12-v2" --cache-dir ~/.turboprop-cache

# Index with custom performance settings
tp index --repo . --worker-threads 8 --batch-size 64
```

### Search Command

The `search` command finds relevant code using semantic similarity:

```bash
tp search <QUERY> [OPTIONS]
```

#### Options:
- `<QUERY>`: Search query (natural language or keywords)
- `--repo <PATH>`: Repository path to search in (default: current directory)
- `--limit <N>`: Maximum number of results to return (default: 10)
- `--threshold <FLOAT>`: Minimum similarity threshold (0.0 to 1.0)
- `--output <FORMAT>`: Output format: 'json' (default) or 'text'
- `--filetype <EXT>`: Filter results by file extension (e.g., '.rs', '.js', '.py')
- `--filter <PATTERN>`: Filter results by glob pattern (e.g., '*.rs', 'src/**/*.js')

#### Examples:

```bash
# Basic search
tp search "user authentication" --repo .

# Search with filters and limits
tp search "database connection" --repo . --filetype .rs --limit 5

# Get human-readable output
tp search "error handling" --repo . --output text

# High-precision search
tp search "jwt token validation" --repo . --threshold 0.8

# Search in specific directory
tp search "api routes" --repo ./backend

# Filter by glob pattern
tp search "authentication" --repo . --filter "src/*.js"

# Recursive glob patterns
tp search "error handling" --repo . --filter "**/*.{rs,py}"

# Combine filters
tp search "database" --repo . --filetype .rs --filter "src/**/*.rs"
```

## Glob Pattern Filtering

TurboProp supports powerful glob pattern filtering to search within specific files or directories. Glob patterns use Unix shell-style wildcards to match file paths.

### Basic Wildcards

| Wildcard | Description | Example |
|----------|-------------|---------|
| `*` | Match any characters within a directory | `*.rs` matches all Rust files |
| `?` | Match exactly one character | `file?.rs` matches `file1.rs`, `fileA.rs` |
| `**` | Match any characters across directories | `**/*.js` matches JS files anywhere |
| `[abc]` | Match any character in the set | `file[123].rs` matches `file1.rs`, `file2.rs`, `file3.rs` |
| `[!abc]` | Match any character NOT in the set | `file[!0-9].rs` matches `filea.rs` but not `file1.rs` |
| `{a,b}` | Match any of the alternatives | `*.{js,ts}` matches both `.js` and `.ts` files |

### Common Pattern Examples

#### File Type Filtering
```bash
# All Rust files anywhere in the codebase
tp search "async function" --filter "*.rs"

# All JavaScript and TypeScript files
tp search "react component" --filter "*.{js,ts,jsx,tsx}"

# All configuration files
tp search "database" --filter "*.{json,yaml,yml,toml,ini}"
```

#### Directory-Specific Filtering
```bash
# Files only in the src directory
tp search "main function" --filter "src/*.rs"

# Files only in tests directory
tp search "test case" --filter "tests/*.py"

# Files in specific subdirectories
tp search "handler" --filter "src/api/*.js"
```

#### Recursive Directory Filtering
```bash
# Python files anywhere in the project
tp search "authentication" --filter "**/*.py"

# Test files in any subdirectory
tp search "unit test" --filter "**/test_*.rs"

# Source files in src and all subdirectories
tp search "database connection" --filter "src/**/*.{rs,py,js}"

# Handler files in nested API directories
tp search "request handler" --filter "**/api/**/handlers/*.rs"
```

#### Advanced Pattern Examples
```bash
# Test files with specific naming patterns
tp search "integration test" --filter "tests/**/*_{test,spec}.{js,ts}"

# Source files excluding certain directories
tp search "function definition" --filter "src/**/*.rs" --filter "!**/target/**"

# Files in multiple specific directories
tp search "configuration" --filter "{src,config,scripts}/**/*.{json,yaml}"

# Files with numeric suffixes
tp search "version" --filter "**/*[0-9].{js,py,rs}"
```

### Pattern Behavior

**Path Matching**: Patterns match against the entire file path, not just the filename:
- `*.rs` matches `main.rs`, `src/main.rs`, and `lib/nested/file.rs`
- `src/*.rs` matches `src/main.rs` but not `src/nested/file.rs`
- `src/**/*.rs` matches both `src/main.rs` and `src/nested/file.rs`

**Case Sensitivity**: Patterns are case-sensitive by default:
- `*.RS` matches `FILE.RS` but not `file.rs`
- `*.rs` matches `file.rs` but not `FILE.RS`

**Path Separators**: Always use forward slashes (`/`) in patterns:
- ✅ `src/api/*.js` (correct)
- ❌ `src\\api\\*.js` (incorrect)

**Combining with File Type Filter**: You can use both `--filter` and `--filetype` together:
```bash
# Search for Rust files in src directory only
tp search "async" --filetype .rs --filter "src/**/*"
```

### Performance Tips

- **Simple patterns are faster**: `*.rs` is faster than `**/*.rs`
- **Be specific when possible**: `src/*.js` is faster than `**/*.js` if you know files are in `src/`
- **Avoid excessive wildcards**: Patterns with many `**` can be slower on large codebases
- **Use file type filter for extensions**: `--filetype .rs` is optimized compared to `--filter "*.rs"`

### Troubleshooting Glob Patterns

**Pattern doesn't match expected files**:
- Check case sensitivity: `*.RS` vs `*.rs`
- Verify path structure: `src/*.js` only matches direct children of `src/`
- Use `**` for recursive matching: `src/**/*.js` matches nested files

**Pattern matching too many files**:
- Be more specific: use `src/*.js` instead of `*.js`
- Add more path components: `src/components/*.jsx`
- Use character classes: `test_[0-9]*.rs` instead of `test_*.rs`

**Complex patterns not working**:
- Test simpler patterns first: start with `*.ext` then add complexity
- Check for typos in braces: `{js,ts}` not `{js, ts}` (no spaces)
- Validate bracket expressions: `[a-z]` not `[a-Z]`

For more pattern examples and troubleshooting, see the `TROUBLESHOOTING.md` file.

## Configuration

TurboProp supports optional configuration via a `.turboprop.yml` file in your repository root:

```yaml
# .turboprop.yml
max_filesize: "2mb"
model: "sentence-transformers/all-MiniLM-L6-v2"
cache_dir: "~/.turboprop-cache"
worker_threads: 4
batch_size: 32
default_output: "json"
similarity_threshold: 0.3
```

## Output Formats

### JSON Output (Default)
```json
{
  "file": "src/auth.rs",
  "score": 0.8234,
  "content": "fn authenticate_user(token: &str) -> Result<User, AuthError> { ... }"
}
```

### Text Output
```
Score: 0.82 | src/auth.rs
fn authenticate_user(token: &str) -> Result<User, AuthError> {
    // JWT token validation logic
    ...
}
```

## Performance Characteristics

- **Indexing Speed**: ~100-500 files/second (depending on file size and hardware)
- **Search Speed**: ~10-50ms per query (after initial model loading)
- **Memory Usage**: ~50-200MB (varies with model and index size)
- **Storage**: Index size is typically 10-30% of source code size

### Recommended Limits
- **File Count**: Up to 10,000 files (tested)
- **File Size**: Up to 2MB per file (configurable)
- **Total Codebase**: Up to 500MB of source code

## Supported File Types

TurboProp works with any text-based file but is optimized for common programming languages:

- **Web**: `.js`, `.ts`, `.jsx`, `.tsx`, `.html`, `.css`, `.scss`, `.vue`
- **Backend**: `.py`, `.rs`, `.go`, `.java`, `.kt`, `.scala`, `.rb`, `.php`
- **Systems**: `.c`, `.cpp`, `.h`, `.hpp`, `.cs`, `.swift`
- **Data**: `.sql`, `.json`, `.yaml`, `.yml`, `.xml`, `.toml`
- **Docs**: `.md`, `.txt`, `.rst`
- **Config**: `.env`, `.ini`, `.conf`, `.cfg`

## Integration Examples

### With Git Hooks
Add to `.git/hooks/post-commit`:
```bash
#!/bin/bash
tp index --repo . --max-filesize 2mb
```

### With IDEs
Many IDEs can be configured to run external tools. Add TurboProp as a custom search tool.

### With CI/CD
```bash
# In your CI script
tp index --repo . --max-filesize 2mb
tp search "security vulnerability" --repo . --output json > security-search-results.json
```

## Troubleshooting

### Common Issues

**Index not found**
```bash
Error: No index found in repository
```
Solution: Run `tp index --repo .` first to create an index.

**Model download fails**
```bash
Error: Failed to download model
```
Solution: Check internet connection or specify a local cache directory with `--cache-dir`.

**Large files skipped**
```bash
Warning: Skipping large file (>2MB)
```
Solution: Increase limit with `--max-filesize 5mb` or exclude large files.

**Out of memory**
```bash
Error: Out of memory during indexing
```
Solution: Reduce `--batch-size` or `--worker-threads`, or exclude large files.

### Getting Help

```bash
tp --help              # General help
tp index --help        # Index command help
tp search --help       # Search command help
```

## Development

### Building from Source
```bash
git clone https://github.com/your-org/turboprop-rust
cd turboprop-rust
cargo build --release
```

### Running Tests
```bash
cargo test                    # Run all tests
cargo test --test integration # Run integration tests only
cargo bench                   # Run benchmarks
```

### Dependencies
- **clap**: CLI parsing and help generation
- **tokio**: Async runtime for I/O operations  
- **serde**: JSON serialization
- **fastembed**: Machine learning embeddings
- **git2**: Git repository integration
- **notify**: File system watching
- **walkdir**: Directory traversal

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for your changes
4. Ensure all tests pass: `cargo test`
5. Submit a pull request

## License

Licensed under either of:
- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.