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
```

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