# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Building and Testing
```bash
# Build the project (development)
cargo build

# Build for production (optimized)
cargo build --release

# Run all tests (unit + integration)
cargo test

# Run specific test suites
cargo test --test integration_tests    # Integration tests
cargo test --test complete_workflow    # End-to-end workflow tests
cargo test chunking_tests             # Unit tests for chunking module

# Run benchmarks
cargo bench

# Fast compilation check without building
cargo check
```

### Code Quality
```bash
# Run clippy linter
cargo clippy

# Format code
cargo fmt

# Check formatting without changing files
cargo fmt --check
```

### Running the CLI Tool
```bash
# Run the CLI tool during development
cargo run --bin tp -- index --repo . --max-filesize 2mb
cargo run --bin tp -- search "jwt authentication" --repo .

# After building, binary is available as:
./target/release/tp index --repo .
./target/release/tp search "error handling"
```

### Documentation and Manual Pages
```bash
# Install man pages (requires build first)
./install-man-pages.sh

# View manual after installation
man tp
```

## Architecture Overview

TurboProp is a semantic code search and indexing tool built in Rust. The architecture follows a multi-stage pipeline design:

### Core Pipeline Flow
1. **File Discovery** (`src/files.rs`, `src/git.rs`) - Discovers files to index using git integration and filters
2. **Content Processing** (`src/content.rs`) - Reads and preprocesses file content 
3. **Chunking** (`src/chunking.rs`) - Breaks large files into searchable chunks
4. **Embedding Generation** (`src/embeddings.rs`) - Creates ML vector embeddings using fastembed
5. **Index Storage** (`src/index.rs`, `src/storage.rs`) - Persists embeddings and metadata for fast retrieval

### Key Components

**CLI Interface** (`src/cli.rs`, `src/commands/`)
- Built with clap for argument parsing
- Two main commands: `index` and `search`
- Enhanced command implementations in `src/commands/` with progress tracking

**Configuration System** (`src/config.rs`)
- YAML configuration support via `.turboprop.yml`
- Hierarchical config: CLI args → config file → environment → defaults
- Strongly typed configuration structs for embedding, chunking, and file discovery

**Semantic Search Engine** (`src/search.rs`, `src/embeddings.rs`)
- Uses HuggingFace sentence-transformer models via fastembed
- Default model: `sentence-transformers/all-MiniLM-L6-v2`
- Cosine similarity search with configurable thresholds

**Persistent Storage** (`src/index.rs`, `src/storage.rs`)
- `PersistentChunkIndex` extends in-memory `ChunkIndex` with disk persistence
- Incremental updates to avoid full re-indexing
- Binary serialization for fast loading

**File Processing** (`src/files.rs`, `src/filters.rs`)
- Git-aware file discovery (respects .gitignore, only indexes tracked files)
- Glob pattern filtering with `--filter` flag
- File type filtering with `--filetype` flag
- Size-based filtering with `--max-filesize`

### Module Organization

The codebase is organized into focused modules:
- `cli.rs` - Command-line interface definitions
- `commands/` - CLI command implementations with enhanced UX
- `config.rs` - Configuration loading and management
- `embeddings.rs` - ML embedding generation
- `index.rs` - Core indexing and vector storage
- `search.rs` - Search algorithms and result processing
- `files.rs` - File discovery and git integration
- `chunking.rs` - Text chunking strategies
- `types.rs` - Common data structures and strongly-typed wrappers
- `error*.rs` - Comprehensive error handling and classification
- `parallel.rs`, `streaming.rs` - Performance optimizations

### Testing Structure

**Unit Tests** - Located in each module file (`#[cfg(test)]`)

**Integration Tests** (`tests/` directory)
- `integration_tests.rs` - Basic integration scenarios
- `complete_workflow_tests.rs` - End-to-end workflow validation
- `search_tests.rs` - Search functionality testing
- `index_tests.rs` - Index building and persistence
- `embedding_tests.rs` - ML embedding generation
- `error_handling_tests.rs` - Error scenarios
- `common.rs` - Shared test utilities and fixtures

**Benchmarks** (`benches/performance.rs`)
- Performance benchmarks for indexing and search operations
- Uses criterion for statistical benchmarking

## Configuration

The tool supports configuration via `.turboprop.yml` files in the repository root:

```yaml
# Example .turboprop.yml
max_filesize: "2mb"
model: "sentence-transformers/all-MiniLM-L6-v2"
cache_dir: "~/.turboprop-cache"
worker_threads: 4
batch_size: 32
default_limit: 10
similarity_threshold: 0.3
```

## Key Development Patterns

**Async/Await**: Heavy use of tokio for I/O operations, especially file reading and embedding generation

**Error Handling**: Comprehensive error handling using `anyhow::Result<T>` with custom error types in `error.rs`

**Strong Typing**: Uses newtype patterns for domain concepts (`ChunkId`, `TokenCount`, `ModelName`)

**Git Integration**: Built-in git repository awareness via `git2` crate

**ML Integration**: Embedding generation using `fastembed` crate with HuggingFace models

**Performance**: Parallel processing, streaming, and compression optimizations for large codebases

## Common Tasks

**Adding a new file filter**: Extend `src/filters.rs` and update the CLI in `src/cli.rs`

**Adding a new embedding model**: Update `src/embeddings.rs` and add model configuration

**Extending search capabilities**: Modify `src/search.rs` and corresponding command in `src/commands/search.rs`

**Adding new chunk processing**: Update `src/chunking.rs` with new chunking strategies

**Performance optimization**: Check `src/parallel.rs` and `src/streaming.rs` for parallelization patterns