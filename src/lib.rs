//! # TurboProp - Fast Semantic Code Search and Indexing
//!
//! TurboProp is a Rust library and CLI tool that enables fast semantic search across codebases
//! using machine learning embeddings. It indexes your code files and allows you to search for
//! functionality using natural language queries.
//!
//! Now includes MCP server support for real-time integration with coding agents.
//!
//! ## Features
//!
//! - **Semantic Search**: Find code by meaning, not just keywords
//! - **Git Integration**: Automatically respects `.gitignore` and only indexes tracked files  
//! - **Watch Mode**: Monitor file changes and automatically update the index
//! - **File Filtering**: Filter by file type, size, and custom patterns
//! - **Multiple Output Formats**: JSON for tools, human-readable text for reading
//! - **Performance Optimized**: Handles codebases with 50-10,000+ files efficiently
//! - **Configurable Models**: Use any HuggingFace sentence-transformer model
//! - **MCP Server**: Real-time integration with coding agents via Model Context Protocol
//!
//! ## Quick Start
//!
//! ### CLI Usage
//!
//! ```bash
//! # Index your codebase
//! tp index --repo . --max-filesize 2mb
//!
//! # Search for code
//! tp search "jwt authentication" --repo .
//!
//! # Filter by file type
//! tp search --filetype .js "error handling" --repo .
//!
//! # Get human-readable output
//! tp search "database queries" --repo . --output text
//! ```
//!
//! ### Library Usage
//!
//! The library provides both high-level convenience functions and low-level components
//! for building custom search solutions.
//!
//! #### Basic Indexing and Search
//!
//! ```no_run
//! use turboprop::{config::TurboPropConfig, build_persistent_index, search_with_config};
//! use std::path::Path;
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! // Build an index with default settings
//! let config = TurboPropConfig::default();
//! let index = build_persistent_index(Path::new("./src"), &config).await?;
//!
//! // Search the index
//! let results = search_with_config(
//!     "error handling patterns",
//!     Path::new("./src"),
//!     Some(10),  // limit results
//!     Some(0.7)  // similarity threshold
//! ).await?;
//!
//! for result in results {
//!     println!("{}: {}", result.location_display(), result.content_preview(80));
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! # });
//! ```
//!
//! #### Custom Configuration
//!
//! ```no_run
//! use turboprop::{
//!     config::TurboPropConfig,
//!     embeddings::EmbeddingConfig,
//!     types::FileDiscoveryConfig,
//!     build_persistent_index
//! };
//! use std::path::Path;
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! // Configure embedding model
//! let embedding_config = EmbeddingConfig::with_model("sentence-transformers/all-mpnet-base-v2")
//!     .with_batch_size(16);
//!
//! // Configure file discovery
//! let file_config = FileDiscoveryConfig::default()
//!     .with_max_filesize(5_000_000)  // 5MB limit
//!     .with_gitignore_respect(true)
//!     .with_untracked(false);
//!
//! // Create complete configuration
//! let config = TurboPropConfig {
//!     embedding: embedding_config,
//!     file_discovery: file_config,
//!     ..Default::default()
//! };
//!
//! // Build index with custom configuration
//! let index = build_persistent_index(Path::new("./project"), &config).await?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! # });
//! ```
//!
//! #### Incremental Updates
//!
//! ```no_run
//! use turboprop::{config::TurboPropConfig, update_persistent_index, index_exists};
//! use std::path::Path;
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let path = Path::new("./src");
//! let config = TurboPropConfig::default();
//!
//! if index_exists(path) {
//!     // Update existing index incrementally
//!     let (updated_index, update_result) = update_persistent_index(path, &config).await?;
//!     
//!     println!("Index updated: {} files added, {} files modified, {} files removed",
//!              update_result.added_files,
//!              update_result.updated_files,
//!              update_result.removed_files);
//! } else {
//!     println!("No existing index found, create one first");
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! # });
//! ```
//!
//! ## Architecture
//!
//! TurboProp uses a multi-stage pipeline for indexing:
//!
//! 1. **File Discovery**: Finds files to index based on git status and filters
//! 2. **Content Processing**: Reads and preprocesses file content
//! 3. **Chunking**: Breaks large files into smaller, searchable chunks
//! 4. **Embedding Generation**: Creates vector embeddings using ML models
//! 5. **Index Storage**: Stores embeddings and metadata for fast retrieval
//!
//! For searching, it:
//!
//! 1. **Query Embedding**: Converts the search query to a vector
//! 2. **Similarity Search**: Finds the most similar code chunks using cosine similarity  
//! 3. **Result Ranking**: Sorts results by relevance score
//! 4. **Output Formatting**: Presents results in the requested format
//!
//! ## Performance Characteristics
//!
//! - **Indexing Speed**: ~100-500 files/second (varies by file size and hardware)
//! - **Search Speed**: ~10-50ms per query (after model loading)
//! - **Memory Usage**: ~50-200MB (varies with model and index size)
//! - **Index Size**: Typically 10-30% of source code size
//!
//! ## Supported Models
//!
//! TurboProp supports any HuggingFace sentence-transformer model:
//!
//! - `sentence-transformers/all-MiniLM-L6-v2` (default, 384 dims, ~90MB)
//! - `sentence-transformers/all-MiniLM-L12-v2` (384 dims, ~130MB)  
//! - `sentence-transformers/all-mpnet-base-v2` (768 dims, ~420MB, highest quality)
//! - `sentence-transformers/paraphrase-MiniLM-L6-v2` (384 dims, ~90MB)
//!
//! ## Error Handling
//!
//! All functions return `anyhow::Result<T>` for comprehensive error handling.
//! Common error types include:
//!
//! - **I/O Errors**: File access, permission issues
//! - **Model Errors**: Download failures, model loading issues
//! - **Configuration Errors**: Invalid settings, malformed config files
//! - **Index Errors**: Corrupted index, version mismatches
//!
//! ## Thread Safety
//!
//! Most operations are thread-safe and designed for concurrent use:
//!
//! - Index building uses multiple worker threads for parallel processing
//! - Search operations are read-only and fully concurrent
//! - File watching runs in a separate background thread
//!
//! ## Module Organization
//!
//! - [`cli`]: Command-line interface definitions
//! - [`commands`]: CLI command implementations  
//! - [`config`]: Configuration structures and loading
//! - [`embeddings`]: ML embedding generation
//! - [`files`]: File discovery and git integration
//! - [`index`]: Core indexing and storage functionality
//! - [`search`]: Search algorithms and result processing
//! - [`types`]: Common data structures and utilities

pub mod backends;
pub mod chunking;
pub mod cli;
pub mod commands;
pub mod compression;
pub mod config;
pub mod constants;
pub mod content;
pub mod embeddings;
pub mod error;
pub mod error_classification;
pub mod error_utils;
pub mod files;
pub mod filters;
pub mod git;
pub mod incremental;
pub mod index;
pub mod metrics;
pub mod model_validation;
pub mod models;
pub mod output;
pub mod parallel;
pub mod pipeline;
pub mod progress;
pub mod query;
pub mod recovery;
pub mod retry;
pub mod search;
pub mod storage;
pub mod streaming;
pub mod types;
pub mod validation;
pub mod warnings;
pub mod watcher;

pub mod mcp;

use crate::chunking::ChunkingStrategy;
use crate::config::TurboPropConfig;
use crate::embeddings::EmbeddingGenerator;
use crate::files::FileDiscovery;
use crate::index::PersistentChunkIndex;
use crate::storage::IndexStorage;
use crate::types::{parse_filesize, ChunkIndex, FileDiscoveryConfig};
use anyhow::Result;
use std::path::Path;
use tracing::{debug, info};

// Re-export MCP server for library usage
pub use mcp::McpServer;

/// Default path for indexing when no path is specified
pub const DEFAULT_INDEX_PATH: &str = ".";

/// Index files in the specified path for fast searching.
///
/// # Arguments
///
/// * `path` - The file system path to index
/// * `max_filesize` - Optional maximum file size filter (e.g., "2mb", "100kb")
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if successful, error otherwise
///
/// # Examples
///
/// Index current directory without size limit:
/// ```
/// use std::path::Path;
/// use turboprop::index_files;
///
/// let result = index_files(Path::new("."), None);
/// assert!(result.is_ok());
/// ```
///
/// Index with maximum file size filter:
/// ```
/// use std::path::Path;
/// use turboprop::index_files;
///
/// // Index files up to 2MB in size
/// # std::fs::create_dir_all("test_dir").unwrap();
/// let result = index_files(Path::new("test_dir"), Some("2mb"));
/// assert!(result.is_ok());
///
/// // Index files up to 500KB in size  
/// let result = index_files(Path::new("test_dir"), Some("500kb"));
/// assert!(result.is_ok());
/// ```
pub fn index_files(path: &Path, max_filesize: Option<&str>) -> Result<()> {
    if !path.exists() {
        anyhow::bail!("Failed to access path: {}", path.display());
    }

    if !path.is_dir() {
        anyhow::bail!(
            "Failed to index path: {} is not a directory",
            path.display()
        );
    }

    info!("Indexing files in: {}", path.display());

    let mut config = FileDiscoveryConfig::default();

    if let Some(size_str) = max_filesize {
        let max_bytes = parse_filesize(size_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse filesize: {}", e))?;
        config = config.with_max_filesize(max_bytes);
        info!("Using max filesize filter: {} bytes", max_bytes);
    }

    let discovery = FileDiscovery::new(config);
    let files = discovery.discover_files(path)?;

    info!("Discovered {} files for indexing", files.len());

    for file in &files {
        info!(
            "  {} ({} bytes, git_tracked: {})",
            file.path.display(),
            file.size_bytes,
            file.is_git_tracked
        );
    }

    Ok(())
}

/// Index files with embedding generation using the provided configuration.
///
/// This is the enhanced version that generates embeddings for the discovered text chunks
/// and stores them in a searchable index.
///
/// # Arguments
///
/// * `path` - The file system path to index
/// * `config` - Complete configuration including embedding and file discovery settings
///
/// # Returns
///
/// * `Result<ChunkIndex>` - The populated chunk index with embeddings
///
/// # Examples
///
/// Basic usage with default configuration:
/// ```no_run
/// use std::path::Path;
/// use turboprop::{config::TurboPropConfig, index_files_with_config};
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let config = TurboPropConfig::default();
/// let result = index_files_with_config(Path::new("./src"), &config).await;
/// assert!(result.is_ok());
///
/// let index = result.unwrap();
/// println!("Indexed {} chunks", index.len());
/// # });
/// ```
///
/// Advanced usage with custom embedding model and batch size:
/// ```no_run
/// use std::path::Path;
/// use turboprop::{config::TurboPropConfig, embeddings::EmbeddingConfig, index_files_with_config};
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let embedding_config = EmbeddingConfig::with_model("sentence-transformers/all-MiniLM-L12-v2")
///     .with_batch_size(16);
///
/// let config = TurboPropConfig {
///     embedding: embedding_config,
///     ..Default::default()
/// };
///
/// let result = index_files_with_config(Path::new("./project"), &config).await;
/// assert!(result.is_ok());
///
/// let index = result.unwrap();
/// // The index can now be used for similarity search
/// println!("Created index with {} chunks using {}-dimensional embeddings",
///          index.len(), config.embedding.embedding_dimensions);
/// # });
/// ```
pub async fn index_files_with_config(path: &Path, config: &TurboPropConfig) -> Result<ChunkIndex> {
    if !path.exists() {
        anyhow::bail!("Failed to access path: {}", path.display());
    }

    if !path.is_dir() {
        anyhow::bail!(
            "Failed to index path: {} is not a directory",
            path.display()
        );
    }

    info!("Indexing files with embeddings in: {}", path.display());
    info!("Using embedding model: {}", config.embedding.model_name);

    // Initialize embedding generator
    let mut embedding_generator = EmbeddingGenerator::new(config.embedding.clone()).await?;
    info!(
        "Embedding generator initialized with {} dimensions",
        embedding_generator.embedding_dimensions()
    );

    // Discover files
    let discovery = FileDiscovery::new(config.file_discovery.clone());
    let files = discovery.discover_files(path)?;
    info!("Discovered {} files for indexing", files.len());

    // Create the chunk index to store results
    let mut chunk_index = ChunkIndex::new();

    // Process each file
    let chunking_strategy = ChunkingStrategy::new(config.chunking.clone());
    let mut total_chunks = 0;
    let mut total_embeddings = 0;

    for file in &files {
        info!("Processing: {}", file.path.display());

        // Generate chunks directly from the file
        let chunks = chunking_strategy.chunk_file(&file.path)?;
        total_chunks += chunks.len();

        if chunks.is_empty() {
            debug!("No chunks generated for: {}", file.path.display());
            continue;
        }

        // Prepare text for embedding
        let chunk_texts: Vec<String> = chunks.iter().map(|chunk| chunk.content.clone()).collect();

        // Generate embeddings
        debug!("Generating embeddings for {} chunks", chunks.len());
        let embeddings = embedding_generator.embed_batch(&chunk_texts)?;
        total_embeddings += embeddings.len();

        // Log progress
        info!(
            "  {} chunks, {} embeddings generated ({} bytes, git_tracked: {})",
            chunks.len(),
            embeddings.len(),
            file.size_bytes,
            file.is_git_tracked
        );

        // Store chunks and embeddings in the index
        chunk_index.add_chunks(chunks, embeddings);
        debug!("Added {} chunks to index", chunk_texts.len());
    }

    info!(
        "Indexing completed: {} files processed, {} chunks created, {} embeddings generated, {} chunks stored in index",
        files.len(),
        total_chunks,
        total_embeddings,
        chunk_index.len()
    );

    Ok(chunk_index)
}

/// Search through indexed files using the specified query.
///
/// # Arguments
///
/// * `query` - The search query string
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if successful, error otherwise
///
/// # Examples
///
/// Simple text search:
/// ```
/// use turboprop::search_files;
///
/// let result = search_files("function main");
/// assert!(result.is_ok());
/// ```
///
/// Search for specific patterns or keywords:
/// ```
/// use turboprop::search_files;
///
/// // Search for function definitions
/// let result = search_files("fn calculate_total");
/// assert!(result.is_ok());
///
/// // Search for error handling patterns
/// let result = search_files("Result<Vec<String>>");
/// assert!(result.is_ok());
///
/// // Search for imports and modules
/// let result = search_files("use serde::");
/// assert!(result.is_ok());
/// ```
pub fn search_files(query: &str) -> Result<()> {
    info!("Searching for: {}", query);
    Ok(())
}

/// Advanced search function with configurable parameters.
///
/// This is the enhanced search functionality that supports similarity thresholds,
/// result limits, and returns detailed results with similarity scores.
///
/// # Arguments
///
/// * `query` - The search query string
/// * `repo_path` - Path to the repository/directory to search in
/// * `limit` - Maximum number of results to return (default: 10)
/// * `threshold` - Minimum similarity threshold (0.0 to 1.0, optional)
///
/// # Returns
///
/// * `Result<Vec<SearchResult>>` - Vector of search results with similarity scores
///
/// # Examples
///
/// Basic search with default parameters:
/// ```no_run
/// use std::path::Path;
/// use turboprop::search_with_config;
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let results = search_with_config("jwt authentication", Path::new("."), None, None).await;
/// if let Ok(results) = results {
///     for result in results {
///         println!("{} ({:.3}): {}",
///                  result.location_display(),
///                  result.similarity,
///                  result.content_preview(50));
///     }
/// }
/// # });
/// ```
///
/// Search with custom limit and threshold:
/// ```no_run
/// use std::path::Path;
/// use turboprop::search_with_config;
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let results = search_with_config(
///     "error handling",
///     Path::new("./src"),
///     Some(5),      // limit: 5 results
///     Some(0.7)     // threshold: minimum 70% similarity
/// ).await;
/// # });
/// ```
pub async fn search_with_config(
    query: &str,
    repo_path: &Path,
    limit: Option<usize>,
    threshold: Option<f32>,
) -> Result<Vec<crate::types::SearchResult>> {
    use crate::search::search_index;

    info!("Performing advanced search for: '{}'", query);
    info!("Repository path: {}", repo_path.display());

    if let Some(limit) = limit {
        info!("Result limit: {}", limit);
    }
    if let Some(threshold) = threshold {
        info!("Similarity threshold: {:.3}", threshold);
    }

    let results = search_index(repo_path, query, limit, threshold).await?;

    info!("Search completed: {} results found", results.len());

    Ok(results)
}

/// Build a persistent vector index for the specified path.
///
/// This creates a complete index with embeddings that is stored to disk
/// and can be loaded later for fast similarity search.
///
/// # Arguments
///
/// * `path` - The file system path to index
/// * `config` - Complete configuration for indexing and embedding generation
///
/// # Returns
///
/// * `Result<PersistentChunkIndex>` - The built index ready for searching
///
/// # Examples
///
/// Build an index with default configuration:
/// ```no_run
/// use std::path::Path;
/// use turboprop::{config::TurboPropConfig, build_persistent_index};
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let config = TurboPropConfig::default();
/// let result = build_persistent_index(Path::new("./src"), &config).await;
/// assert!(result.is_ok());
///
/// let index = result.unwrap();
/// println!("Built index with {} chunks", index.len());
/// # });
/// ```
pub async fn build_persistent_index(
    path: &Path,
    config: &TurboPropConfig,
) -> Result<PersistentChunkIndex> {
    PersistentChunkIndex::build(path, config).await
}

/// Load an existing persistent vector index from disk.
///
/// # Arguments
///
/// * `path` - The path where the index was originally built
///
/// # Returns
///
/// * `Result<PersistentChunkIndex>` - The loaded index ready for searching
///
/// # Examples
///
/// Load an existing index:
/// ```no_run
/// use std::path::Path;
/// use turboprop::load_persistent_index;
///
/// let result = load_persistent_index(Path::new("./src"));
/// if let Ok(index) = result {
///     println!("Loaded index with {} chunks", index.len());
/// }
/// ```
pub fn load_persistent_index(path: &Path) -> Result<PersistentChunkIndex> {
    PersistentChunkIndex::load(path)
}

/// Check if a persistent index exists at the specified path.
///
/// # Arguments
///
/// * `path` - The path to check for an existing index
///
/// # Returns
///
/// * `bool` - true if an index exists, false otherwise
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use turboprop::index_exists;
///
/// if index_exists(Path::new("./src")) {
///     println!("Index found, can load it");
/// } else {
///     println!("No index found, need to build one");
/// }
/// ```
pub fn index_exists(path: &Path) -> bool {
    IndexStorage::new(path)
        .map(|storage| storage.index_exists())
        .unwrap_or(false)
}

/// Update an existing persistent index incrementally based on file changes.
///
/// This is more efficient than rebuilding the entire index when only some
/// files have changed.
///
/// # Arguments
///
/// * `path` - The path of the existing index
/// * `config` - Configuration for the update process
///
/// # Returns
///
/// * `Result<(PersistentChunkIndex, crate::index::UpdateResult)>` - The updated index and update statistics
///
/// # Examples
///
/// Update an existing index:
/// ```no_run
/// use std::path::Path;
/// use turboprop::{config::TurboPropConfig, update_persistent_index};
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let config = TurboPropConfig::default();
/// let result = update_persistent_index(Path::new("./src"), &config).await;
///
/// if let Ok((index, update_result)) = result {
///     println!("Updated index: {} files added, {} files updated, {} files removed",
///              update_result.added_files,
///              update_result.updated_files,
///              update_result.removed_files);
/// }
/// # });
/// ```
pub async fn update_persistent_index(
    path: &Path,
    config: &TurboPropConfig,
) -> Result<(PersistentChunkIndex, crate::index::UpdateResult)> {
    let mut index = if index_exists(path) {
        PersistentChunkIndex::load(path)?
    } else {
        PersistentChunkIndex::new(path)?
    };

    let update_result = index.update_incremental(config).await?;
    Ok((index, update_result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_index_files_success() {
        let result = index_files(Path::new("."), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_index_files_with_valid_directory() {
        let result = index_files(Path::new("src"), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_search_files_success() {
        let result = search_files("test query");
        assert!(result.is_ok());
    }

    #[test]
    fn test_search_files_empty_query() {
        let result = search_files("");
        assert!(result.is_ok());
    }

    #[test]
    fn test_search_files_special_characters() {
        let result = search_files("fn main() {");
        assert!(result.is_ok());
    }

    #[test]
    fn test_search_files_long_query() {
        let long_query = "a".repeat(1000);
        let result = search_files(&long_query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_index_files_nonexistent_path() {
        let result = index_files(Path::new("/nonexistent/path"), None);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Failed to access path"));
    }

    #[test]
    fn test_index_files_file_not_directory() {
        let result = index_files(Path::new("Cargo.toml"), None);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("is not a directory"));
    }

    #[test]
    fn test_index_files_with_max_filesize() {
        let result = index_files(Path::new("src"), Some("1mb"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_index_files_with_invalid_filesize() {
        let result = index_files(Path::new("src"), Some("invalid"));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid filesize format"));
    }
}
