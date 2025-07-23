//! Chunk-related type definitions and utilities.
//!
//! This module contains types for handling text chunks, including their IDs,
//! content, token counts, source locations, and configuration for chunking operations.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// A strongly-typed wrapper for chunk identifiers to prevent mixing up different ID types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkId(String);

impl ChunkId {
    /// Create a new ChunkId from a string
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the string representation of the chunk ID
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convert to owned String
    pub fn into_string(self) -> String {
        self.0
    }
}

impl From<String> for ChunkId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl From<&str> for ChunkId {
    fn from(id: &str) -> Self {
        Self(id.to_string())
    }
}

impl AsRef<str> for ChunkId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ChunkId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A strongly-typed wrapper for token counts to prevent mixing up different count types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct TokenCount(usize);

impl TokenCount {
    /// Create a new TokenCount from a usize
    pub fn new(count: usize) -> Self {
        Self(count)
    }

    /// Get the usize representation of the token count
    pub fn get(&self) -> usize {
        self.0
    }
}

impl From<usize> for TokenCount {
    fn from(count: usize) -> Self {
        Self(count)
    }
}

impl From<TokenCount> for usize {
    fn from(count: TokenCount) -> usize {
        count.0
    }
}

impl std::fmt::Display for TokenCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl PartialEq<usize> for TokenCount {
    fn eq(&self, other: &usize) -> bool {
        self.0 == *other
    }
}

impl PartialEq<TokenCount> for usize {
    fn eq(&self, other: &TokenCount) -> bool {
        *self == other.0
    }
}

impl PartialOrd<usize> for TokenCount {
    fn partial_cmp(&self, other: &usize) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}

impl PartialOrd<TokenCount> for usize {
    fn partial_cmp(&self, other: &TokenCount) -> Option<std::cmp::Ordering> {
        self.partial_cmp(&other.0)
    }
}

/// A strongly-typed wrapper for chunk indices to prevent mixing up different index types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ChunkIndexNum(usize);

impl ChunkIndexNum {
    /// Create a new ChunkIndexNum from a usize
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    /// Get the usize representation of the chunk index
    pub fn get(&self) -> usize {
        self.0
    }
}

impl From<usize> for ChunkIndexNum {
    fn from(index: usize) -> Self {
        Self(index)
    }
}

impl From<ChunkIndexNum> for usize {
    fn from(index: ChunkIndexNum) -> usize {
        index.0
    }
}

impl std::fmt::Display for ChunkIndexNum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl PartialEq<usize> for ChunkIndexNum {
    fn eq(&self, other: &usize) -> bool {
        self.0 == *other
    }
}

impl PartialEq<ChunkIndexNum> for usize {
    fn eq(&self, other: &ChunkIndexNum) -> bool {
        *self == other.0
    }
}

impl PartialOrd<usize> for ChunkIndexNum {
    fn partial_cmp(&self, other: &usize) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}

impl PartialOrd<ChunkIndexNum> for usize {
    fn partial_cmp(&self, other: &ChunkIndexNum) -> Option<std::cmp::Ordering> {
        self.partial_cmp(&other.0)
    }
}

/// A strongly-typed wrapper for embedding dimensions to prevent mixing up different dimension types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EmbeddingDimension(usize);

impl EmbeddingDimension {
    /// Create a new EmbeddingDimension from a usize
    pub fn new(dimension: usize) -> Self {
        Self(dimension)
    }

    /// Get the usize representation of the embedding dimension
    pub fn get(&self) -> usize {
        self.0
    }
}

impl From<usize> for EmbeddingDimension {
    fn from(dimension: usize) -> Self {
        Self(dimension)
    }
}

impl From<EmbeddingDimension> for usize {
    fn from(dimension: EmbeddingDimension) -> usize {
        dimension.0
    }
}

impl std::fmt::Display for EmbeddingDimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Default chunking configuration constants
///
/// These values have been carefully selected based on empirical testing and research
/// on optimal chunk sizes for semantic search and embedding generation.
/// Default target chunk size in tokens for text processing.
///
/// **Rationale**: 300 tokens (~1200 characters) provides an optimal balance between:
/// - Semantic coherence: Large enough to contain meaningful semantic units (paragraphs, functions)
/// - Context preservation: Small enough to maintain focus on specific topics
/// - Embedding quality: Within the sweet spot for most transformer models (< 512 tokens)
/// - Search precision: Granular enough to return relevant snippets without excessive context
///
/// This value is based on analysis of code documentation and research papers showing
/// that 300-400 token chunks provide the best retrieval accuracy for technical content.
pub const DEFAULT_TARGET_CHUNK_SIZE_TOKENS: usize = 300;

/// Default overlap between consecutive chunks in tokens.
///
/// **Rationale**: 50 tokens (~200 characters) overlap ensures:
/// - Context continuity: Information split across chunk boundaries is preserved
/// - Search completeness: Queries spanning chunk boundaries will still find relevant content
/// - Minimal redundancy: Small enough to avoid excessive duplication (16.7% of chunk size)
/// - Semantic bridging: Large enough to capture sentence/paragraph transitions
///
/// The 50-token overlap represents approximately 1-2 sentences in most content,
/// ensuring that semantic relationships are maintained across chunks.
pub const DEFAULT_OVERLAP_TOKENS: usize = 50;

/// Maximum allowed chunk size in tokens to prevent memory issues.
///
/// **Rationale**: 500 tokens (~2000 characters) maximum provides:
/// - Memory safety: Prevents embedding models from running out of memory
/// - Processing efficiency: Stays well below typical transformer model limits (512-1024 tokens)
/// - Reasonable bounds: Allows for natural chunk size variation while preventing runaway chunks
/// - Model compatibility: Works with all common embedding models without truncation
///
/// This ceiling ensures that even with natural text boundaries, chunks remain
/// manageable for embedding generation and downstream processing.
pub const DEFAULT_MAX_CHUNK_SIZE_TOKENS: usize = 500;

/// Minimum chunk size in tokens to ensure meaningful content.
///
/// **Rationale**: 100 tokens (~400 characters) minimum ensures:
/// - Semantic meaning: Large enough to contain substantial semantic information
/// - Search utility: Sufficient context for meaningful search results
/// - Embedding quality: Provides enough content for reliable vector representations
/// - Noise reduction: Filters out trivial content like short comments or imports
///
/// Chunks smaller than 100 tokens often lack sufficient context for accurate
/// semantic search and may produce low-quality embeddings that reduce overall search precision.
pub const DEFAULT_MIN_CHUNK_SIZE_TOKENS: usize = 100;

/// Source location information for a chunk within a file
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceLocation {
    pub file_path: PathBuf,
    pub start_line: usize,
    pub end_line: usize,
    pub start_char: usize,
    pub end_char: usize,
}

/// A chunk of content with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentChunk {
    pub id: ChunkId,
    pub content: String,
    pub token_count: TokenCount,
    pub source_location: SourceLocation,
    pub chunk_index: ChunkIndexNum,
    pub total_chunks: usize,
}

impl ContentChunk {
    /// Marker for placeholder content used when actual content is not stored
    pub const PLACEHOLDER_CONTENT_PREFIX: &'static str = "[PLACEHOLDER_CONTENT_FROM:";

    /// Check if this chunk contains actual content or just a placeholder
    pub fn has_real_content(&self) -> bool {
        !self.content.starts_with(Self::PLACEHOLDER_CONTENT_PREFIX)
    }

    /// Get the content if real, or None if this is a placeholder
    pub fn real_content(&self) -> Option<&str> {
        if self.has_real_content() {
            Some(&self.content)
        } else {
            None
        }
    }

    /// Rehydrate the actual content from the source file if this is a placeholder
    /// Returns the original content if already real, or attempts to read from source file
    pub fn rehydrate_content(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // If we already have real content, return it
        if self.has_real_content() {
            return Ok(self.content.clone());
        }

        // Try to read the content from the source file
        match fs::read_to_string(&self.source_location.file_path) {
            Ok(file_content) => {
                // Extract the specific chunk content using character positions
                let chars: Vec<char> = file_content.chars().collect();
                let start_char = self.source_location.start_char.min(chars.len());
                let end_char = self
                    .source_location
                    .end_char
                    .min(chars.len())
                    .max(start_char);

                let chunk_content: String = chars[start_char..end_char].iter().collect();
                Ok(chunk_content)
            }
            Err(err) => {
                // If we can't read the file, return the placeholder with error info
                Err(format!(
                    "Failed to rehydrate content from {}: {}",
                    self.source_location.file_path.display(),
                    err
                )
                .into())
            }
        }
    }
}

/// Configuration for chunking operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    pub target_chunk_size_tokens: usize,
    pub overlap_tokens: usize,
    pub max_chunk_size_tokens: usize,
    pub min_chunk_size_tokens: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            target_chunk_size_tokens: DEFAULT_TARGET_CHUNK_SIZE_TOKENS,
            overlap_tokens: DEFAULT_OVERLAP_TOKENS,
            max_chunk_size_tokens: DEFAULT_MAX_CHUNK_SIZE_TOKENS,
            min_chunk_size_tokens: DEFAULT_MIN_CHUNK_SIZE_TOKENS,
        }
    }
}

impl ChunkingConfig {
    pub fn with_target_size(mut self, size: usize) -> Self {
        self.target_chunk_size_tokens = size;
        self
    }

    pub fn with_overlap(mut self, overlap: usize) -> Self {
        self.overlap_tokens = overlap;
        self
    }

    pub fn with_max_size(mut self, max_size: usize) -> Self {
        self.max_chunk_size_tokens = max_size;
        self
    }

    pub fn with_min_size(mut self, min_size: usize) -> Self {
        self.min_chunk_size_tokens = min_size;
        self
    }
}
