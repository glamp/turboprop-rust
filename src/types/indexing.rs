//! Index-related type definitions and functionality.
//!
//! This module contains types for handling indexing operations, including
//! indexed chunks, search results, index statistics, and the main chunk index.

use serde::{Deserialize, Serialize};
use std::path::Path;

use super::chunks::ContentChunk;

/// Unique identifier type for indexes to prevent mixing up different types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IndexId(pub String);

impl IndexId {
    /// Create a new index ID from a path
    pub fn from_path(path: &Path) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        path.hash(&mut hasher);
        let hash = hasher.finish();

        Self(format!("idx_{:016x}", hash))
    }

    /// Create a new index ID from a string
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for IndexId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for IndexId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// A content chunk with its associated embedding vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedChunk {
    pub chunk: ContentChunk,
    pub embedding: Vec<f32>,
}

/// In-memory index of chunks with their embeddings for similarity search
#[derive(Debug, Default)]
pub struct ChunkIndex {
    chunks: Vec<IndexedChunk>,
}

impl ChunkIndex {
    pub fn new() -> Self {
        Self { chunks: Vec::new() }
    }

    pub fn add_chunk(&mut self, chunk: ContentChunk, embedding: Vec<f32>) {
        self.chunks.push(IndexedChunk { chunk, embedding });
    }

    /// Add an already-created IndexedChunk efficiently without cloning
    pub fn add_indexed_chunk(&mut self, indexed_chunk: IndexedChunk) {
        self.chunks.push(indexed_chunk);
    }

    /// Add multiple IndexedChunks efficiently without cloning  
    pub fn add_indexed_chunks(&mut self, indexed_chunks: Vec<IndexedChunk>) {
        self.chunks.extend(indexed_chunks);
    }

    pub fn add_chunks(&mut self, chunks: Vec<ContentChunk>, embeddings: Vec<Vec<f32>>) {
        for (chunk, embedding) in chunks.into_iter().zip(embeddings.into_iter()) {
            self.add_chunk(chunk, embedding);
        }
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn get_chunks(&self) -> &[IndexedChunk] {
        &self.chunks
    }

    pub fn similarity_search(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Vec<(f32, &IndexedChunk)> {
        // Input validation for query embedding
        if query_embedding.is_empty() {
            tracing::warn!("Empty query embedding provided to similarity search");
            return Vec::new();
        }

        // Check for non-finite values in query embedding
        if query_embedding.iter().any(|&x| !x.is_finite()) {
            tracing::warn!("Non-finite values detected in query embedding");
            return Vec::new();
        }

        // Validate limit bounds
        if limit == 0 {
            tracing::debug!("Zero limit provided to similarity search");
            return Vec::new();
        }

        let mut results: Vec<(f32, &IndexedChunk)> = self
            .chunks
            .iter()
            .map(|indexed_chunk| {
                let similarity = super::similarity::cosine_similarity(query_embedding, &indexed_chunk.embedding);
                (similarity, indexed_chunk)
            })
            .collect();

        // Safe sorting that handles NaN values properly
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        results.into_iter().take(limit).collect()
    }
}

/// A search result with similarity score and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Similarity score (0.0 to 1.0, higher is more similar)
    pub similarity: f32,
    /// The matching chunk (owned)
    pub chunk: IndexedChunk,
    /// Rank in the result set (0-based)
    pub rank: usize,
}

impl SearchResult {
    /// Create a new search result
    pub fn new(similarity: f32, chunk: IndexedChunk, rank: usize) -> Self {
        Self {
            similarity: similarity.max(0.0), // Clamp negative values to 0.0
            chunk,
            rank,
        }
    }

    /// Get a formatted display of the file location
    pub fn location_display(&self) -> String {
        format!(
            "{}:{}",
            self.chunk.chunk.source_location.file_path.display(),
            self.chunk.chunk.source_location.start_line
        )
    }

    /// Get a preview of the chunk content (first N characters)
    pub fn content_preview(&self, max_chars: usize) -> String {
        let content = &self.chunk.chunk.content;
        if content.len() <= max_chars {
            content.clone()
        } else {
            format!("{}...", &content[..max_chars])
        }
    }
}

/// Statistics about an index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// Total number of chunks in the index
    pub total_chunks: usize,
    /// Number of unique files indexed
    pub unique_files: usize,
    /// Total number of tokens across all chunks
    pub total_tokens: usize,
    /// Average chunk size in tokens
    pub average_chunk_size: f64,
    /// Embedding dimensions used
    pub embedding_dimensions: usize,
    /// Timestamp when stats were calculated
    pub calculated_at: std::time::SystemTime,
}

impl IndexStats {
    /// Calculate statistics from a collection of indexed chunks
    pub fn calculate(indexed_chunks: &[IndexedChunk]) -> Self {
        let total_chunks = indexed_chunks.len();
        let unique_files = indexed_chunks
            .iter()
            .map(|chunk| &chunk.chunk.source_location.file_path)
            .collect::<std::collections::HashSet<_>>()
            .len();

        let total_tokens: usize = indexed_chunks
            .iter()
            .map(|chunk| chunk.chunk.token_count.get())
            .sum();

        let average_chunk_size = if total_chunks > 0 {
            total_tokens as f64 / total_chunks as f64
        } else {
            0.0
        };

        let embedding_dimensions = indexed_chunks
            .first()
            .map(|chunk| chunk.embedding.len())
            .unwrap_or(0);

        Self {
            total_chunks,
            unique_files,
            total_tokens,
            average_chunk_size,
            embedding_dimensions,
            calculated_at: std::time::SystemTime::now(),
        }
    }
}