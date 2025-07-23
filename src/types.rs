use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

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
pub const DEFAULT_TARGET_CHUNK_SIZE_TOKENS: usize = 300;
pub const DEFAULT_OVERLAP_TOKENS: usize = 50;
pub const DEFAULT_MAX_CHUNK_SIZE_TOKENS: usize = 500;
pub const DEFAULT_MIN_CHUNK_SIZE_TOKENS: usize = 100;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub last_modified: std::time::SystemTime,
    pub is_git_tracked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDiscoveryConfig {
    pub max_filesize_bytes: Option<u64>,
    pub respect_gitignore: bool,
    pub include_untracked: bool,
}

impl Default for FileDiscoveryConfig {
    fn default() -> Self {
        Self {
            max_filesize_bytes: None,
            respect_gitignore: true,
            include_untracked: false,
        }
    }
}

impl FileDiscoveryConfig {
    pub fn with_max_filesize(mut self, max_size: u64) -> Self {
        self.max_filesize_bytes = Some(max_size);
        self
    }

    pub fn with_gitignore_respect(mut self, respect: bool) -> Self {
        self.respect_gitignore = respect;
        self
    }

    pub fn with_untracked(mut self, include: bool) -> Self {
        self.include_untracked = include;
        self
    }
}

pub fn parse_filesize(input: &str) -> Result<u64, String> {
    let input = input.to_lowercase();

    if let Some(stripped) = input.strip_suffix("kb") {
        stripped
            .parse::<u64>()
            .map(|n| n * 1024)
            .map_err(|_| format!("Invalid number in filesize: {}", input))
    } else if let Some(stripped) = input.strip_suffix("mb") {
        stripped
            .parse::<u64>()
            .map(|n| n * 1024 * 1024)
            .map_err(|_| format!("Invalid number in filesize: {}", input))
    } else if let Some(stripped) = input.strip_suffix("gb") {
        stripped
            .parse::<u64>()
            .map(|n| n * 1024 * 1024 * 1024)
            .map_err(|_| format!("Invalid number in filesize: {}", input))
    } else if let Some(stripped) = input.strip_suffix("b") {
        stripped
            .parse::<u64>()
            .map_err(|_| format!("Invalid number in filesize: {}", input))
    } else {
        input
            .parse::<u64>()
            .map_err(|_| format!("Invalid filesize format: {}", input))
    }
}

/// Document chunk with metadata and embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub content: String,
    pub embedding: Vec<f32>,
    pub metadata: crate::storage::ChunkMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedChunk {
    pub chunk: ContentChunk,
    pub embedding: Vec<f32>,
}

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
                let similarity = cosine_similarity(query_embedding, &indexed_chunk.embedding);
                (similarity, indexed_chunk)
            })
            .collect();

        // Safe sorting that handles NaN values properly
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        results.into_iter().take(limit).collect()
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    // Use checked arithmetic to prevent overflow
    let mut dot_product = 0.0_f32;
    let mut sum_a_squared = 0.0_f32;
    let mut sum_b_squared = 0.0_f32;

    for (x, y) in a.iter().zip(b.iter()) {
        // Check for infinity or NaN before operations
        if !x.is_finite() || !y.is_finite() {
            tracing::warn!("Non-finite values detected in cosine similarity calculation");
            return 0.0;
        }

        // Safely compute dot product with overflow checking
        let product = x * y;
        if !product.is_finite() {
            tracing::warn!("Overflow detected in dot product calculation");
            return 0.0;
        }
        dot_product += product;

        // Safely compute squared magnitudes with overflow checking
        let x_squared = x * x;
        let y_squared = y * y;
        if !x_squared.is_finite() || !y_squared.is_finite() {
            tracing::warn!("Overflow detected in magnitude calculation");
            return 0.0;
        }
        sum_a_squared += x_squared;
        sum_b_squared += y_squared;
    }

    // Check for valid intermediate results
    if !dot_product.is_finite() || !sum_a_squared.is_finite() || !sum_b_squared.is_finite() {
        return 0.0;
    }

    let magnitude_a = sum_a_squared.sqrt();
    let magnitude_b = sum_b_squared.sqrt();

    if magnitude_a == 0.0
        || magnitude_b == 0.0
        || !magnitude_a.is_finite()
        || !magnitude_b.is_finite()
    {
        return 0.0;
    }

    let result = dot_product / (magnitude_a * magnitude_b);

    // Final check for valid result
    if !result.is_finite() {
        tracing::warn!("Invalid result in cosine similarity calculation");
        return 0.0;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_filesize() {
        assert_eq!(parse_filesize("100"), Ok(100));
        assert_eq!(parse_filesize("100b"), Ok(100));
        assert_eq!(parse_filesize("2kb"), Ok(2048));
        assert_eq!(parse_filesize("2KB"), Ok(2048));
        assert_eq!(parse_filesize("5mb"), Ok(5 * 1024 * 1024));
        assert_eq!(parse_filesize("1gb"), Ok(1024 * 1024 * 1024));

        assert!(parse_filesize("invalid").is_err());
        assert!(parse_filesize("2.5mb").is_err());
    }

    #[test]
    fn test_file_discovery_config() {
        let config = FileDiscoveryConfig::default()
            .with_max_filesize(1024)
            .with_gitignore_respect(false)
            .with_untracked(true);

        assert_eq!(config.max_filesize_bytes, Some(1024));
        assert!(!config.respect_gitignore);
        assert!(config.include_untracked);
    }

    #[test]
    fn test_index_id() {
        let id1 = IndexId::new("test-index");
        let id2 = IndexId::from("another-index".to_string());
        let id3 = IndexId::from_path(Path::new("/some/path"));

        assert_eq!(id1.to_string(), "test-index");
        assert_eq!(id2.to_string(), "another-index");
        assert!(id3.to_string().starts_with("idx_"));

        // Same path should generate same ID
        let id4 = IndexId::from_path(Path::new("/some/path"));
        assert_eq!(id3, id4);
    }

    #[test]
    fn test_search_result() {
        // Test-specific constant for content preview length
        const TEST_CONTENT_PREVIEW_LENGTH: usize = 100;

        let chunk = ContentChunk {
            id: "test-chunk".into(),
            content: "This is a test chunk with some content".to_string(),
            token_count: 8.into(),
            source_location: SourceLocation {
                file_path: PathBuf::from("test.rs"),
                start_line: 10,
                end_line: 12,
                start_char: 0,
                end_char: 38,
            },
            chunk_index: 0.into(),
            total_chunks: 1,
        };

        let indexed_chunk = IndexedChunk {
            chunk,
            embedding: vec![0.1, 0.2, 0.3],
        };

        let result = SearchResult::new(0.85, indexed_chunk, 0);

        assert_eq!(result.similarity, 0.85);
        assert_eq!(result.rank, 0);
        assert_eq!(result.location_display(), "test.rs:10");
        assert_eq!(result.content_preview(20), "This is a test chunk...");
        assert_eq!(
            result.content_preview(TEST_CONTENT_PREVIEW_LENGTH),
            "This is a test chunk with some content"
        );
    }

    #[test]
    fn test_index_stats() {
        let chunks = vec![
            IndexedChunk {
                chunk: ContentChunk {
                    id: "chunk1".into(),
                    content: "First chunk".to_string(),
                    token_count: 2.into(),
                    source_location: SourceLocation {
                        file_path: PathBuf::from("file1.rs"),
                        start_line: 1,
                        end_line: 1,
                        start_char: 0,
                        end_char: 11,
                    },
                    chunk_index: 0.into(),
                    total_chunks: 2,
                },
                embedding: vec![0.1, 0.2],
            },
            IndexedChunk {
                chunk: ContentChunk {
                    id: "chunk2".into(),
                    content: "Second chunk with more tokens".to_string(),
                    token_count: 5.into(),
                    source_location: SourceLocation {
                        file_path: PathBuf::from("file1.rs"),
                        start_line: 2,
                        end_line: 2,
                        start_char: 0,
                        end_char: 29,
                    },
                    chunk_index: 1.into(),
                    total_chunks: 2,
                },
                embedding: vec![0.3, 0.4],
            },
            IndexedChunk {
                chunk: ContentChunk {
                    id: "chunk3".into(),
                    content: "Third chunk from different file".to_string(),
                    token_count: 6.into(),
                    source_location: SourceLocation {
                        file_path: PathBuf::from("file2.rs"),
                        start_line: 1,
                        end_line: 1,
                        start_char: 0,
                        end_char: 31,
                    },
                    chunk_index: 0.into(),
                    total_chunks: 1,
                },
                embedding: vec![0.5, 0.6],
            },
        ];

        let stats = IndexStats::calculate(&chunks);

        assert_eq!(stats.total_chunks, 3);
        assert_eq!(stats.unique_files, 2);
        assert_eq!(stats.total_tokens, 13);
        assert_eq!(stats.average_chunk_size, 13.0 / 3.0);
        assert_eq!(stats.embedding_dimensions, 2);
    }

    #[test]
    fn test_index_stats_empty() {
        let stats = IndexStats::calculate(&[]);

        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.unique_files, 0);
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.average_chunk_size, 0.0);
        assert_eq!(stats.embedding_dimensions, 0);
    }

    #[test]
    fn test_content_chunk_has_real_content() {
        let real_chunk = ContentChunk {
            id: "test".into(),
            content: "fn main() {}".to_string(),
            token_count: 3.into(),
            source_location: SourceLocation {
                file_path: PathBuf::from("test.rs"),
                start_line: 1,
                end_line: 1,
                start_char: 0,
                end_char: 12,
            },
            chunk_index: 0.into(),
            total_chunks: 1,
        };

        let placeholder_chunk = ContentChunk {
            id: "test".into(),
            content: "[PLACEHOLDER_CONTENT_FROM:/path/to/file.rs:1]".to_string(),
            token_count: 3.into(),
            source_location: SourceLocation {
                file_path: PathBuf::from("test.rs"),
                start_line: 1,
                end_line: 1,
                start_char: 0,
                end_char: 12,
            },
            chunk_index: 0.into(),
            total_chunks: 1,
        };

        assert!(real_chunk.has_real_content());
        assert!(!placeholder_chunk.has_real_content());

        assert_eq!(real_chunk.real_content(), Some("fn main() {}"));
        assert_eq!(placeholder_chunk.real_content(), None);
    }

    #[test]
    fn test_content_chunk_rehydrate_with_real_content() {
        let chunk = ContentChunk {
            id: "test".into(),
            content: "fn main() {}".to_string(),
            token_count: 3.into(),
            source_location: SourceLocation {
                file_path: PathBuf::from("test.rs"),
                start_line: 1,
                end_line: 1,
                start_char: 0,
                end_char: 12,
            },
            chunk_index: 0.into(),
            total_chunks: 1,
        };

        // Should return the original content since it's already real
        let result = chunk.rehydrate_content().unwrap();
        assert_eq!(result, "fn main() {}");
    }

    #[test]
    fn test_content_chunk_rehydrate_missing_file() {
        let chunk = ContentChunk {
            id: "test".into(),
            content: "[PLACEHOLDER_CONTENT_FROM:/nonexistent/file.rs:1]".to_string(),
            token_count: 3.into(),
            source_location: SourceLocation {
                file_path: PathBuf::from("/nonexistent/file.rs"),
                start_line: 1,
                end_line: 1,
                start_char: 0,
                end_char: 12,
            },
            chunk_index: 0.into(),
            total_chunks: 1,
        };

        // Should return an error for non-existent file
        let result = chunk.rehydrate_content();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Failed to rehydrate content"));
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceLocation {
    pub file_path: PathBuf,
    pub start_line: usize,
    pub end_line: usize,
    pub start_char: usize,
    pub end_char: usize,
}

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

impl From<&str> for IndexId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Search result with similarity score and owned chunk data
#[derive(Debug, Clone)]
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
