use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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
        let mut results: Vec<(f32, &IndexedChunk)> = self
            .chunks
            .iter()
            .map(|indexed_chunk| {
                let similarity = cosine_similarity(query_embedding, &indexed_chunk.embedding);
                (similarity, indexed_chunk)
            })
            .collect();

        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        results.into_iter().take(limit).collect()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }

    dot_product / (magnitude_a * magnitude_b)
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
        let chunk = ContentChunk {
            id: "test-chunk".to_string(),
            content: "This is a test chunk with some content".to_string(),
            token_count: 8,
            source_location: SourceLocation {
                file_path: PathBuf::from("test.rs"),
                start_line: 10,
                end_line: 12,
                start_char: 0,
                end_char: 38,
            },
            chunk_index: 0,
            total_chunks: 1,
        };

        let indexed_chunk = IndexedChunk {
            chunk,
            embedding: vec![0.1, 0.2, 0.3],
        };

        let result = SearchResult::new(0.85, &indexed_chunk, 0);

        assert_eq!(result.similarity, 0.85);
        assert_eq!(result.rank, 0);
        assert_eq!(result.location_display(), "test.rs:10");
        assert_eq!(result.content_preview(20), "This is a test chunk...");
        assert_eq!(
            result.content_preview(100),
            "This is a test chunk with some content"
        );
    }

    #[test]
    fn test_index_stats() {
        let chunks = vec![
            IndexedChunk {
                chunk: ContentChunk {
                    id: "chunk1".to_string(),
                    content: "First chunk".to_string(),
                    token_count: 2,
                    source_location: SourceLocation {
                        file_path: PathBuf::from("file1.rs"),
                        start_line: 1,
                        end_line: 1,
                        start_char: 0,
                        end_char: 11,
                    },
                    chunk_index: 0,
                    total_chunks: 2,
                },
                embedding: vec![0.1, 0.2],
            },
            IndexedChunk {
                chunk: ContentChunk {
                    id: "chunk2".to_string(),
                    content: "Second chunk with more tokens".to_string(),
                    token_count: 5,
                    source_location: SourceLocation {
                        file_path: PathBuf::from("file1.rs"),
                        start_line: 2,
                        end_line: 2,
                        start_char: 0,
                        end_char: 29,
                    },
                    chunk_index: 1,
                    total_chunks: 2,
                },
                embedding: vec![0.3, 0.4],
            },
            IndexedChunk {
                chunk: ContentChunk {
                    id: "chunk3".to_string(),
                    content: "Third chunk from different file".to_string(),
                    token_count: 6,
                    source_location: SourceLocation {
                        file_path: PathBuf::from("file2.rs"),
                        start_line: 1,
                        end_line: 1,
                        start_char: 0,
                        end_char: 31,
                    },
                    chunk_index: 0,
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
    pub id: String,
    pub content: String,
    pub token_count: usize,
    pub source_location: SourceLocation,
    pub chunk_index: usize,
    pub total_chunks: usize,
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
            target_chunk_size_tokens: 300,
            overlap_tokens: 50,
            max_chunk_size_tokens: 500,
            min_chunk_size_tokens: 100,
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

/// Search result with similarity score and chunk reference
#[derive(Debug, Clone)]
pub struct SearchResult<'a> {
    /// Similarity score (0.0 to 1.0, higher is more similar)
    pub similarity: f32,
    /// Reference to the matching chunk
    pub chunk: &'a IndexedChunk,
    /// Rank in the result set (0-based)
    pub rank: usize,
}

impl<'a> SearchResult<'a> {
    /// Create a new search result
    pub fn new(similarity: f32, chunk: &'a IndexedChunk, rank: usize) -> Self {
        Self {
            similarity,
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
            .map(|chunk| chunk.chunk.token_count)
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
