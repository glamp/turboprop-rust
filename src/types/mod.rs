//! Type definitions for TurboProp.
//!
//! This module provides strongly-typed wrappers, enums, and data structures
//! used throughout the TurboProp codebase. The types are organized into
//! domain-specific sub-modules for better maintainability.
//!
//! # Organization
//!
//! - `models` - Model-related types (ModelName, ModelBackend, etc.)
//! - `chunks` - Chunk-related types (ChunkId, ContentChunk, etc.)
//! - `indexing` - Index-related types (IndexedChunk, SearchResult, etc.)
//! - `mcp` - MCP server types (Port, TimeoutSeconds, etc.)
//! - `files` - File-related types (FileMetadata, FileDiscoveryConfig, etc.)
//! - `similarity` - Similarity calculation utilities

pub mod chunks;
pub mod files;
pub mod indexing;
pub mod mcp;
pub mod models;
pub mod similarity;

// Re-export model types
pub use models::{CachePath, ModelBackend, ModelName, ModelType};

// Re-export chunk types
pub use chunks::{
    ChunkId, ChunkIndexNum, ChunkingConfig, ContentChunk, EmbeddingDimension, SourceLocation,
    TokenCount, DEFAULT_MAX_CHUNK_SIZE_TOKENS, DEFAULT_MIN_CHUNK_SIZE_TOKENS,
    DEFAULT_OVERLAP_TOKENS, DEFAULT_TARGET_CHUNK_SIZE_TOKENS,
};

// Re-export indexing types
pub use indexing::{ChunkIndex, IndexId, IndexStats, IndexedChunk, SearchResult};

// Re-export file types
pub use files::{parse_filesize, DocumentChunk, FileDiscoveryConfig, FileMetadata};

// Re-export similarity utilities
pub use similarity::cosine_similarity;

// Re-export MCP types
pub use mcp::{
    ConnectionLimit, ErrorCode, InvalidConnectionLimitError, InvalidMessageSizeError,
    InvalidPortError, MessageSize, Port, PreviewLength, TimeoutSeconds,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

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
        use chunks::{ChunkId, ChunkIndexNum, SourceLocation, TokenCount};
        use std::path::PathBuf;

        // Test-specific constant for content preview length
        const TEST_CONTENT_PREVIEW_LENGTH: usize = 100;

        let chunk = ContentChunk {
            id: ChunkId::new("test-chunk"),
            content: "This is a test chunk with some content that should be searchable."
                .to_string(),
            token_count: TokenCount::new(10),
            source_location: SourceLocation {
                file_path: PathBuf::from("test.rs"),
                start_line: 1,
                end_line: 1,
                start_char: 0,
                end_char: 67,
            },
            chunk_index: ChunkIndexNum::new(0),
            total_chunks: 1,
        };

        let indexed_chunk = IndexedChunk {
            chunk,
            embedding: vec![0.1, 0.2, 0.3],
        };

        let search_result = SearchResult::new(0.95, indexed_chunk, 0);

        assert_eq!(search_result.similarity, 0.95);
        assert_eq!(search_result.rank, 0);
        assert_eq!(search_result.location_display(), "test.rs:1");

        let preview = search_result.content_preview(TEST_CONTENT_PREVIEW_LENGTH);
        assert_eq!(
            preview,
            "This is a test chunk with some content that should be searchable."
        );

        // Test truncation
        let short_preview = search_result.content_preview(10);
        assert_eq!(short_preview, "This is a ...");
    }

    #[test]
    fn test_chunk_index_operations() {
        use chunks::{ChunkId, ChunkIndexNum, SourceLocation, TokenCount};
        use std::path::PathBuf;

        let mut index = ChunkIndex::new();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);

        let chunk = ContentChunk {
            id: ChunkId::new("test-chunk"),
            content: "Test content".to_string(),
            token_count: TokenCount::new(2),
            source_location: SourceLocation {
                file_path: PathBuf::from("test.rs"),
                start_line: 1,
                end_line: 1,
                start_char: 0,
                end_char: 12,
            },
            chunk_index: ChunkIndexNum::new(0),
            total_chunks: 1,
        };

        let embedding = vec![0.1, 0.2, 0.3];
        index.add_chunk(chunk, embedding);

        assert!(!index.is_empty());
        assert_eq!(index.len(), 1);

        let chunks = index.get_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk.content, "Test content");
    }

    #[test]
    fn test_content_chunk_has_real_content() {
        use chunks::{ChunkId, ChunkIndexNum, SourceLocation, TokenCount};
        use std::path::PathBuf;

        let real_chunk = ContentChunk {
            id: ChunkId::new("real"),
            content: "fn main() {}".to_string(),
            token_count: TokenCount::new(3),
            source_location: SourceLocation {
                file_path: PathBuf::from("test.rs"),
                start_line: 1,
                end_line: 1,
                start_char: 0,
                end_char: 12,
            },
            chunk_index: ChunkIndexNum::new(0),
            total_chunks: 1,
        };

        assert!(real_chunk.has_real_content());
        assert_eq!(real_chunk.real_content(), Some("fn main() {}"));

        let placeholder_chunk = ContentChunk {
            id: ChunkId::new("placeholder"),
            content: "[PLACEHOLDER_CONTENT_FROM:test.rs:1]".to_string(),
            token_count: TokenCount::new(1),
            source_location: SourceLocation {
                file_path: PathBuf::from("test.rs"),
                start_line: 1,
                end_line: 1,
                start_char: 0,
                end_char: 12,
            },
            chunk_index: ChunkIndexNum::new(0),
            total_chunks: 1,
        };

        assert!(!placeholder_chunk.has_real_content());
        assert_eq!(placeholder_chunk.real_content(), None);
    }

    #[test]
    fn test_index_stats_calculation() {
        use chunks::{ChunkId, ChunkIndexNum, SourceLocation, TokenCount};
        use std::path::PathBuf;

        let chunks = vec![
            IndexedChunk {
                chunk: ContentChunk {
                    id: ChunkId::new("chunk1"),
                    content: "Content 1".to_string(),
                    token_count: TokenCount::new(2),
                    source_location: SourceLocation {
                        file_path: PathBuf::from("file1.rs"),
                        start_line: 1,
                        end_line: 1,
                        start_char: 0,
                        end_char: 9,
                    },
                    chunk_index: ChunkIndexNum::new(0),
                    total_chunks: 1,
                },
                embedding: vec![0.1, 0.2, 0.3],
            },
            IndexedChunk {
                chunk: ContentChunk {
                    id: ChunkId::new("chunk2"),
                    content: "Content 2".to_string(),
                    token_count: TokenCount::new(4),
                    source_location: SourceLocation {
                        file_path: PathBuf::from("file2.rs"),
                        start_line: 1,
                        end_line: 1,
                        start_char: 0,
                        end_char: 9,
                    },
                    chunk_index: ChunkIndexNum::new(0),
                    total_chunks: 1,
                },
                embedding: vec![0.4, 0.5, 0.6],
            },
        ];

        let stats = IndexStats::calculate(&chunks);

        assert_eq!(stats.total_chunks, 2);
        assert_eq!(stats.unique_files, 2);
        assert_eq!(stats.total_tokens, 6);
        assert_eq!(stats.average_chunk_size, 3.0);
        assert_eq!(stats.embedding_dimensions, 3);
    }
}
