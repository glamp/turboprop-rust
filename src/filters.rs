//! Search result filtering functionality.
//!
//! This module provides filtering capabilities for search results, including
//! filtering by file type/extension.

use anyhow::Result;
use std::path::Path;

use crate::types::SearchResult;

/// Maximum allowed length for file extensions (including the dot)
const MAX_EXTENSION_LENGTH: usize = 10;

/// Configuration for filtering search results
#[derive(Debug, Clone, Default)]
pub struct FilterConfig {
    /// File extension filter (e.g., ".rs", ".js", ".py")
    pub file_extension: Option<String>,
}

impl FilterConfig {
    /// Create a new filter configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the file extension filter
    pub fn with_file_extension(mut self, extension: String) -> Self {
        // Normalize extension to start with a dot
        let normalized = if extension.starts_with('.') {
            extension
        } else {
            format!(".{}", extension)
        };
        self.file_extension = Some(normalized);
        self
    }
}

/// Filter for search results
pub struct SearchFilter {
    config: FilterConfig,
}

impl SearchFilter {
    /// Create a new search filter with the given configuration
    pub fn new(config: FilterConfig) -> Self {
        Self { config }
    }

    /// Create a search filter from optional command line arguments
    pub fn from_cli_args(filetype: Option<String>) -> Self {
        let mut config = FilterConfig::new();

        if let Some(extension) = filetype {
            config = config.with_file_extension(extension);
        }

        Self::new(config)
    }

    /// Apply all configured filters to search results
    pub fn apply_filters(&self, results: Vec<SearchResult>) -> Result<Vec<SearchResult>> {
        let mut filtered = results;

        // Apply file extension filter if configured
        if let Some(ref extension) = self.config.file_extension {
            filtered = self.filter_by_extension(filtered, extension)?;
        }

        Ok(filtered)
    }

    /// Filter results by file extension
    fn filter_by_extension(
        &self,
        results: Vec<SearchResult>,
        extension: &str,
    ) -> Result<Vec<SearchResult>> {
        Ok(results
            .into_iter()
            .filter(|result| {
                let file_path = &result.chunk.chunk.source_location.file_path;
                self.matches_extension(file_path, extension)
            })
            .collect())
    }

    /// Check if a file path matches the given extension
    fn matches_extension(&self, path: &Path, target_extension: &str) -> bool {
        if let Some(file_extension) = path.extension() {
            let file_ext_str = format!(".{}", file_extension.to_string_lossy());
            file_ext_str.eq_ignore_ascii_case(target_extension)
        } else {
            false
        }
    }

    /// Get a description of active filters for logging/display
    pub fn describe_filters(&self) -> Vec<String> {
        let mut descriptions = Vec::new();

        if let Some(ref extension) = self.config.file_extension {
            descriptions.push(format!("File extension: {}", extension));
        }

        if descriptions.is_empty() {
            descriptions.push("No filters active".to_string());
        }

        descriptions
    }

    /// Check if any filters are active
    pub fn has_active_filters(&self) -> bool {
        self.config.file_extension.is_some()
    }
}

/// Validate and normalize file extension input
pub fn normalize_file_extension(input: &str) -> Result<String> {
    let input = input.trim();

    if input.is_empty() {
        anyhow::bail!("File extension cannot be empty");
    }

    // Handle common cases and normalize
    let normalized = if input.starts_with('.') {
        input.to_lowercase()
    } else {
        format!(".{}", input.to_lowercase())
    };

    // Validate that extension contains only allowed characters
    if !normalized.chars().skip(1).all(|c| c.is_alphanumeric()) {
        anyhow::bail!(
            "Invalid file extension: '{}'. Extensions should contain only alphanumeric characters",
            input
        );
    }

    if normalized.len() < 2 {
        anyhow::bail!(
            "File extension too short: '{}'. Must be at least one character after the dot",
            input
        );
    }

    if normalized.len() > MAX_EXTENSION_LENGTH {
        anyhow::bail!(
            "File extension too long: '{}'. Must be {} characters or less",
            input,
            MAX_EXTENSION_LENGTH
        );
    }

    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ChunkId, ChunkIndexNum, ContentChunk, IndexedChunk, SearchResult, SourceLocation,
        TokenCount,
    };
    use std::path::PathBuf;

    fn create_test_result(file_path: &str, similarity: f32) -> SearchResult {
        let chunk = ContentChunk {
            id: ChunkId::new("test-chunk"),
            content: "test content".to_string(),
            token_count: TokenCount::new(2),
            source_location: SourceLocation {
                file_path: PathBuf::from(file_path),
                start_line: 1,
                end_line: 1,
                start_char: 0,
                end_char: 12,
            },
            chunk_index: ChunkIndexNum::new(0),
            total_chunks: 1,
        };

        let indexed_chunk = IndexedChunk {
            chunk,
            embedding: vec![0.1, 0.2, 0.3],
        };

        SearchResult::new(similarity, indexed_chunk, 0)
    }

    #[test]
    fn test_filter_config_creation() {
        let config = FilterConfig::new();
        assert!(config.file_extension.is_none());

        let config = FilterConfig::new().with_file_extension("rs".to_string());
        assert_eq!(config.file_extension, Some(".rs".to_string()));

        let config = FilterConfig::new().with_file_extension(".js".to_string());
        assert_eq!(config.file_extension, Some(".js".to_string()));
    }

    #[test]
    fn test_search_filter_from_cli_args() {
        let filter = SearchFilter::from_cli_args(None);
        assert!(!filter.has_active_filters());

        let filter = SearchFilter::from_cli_args(Some("rs".to_string()));
        assert!(filter.has_active_filters());
        assert_eq!(filter.config.file_extension, Some(".rs".to_string()));
    }

    #[test]
    fn test_extension_filtering() {
        let results = vec![
            create_test_result("src/main.rs", 0.9),
            create_test_result("src/lib.js", 0.8),
            create_test_result("src/test.py", 0.7),
            create_test_result("README.md", 0.6),
        ];

        let filter = SearchFilter::from_cli_args(Some("rs".to_string()));
        let filtered = filter.apply_filters(results).unwrap();

        assert_eq!(filtered.len(), 1);
        assert!(filtered[0]
            .chunk
            .chunk
            .source_location
            .file_path
            .to_str()
            .unwrap()
            .ends_with(".rs"));
    }

    #[test]
    fn test_extension_case_insensitive() {
        let results = vec![
            create_test_result("src/Main.RS", 0.9),
            create_test_result("src/lib.JS", 0.8),
        ];

        let filter = SearchFilter::from_cli_args(Some("rs".to_string()));
        let filtered = filter.apply_filters(results).unwrap();

        assert_eq!(filtered.len(), 1);
        assert!(filtered[0]
            .chunk
            .chunk
            .source_location
            .file_path
            .to_str()
            .unwrap()
            .ends_with(".RS"));
    }

    #[test]
    fn test_no_extension_files() {
        let results = vec![
            create_test_result("Dockerfile", 0.9),
            create_test_result("Makefile", 0.8),
            create_test_result("src/main.rs", 0.7),
        ];

        let filter = SearchFilter::from_cli_args(Some("rs".to_string()));
        let filtered = filter.apply_filters(results).unwrap();

        assert_eq!(filtered.len(), 1);
        assert!(filtered[0]
            .chunk
            .chunk
            .source_location
            .file_path
            .to_str()
            .unwrap()
            .ends_with(".rs"));
    }

    #[test]
    fn test_matches_extension() {
        let filter = SearchFilter::new(FilterConfig::new());

        assert!(filter.matches_extension(Path::new("test.rs"), ".rs"));
        assert!(filter.matches_extension(Path::new("test.RS"), ".rs"));
        assert!(filter.matches_extension(Path::new("test.js"), ".js"));
        assert!(!filter.matches_extension(Path::new("test.rs"), ".js"));
        assert!(!filter.matches_extension(Path::new("test"), ".rs"));
        assert!(!filter.matches_extension(Path::new("test."), ".rs"));
    }

    #[test]
    fn test_describe_filters() {
        let filter = SearchFilter::from_cli_args(None);
        let descriptions = filter.describe_filters();
        assert_eq!(descriptions, vec!["No filters active"]);

        let filter = SearchFilter::from_cli_args(Some("rs".to_string()));
        let descriptions = filter.describe_filters();
        assert_eq!(descriptions, vec!["File extension: .rs"]);
    }

    #[test]
    fn test_normalize_file_extension() {
        assert_eq!(normalize_file_extension("rs").unwrap(), ".rs");
        assert_eq!(normalize_file_extension(".js").unwrap(), ".js");
        assert_eq!(normalize_file_extension("PY").unwrap(), ".py");
        assert_eq!(normalize_file_extension(".TS").unwrap(), ".ts");

        // Invalid cases
        assert!(normalize_file_extension("").is_err());
        assert!(normalize_file_extension("rs.").is_err()); // Contains non-alphanumeric
        assert!(normalize_file_extension("r s").is_err()); // Contains space
        assert!(normalize_file_extension("verylongextension").is_err()); // Too long
    }

    #[test]
    fn test_has_active_filters() {
        let filter = SearchFilter::from_cli_args(None);
        assert!(!filter.has_active_filters());

        let filter = SearchFilter::from_cli_args(Some("rs".to_string()));
        assert!(filter.has_active_filters());
    }
}
