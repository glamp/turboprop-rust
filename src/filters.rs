//! Search result filtering functionality.
//!
//! This module provides filtering capabilities for search results, including
//! filtering by file type/extension.

use anyhow::Result;
use glob::Pattern;
use std::path::Path;

use crate::types::SearchResult;

/// Maximum allowed length for file extensions (including the dot)
const MAX_EXTENSION_LENGTH: usize = 10;

/// Maximum allowed length for glob patterns
const MAX_GLOB_PATTERN_LENGTH: usize = 1000;

/// A validated glob pattern wrapper
#[derive(Debug, Clone)]
pub struct GlobPattern {
    /// The original pattern string
    pattern: String,
    /// The compiled glob pattern
    compiled: Pattern,
}

impl GlobPattern {
    /// Create a new GlobPattern from a string, validating it first
    pub fn new(pattern: &str) -> Result<Self> {
        validate_glob_pattern(pattern)?;
        let compiled = Pattern::new(pattern)
            .map_err(|e| anyhow::anyhow!("Invalid glob pattern '{}': {}", pattern, e))?;

        Ok(Self {
            pattern: pattern.to_string(),
            compiled,
        })
    }

    /// Get the original pattern string
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Check if a path matches this glob pattern
    pub fn matches(&self, path: &Path) -> bool {
        // Convert path to string for matching
        if let Some(path_str) = path.to_str() {
            self.compiled.matches(path_str)
        } else {
            // If path contains invalid UTF-8, it won't match
            false
        }
    }
}

/// Validate a glob pattern string
pub fn validate_glob_pattern(pattern: &str) -> Result<()> {
    let pattern = pattern.trim();

    // Check for empty pattern
    if pattern.is_empty() {
        anyhow::bail!("Glob pattern cannot be empty");
    }

    // Check pattern length
    if pattern.len() > MAX_GLOB_PATTERN_LENGTH {
        anyhow::bail!(
            "Glob pattern too long: {} characters. Maximum allowed is {}",
            pattern.len(),
            MAX_GLOB_PATTERN_LENGTH
        );
    }

    // Validate that the pattern is valid UTF-8 (already guaranteed by &str)
    // and contains reasonable characters
    if pattern.chars().any(|c| c.is_control() && c != '\t') {
        anyhow::bail!(
            "Glob pattern contains invalid control characters: '{}'",
            pattern
        );
    }

    // Try to compile the pattern to check for syntax errors
    Pattern::new(pattern)
        .map_err(|e| anyhow::anyhow!("Invalid glob pattern syntax '{}': {}", pattern, e))?;

    Ok(())
}

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

    mod test_glob_pattern {
        use super::*;

        #[test]
        fn test_glob_pattern_creation_valid() {
            // Test valid patterns
            let pattern = GlobPattern::new("*.rs").unwrap();
            assert_eq!(pattern.pattern(), "*.rs");

            let pattern = GlobPattern::new("src/*.js").unwrap();
            assert_eq!(pattern.pattern(), "src/*.js");

            let pattern = GlobPattern::new("**/test_*.py").unwrap();
            assert_eq!(pattern.pattern(), "**/test_*.py");

            let pattern = GlobPattern::new("dir/*/file.txt").unwrap();
            assert_eq!(pattern.pattern(), "dir/*/file.txt");
        }

        #[test]
        fn test_glob_pattern_creation_invalid() {
            // Test empty pattern
            assert!(GlobPattern::new("").is_err());
            assert!(GlobPattern::new("   ").is_err());

            // Test pattern with control characters (except tab)
            assert!(GlobPattern::new("file\x00.txt").is_err());
            assert!(GlobPattern::new("file\x01.txt").is_err());

            // Test extremely long pattern
            let long_pattern = "a".repeat(MAX_GLOB_PATTERN_LENGTH + 1);
            assert!(GlobPattern::new(&long_pattern).is_err());
        }

        #[test]
        fn test_glob_pattern_matching() {
            // Test simple wildcard patterns - these match against the full path
            let pattern = GlobPattern::new("*.rs").unwrap();
            assert!(pattern.matches(Path::new("main.rs")));
            assert!(pattern.matches(Path::new("lib.rs")));
            assert!(!pattern.matches(Path::new("main.js")));
            // *.rs matches paths ending in .rs, including those with directories
            assert!(pattern.matches(Path::new("src/main.rs")));
            assert!(pattern.matches(Path::new("deep/nested/file.rs")));

            // Test directory patterns - understand how * works with directories
            let pattern = GlobPattern::new("src/*.rs").unwrap();
            assert!(pattern.matches(Path::new("src/main.rs")));
            assert!(pattern.matches(Path::new("src/lib.rs")));
            assert!(!pattern.matches(Path::new("main.rs"))); // No src/ prefix
            assert!(!pattern.matches(Path::new("tests/main.rs"))); // Wrong directory
                                                                   // The * in src/*.rs can match paths with slashes, so this actually matches
            assert!(pattern.matches(Path::new("src/nested/main.rs"))); // * can match nested/main

            // Test recursive patterns - ** matches any number of directories
            let pattern = GlobPattern::new("**/test_*.py").unwrap();
            assert!(pattern.matches(Path::new("test_main.py")));
            assert!(pattern.matches(Path::new("src/test_lib.py")));
            assert!(pattern.matches(Path::new("tests/unit/test_utils.py")));
            assert!(!pattern.matches(Path::new("main.py")));
            assert!(!pattern.matches(Path::new("src/lib.py")));

            // Test patterns that should match any file with extension regardless of path depth
            let pattern = GlobPattern::new("**/*.rs").unwrap();
            assert!(pattern.matches(Path::new("main.rs")));
            assert!(pattern.matches(Path::new("src/main.rs")));
            assert!(pattern.matches(Path::new("deep/nested/dir/file.rs")));
            assert!(!pattern.matches(Path::new("main.js")));
        }

        #[test]
        fn test_glob_pattern_case_sensitivity() {
            // Glob patterns should be case-sensitive by default
            let pattern = GlobPattern::new("*.RS").unwrap();
            assert!(pattern.matches(Path::new("main.RS")));
            assert!(!pattern.matches(Path::new("main.rs"))); // Different case
        }

        #[test]
        fn test_glob_pattern_edge_cases() {
            // Test pattern with tab character (should be allowed)
            let pattern = GlobPattern::new("file\twith\ttab.txt");
            assert!(pattern.is_ok());

            // Test Unicode characters
            let pattern = GlobPattern::new("файл_*.txt").unwrap();
            assert!(pattern.matches(Path::new("файл_test.txt")));
            assert!(!pattern.matches(Path::new("file_test.txt")));

            // Test very specific patterns
            let pattern = GlobPattern::new("exact_file.txt").unwrap();
            assert!(pattern.matches(Path::new("exact_file.txt")));
            assert!(!pattern.matches(Path::new("exact_file.rs")));
            assert!(!pattern.matches(Path::new("other_exact_file.txt")));
        }

        #[test]
        fn test_validate_glob_pattern() {
            // Valid patterns
            assert!(validate_glob_pattern("*.rs").is_ok());
            assert!(validate_glob_pattern("src/**/*.js").is_ok());
            assert!(validate_glob_pattern("test_*.py").is_ok());
            assert!(validate_glob_pattern("file.txt").is_ok());
            assert!(validate_glob_pattern("dir/*/file.?").is_ok());

            // Invalid patterns
            assert!(validate_glob_pattern("").is_err());
            assert!(validate_glob_pattern("   ").is_err());

            // Pattern with control characters
            assert!(validate_glob_pattern("file\x00.txt").is_err());
            assert!(validate_glob_pattern("file\n.txt").is_err());

            // Pattern too long
            let long_pattern = "a".repeat(MAX_GLOB_PATTERN_LENGTH + 1);
            assert!(validate_glob_pattern(&long_pattern).is_err());

            // Tab should be allowed
            assert!(validate_glob_pattern("file\ttab.txt").is_ok());
        }

        #[test]
        fn test_glob_pattern_common_use_cases() {
            // Test common patterns from the specification
            let test_cases = vec![
                ("*.ext", "file.ext", true),
                ("*.ext", "file.other", false),
                ("dir/*.ext", "dir/file.ext", true),
                ("dir/*.ext", "other/file.ext", false),
                ("**/pattern", "pattern", true),
                ("**/pattern", "deep/nested/pattern", true),
                ("**/pattern", "deep/nested/other", false),
                ("src/*.js", "src/main.js", true),
                ("src/*.js", "src/lib.js", true),
                ("src/*.js", "tests/main.js", false),
                ("**/*.rs", "main.rs", true),
                ("**/*.rs", "src/main.rs", true),
                ("**/*.rs", "tests/unit/helper.rs", true),
                ("**/*.rs", "main.js", false),
            ];

            for (pattern_str, path_str, should_match) in test_cases {
                let pattern = GlobPattern::new(pattern_str).unwrap();
                let path = Path::new(path_str);
                assert_eq!(
                    pattern.matches(path),
                    should_match,
                    "Pattern '{}' vs path '{}' should {}match",
                    pattern_str,
                    path_str,
                    if should_match { "" } else { "not " }
                );
            }
        }

        #[test]
        fn test_glob_pattern_invalid_utf8_handling() {
            // Test that patterns handle invalid UTF-8 paths gracefully
            let _pattern = GlobPattern::new("*.txt").unwrap();

            // We can't easily create an invalid UTF-8 Path in safe Rust,
            // but we can verify that our matching function handles the None case
            // This is covered by the matches() implementation returning false
            // for paths that can't be converted to strings
        }
    }
}
