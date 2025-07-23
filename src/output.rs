//! Output formatting for search results.
//!
//! This module provides different output formats for search results including
//! JSON for LLM consumption and human-readable text for terminal display.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::config::SearchConfig;
use crate::types::SearchResult;

/// Default number of content lines to show in text output
pub const DEFAULT_MAX_CONTENT_LINES: usize = 3;

/// Supported output formats for search results
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputFormat {
    /// Line-delimited JSON for LLM consumption (default)
    Json,
    /// Human-readable text for terminal display
    Text,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(OutputFormat::Json),
            "text" => Ok(OutputFormat::Text),
            _ => Err(format!(
                "Invalid output format: '{}'. Valid options are 'json' or 'text'",
                s
            )),
        }
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Json => write!(f, "json"),
            OutputFormat::Text => write!(f, "text"),
        }
    }
}

/// JSON representation of a search result for LLM consumption
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonSearchResult {
    /// File path relative to search root
    pub file: String,
    /// Starting line number (1-based)
    pub line_start: usize,
    /// Ending line number (1-based)
    pub line_end: usize,
    /// Similarity score (0.0 to 1.0)
    pub score: f32,
    /// Content of the matching chunk
    pub content: String,
    /// Rank in the result set (0-based)
    pub rank: usize,
}


/// Result formatter that handles different output formats
pub struct ResultFormatter {
    format: OutputFormat,
    max_content_lines: usize,
    search_config: SearchConfig,
}

impl ResultFormatter {
    /// Create a new result formatter with the specified format and search config
    pub fn new(format: OutputFormat, search_config: SearchConfig) -> Self {
        Self {
            format,
            max_content_lines: DEFAULT_MAX_CONTENT_LINES,
            search_config,
        }
    }

    /// Create a new result formatter with custom configuration
    pub fn with_config(format: OutputFormat, max_content_lines: usize, search_config: SearchConfig) -> Self {
        Self {
            format,
            max_content_lines,
            search_config,
        }
    }

    /// Format and print search results to stdout
    pub fn print_results(&self, results: &[SearchResult], query: &str) -> Result<()> {
        match self.format {
            OutputFormat::Json => self.print_json_results(results),
            OutputFormat::Text => self.print_text_results(results, query),
        }
    }

    /// Print results in line-delimited JSON format
    fn print_json_results(&self, results: &[SearchResult]) -> Result<()> {
        for result in results {
            let json_result = self.create_json_result(result);
            let json_line = serde_json::to_string(&json_result)?;
            println!("{}", json_line);
        }
        Ok(())
    }

    /// Create a JsonSearchResult with optional content rehydration
    fn create_json_result(&self, result: &SearchResult) -> JsonSearchResult {
        let content = if self.search_config.rehydrate_content {
            // Try to rehydrate the content, fallback to original if it fails
            result.chunk.chunk.rehydrate_content()
                .unwrap_or_else(|_| result.chunk.chunk.content.clone())
        } else {
            result.chunk.chunk.content.clone()
        };

        JsonSearchResult {
            file: result
                .chunk
                .chunk
                .source_location
                .file_path
                .to_string_lossy()
                .into_owned(),
            line_start: result.chunk.chunk.source_location.start_line,
            line_end: result.chunk.chunk.source_location.end_line,
            score: result.similarity,
            content,
            rank: result.rank,
        }
    }

    /// Print results in human-readable text format
    fn print_text_results(&self, results: &[SearchResult], query: &str) -> Result<()> {
        if results.is_empty() {
            println!("No results found for query: '{}'", query);
            println!("Try adjusting your search terms or lowering the similarity threshold.");
            return Ok(());
        }

        println!("Found {} results for query: '{}'", results.len(), query);
        println!();

        for (i, result) in results.iter().enumerate() {
            // Header with rank, file location, and similarity score
            println!(
                "{}. 📄 {} (similarity: {:.1}%)",
                i + 1,
                result.location_display(),
                result.similarity * 100.0
            );

            // Content preview with proper indentation
            let content = if self.search_config.rehydrate_content {
                // Try to rehydrate the content, fallback to original if it fails
                result.chunk.chunk.rehydrate_content()
                    .unwrap_or_else(|_| result.chunk.chunk.content.clone())
            } else {
                result.chunk.chunk.content.clone()
            };
            
            let lines: Vec<&str> = content.lines().collect();
            let max_lines = self.max_content_lines;

            for (line_idx, line) in lines.iter().take(max_lines).enumerate() {
                let line_number = result.chunk.chunk.source_location.start_line + line_idx;
                println!("  {:4}│ {}", line_number, line.trim());
            }

            if lines.len() > max_lines {
                println!("  ...│ ...");
            }

            println!(); // Empty line between results
        }

        println!("Search completed successfully");
        Ok(())
    }

    /// Print a message when no results are found
    pub fn print_no_results(&self, query: &str, threshold: Option<f32>) -> Result<()> {
        match self.format {
            OutputFormat::Json => {
                // For JSON output, we just print nothing (no results)
                Ok(())
            }
            OutputFormat::Text => {
                println!("No results found for query: '{}'", query);
                if let Some(threshold) = threshold {
                    println!(
                        "Try lowering the similarity threshold (currently {:.1}%)",
                        threshold * 100.0
                    );
                }
                println!("Or try different search terms that might better match your content.");
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ChunkId, ChunkIndexNum, ContentChunk, IndexedChunk, SearchResult, SourceLocation,
        TokenCount,
    };
    use std::path::PathBuf;

    fn create_test_search_result(similarity: f32, file_path: &str, content: &str) -> SearchResult {
        let chunk = ContentChunk {
            id: ChunkId::new("test-chunk"),
            content: content.to_string(),
            token_count: TokenCount::new(10),
            source_location: SourceLocation {
                file_path: PathBuf::from(file_path),
                start_line: 42,
                end_line: 44,
                start_char: 0,
                end_char: content.len(),
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
    fn test_output_format_from_str() {
        assert_eq!("json".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert_eq!("text".parse::<OutputFormat>().unwrap(), OutputFormat::Text);
        assert_eq!("JSON".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert_eq!("TEXT".parse::<OutputFormat>().unwrap(), OutputFormat::Text);

        assert!("invalid".parse::<OutputFormat>().is_err());
        assert!("xml".parse::<OutputFormat>().is_err());
    }

    #[test]
    fn test_output_format_display() {
        assert_eq!(OutputFormat::Json.to_string(), "json");
        assert_eq!(OutputFormat::Text.to_string(), "text");
    }

    #[test]
    fn test_json_search_result_conversion() {
        let result = create_test_search_result(
            0.85,
            "src/main.rs",
            "fn main() {\n    println!(\"Hello\");\n}",
        );
        let search_config = SearchConfig::default();
        let formatter = ResultFormatter::new(OutputFormat::Json, search_config);
        let json_result = formatter.create_json_result(&result);

        assert_eq!(json_result.file, "src/main.rs");
        assert_eq!(json_result.line_start, 42);
        assert_eq!(json_result.line_end, 44);
        assert_eq!(json_result.score, 0.85);
        assert_eq!(
            json_result.content,
            "fn main() {\n    println!(\"Hello\");\n}"
        );
        assert_eq!(json_result.rank, 0);
    }

    #[test]
    fn test_json_serialization() {
        let result = create_test_search_result(0.75, "test.rs", "test content");
        let search_config = SearchConfig::default();
        let formatter = ResultFormatter::new(OutputFormat::Json, search_config);
        let json_result = formatter.create_json_result(&result);

        let serialized = serde_json::to_string(&json_result).unwrap();
        assert!(serialized.contains("\"file\":\"test.rs\""));
        assert!(serialized.contains("\"score\":0.75"));
        assert!(serialized.contains("\"content\":\"test content\""));

        // Test that we can deserialize it back
        let deserialized: JsonSearchResult = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.file, json_result.file);
        assert_eq!(deserialized.score, json_result.score);
    }

    #[test]
    fn test_result_formatter_creation() {
        let search_config = SearchConfig::default();
        let formatter = ResultFormatter::new(OutputFormat::Json, search_config.clone());
        assert_eq!(formatter.format, OutputFormat::Json);

        let formatter = ResultFormatter::new(OutputFormat::Text, search_config);
        assert_eq!(formatter.format, OutputFormat::Text);
    }

    #[test]
    fn test_content_rehydration_enabled() {
        // Test with rehydration enabled (default)  
        let mut search_config = SearchConfig::default();
        search_config.rehydrate_content = true;
        
        let result = create_test_search_result(0.85, "src/main.rs", "fn main() {}");
        let formatter = ResultFormatter::new(OutputFormat::Json, search_config);
        let json_result = formatter.create_json_result(&result);
        
        // Since test content is real content (not placeholder), it should remain unchanged
        assert!(json_result.content.contains("fn main() {}"));
    }

    #[test]
    fn test_content_rehydration_disabled() {
        // Test with rehydration disabled
        let mut search_config = SearchConfig::default();
        search_config.rehydrate_content = false;
        
        let result = create_test_search_result(0.85, "src/main.rs", "fn main() {}");
        let formatter = ResultFormatter::new(OutputFormat::Json, search_config);
        let json_result = formatter.create_json_result(&result);
        
        // Content should be unchanged since we're not rehydrating
        assert!(json_result.content.contains("fn main() {}"));
    }
}
