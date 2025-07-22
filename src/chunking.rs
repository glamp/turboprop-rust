use crate::content::{ContentProcessor, ProcessedContent};
use crate::types::{ChunkingConfig, ContentChunk, SourceLocation};
use anyhow::{Context, Result};
use std::path::Path;
use unicode_segmentation::UnicodeSegmentation;

pub struct ChunkingStrategy {
    config: ChunkingConfig,
    content_processor: ContentProcessor,
}

impl ChunkingStrategy {
    pub fn new(config: ChunkingConfig) -> Self {
        Self {
            config,
            content_processor: ContentProcessor::new(),
        }
    }

    pub fn with_content_processor(mut self, processor: ContentProcessor) -> Self {
        self.content_processor = processor;
        self
    }

    pub fn chunk_file(&self, file_path: &Path) -> Result<Vec<ContentChunk>> {
        let processed = self
            .content_processor
            .process_file(file_path)
            .with_context(|| {
                format!(
                    "Failed to read and process file content for chunking: {}",
                    file_path.display()
                )
            })?;

        if processed.is_binary || processed.content.is_empty() {
            return Ok(vec![]);
        }

        self.chunk_content(&processed, file_path)
    }

    pub fn chunk_content(
        &self,
        processed: &ProcessedContent,
        file_path: &Path,
    ) -> Result<Vec<ContentChunk>> {
        let content = &processed.content;

        if content.is_empty() {
            return Ok(vec![]);
        }

        let tokens = self.tokenize_content(content);
        let token_count = tokens.len();

        if token_count <= self.config.target_chunk_size_tokens {
            let source_location = SourceLocation {
                file_path: file_path.to_path_buf(),
                start_line: 1,
                end_line: processed.line_count.max(1),
                start_char: 0,
                end_char: processed.char_count,
            };

            let chunk = ContentChunk {
                id: format!("{}:0", file_path.display()),
                content: content.clone(),
                token_count,
                source_location,
                chunk_index: 0,
                total_chunks: 1,
            };

            return Ok(vec![chunk]);
        }

        self.create_overlapping_chunks(&tokens, content, file_path, processed.line_count)
    }

    fn create_overlapping_chunks(
        &self,
        tokens: &[String],
        content: &str,
        file_path: &Path,
        _total_lines: usize,
    ) -> Result<Vec<ContentChunk>> {
        let mut chunks = Vec::new();
        let mut start_token_idx = 0;
        let mut chunk_index = 0;

        // Pre-compute token positions for efficiency
        let token_positions = self.compute_token_positions(content, tokens);

        while start_token_idx < tokens.len() {
            // Calculate the desired end position respecting max_chunk_size_tokens
            let desired_end = start_token_idx + self.config.target_chunk_size_tokens;
            let max_end = start_token_idx + self.config.max_chunk_size_tokens;

            let end_token_idx = std::cmp::min(std::cmp::min(desired_end, max_end), tokens.len());

            // Ensure we have at least min_chunk_size_tokens unless we're at the end or target size is small
            let remaining_tokens = tokens.len() - start_token_idx;
            if remaining_tokens < self.config.min_chunk_size_tokens
                && start_token_idx > 0
                && self.config.target_chunk_size_tokens >= self.config.min_chunk_size_tokens
            {
                // Only apply minimum size constraint if target size is also above minimum
                break;
            }

            let chunk_tokens = &tokens[start_token_idx..end_token_idx];
            let chunk_content = chunk_tokens.join(" ");

            // Validate chunk size
            if chunk_tokens.len() > self.config.max_chunk_size_tokens {
                return Err(anyhow::anyhow!(
                    "Failed to create chunk: size {} exceeds maximum allowed size {}",
                    chunk_tokens.len(),
                    self.config.max_chunk_size_tokens
                ));
            }

            // Get positions from pre-computed data
            let (start_char, end_char) = if start_token_idx < token_positions.len()
                && end_token_idx > 0
                && end_token_idx - 1 < token_positions.len()
            {
                (
                    token_positions[start_token_idx].0,
                    token_positions[end_token_idx - 1].1,
                )
            } else {
                (0, chunk_content.len())
            };

            let (start_line, end_line) =
                self.calculate_line_positions(content, start_char, end_char);

            let source_location = SourceLocation {
                file_path: file_path.to_path_buf(),
                start_line: start_line + 1,
                end_line: end_line + 1,
                start_char,
                end_char,
            };

            let chunk = ContentChunk {
                id: format!("{}:{}", file_path.display(), chunk_index),
                content: chunk_content,
                token_count: chunk_tokens.len(),
                source_location,
                chunk_index,
                total_chunks: 0,
            };

            chunks.push(chunk);

            if end_token_idx >= tokens.len() {
                break;
            }

            let overlap_start = if end_token_idx >= self.config.overlap_tokens {
                end_token_idx - self.config.overlap_tokens
            } else {
                end_token_idx
            };

            start_token_idx = overlap_start;
            chunk_index += 1;
        }

        let total_chunks = chunks.len();
        for chunk in &mut chunks {
            chunk.total_chunks = total_chunks;
        }

        Ok(chunks)
    }

    fn compute_token_positions(&self, content: &str, tokens: &[String]) -> Vec<(usize, usize)> {
        let mut positions = Vec::new();
        let mut search_start = 0;

        // Find actual token positions in the original content
        for token in tokens {
            if let Some(pos) = content[search_start..].find(token) {
                let absolute_start = search_start + pos;
                let absolute_end = absolute_start + token.len();
                positions.push((absolute_start, absolute_end));
                search_start = absolute_end;
            } else {
                // If we can't find the token, use the last known position
                let last_end = positions.last().map(|(_, end)| *end).unwrap_or(0);
                positions.push((last_end, last_end + token.len()));
            }
        }

        positions
    }

    fn calculate_line_positions(
        &self,
        content: &str,
        start_char: usize,
        end_char: usize,
    ) -> (usize, usize) {
        let mut char_count = 0;
        let mut start_line = 0;
        let mut end_line = 0;
        let mut found_start = false;
        let mut found_end = false;

        for (line_idx, line) in content.lines().enumerate() {
            let line_len = line.len();
            let line_end = char_count + line_len;

            // Check if start_char is in this line
            if !found_start && char_count <= start_char && start_char <= line_end {
                start_line = line_idx;
                found_start = true;
            }

            // Check if end_char is in this line
            if char_count <= end_char && end_char <= line_end {
                end_line = line_idx;
                found_end = true;
                if found_start {
                    break;
                }
            }

            // Account for newline character, but check if there's actually a next line
            let next_char_pos = char_count + line_len;
            if next_char_pos < content.len() {
                // There's more content, so there was a newline
                char_count = next_char_pos + 1;
            } else {
                // This is the last line, no newline at the end
                break;
            }
        }

        // Handle edge case where positions are beyond content
        if !found_end && end_char >= content.len() {
            end_line = content.lines().count().max(1) - 1;
        }

        (start_line, end_line)
    }

    fn tokenize_content(&self, content: &str) -> Vec<String> {
        content.unicode_words().map(|s| s.to_string()).collect()
    }
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        Self::new(ChunkingConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChunkingConfig;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_chunk_small_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let content = "Hello world! This is a small test file.";
        temp_file.write_all(content.as_bytes()).unwrap();

        let config = ChunkingConfig::default().with_target_size(10);
        let strategy = ChunkingStrategy::new(config);

        let chunks = strategy.chunk_file(temp_file.path()).unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[0].total_chunks, 1);
        assert_eq!(chunks[0].source_location.start_line, 1);
    }

    #[test]
    fn test_chunk_large_file_with_overlap() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let content = "This is a longer piece of text that should be split into multiple chunks with overlap between them to ensure continuity.";
        temp_file.write_all(content.as_bytes()).unwrap();

        let config = ChunkingConfig::default()
            .with_target_size(8)
            .with_overlap(2);
        let strategy = ChunkingStrategy::new(config);

        let chunks = strategy.chunk_file(temp_file.path()).unwrap();

        assert!(chunks.len() > 1);

        for chunk in &chunks {
            assert!(chunk.token_count <= 8);
            assert!(chunk.total_chunks == chunks.len());
        }

        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_index, i);
        }
    }

    #[test]
    fn test_chunk_empty_file() {
        let temp_file = NamedTempFile::new().unwrap();

        let strategy = ChunkingStrategy::default();
        let chunks = strategy.chunk_file(temp_file.path()).unwrap();

        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_multiline_content() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let content = "Line one has some words.\nLine two has different words.\nLine three completes the test.";
        temp_file.write_all(content.as_bytes()).unwrap();

        let config = ChunkingConfig::default().with_target_size(6);
        let strategy = ChunkingStrategy::new(config);

        let chunks = strategy.chunk_file(temp_file.path()).unwrap();

        assert!(!chunks.is_empty());

        for chunk in &chunks {
            assert!(chunk.source_location.start_line >= 1);
            assert!(chunk.source_location.end_line >= chunk.source_location.start_line);
        }
    }

    #[test]
    fn test_tokenize_content() {
        let strategy = ChunkingStrategy::default();
        let content = "Hello, world! This is a test.";
        let tokens = strategy.tokenize_content(content);

        let expected = vec!["Hello", "world", "This", "is", "a", "test"];
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_chunk_content_directly() {
        let processed = ProcessedContent {
            content: "Short content here".to_string(),
            encoding: "UTF-8".to_string(),
            is_binary: false,
            line_count: 1,
            char_count: 18,
        };

        let strategy = ChunkingStrategy::default();
        let chunks = strategy
            .chunk_content(&processed, Path::new("test.txt"))
            .unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "Short content here");
    }
}
