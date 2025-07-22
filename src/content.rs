use anyhow::{Context, Result};
use encoding_rs::{Encoding, UTF_8};
use std::fs;
use std::path::Path;

pub struct ContentProcessor {
    max_file_size_bytes: Option<u64>,
    binary_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct ProcessedContent {
    pub content: String,
    pub encoding: String,
    pub is_binary: bool,
    pub line_count: usize,
    pub char_count: usize,
}

impl ContentProcessor {
    pub fn new() -> Self {
        Self {
            max_file_size_bytes: None,
            binary_threshold: 0.01,
        }
    }

    pub fn with_max_file_size(mut self, max_size: u64) -> Self {
        self.max_file_size_bytes = Some(max_size);
        self
    }

    pub fn with_binary_threshold(mut self, threshold: f64) -> Self {
        self.binary_threshold = threshold;
        self
    }

    pub fn process_file(&self, file_path: &Path) -> Result<ProcessedContent> {
        let metadata = fs::metadata(file_path)
            .with_context(|| format!("Failed to get metadata for: {}", file_path.display()))?;

        if let Some(max_size) = self.max_file_size_bytes {
            if metadata.len() > max_size {
                anyhow::bail!(
                    "Failed to read file: {} bytes exceeds size limit",
                    metadata.len()
                );
            }
        }

        let raw_bytes = fs::read(file_path)
            .with_context(|| format!("Failed to read file: {}", file_path.display()))?;

        if raw_bytes.is_empty() {
            return Ok(ProcessedContent {
                content: String::new(),
                encoding: "UTF-8".to_string(),
                is_binary: false,
                line_count: 0,
                char_count: 0,
            });
        }

        if self.is_binary_content(&raw_bytes) {
            return Ok(ProcessedContent {
                content: String::new(),
                encoding: "binary".to_string(),
                is_binary: true,
                line_count: 0,
                char_count: 0,
            });
        }

        let (encoding, content) = self.decode_content(&raw_bytes)?;
        let line_count = content.lines().count();
        let char_count = content.chars().count();

        Ok(ProcessedContent {
            content,
            encoding: encoding.name().to_string(),
            is_binary: false,
            line_count,
            char_count,
        })
    }

    fn is_binary_content(&self, bytes: &[u8]) -> bool {
        const SAMPLE_SIZE: usize = 8000;
        let sample = if bytes.len() > SAMPLE_SIZE {
            &bytes[..SAMPLE_SIZE]
        } else {
            bytes
        };

        let null_count = sample.iter().filter(|&&b| b == 0).count();
        let null_percentage = (null_count as f64) / (sample.len() as f64);

        null_percentage > self.binary_threshold
    }

    fn decode_content(&self, bytes: &[u8]) -> Result<(&'static Encoding, String)> {
        let (encoding, _) = Encoding::for_bom(bytes).unwrap_or((UTF_8, 0));
        let (cow, actual_encoding, _) = encoding.decode(bytes);
        Ok((actual_encoding, cow.into_owned()))
    }
}

impl Default for ContentProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_process_empty_file() {
        let temp_file = NamedTempFile::new().unwrap();

        let processor = ContentProcessor::new();
        let result = processor.process_file(temp_file.path()).unwrap();

        assert_eq!(result.content, "");
        assert_eq!(result.encoding, "UTF-8");
        assert!(!result.is_binary);
        assert_eq!(result.line_count, 0);
        assert_eq!(result.char_count, 0);
    }

    #[test]
    fn test_process_text_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let test_content = "Hello, world!\nThis is a test file.\n";
        temp_file.write_all(test_content.as_bytes()).unwrap();

        let processor = ContentProcessor::new();
        let result = processor.process_file(temp_file.path()).unwrap();

        assert_eq!(result.content, test_content);
        assert_eq!(result.encoding, "UTF-8");
        assert!(!result.is_binary);
        assert_eq!(result.line_count, 2);
        assert_eq!(result.char_count, test_content.chars().count());
    }

    #[test]
    fn test_process_binary_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let binary_content = vec![0u8; 100];
        temp_file.write_all(&binary_content).unwrap();

        let processor = ContentProcessor::new();
        let result = processor.process_file(temp_file.path()).unwrap();

        assert_eq!(result.content, "");
        assert_eq!(result.encoding, "binary");
        assert!(result.is_binary);
        assert_eq!(result.line_count, 0);
        assert_eq!(result.char_count, 0);
    }

    #[test]
    fn test_max_file_size_limit() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let large_content = "x".repeat(1000);
        temp_file.write_all(large_content.as_bytes()).unwrap();

        let processor = ContentProcessor::new().with_max_file_size(500);
        let result = processor.process_file(temp_file.path());

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Failed to read file"));
    }

    #[test]
    fn test_nonexistent_file() {
        let processor = ContentProcessor::new();
        let result = processor.process_file(Path::new("/nonexistent/file.txt"));

        assert!(result.is_err());
    }
}
