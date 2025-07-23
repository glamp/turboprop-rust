//! Mock embedding generator for testing purposes.
//!
//! This module provides a deterministic fake embedding generator that can be used
//! in tests to avoid depending on actual ML models or network downloads.

use anyhow::Result;

use crate::constants;
use super::config::EmbeddingConfig;

/// Mock embedding generator for testing
#[cfg(any(test, feature = "test-utils"))]
pub struct MockEmbeddingGenerator {
    config: EmbeddingConfig,
}

#[cfg(any(test, feature = "test-utils"))]
impl MockEmbeddingGenerator {
    /// Create a new mock embedding generator for testing
    pub fn new(config: EmbeddingConfig) -> Self {
        Self { config }
    }

    /// Generate a deterministic fake embedding for testing
    pub fn embed_single(&mut self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Ok(vec![0.0; self.config.embedding_dimensions]);
        }

        // Generate deterministic embeddings based on text content
        let mut embedding = Vec::with_capacity(self.config.embedding_dimensions);
        let text_hash = text.chars().map(|c| c as usize).sum::<usize>();

        for i in 0..self.config.embedding_dimensions {
            let value = constants::test::TEST_EMBEDDING_BASE_VALUE
                + ((text_hash + i) % constants::test::TEXT_HASH_MODULO) as f32
                    * constants::test::TEST_EMBEDDING_VARIATION_FACTOR;
            embedding.push(value);
        }

        Ok(embedding)
    }

    /// Generate fake embeddings for a batch of texts
    pub fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|text| self.embed_single(text)).collect()
    }

    /// Get embedding dimensions
    pub fn embedding_dimensions(&self) -> usize {
        self.config.embedding_dimensions
    }

    /// Get model name
    pub fn model_name(&self) -> &str {
        &self.config.model_name
    }
}