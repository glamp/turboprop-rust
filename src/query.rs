//! Query processing and embedding generation for search functionality.
//!
//! This module handles converting search queries into vector embeddings using
//! the same model and configuration as the target index.

use anyhow::{Context, Result};
use std::path::Path;
use tracing::{debug, info};

use crate::config::TurboPropConfig;
use crate::embeddings::EmbeddingGenerator;
use crate::index::PersistentChunkIndex;

/// Query processor that generates embeddings for search queries
pub struct QueryProcessor {
    embedding_generator: EmbeddingGenerator,
}

impl QueryProcessor {
    /// Create a new query processor using the same configuration as the target index
    pub async fn from_index_config(index: &PersistentChunkIndex) -> Result<Self> {
        let index_config = index.config();

        // Convert IndexConfig to EmbeddingConfig
        let embedding_config = crate::embeddings::EmbeddingConfig {
            model_name: index_config.model_name.clone(),
            cache_dir: std::path::PathBuf::from(".turboprop/models"),
            batch_size: index_config.batch_size,
            embedding_dimensions: index_config.embedding_dimensions,
        };

        let embedding_generator = EmbeddingGenerator::new(embedding_config)
            .await
            .context("Failed to initialize embedding generator for query processing")?;

        Ok(Self {
            embedding_generator,
        })
    }

    /// Create a new query processor from explicit configuration
    pub async fn from_config(config: &TurboPropConfig) -> Result<Self> {
        let embedding_generator = EmbeddingGenerator::new(config.embedding.clone())
            .await
            .context("Failed to initialize embedding generator for query processing")?;

        Ok(Self {
            embedding_generator,
        })
    }

    /// Generate embedding for a search query
    pub fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        if query.trim().is_empty() {
            anyhow::bail!("Query cannot be empty");
        }

        debug!("Generating embedding for query: '{}'", query);

        let embeddings = self
            .embedding_generator
            .embed_batch(&[query.to_string()])
            .context("Failed to generate query embedding")?;

        match embeddings.into_iter().next() {
            Some(embedding) => {
                info!(
                    "Generated query embedding with {} dimensions",
                    embedding.len()
                );
                Ok(embedding)
            }
            None => anyhow::bail!("Failed to generate embedding for query"),
        }
    }

    /// Get the embedding dimensions for this processor
    pub fn embedding_dimensions(&self) -> usize {
        self.embedding_generator.embedding_dimensions()
    }
}

/// Validate that a query is suitable for searching
pub fn validate_query(query: &str) -> Result<()> {
    let trimmed = query.trim();

    if trimmed.is_empty() {
        anyhow::bail!("Search query cannot be empty");
    }

    if trimmed.len() > 1000 {
        anyhow::bail!("Search query is too long (maximum 1000 characters)");
    }

    Ok(())
}

/// Create a query processor from an index path by loading the index configuration
pub async fn create_query_processor_from_path<P: AsRef<Path>>(
    index_path: P,
) -> Result<QueryProcessor> {
    let index = PersistentChunkIndex::load(index_path.as_ref())
        .context("Failed to load index for query processing")?;

    QueryProcessor::from_index_config(&index).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TurboPropConfig;

    #[test]
    fn test_validate_query() {
        // Valid queries
        assert!(validate_query("test query").is_ok());
        assert!(validate_query("jwt authentication").is_ok());
        assert!(validate_query("function main").is_ok());

        // Invalid queries
        assert!(validate_query("").is_err());
        assert!(validate_query("   ").is_err());
        assert!(validate_query(&"a".repeat(1001)).is_err());
    }

    #[tokio::test]
    async fn test_query_processor_creation() {
        let config = TurboPropConfig::default();
        let result = QueryProcessor::from_config(&config).await;

        // This might fail in CI without proper model setup, so we just check the error is reasonable
        match result {
            Ok(processor) => {
                assert!(processor.embedding_dimensions() > 0);
            }
            Err(e) => {
                // Expected in environments without model access
                assert!(e.to_string().contains("embedding") || e.to_string().contains("model"));
            }
        }
    }

    #[tokio::test]
    async fn test_embed_query_validation() {
        let config = TurboPropConfig::default();

        // Try to create processor, but handle potential model loading failures
        if let Ok(mut processor) = QueryProcessor::from_config(&config).await {
            // Test empty query rejection
            let result = processor.embed_query("");
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("empty"));

            // Test whitespace-only query rejection
            let result = processor.embed_query("   ");
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("empty"));
        }
    }
}
