//! Embedding generation module for converting text chunks to vector representations.
//!
//! This module provides functionality for generating embeddings from text chunks
//! using pre-trained transformer models via different backends including FastEmbed,
//! GGUF/Candle, and HuggingFace.
//!
//! # Architecture
//!
//! The module is organized into several focused sub-modules:
//!
//! - `generator` - Main EmbeddingGenerator implementation
//! - `config` - Configuration structures (EmbeddingConfig, EmbeddingOptions)
//! - `backends` - Backend enumeration and delegation (EmbeddingBackendType)
//! - `batch` - Batch processing utilities (BatchProcessor trait)
//! - `mock` - Test utilities (feature-gated)
//!
//! # Usage
//!
//! ```no_run
//! use turboprop::embeddings::{EmbeddingGenerator, EmbeddingConfig};
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let config = EmbeddingConfig::default();
//! let mut generator = EmbeddingGenerator::new(config).await?;
//!
//! let text = "Hello, world!";
//! let embedding = generator.embed_single(text)?;
//! println!("Generated embedding with {} dimensions", embedding.len());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! # });
//! ```

pub mod backends;
pub mod batch;
pub mod config;
pub mod generator;
pub mod optimized;

#[cfg(any(test, feature = "test-utils"))]
pub mod mock;

// Re-export main public API
pub use backends::EmbeddingBackendType;
pub use batch::{process_texts_in_batches, BatchProcessor};
pub use config::{EmbeddingConfig, EmbeddingOptions, DEFAULT_EMBEDDING_DIMENSIONS, DEFAULT_MODEL};
pub use generator::EmbeddingGenerator;
pub use optimized::{OptimizedEmbeddingGenerator, PerformanceReport};

#[cfg(any(test, feature = "test-utils"))]
pub use mock::MockEmbeddingGenerator;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_embedding_config_default() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.model_name, DEFAULT_MODEL);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.cache_dir, PathBuf::from(".turboprop/models"));
    }

    #[tokio::test]
    async fn test_embedding_config_with_model() {
        let config = EmbeddingConfig::with_model("custom-model");
        assert_eq!(config.model_name, "custom-model");
        assert_eq!(config.batch_size, 32); // Should keep default
    }

    #[tokio::test]
    async fn test_embedding_config_with_cache_dir() {
        let config = EmbeddingConfig::default().with_cache_dir("/tmp/models");
        assert_eq!(config.cache_dir, PathBuf::from("/tmp/models"));
    }

    #[tokio::test]
    async fn test_embedding_config_with_batch_size() {
        let config = EmbeddingConfig::default().with_batch_size(16);
        assert_eq!(config.batch_size, 16);
    }

    #[test]
    fn test_mock_embedding_generator_initialization() {
        let config = EmbeddingConfig::default();
        let generator = MockEmbeddingGenerator::new(config);

        assert_eq!(generator.model_name(), DEFAULT_MODEL);
        assert_eq!(
            generator.embedding_dimensions(),
            DEFAULT_EMBEDDING_DIMENSIONS
        );
    }

    #[test]
    fn test_mock_embed_single_empty_string() {
        let config = EmbeddingConfig::default();
        let mut generator = MockEmbeddingGenerator::new(config);

        let embedding = generator.embed_single("").unwrap();
        assert_eq!(embedding.len(), DEFAULT_EMBEDDING_DIMENSIONS);
        assert!(embedding.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_mock_embed_single_deterministic() {
        let config = EmbeddingConfig::default();
        let mut generator = MockEmbeddingGenerator::new(config);

        let text = "test text";
        let embedding1 = generator.embed_single(text).unwrap();
        let embedding2 = generator.embed_single(text).unwrap();

        assert_eq!(embedding1, embedding2); // Should be deterministic
        assert_eq!(embedding1.len(), DEFAULT_EMBEDDING_DIMENSIONS);
    }

    #[test]
    fn test_mock_embed_batch() {
        let config = EmbeddingConfig::default();
        let mut generator = MockEmbeddingGenerator::new(config);

        let texts = vec!["text1".to_string(), "text2".to_string()];
        let embeddings = generator.embed_batch(&texts).unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), DEFAULT_EMBEDDING_DIMENSIONS);
        assert_eq!(embeddings[1].len(), DEFAULT_EMBEDDING_DIMENSIONS);
        assert_ne!(embeddings[0], embeddings[1]); // Should be different for different texts
    }

    #[test]
    fn test_embedding_options_with_instruction() {
        let options = EmbeddingOptions::with_instruction("test instruction");
        assert_eq!(options.instruction.as_deref(), Some("test instruction"));
        assert!(options.normalize);
        assert_eq!(options.max_length, None);
    }

    #[test]
    fn test_embedding_options_without_normalization() {
        let options = EmbeddingOptions::without_normalization();
        assert_eq!(options.instruction, None);
        assert!(!options.normalize);
        assert_eq!(options.max_length, None);
    }

    #[test]
    fn test_embedding_options_with_max_length() {
        let options = EmbeddingOptions::with_max_length(100);
        assert_eq!(options.instruction, None);
        assert!(options.normalize);
        assert_eq!(options.max_length, Some(100));
    }
}
