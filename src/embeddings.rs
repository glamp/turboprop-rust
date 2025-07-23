//! Embedding generation module for converting text chunks to vector representations.
//!
//! This module provides functionality for generating embeddings from text chunks
//! using pre-trained transformer models via the fastembed library.

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::path::PathBuf;
use tracing::{debug, info, warn};

use crate::error::TurboPropError;
use crate::models::ModelManager;

/// Default embedding model to use if none specified
pub const DEFAULT_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// Expected embedding dimensions for the default model
pub const DEFAULT_EMBEDDING_DIMENSIONS: usize = 384;

/// Maximum length for text in error messages before truncation
const ERROR_MESSAGE_TEXT_PREVIEW_LENGTH: usize = 50;

/// Configuration for embedding generation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingConfig {
    /// The model identifier to use for embedding generation
    pub model_name: String,
    /// Directory to cache downloaded models
    pub cache_dir: PathBuf,
    /// Batch size for processing multiple texts at once
    pub batch_size: usize,
    /// Expected embedding dimensions for the model
    pub embedding_dimensions: usize,
    /// Threshold for warning about large batch sizes that might cause memory issues
    pub batch_size_warning_threshold: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_name: DEFAULT_MODEL.to_string(),
            cache_dir: PathBuf::from(".turboprop/models"),
            batch_size: 32,
            embedding_dimensions: DEFAULT_EMBEDDING_DIMENSIONS,
            batch_size_warning_threshold: 1000,
        }
    }
}

impl EmbeddingConfig {
    /// Create a new configuration with a custom model name
    pub fn with_model(model_name: impl Into<String>) -> Self {
        let model_name = model_name.into();

        // Look up model dimensions from available models
        let embedding_dimensions = ModelManager::get_available_models()
            .iter()
            .find(|model| model.name == model_name)
            .map(|model| model.dimensions)
            .unwrap_or(DEFAULT_EMBEDDING_DIMENSIONS);

        Self {
            model_name,
            embedding_dimensions,
            ..Default::default()
        }
    }

    /// Set the cache directory for model storage
    pub fn with_cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = cache_dir.into();
        self
    }

    /// Set the batch size for processing
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the embedding dimensions for the model
    pub fn with_embedding_dimensions(mut self, dimensions: usize) -> Self {
        self.embedding_dimensions = dimensions;
        self
    }

    /// Set the batch size warning threshold
    pub fn with_batch_size_warning_threshold(mut self, threshold: usize) -> Self {
        self.batch_size_warning_threshold = threshold;
        self
    }
}

/// Main embedding generator that handles model loading and text embedding
pub struct EmbeddingGenerator {
    model: TextEmbedding,
    config: EmbeddingConfig,
}

impl std::fmt::Debug for EmbeddingGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingGenerator")
            .field("config", &self.config)
            .field("model_name", &self.config.model_name)
            .finish()
    }
}

impl EmbeddingGenerator {
    /// Initialize a new embedding generator with the specified configuration
    ///
    /// This will download the model if it's not already cached locally.
    /// Progress indicators will be shown during model download.
    pub async fn new(config: EmbeddingConfig) -> Result<Self> {
        info!(
            "Initializing embedding generator with model: {}",
            config.model_name
        );

        // Ensure cache directory exists
        if !config.cache_dir.exists() {
            std::fs::create_dir_all(&config.cache_dir).with_context(|| {
                format!(
                    "Failed to create cache directory: {}",
                    config.cache_dir.display()
                )
            })?;
            debug!("Created cache directory: {}", config.cache_dir.display());
        }

        // Parse the model name to get the EmbeddingModel enum variant
        let embedding_model = match config.model_name.as_str() {
            "sentence-transformers/all-MiniLM-L6-v2" => EmbeddingModel::AllMiniLML6V2,
            "sentence-transformers/all-MiniLM-L12-v2" => EmbeddingModel::AllMiniLML12V2,
            _ => {
                warn!(
                    "Unknown model '{}', falling back to default",
                    config.model_name
                );
                EmbeddingModel::AllMiniLML6V2
            }
        };

        // Initialize the model with cache directory
        let init_options =
            InitOptions::new(embedding_model).with_cache_dir(config.cache_dir.clone());

        info!("Loading embedding model (this may download the model on first use)...");
        let model = TextEmbedding::try_new(init_options).with_context(|| {
            format!(
                "Failed to initialize embedding model: {}",
                config.model_name
            )
        })?;

        info!("Embedding model loaded successfully");

        Ok(Self { model, config })
    }

    /// Generate embeddings for a single text chunk
    ///
    /// Returns a vector of floating point values representing the embedding.
    /// For the default model, this will be 384 dimensions.
    pub fn embed_single(&mut self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Ok(vec![0.0; self.config.embedding_dimensions]);
        }

        let embeddings = self.model.embed(vec![text], None).with_context(|| {
            format!(
                "Failed to generate embedding for text: {}",
                if text.len() > ERROR_MESSAGE_TEXT_PREVIEW_LENGTH {
                    &text[..ERROR_MESSAGE_TEXT_PREVIEW_LENGTH]
                } else {
                    text
                }
            )
        })?;

        embeddings.into_iter().next().ok_or_else(|| {
            TurboPropError::other("Failed to generate embedding: no output for input text").into()
        })
    }

    /// Generate embeddings for multiple text chunks in batches
    ///
    /// This is more efficient than calling embed_single multiple times.
    /// Returns embeddings in the same order as the input texts.
    pub fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let mut all_embeddings = Vec::with_capacity(texts.len());

        // Process in batches to avoid memory issues with large inputs
        for chunk in text_refs.chunks(self.config.batch_size) {
            debug!("Processing batch of {} texts", chunk.len());

            let batch_embeddings = self.model.embed(chunk.to_vec(), None).with_context(|| {
                format!(
                    "Failed to generate embeddings for batch of {} texts",
                    chunk.len()
                )
            })?;

            all_embeddings.extend(batch_embeddings);
        }

        debug!("Generated {} embeddings total", all_embeddings.len());
        Ok(all_embeddings)
    }

    /// Generate embeddings for multiple text chunks with optimized batching
    ///
    /// This method uses larger batch sizes for better throughput on large datasets.
    /// Automatically adjusts batch size based on available memory and text length.
    pub fn embed_batch_optimized(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Check if batch size should be adjusted for performance
        let optimal_batch_size = self.calculate_optimal_batch_size(texts);
        debug!(
            "Using optimal batch size: {} for {} texts",
            optimal_batch_size,
            texts.len()
        );

        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let mut all_embeddings = Vec::with_capacity(texts.len());

        // Process with optimal batching
        let start_time = std::time::Instant::now();
        for (batch_idx, chunk) in text_refs.chunks(optimal_batch_size).enumerate() {
            let batch_start = std::time::Instant::now();

            let batch_embeddings = self.model.embed(chunk.to_vec(), None).with_context(|| {
                format!(
                    "Failed to generate embeddings for batch {} of {} texts",
                    batch_idx + 1,
                    chunk.len()
                )
            })?;

            all_embeddings.extend(batch_embeddings);

            let batch_time = batch_start.elapsed();
            debug!(
                "Batch {} completed in {:.2}ms ({:.1} texts/sec)",
                batch_idx + 1,
                batch_time.as_secs_f64() * 1000.0,
                chunk.len() as f64 / batch_time.as_secs_f64()
            );
        }

        let total_time = start_time.elapsed();
        info!(
            "Generated {} embeddings in {:.2}s ({:.1} texts/sec)",
            all_embeddings.len(),
            total_time.as_secs_f64(),
            texts.len() as f64 / total_time.as_secs_f64()
        );

        Ok(all_embeddings)
    }

    /// Calculate optimal batch size based on text characteristics and system resources
    fn calculate_optimal_batch_size(&self, texts: &[String]) -> usize {
        if texts.is_empty() {
            return self.config.batch_size;
        }

        // Calculate average text length
        let avg_length = texts.iter().map(|t| t.len()).sum::<usize>() / texts.len();

        // Adjust batch size based on text length and available memory
        let base_batch_size = self.config.batch_size;
        let adjusted_size = if avg_length > 1000 {
            // Longer texts require smaller batches
            base_batch_size / 2
        } else if avg_length < 100 {
            // Shorter texts can use larger batches
            base_batch_size * 2
        } else {
            base_batch_size
        };

        // Apply memory-based scaling
        if let Ok(mem_info) = sys_info::mem_info() {
            let available_gb = mem_info.avail / 1024 / 1024; // Convert KB to GB
            let memory_factor = if available_gb >= 8 {
                2.0 // Plenty of memory
            } else if available_gb >= 4 {
                1.5 // Moderate memory
            } else {
                0.75 // Limited memory
            };

            let memory_adjusted = (adjusted_size as f64 * memory_factor) as usize;
            memory_adjusted.clamp(1, 256) // Clamp between 1 and 256
        } else {
            adjusted_size.clamp(1, 128) // Conservative default
        }
    }

    /// Prepare texts for embedding by cleaning and normalizing them
    pub fn preprocess_texts(&self, texts: &[String]) -> Vec<String> {
        texts
            .iter()
            .map(|text| {
                // Remove excessive whitespace and normalize
                text.split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .trim()
                    .to_string()
            })
            .collect()
    }

    /// Get the embedding dimensions for this model
    pub fn embedding_dimensions(&self) -> usize {
        self.config.embedding_dimensions
    }

    /// Get the model name being used
    pub fn model_name(&self) -> &str {
        &self.config.model_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

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

    #[tokio::test]
    async fn test_embedding_generator_initialization() {
        let temp_dir = TempDir::new().unwrap();
        let config = EmbeddingConfig::default().with_cache_dir(temp_dir.path().to_path_buf());

        // This test requires network access to download the model
        // Skip if we're in an offline environment
        if std::env::var("OFFLINE_TESTS").is_ok() {
            return;
        }

        let generator = EmbeddingGenerator::new(config).await;
        assert!(
            generator.is_ok(),
            "Failed to initialize generator: {:?}",
            generator.unwrap_err()
        );

        let generator = generator.unwrap();
        assert_eq!(generator.model_name(), DEFAULT_MODEL);
        assert_eq!(
            generator.embedding_dimensions(),
            DEFAULT_EMBEDDING_DIMENSIONS
        );
    }

    #[tokio::test]
    async fn test_embed_single_empty_string() {
        let temp_dir = TempDir::new().unwrap();
        let config = EmbeddingConfig::default().with_cache_dir(temp_dir.path().to_path_buf());

        if std::env::var("OFFLINE_TESTS").is_ok() {
            return;
        }

        let mut generator = EmbeddingGenerator::new(config).await.unwrap();
        let embedding = generator.embed_single("").unwrap();
        assert_eq!(embedding.len(), DEFAULT_EMBEDDING_DIMENSIONS);
        assert!(embedding.iter().all(|&x| x == 0.0));
    }
}
