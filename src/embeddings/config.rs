//! Configuration structures for embedding generation.
//!
//! This module provides configuration options for embedding generation,
//! including model selection, caching, and batch processing settings.

use std::path::PathBuf;

use crate::constants;
use crate::models::ModelManager;

/// Default embedding model to use if none specified
pub const DEFAULT_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// Expected embedding dimensions for the default model
pub const DEFAULT_EMBEDDING_DIMENSIONS: usize = 384;

/// Configuration for embedding generation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingConfig {
    /// The model identifier to use for embedding generation
    pub model_name: String,
    /// Directory to cache downloaded models
    pub cache_dir: PathBuf,
    /// Batch size for processing multiple texts at once (general default)
    pub batch_size: usize,
    /// Optimal batch size for FastEmbed models
    pub fastembed_batch_size: usize,
    /// Optimal batch size for GGUF models
    pub gguf_batch_size: usize,
    /// Optimal batch size for HuggingFace models
    pub huggingface_batch_size: usize,
    /// Size of the LRU cache for caching embedding results
    pub cache_size: usize,
    /// Interval for sampling system resources (in milliseconds)
    pub resource_sampling_interval_ms: u64,
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
            fastembed_batch_size: 32,
            gguf_batch_size: 8,
            huggingface_batch_size: 16,
            cache_size: 1000,
            resource_sampling_interval_ms: 1000, // Sample every second by default
            embedding_dimensions: DEFAULT_EMBEDDING_DIMENSIONS,
            batch_size_warning_threshold: constants::text::BATCH_SIZE_WARNING_THRESHOLD,
        }
    }
}

impl EmbeddingConfig {
    /// Create a new configuration with a custom model name
    pub fn with_model(model_name: impl Into<String>) -> Self {
        let model_name = model_name.into();

        // Look up model dimensions from available models
        let model_manager = ModelManager::default();
        let embedding_dimensions = model_manager
            .get_available_models()
            .iter()
            .find(|model| model.name.as_str() == model_name)
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

/// Options for embedding generation with advanced features
#[derive(Debug, Clone)]
pub struct EmbeddingOptions {
    /// Optional instruction to guide embedding generation (Qwen3 feature)
    pub instruction: Option<String>,
    /// Whether to normalize embeddings (enabled by default)
    pub normalize: bool,
    /// Maximum sequence length to truncate to
    pub max_length: Option<usize>,
}

impl Default for EmbeddingOptions {
    fn default() -> Self {
        Self {
            instruction: None,
            normalize: true,
            max_length: None,
        }
    }
}

impl EmbeddingOptions {
    /// Create options with an instruction
    pub fn with_instruction(instruction: impl Into<String>) -> Self {
        Self {
            instruction: Some(instruction.into()),
            ..Default::default()
        }
    }

    /// Create options without normalization
    pub fn without_normalization() -> Self {
        Self {
            normalize: false,
            ..Default::default()
        }
    }

    /// Create options with max length limit
    pub fn with_max_length(max_length: usize) -> Self {
        Self {
            max_length: Some(max_length),
            ..Default::default()
        }
    }
}
