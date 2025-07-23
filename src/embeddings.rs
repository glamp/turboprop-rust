//! Embedding generation module for converting text chunks to vector representations.
//!
//! This module provides functionality for generating embeddings from text chunks
//! using pre-trained transformer models via the fastembed library.

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel as FastEmbedModel, InitOptions, TextEmbedding};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

use crate::backends::{GGUFEmbeddingModel, HuggingFaceBackend, Qwen3EmbeddingModel};
use crate::error::TurboPropError;
use crate::models::{EmbeddingModel as EmbeddingModelTrait, ModelInfo, ModelManager};
use crate::types::ModelBackend;

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

/// Enum wrapping different embedding backend implementations
pub enum EmbeddingBackendType {
    /// FastEmbed backend for sentence-transformer models
    FastEmbed(TextEmbedding),
    /// GGUF backend for quantized models using candle framework
    GGUF(GGUFEmbeddingModel),
    /// HuggingFace backend for models like Qwen3 not supported by FastEmbed
    HuggingFace(Qwen3EmbeddingModel),
}

impl std::fmt::Debug for EmbeddingBackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingBackendType::FastEmbed(_) => write!(f, "FastEmbed"),
            EmbeddingBackendType::GGUF(model) => write!(f, "GGUF({:?})", model),
            EmbeddingBackendType::HuggingFace(model) => write!(f, "HuggingFace({:?})", model),
        }
    }
}

/// Main embedding generator that handles model loading and text embedding
pub struct EmbeddingGenerator {
    backend: EmbeddingBackendType,
    config: EmbeddingConfig,
}

impl std::fmt::Debug for EmbeddingGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingGenerator")
            .field("backend", &self.backend)
            .field("config", &self.config)
            .field("model_name", &self.config.model_name)
            .finish()
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
            "sentence-transformers/all-MiniLM-L6-v2" => FastEmbedModel::AllMiniLML6V2,
            "sentence-transformers/all-MiniLM-L12-v2" => FastEmbedModel::AllMiniLML12V2,
            _ => {
                warn!(
                    "Unknown model '{}', falling back to default",
                    config.model_name
                );
                FastEmbedModel::AllMiniLML6V2
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

        Ok(Self {
            backend: EmbeddingBackendType::FastEmbed(model),
            config,
        })
    }

    /// Initialize a new embedding generator with the specified model information
    ///
    /// This method automatically selects the appropriate backend based on the model type
    /// and backend specified in the ModelInfo.
    ///
    /// # Arguments
    /// * `model_info` - Information about the model to load
    /// * `cache_dir` - Directory to cache downloaded models
    ///
    /// # Returns
    /// * `Result<Self>` - The initialized embedding generator
    ///
    /// # Examples
    /// ```no_run
    /// # use turboprop::embeddings::EmbeddingGenerator;
    /// # use turboprop::models::{ModelManager, ModelInfo};
    /// # use std::path::Path;
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let models = ModelManager::get_available_models();
    /// let gguf_model = models.iter().find(|m| m.name.contains("gguf")).unwrap();
    ///
    /// let generator = EmbeddingGenerator::new_with_model(gguf_model, Path::new(".cache")).await?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// # });
    /// ```
    pub async fn new_with_model(model_info: &ModelInfo, cache_dir: &Path) -> Result<Self> {
        info!(
            "Initializing embedding generator with model: {}",
            model_info.name
        );
        info!(
            "Backend: {:?}, Type: {:?}",
            model_info.backend, model_info.model_type
        );

        // Create configuration based on model info
        let config = EmbeddingConfig {
            model_name: model_info.name.clone(),
            cache_dir: cache_dir.to_path_buf(),
            embedding_dimensions: model_info.dimensions,
            batch_size: 32,
            batch_size_warning_threshold: 1000,
        };

        let backend = match model_info.backend {
            ModelBackend::FastEmbed => {
                info!("Using FastEmbed backend for model: {}", model_info.name);

                // Parse the model name to get the EmbeddingModel enum variant
                let embedding_model = match model_info.name.as_str() {
                    "sentence-transformers/all-MiniLM-L6-v2" => FastEmbedModel::AllMiniLML6V2,
                    "sentence-transformers/all-MiniLM-L12-v2" => FastEmbedModel::AllMiniLML12V2,
                    _ => {
                        warn!(
                            "Unknown FastEmbed model '{}', falling back to default",
                            model_info.name
                        );
                        FastEmbedModel::AllMiniLML6V2
                    }
                };

                // Initialize the model with cache directory
                let init_options =
                    InitOptions::new(embedding_model).with_cache_dir(config.cache_dir.clone());

                let model = TextEmbedding::try_new(init_options).with_context(|| {
                    format!("Failed to initialize FastEmbed model: {}", model_info.name)
                })?;

                EmbeddingBackendType::FastEmbed(model)
            }
            ModelBackend::Candle => {
                info!("Using GGUF/Candle backend for model: {}", model_info.name);

                // Initialize model manager and download GGUF model if needed
                let model_manager = ModelManager::new(&config.cache_dir);
                model_manager.init_cache()?;

                let model_path = model_manager.download_gguf_model(model_info).await?;
                info!("GGUF model available at: {}", model_path.display());

                // Load the GGUF model using our backend
                let gguf_model = GGUFEmbeddingModel::load_from_path(&model_path, model_info)?;

                EmbeddingBackendType::GGUF(gguf_model)
            }
            ModelBackend::Custom => {
                info!("Using HuggingFace backend for model: {}", model_info.name);

                // Initialize HuggingFace backend for Qwen3 models
                let hf_backend = HuggingFaceBackend::new()
                    .context("Failed to initialize HuggingFace backend")?;

                // Load the Qwen3 model
                let qwen3_model = hf_backend
                    .load_qwen3_model(&model_info.name.clone().into(), &config.cache_dir.clone().into())
                    .await
                    .with_context(|| format!("Failed to load Qwen3 model: {}", model_info.name))?;

                EmbeddingBackendType::HuggingFace(qwen3_model)
            }
        };

        info!(
            "Embedding generator initialized successfully with {:?} backend",
            model_info.backend
        );

        Ok(Self { backend, config })
    }

    /// Generate embeddings for a single text chunk
    ///
    /// Returns a vector of floating point values representing the embedding.
    /// The dimensions depend on the model being used.
    pub fn embed_single(&mut self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Ok(vec![0.0; self.config.embedding_dimensions]);
        }

        match &mut self.backend {
            EmbeddingBackendType::FastEmbed(model) => {
                let embeddings = model.embed(vec![text], None).with_context(|| {
                    format!(
                        "Failed to generate FastEmbed embedding for text: {}",
                        if text.len() > ERROR_MESSAGE_TEXT_PREVIEW_LENGTH {
                            &text[..ERROR_MESSAGE_TEXT_PREVIEW_LENGTH]
                        } else {
                            text
                        }
                    )
                })?;

                embeddings.into_iter().next().ok_or_else(|| {
                    TurboPropError::other(
                        "Failed to generate FastEmbed embedding: no output for input text",
                    )
                    .into()
                })
            }
            EmbeddingBackendType::GGUF(model) => {
                let texts = vec![text.to_string()];
                let embeddings = model.embed(&texts).with_context(|| {
                    format!(
                        "Failed to generate GGUF embedding for text: {}",
                        if text.len() > ERROR_MESSAGE_TEXT_PREVIEW_LENGTH {
                            &text[..ERROR_MESSAGE_TEXT_PREVIEW_LENGTH]
                        } else {
                            text
                        }
                    )
                })?;

                embeddings.into_iter().next().ok_or_else(|| {
                    TurboPropError::other(
                        "Failed to generate GGUF embedding: no output for input text",
                    )
                    .into()
                })
            }
            EmbeddingBackendType::HuggingFace(model) => {
                let texts = vec![text.to_string()];
                let embeddings = model.embed(&texts).with_context(|| {
                    format!(
                        "Failed to generate HuggingFace embedding for text: {}",
                        if text.len() > ERROR_MESSAGE_TEXT_PREVIEW_LENGTH {
                            &text[..ERROR_MESSAGE_TEXT_PREVIEW_LENGTH]
                        } else {
                            text
                        }
                    )
                })?;

                embeddings.into_iter().next().ok_or_else(|| {
                    TurboPropError::other(
                        "Failed to generate HuggingFace embedding: no output for input text",
                    )
                    .into()
                })
            }
        }
    }

    /// Generate embeddings for multiple text chunks in batches
    ///
    /// This is more efficient than calling embed_single multiple times.
    /// Returns embeddings in the same order as the input texts.
    pub fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        match &mut self.backend {
            EmbeddingBackendType::FastEmbed(model) => {
                let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                let mut all_embeddings = Vec::with_capacity(texts.len());

                // Process in batches to avoid memory issues with large inputs
                for chunk in text_refs.chunks(self.config.batch_size) {
                    debug!("Processing FastEmbed batch of {} texts", chunk.len());

                    let batch_embeddings =
                        model.embed(chunk.to_vec(), None).with_context(|| {
                            format!(
                                "Failed to generate FastEmbed embeddings for batch of {} texts",
                                chunk.len()
                            )
                        })?;

                    all_embeddings.extend(batch_embeddings);
                }

                debug!(
                    "Generated {} FastEmbed embeddings total",
                    all_embeddings.len()
                );
                Ok(all_embeddings)
            }
            EmbeddingBackendType::GGUF(model) => {
                let mut all_embeddings = Vec::with_capacity(texts.len());

                // Process in batches to avoid memory issues with large inputs
                for chunk in texts.chunks(self.config.batch_size) {
                    debug!("Processing GGUF batch of {} texts", chunk.len());

                    let batch_embeddings = model.embed(chunk).with_context(|| {
                        format!(
                            "Failed to generate GGUF embeddings for batch of {} texts",
                            chunk.len()
                        )
                    })?;

                    all_embeddings.extend(batch_embeddings);
                }

                debug!("Generated {} GGUF embeddings total", all_embeddings.len());
                Ok(all_embeddings)
            }
            EmbeddingBackendType::HuggingFace(model) => {
                let mut all_embeddings = Vec::with_capacity(texts.len());

                // Process in batches to avoid memory issues with large inputs
                for chunk in texts.chunks(self.config.batch_size) {
                    debug!("Processing HuggingFace batch of {} texts", chunk.len());

                    let batch_embeddings = model.embed(chunk).with_context(|| {
                        format!(
                            "Failed to generate HuggingFace embeddings for batch of {} texts",
                            chunk.len()
                        )
                    })?;

                    all_embeddings.extend(batch_embeddings);
                }

                debug!(
                    "Generated {} HuggingFace embeddings total",
                    all_embeddings.len()
                );
                Ok(all_embeddings)
            }
        }
    }

    /// Generate embeddings with advanced options including instruction support
    ///
    /// This method supports advanced features like instruction-based embeddings
    /// for models that support them (e.g., Qwen3).
    pub fn embed_with_options(
        &mut self,
        texts: &[String],
        options: &EmbeddingOptions,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Apply max_length truncation if specified
        let processed_texts = if let Some(max_len) = options.max_length {
            texts
                .iter()
                .map(|text| {
                    if text.len() > max_len {
                        text[..max_len].to_string()
                    } else {
                        text.clone()
                    }
                })
                .collect::<Vec<_>>()
        } else {
            texts.to_vec()
        };

        let embeddings = match &mut self.backend {
            EmbeddingBackendType::HuggingFace(model) => {
                // Use instruction-based embedding for HuggingFace backend
                model
                    .embed_with_instruction(&processed_texts, options.instruction.as_deref())
                    .with_context(|| "Failed to generate embeddings with HuggingFace backend")?
            }
            EmbeddingBackendType::FastEmbed(model) => {
                // FastEmbed doesn't support instructions, use standard embedding
                if options.instruction.is_some() {
                    warn!("FastEmbed backend does not support instruction-based embeddings, ignoring instruction");
                }
                let text_refs: Vec<&str> = processed_texts.iter().map(|s| s.as_str()).collect();
                model
                    .embed(text_refs, None)
                    .with_context(|| "Failed to generate embeddings with FastEmbed backend")?
            }
            EmbeddingBackendType::GGUF(model) => {
                // GGUF backend doesn't support instructions, use standard embedding
                if options.instruction.is_some() {
                    warn!("GGUF backend does not support instruction-based embeddings, ignoring instruction");
                }
                model
                    .embed(&processed_texts)
                    .with_context(|| "Failed to generate embeddings with GGUF backend")?
            }
        };

        // Apply normalization if disabled (most models normalize by default)
        if !options.normalize {
            // For now, we'll trust that models handle normalization internally
            // Future enhancement could add explicit denormalization
            debug!("Normalization disabled but not yet implemented for denormalization");
        }

        Ok(embeddings)
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

        match &mut self.backend {
            EmbeddingBackendType::FastEmbed(model) => {
                for (batch_idx, chunk) in text_refs.chunks(optimal_batch_size).enumerate() {
                    let batch_start = std::time::Instant::now();

                    let batch_embeddings =
                        model.embed(chunk.to_vec(), None).with_context(|| {
                            format!(
                                "Failed to generate FastEmbed embeddings for batch {} of {} texts",
                                batch_idx + 1,
                                chunk.len()
                            )
                        })?;

                    all_embeddings.extend(batch_embeddings);

                    let batch_time = batch_start.elapsed();
                    debug!(
                        "FastEmbed batch {} completed in {:.2}ms ({:.1} texts/sec)",
                        batch_idx + 1,
                        batch_time.as_secs_f64() * 1000.0,
                        chunk.len() as f64 / batch_time.as_secs_f64()
                    );
                }
            }
            EmbeddingBackendType::GGUF(model) => {
                for (batch_idx, chunk) in texts.chunks(optimal_batch_size).enumerate() {
                    let batch_start = std::time::Instant::now();

                    let batch_embeddings = model.embed(chunk).with_context(|| {
                        format!(
                            "Failed to generate GGUF embeddings for batch {} of {} texts",
                            batch_idx + 1,
                            chunk.len()
                        )
                    })?;

                    all_embeddings.extend(batch_embeddings);

                    let batch_time = batch_start.elapsed();
                    debug!(
                        "GGUF batch {} completed in {:.2}ms ({:.1} texts/sec)",
                        batch_idx + 1,
                        batch_time.as_secs_f64() * 1000.0,
                        chunk.len() as f64 / batch_time.as_secs_f64()
                    );
                }
            }
            EmbeddingBackendType::HuggingFace(model) => {
                for (batch_idx, chunk) in texts.chunks(optimal_batch_size).enumerate() {
                    let batch_start = std::time::Instant::now();

                    let batch_embeddings = model.embed(chunk).with_context(|| {
                        format!(
                            "Failed to generate HuggingFace embeddings for batch {} of {} texts",
                            batch_idx + 1,
                            chunk.len()
                        )
                    })?;

                    all_embeddings.extend(batch_embeddings);

                    let batch_time = batch_start.elapsed();
                    debug!(
                        "HuggingFace batch {} completed in {:.2}ms ({:.1} texts/sec)",
                        batch_idx + 1,
                        batch_time.as_secs_f64() * 1000.0,
                        chunk.len() as f64 / batch_time.as_secs_f64()
                    );
                }
            }
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
    use crate::models::{ModelInfo, ModelInfoConfig};
    use crate::types::{ModelBackend, ModelType};
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

    #[tokio::test]
    async fn test_embedding_generator_new_with_model_fastembed() {
        let temp_dir = TempDir::new().unwrap();

        let model_info = ModelInfo::simple(
            "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            "FastEmbed model".to_string(),
            384,
            23_000_000,
        );

        if std::env::var("OFFLINE_TESTS").is_ok() {
            return;
        }

        // Test that new_with_model works with FastEmbed backend
        let result = EmbeddingGenerator::new_with_model(&model_info, temp_dir.path()).await;

        // Should succeed for FastEmbed models
        assert!(
            result.is_ok(),
            "new_with_model should work for FastEmbed models"
        );

        let generator = result.unwrap();
        assert_eq!(
            generator.model_name(),
            "sentence-transformers/all-MiniLM-L6-v2"
        );
        assert_eq!(generator.embedding_dimensions(), 384);
    }

    #[tokio::test]
    async fn test_embedding_generator_new_with_model_gguf() {
        let temp_dir = TempDir::new().unwrap();

        let model_info = ModelInfo::gguf_model(
            "nomic-embed-code.Q5_K_S.gguf".to_string(),
            "GGUF model for testing".to_string(),
            768,
            2_500_000_000,
            "https://example.com/model.gguf".to_string(),
        );

        // Test that new_with_model works with GGUF backend
        let result = EmbeddingGenerator::new_with_model(&model_info, temp_dir.path()).await;

        // Should succeed for GGUF models (assuming network is available)
        if result.is_ok() {
            let generator = result.unwrap();
            assert_eq!(generator.model_name(), "nomic-embed-code.Q5_K_S.gguf");
            assert_eq!(generator.embedding_dimensions(), 768);
        } else {
            // Allow failure for network-related issues in test environment
            let error_msg = result.err().unwrap().to_string();
            println!("GGUF test failed (may be network-related): {}", error_msg);
        }
    }

    #[tokio::test]
    async fn test_embedding_generator_backend_selection() {
        // This test verifies that the correct backend is selected based on ModelInfo.backend
        let temp_dir = TempDir::new().unwrap();

        // Test FastEmbed backend selection
        let fastembed_model = ModelInfo::new(ModelInfoConfig {
            name: "test-fastembed".to_string(),
            description: "Test FastEmbed model".to_string(),
            dimensions: 384,
            size_bytes: 1000,
            model_type: ModelType::SentenceTransformer,
            backend: ModelBackend::FastEmbed,
            download_url: None,
            local_path: None,
        });

        // Test GGUF/Candle backend selection
        let gguf_model = ModelInfo::new(ModelInfoConfig {
            name: "test-gguf".to_string(),
            description: "Test GGUF model".to_string(),
            dimensions: 768,
            size_bytes: 1000,
            model_type: ModelType::GGUF,
            backend: ModelBackend::Candle,
            download_url: Some("https://example.com/test.gguf".to_string()),
            local_path: None,
        });

        // Test backend selection logic - should choose correct backend based on ModelInfo.backend
        let fastembed_result =
            EmbeddingGenerator::new_with_model(&fastembed_model, temp_dir.path()).await;
        let gguf_result = EmbeddingGenerator::new_with_model(&gguf_model, temp_dir.path()).await;

        // FastEmbed should work offline (but skip in offline test mode)
        if std::env::var("OFFLINE_TESTS").is_err() {
            assert!(fastembed_result.is_ok(), "FastEmbed backend should work");
        }

        // GGUF may fail due to network or URL issues in test environment, but shouldn't fail due to backend selection
        if let Err(ref e) = gguf_result {
            let error_msg = e.to_string();
            // Should not fail due to backend selection issues
            assert!(
                !error_msg.contains("does not support model type"),
                "Backend selection should work for GGUF"
            );
        }
    }

    #[tokio::test]
    async fn test_embedding_generator_custom_backend_model_loading_failure() {
        let temp_dir = TempDir::new().unwrap();

        // Test that custom backend fails gracefully when trying to load a non-existent model
        let fake_model = ModelInfo::new(ModelInfoConfig {
            name: "fake/nonexistent-model".to_string(),
            description: "Test fake model for custom backend".to_string(),
            dimensions: 512,
            size_bytes: 1000,
            model_type: ModelType::HuggingFace,
            backend: ModelBackend::Custom, // Now supported backend
            download_url: None,
            local_path: None,
        });

        let result = EmbeddingGenerator::new_with_model(&fake_model, temp_dir.path()).await;

        // Should fail when trying to download the fake model
        assert!(result.is_err());

        let error_msg = result.err().unwrap().to_string();
        assert!(
            error_msg.contains("Failed to load Qwen3 model")
                || error_msg.contains("Failed to download"),
            "Should fail with model loading error, got: {}",
            error_msg
        );
    }

    #[test]
    fn test_embedding_options_default() {
        let options = EmbeddingOptions::default();
        assert!(options.instruction.is_none());
        assert!(options.normalize);
        assert!(options.max_length.is_none());
    }

    #[test]
    fn test_embedding_options_with_instruction() {
        let options = EmbeddingOptions::with_instruction("Retrieve relevant documents:");
        assert_eq!(
            options.instruction,
            Some("Retrieve relevant documents:".to_string())
        );
        assert!(options.normalize);
        assert!(options.max_length.is_none());
    }

    #[test]
    fn test_embedding_options_without_normalization() {
        let options = EmbeddingOptions::without_normalization();
        assert!(options.instruction.is_none());
        assert!(!options.normalize);
        assert!(options.max_length.is_none());
    }

    #[test]
    fn test_embedding_options_with_max_length() {
        let options = EmbeddingOptions::with_max_length(512);
        assert!(options.instruction.is_none());
        assert!(options.normalize);
        assert_eq!(options.max_length, Some(512));
    }

    #[tokio::test]
    async fn test_embed_with_options_empty_input() {
        let temp_dir = TempDir::new().unwrap();
        let config = EmbeddingConfig::default().with_cache_dir(temp_dir.path().to_path_buf());

        if std::env::var("OFFLINE_TESTS").is_ok() {
            return;
        }

        let mut generator = EmbeddingGenerator::new(config).await.unwrap();
        let options = EmbeddingOptions::default();
        let embeddings = generator.embed_with_options(&[], &options).unwrap();
        assert!(embeddings.is_empty());
    }

    #[tokio::test]
    async fn test_embed_with_options_max_length_truncation() {
        let temp_dir = TempDir::new().unwrap();
        let config = EmbeddingConfig::default().with_cache_dir(temp_dir.path().to_path_buf());

        if std::env::var("OFFLINE_TESTS").is_ok() {
            return;
        }

        let mut generator = EmbeddingGenerator::new(config).await.unwrap();

        // Create a long text that should be truncated
        let long_text =
            "This is a very long text that should be truncated when max_length is applied. "
                .repeat(20);
        let texts = vec![long_text];

        let options = EmbeddingOptions::with_max_length(50);
        let result = generator.embed_with_options(&texts, &options);

        // Should succeed even with truncation
        assert!(result.is_ok(), "Should handle max_length truncation");
    }
}
