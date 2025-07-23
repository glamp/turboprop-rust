//! Embedding generation module for converting text chunks to vector representations.
//!
//! This module provides functionality for generating embeddings from text chunks
//! using pre-trained transformer models via the fastembed library.

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel as FastEmbedModel, InitOptions, TextEmbedding};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

use crate::backends::{GGUFEmbeddingModel, HuggingFaceBackend, Qwen3EmbeddingModel};
use crate::constants;
use crate::error::TurboPropError;
use crate::models::{EmbeddingModel as EmbeddingModelTrait, ModelInfo, ModelManager};
use crate::types::ModelBackend;

/// Trait for processing embeddings in batches with common error handling and logging
trait BatchProcessor<T> {
    /// Process a batch of texts and return embeddings
    fn process_batch(&mut self, texts: &[T]) -> Result<Vec<Vec<f32>>>;

    /// Get the backend name for logging purposes
    fn backend_name(&self) -> &'static str;
}

/// Helper function to process texts in batches using any BatchProcessor
fn process_texts_in_batches<T, P: BatchProcessor<T>>(
    processor: &mut P,
    texts: &[T],
    batch_size: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut all_embeddings = Vec::with_capacity(texts.len());
    let backend_name = processor.backend_name();

    // Process in batches to avoid memory issues with large inputs
    for chunk in texts.chunks(batch_size) {
        debug!("Processing {} batch of {} texts", backend_name, chunk.len());

        let batch_embeddings = processor.process_batch(chunk).with_context(|| {
            format!(
                "Failed to generate {} embeddings for batch of {} texts",
                backend_name,
                chunk.len()
            )
        })?;

        all_embeddings.extend(batch_embeddings);
    }

    debug!(
        "Generated {} {} embeddings total",
        all_embeddings.len(),
        backend_name
    );

    Ok(all_embeddings)
}

/// BatchProcessor implementation for FastEmbed backend
impl BatchProcessor<String> for TextEmbedding {
    fn process_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.embed(text_refs, None)
    }

    fn backend_name(&self) -> &'static str {
        "FastEmbed"
    }
}

/// BatchProcessor implementation for GGUF backend
impl BatchProcessor<String> for GGUFEmbeddingModel {
    fn process_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed(texts)
    }

    fn backend_name(&self) -> &'static str {
        "GGUF"
    }
}

/// BatchProcessor implementation for HuggingFace backend
impl BatchProcessor<String> for Qwen3EmbeddingModel {
    fn process_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed(texts)
    }

    fn backend_name(&self) -> &'static str {
        "HuggingFace"
    }
}

/// Default embedding model to use if none specified
pub const DEFAULT_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// Expected embedding dimensions for the default model
pub const DEFAULT_EMBEDDING_DIMENSIONS: usize = 384;

// constants::text::ERROR_MESSAGE_TEXT_PREVIEW_LENGTH is now defined in constants::text

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
            batch_size_warning_threshold: constants::text::BATCH_SIZE_WARNING_THRESHOLD,
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
    /// let gguf_model = models.iter().find(|m| m.name.as_str().contains("gguf")).unwrap();
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
            model_name: model_info.name.to_string(),
            cache_dir: cache_dir.to_path_buf(),
            embedding_dimensions: model_info.dimensions,
            batch_size: 32,
            batch_size_warning_threshold: constants::text::BATCH_SIZE_WARNING_THRESHOLD,
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
                    .load_qwen3_model(
                        &model_info.name,
                        &config.cache_dir.clone().into(),
                    )
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
                        if text.len() > constants::text::ERROR_MESSAGE_TEXT_PREVIEW_LENGTH {
                            &text[..constants::text::ERROR_MESSAGE_TEXT_PREVIEW_LENGTH]
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
                        if text.len() > constants::text::ERROR_MESSAGE_TEXT_PREVIEW_LENGTH {
                            &text[..constants::text::ERROR_MESSAGE_TEXT_PREVIEW_LENGTH]
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
                        if text.len() > constants::text::ERROR_MESSAGE_TEXT_PREVIEW_LENGTH {
                            &text[..constants::text::ERROR_MESSAGE_TEXT_PREVIEW_LENGTH]
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
                process_texts_in_batches(model, texts, self.config.batch_size)
            }
            EmbeddingBackendType::GGUF(model) => {
                process_texts_in_batches(model, texts, self.config.batch_size)
            }
            EmbeddingBackendType::HuggingFace(model) => {
                process_texts_in_batches(model, texts, self.config.batch_size)
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
                        batch_time.as_secs_f64() * constants::memory::SECONDS_TO_MILLISECONDS,
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
                        batch_time.as_secs_f64() * constants::memory::SECONDS_TO_MILLISECONDS,
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
                        batch_time.as_secs_f64() * constants::memory::SECONDS_TO_MILLISECONDS,
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

    /// Analyze text characteristics to determine batch size adjustments
    fn analyze_text_characteristics(&self, texts: &[String]) -> usize {
        if texts.is_empty() {
            return self.config.batch_size;
        }

        // Calculate average text length
        let avg_length = texts.iter().map(|t| t.len()).sum::<usize>() / texts.len();

        // Adjust batch size based on text length characteristics
        let base_batch_size = self.config.batch_size;
        if avg_length > constants::text::LONG_TEXT_LENGTH_THRESHOLD {
            // Longer texts require smaller batches to avoid memory issues
            base_batch_size / 2
        } else if avg_length < constants::text::SHORT_TEXT_LENGTH_THRESHOLD {
            // Shorter texts can use larger batches for better throughput
            base_batch_size * 2
        } else {
            // Medium-length texts use the configured batch size
            base_batch_size
        }
    }

    /// Calculate memory-based scaling factor for batch size
    fn calculate_memory_scaling(&self) -> f64 {
        match sys_info::mem_info() {
            Ok(mem_info) => {
                let available_gb = mem_info.avail / constants::memory::KB_TO_GB_DIVISOR;
                if available_gb >= constants::memory::HIGH_MEMORY_THRESHOLD_GB {
                    constants::memory::HIGH_MEMORY_SCALING_FACTOR // Plenty of memory
                } else if available_gb >= constants::memory::MEDIUM_MEMORY_THRESHOLD_GB {
                    constants::memory::MEDIUM_MEMORY_SCALING_FACTOR // Moderate memory
                } else {
                    constants::memory::LOW_MEMORY_SCALING_FACTOR // Limited memory
                }
            }
            Err(_) => {
                // Conservative scaling when memory info is unavailable
                constants::memory::LOW_MEMORY_SCALING_FACTOR
            }
        }
    }

    /// Apply final constraints and bounds to the batch size
    fn apply_batch_constraints(&self, batch_size: usize, has_memory_info: bool) -> usize {
        if has_memory_info {
            // With memory information, use full batch size range
            batch_size.clamp(constants::batch::MIN_SIZE, constants::batch::MAX_SIZE)
        } else {
            // Without memory information, use conservative maximum
            batch_size.clamp(
                constants::batch::MIN_SIZE,
                constants::batch::CONSERVATIVE_MAX_SIZE,
            )
        }
    }

    /// Calculate optimal batch size based on text characteristics and system resources
    fn calculate_optimal_batch_size(&self, texts: &[String]) -> usize {
        // Step 1: Analyze text characteristics and get initial batch size
        let text_adjusted_size = self.analyze_text_characteristics(texts);

        // Step 2: Calculate memory-based scaling factor
        let memory_factor = self.calculate_memory_scaling();

        // Step 3: Apply memory scaling to the text-adjusted batch size
        let memory_adjusted_size = (text_adjusted_size as f64 * memory_factor) as usize;

        // Step 4: Apply final constraints and bounds
        let has_memory_info = sys_info::mem_info().is_ok();
        self.apply_batch_constraints(memory_adjusted_size, has_memory_info)
    }

    /// Prepare texts for embedding by cleaning and normalizing them
    pub fn preprocess_texts(&self, texts: &[String]) -> Vec<String> {
        texts
            .iter()
            .map(|text| {
                // Remove excessive whitespace and normalize using iterators
                // Avoid intermediate Vec allocation by manually joining with iterator
                let mut result = String::new();
                let mut first = true;
                for word in text.split_whitespace() {
                    if !first {
                        result.push(' ');
                    }
                    result.push_str(word);
                    first = false;
                }
                result
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

/// Mock embedding generator for unit tests that doesn't require model downloads
#[cfg(any(test, feature = "test-utils"))]
#[derive(Debug)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ModelInfo, ModelInfoConfig};
    use crate::types::{ModelBackend, ModelName, ModelType};

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
    fn test_mock_embed_single_non_empty() {
        let config = EmbeddingConfig::default();
        let mut generator = MockEmbeddingGenerator::new(config);

        let text = "test text";
        let embedding = generator.embed_single(text).unwrap();

        assert_eq!(embedding.len(), DEFAULT_EMBEDDING_DIMENSIONS);
        assert!(embedding.iter().any(|&x| x != 0.0));

        // Test deterministic behavior
        let embedding2 = generator.embed_single(text).unwrap();
        assert_eq!(embedding, embedding2);
    }

    #[test]
    fn test_mock_embed_batch() {
        let config = EmbeddingConfig::default();
        let mut generator = MockEmbeddingGenerator::new(config);

        let texts = vec![
            "first text".to_string(),
            "second text".to_string(),
            "third text".to_string(),
        ];

        let embeddings = generator.embed_batch(&texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        for embedding in &embeddings {
            assert_eq!(embedding.len(), DEFAULT_EMBEDDING_DIMENSIONS);
            assert!(embedding.iter().any(|&x| x != 0.0));
        }

        // Different texts should produce different embeddings
        assert_ne!(embeddings[0], embeddings[1]);
        assert_ne!(embeddings[1], embeddings[2]);
    }

    #[test]
    fn test_model_info_configuration() {
        // Test that ModelInfo can be created with different backends
        let fastembed_model = ModelInfo::new(ModelInfoConfig {
            name: ModelName::from("test-fastembed"),
            description: "Test FastEmbed model".to_string(),
            dimensions: 384,
            size_bytes: constants::test::DEFAULT_MODEL_SIZE_BYTES,
            model_type: ModelType::SentenceTransformer,
            backend: ModelBackend::FastEmbed,
            download_url: None,
            local_path: None,
        });

        assert_eq!(fastembed_model.backend, ModelBackend::FastEmbed);
        assert_eq!(fastembed_model.dimensions, 384);
        assert_eq!(fastembed_model.model_type, ModelType::SentenceTransformer);

        let gguf_model = ModelInfo::new(ModelInfoConfig {
            name: ModelName::from("test-gguf"),
            description: "Test GGUF model".to_string(),
            dimensions: 768,
            size_bytes: constants::test::DEFAULT_MODEL_SIZE_BYTES,
            model_type: ModelType::GGUF,
            backend: ModelBackend::Candle,
            download_url: Some("https://example.com/test.gguf".to_string()),
            local_path: None,
        });

        assert_eq!(gguf_model.backend, ModelBackend::Candle);
        assert_eq!(gguf_model.dimensions, 768);
        assert_eq!(gguf_model.model_type, ModelType::GGUF);
        assert!(gguf_model.download_url.is_some());
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

    #[test]
    fn test_embedding_config_validation() {
        // Test that embedding configurations can be created and validated
        let config = EmbeddingConfig::with_model("custom-model")
            .with_batch_size(64)
            .with_embedding_dimensions(512)
            .with_cache_dir("/tmp/test-cache");

        assert_eq!(config.model_name, "custom-model");
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.embedding_dimensions, 512);
        assert_eq!(config.cache_dir.to_string_lossy(), "/tmp/test-cache");
    }
}

