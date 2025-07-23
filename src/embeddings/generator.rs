//! Main embedding generator implementation.
//!
//! This module provides the EmbeddingGenerator struct that handles model loading,
//! initialization, and text embedding generation using different backends.

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel as FastEmbedModel, InitOptions, TextEmbedding};
use std::path::Path;
use tracing::{debug, info, warn};

use crate::backends::{GGUFEmbeddingModel, HuggingFaceBackend};
use crate::constants;
use crate::error::TurboPropError;
use crate::models::{EmbeddingModel, ModelInfo, ModelManager};
use crate::types::ModelBackend;

use super::backends::EmbeddingBackendType;
use super::batch::process_texts_in_batches;
use super::config::{EmbeddingConfig, EmbeddingOptions};

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
                    .load_qwen3_model(&model_info.name, &config.cache_dir.clone().into())
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
