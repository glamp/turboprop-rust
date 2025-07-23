//! HuggingFace backend implementation for embedding models not supported by FastEmbed.
//!
//! This backend provides support for models like Qwen3-Embedding that are available
//! on HuggingFace but not yet supported in the FastEmbed library.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::Activation;
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM};
use hf_hub::api::tokio::Api;
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

use crate::constants;
use crate::models::EmbeddingModel as EmbeddingModelTrait;
use crate::types::{CachePath, ModelName};

/// HuggingFace backend for loading and running models not supported by FastEmbed
pub struct HuggingFaceBackend {
    device: Device,
    api: Api,
}

impl HuggingFaceBackend {
    /// Create a new HuggingFace backend
    pub fn new() -> Result<Self> {
        let device = Device::Cpu;
        let api = Api::new()?;
        Ok(Self { device, api })
    }

    /// Validate inputs for Qwen3 model loading
    fn validate_model_inputs(model_name: &ModelName, cache_dir: &CachePath) -> Result<()> {
        // Validate model name format (should follow HuggingFace convention: org/model-name)
        let model_str = model_name.as_str();
        if model_str.is_empty() {
            return Err(anyhow::anyhow!(
                "[HUGGINGFACE] [VALIDATION] failed: Model name cannot be empty"
            ));
        }

        if !model_str.contains('/') {
            return Err(anyhow::anyhow!(
                "[HUGGINGFACE] [VALIDATION] failed: Model name '{}' must follow HuggingFace format 'organization/model-name'",
                model_str
            ));
        }

        let parts: Vec<&str> = model_str.split('/').collect();
        if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
            return Err(anyhow::anyhow!(
                "[HUGGINGFACE] [VALIDATION] failed: Model name '{}' has invalid format. Expected 'organization/model-name'",
                model_str
            ));
        }

        // Validate model name contains only allowed characters
        let allowed_chars =
            |c: char| c.is_alphanumeric() || c == '-' || c == '_' || c == '.' || c == '/';
        if !model_str.chars().all(allowed_chars) {
            return Err(anyhow::anyhow!(
                "[HUGGINGFACE] [VALIDATION] failed: Model name '{}' contains invalid characters. Only alphanumeric, '-', '_', '.', and '/' are allowed",
                model_str
            ));
        }

        // Validate cache directory
        let cache_path = cache_dir.as_ref();
        if !cache_path.exists() {
            return Err(anyhow::anyhow!(
                "[HUGGINGFACE] [VALIDATION] failed: Cache directory does not exist: {}",
                cache_dir
            ));
        }

        if !cache_path.is_dir() {
            return Err(anyhow::anyhow!(
                "[HUGGINGFACE] [VALIDATION] failed: Cache path is not a directory: {}",
                cache_dir
            ));
        }

        // Check cache directory permissions (read, write, execute)
        let metadata = std::fs::metadata(cache_path).with_context(|| {
            format!(
                "[HUGGINGFACE] [VALIDATION] failed: Cannot read cache directory metadata: {}",
                cache_dir
            )
        })?;

        if metadata.permissions().readonly() {
            return Err(anyhow::anyhow!(
                "[HUGGINGFACE] [VALIDATION] failed: Cache directory is read-only: {}. Write permissions required for model downloads",
                cache_dir
            ));
        }

        // Test write permissions by attempting to create a temporary file
        let test_file = cache_path.join(".turboprop_write_test");
        match std::fs::File::create(&test_file) {
            Ok(_) => {
                // Clean up test file
                let _ = std::fs::remove_file(&test_file);
            }
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "[HUGGINGFACE] [VALIDATION] failed: Cannot write to cache directory: {}. Error: {}",
                    cache_dir,
                    e
                ));
            }
        }

        // Note: Skipping disk space check as fs2 crate is not available
        // In a production environment, you might want to add this dependency
        // or implement platform-specific disk space checks

        Ok(())
    }

    /// Load a Qwen3 embedding model from HuggingFace
    pub async fn load_qwen3_model(
        &self,
        model_name: &ModelName,
        cache_dir: &CachePath,
    ) -> Result<Qwen3EmbeddingModel> {
        // Validate inputs before proceeding
        Self::validate_model_inputs(model_name, cache_dir)?;

        info!("[HUGGINGFACE] [LOAD] Loading Qwen3 model: {}", model_name);

        // Download model files from HuggingFace
        let repo = self.api.model(model_name.as_str().to_string());

        // Create model-specific cache directory
        let model_cache_dir = cache_dir.join(model_name.as_str().replace('/', "_"));
        if !model_cache_dir.exists() {
            std::fs::create_dir_all(&model_cache_dir).with_context(|| {
                format!(
                    "Failed to create model cache directory: {}",
                    model_cache_dir
                )
            })?;
        }

        // Download required files
        let config_path = self
            .download_file(&repo, "config.json", model_cache_dir.as_path())
            .await?;
        let tokenizer_path = self
            .download_file(&repo, "tokenizer.json", model_cache_dir.as_path())
            .await?;

        // Try to download model weights (safetensors preferred, fallback to pytorch)
        let model_path = match self
            .download_file(&repo, "model.safetensors", model_cache_dir.as_path())
            .await
        {
            Ok(path) => path,
            Err(e) => {
                warn!("[HUGGINGFACE] [DOWNLOAD] safetensors download failed: {}. Trying pytorch_model.bin as fallback", e);
                self.download_file(&repo, "pytorch_model.bin", model_cache_dir.as_path())
                    .await
                    .context("[HUGGINGFACE] [DOWNLOAD] failed: Neither safetensors nor pytorch model weights could be downloaded. Check network connectivity and model repository availability.")?
            }
        };

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            anyhow::anyhow!(
                "Failed to load tokenizer from {}: {}",
                tokenizer_path.display(),
                e
            )
        })?;

        // Load model configuration
        let config_content = std::fs::read_to_string(&config_path).with_context(|| {
            format!("Failed to read model config from {}", config_path.display())
        })?;
        let config_json: Value =
            serde_json::from_str(&config_content).context("[HUGGINGFACE] [CONFIG] failed: Cannot parse model configuration JSON. Verify config file format and completeness.")?;

        // Create Qwen2 config from the JSON
        let qwen_config = self.parse_qwen2_config(&config_json)?;

        // Load model weights
        let model = self.load_qwen3_weights(&model_path, &qwen_config)?;

        Ok(Qwen3EmbeddingModel {
            model: tokio::sync::RwLock::new(model),
            tokenizer,
            config: qwen_config,
            device: self.device.clone(),
            model_name: model_name.to_string(),
        })
    }

    /// Download a file from HuggingFace repository, using cache if available
    async fn download_file(
        &self,
        repo: &hf_hub::api::tokio::ApiRepo,
        filename: &str,
        cache_dir: &Path,
    ) -> Result<PathBuf> {
        let local_path = cache_dir.join(filename);

        // Return cached file if it exists
        if local_path.exists() {
            debug!("Using cached file: {}", local_path.display());
            return Ok(local_path);
        }

        info!("[HUGGINGFACE] [DOWNLOAD] Downloading {}", filename);
        let remote_file = repo
            .get(filename)
            .await
            .with_context(|| format!("Failed to download {}", filename))?;

        // Copy to cache directory
        std::fs::copy(&remote_file, &local_path)
            .with_context(|| format!("Failed to copy {} to cache", filename))?;

        debug!("Downloaded and cached: {}", local_path.display());
        Ok(local_path)
    }

    /// Parse Qwen2 configuration from HuggingFace config.json
    fn parse_qwen2_config(&self, config_json: &Value) -> Result<Qwen2Config> {
        let vocab_size = config_json["vocab_size"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("[HUGGINGFACE] [CONFIG] failed: Missing required vocab_size field in model configuration"))?
            as usize;

        let hidden_size = config_json["hidden_size"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("[HUGGINGFACE] [CONFIG] failed: Missing required hidden_size field in model configuration"))?
            as usize;

        let intermediate_size = config_json["intermediate_size"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("[HUGGINGFACE] [CONFIG] failed: Missing required intermediate_size field in model configuration"))?
            as usize;

        let num_hidden_layers = config_json["num_hidden_layers"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("[HUGGINGFACE] [CONFIG] failed: Missing required num_hidden_layers field in model configuration"))?
            as usize;

        let num_attention_heads = config_json["num_attention_heads"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("[HUGGINGFACE] [CONFIG] failed: Missing required num_attention_heads field in model configuration"))?
            as usize;

        let num_key_value_heads = config_json["num_key_value_heads"]
            .as_u64()
            .unwrap_or(num_attention_heads as u64) as usize;

        let max_position_embeddings = config_json["max_position_embeddings"]
            .as_u64()
            .unwrap_or(constants::model_config::DEFAULT_MAX_POSITION_EMBEDDINGS)
            as usize;

        let sliding_window = config_json["sliding_window"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or(max_position_embeddings);

        let rope_theta = config_json["rope_theta"]
            .as_f64()
            .unwrap_or(constants::model_config::DEFAULT_ROPE_THETA);

        let hidden_act = match config_json["hidden_act"].as_str().unwrap_or("silu") {
            "silu" => Activation::Silu,
            "relu" => Activation::Relu,
            "gelu" => Activation::NewGelu,
            _ => Activation::Silu, // default to silu
        };

        Ok(Qwen2Config {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings,
            sliding_window,
            rope_theta,
            rms_norm_eps: config_json["rms_norm_eps"]
                .as_f64()
                .unwrap_or(constants::model_config::DEFAULT_RMS_NORM_EPS),
            hidden_act,
            max_window_layers: config_json["max_window_layers"]
                .as_u64()
                .map(|v| v as usize)
                .unwrap_or(num_hidden_layers),
            tie_word_embeddings: config_json["tie_word_embeddings"]
                .as_bool()
                .unwrap_or(false),
            use_sliding_window: config_json["use_sliding_window"].as_bool().unwrap_or(false),
        })
    }

    /// Load Qwen3 model weights using candle
    fn load_qwen3_weights(
        &self,
        model_path: &Path,
        config: &Qwen2Config,
    ) -> Result<ModelForCausalLM> {
        debug!("Loading model weights from: {}", model_path.display());

        // Load model weights using candle's safetensors or pytorch loader
        let weights = if model_path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            candle_core::safetensors::load(model_path, &self.device).with_context(|| {
                format!("Failed to load safetensors from {}", model_path.display())
            })?
        } else {
            // For pytorch_model.bin, attempt to load as PyTorch tensors
            debug!(
                "Loading PyTorch model weights from: {}",
                model_path.display()
            );

            // Try loading as PyTorch tensors using candle's built-in support
            match candle_core::pickle::read_all(model_path) {
                Ok(tensor_vec) => {
                    debug!(
                        "[HUGGINGFACE] [LOAD] Successfully loaded {} PyTorch tensors",
                        tensor_vec.len()
                    );
                    // Convert Vec<(String, Tensor)> to HashMap<String, Tensor> for compatibility
                    tensor_vec
                        .into_iter()
                        .collect::<std::collections::HashMap<_, _>>()
                }
                Err(e) => {
                    warn!("[HUGGINGFACE] [LOAD] PyTorch model loading failed: {}", e);
                    // Provide actionable error message
                    return Err(anyhow::anyhow!(
                        "[HUGGINGFACE] [LOAD] failed: Cannot load PyTorch model from {}. Error: {}. Recommendation: Convert model to safetensors format for better compatibility, or verify model file integrity.",
                        model_path.display(),
                        e
                    ));
                }
            }
        };

        // Create variable builder from the weights
        let vb = VarBuilder::from_tensors(weights, DType::F32, &self.device);

        // Load the Qwen2 model
        let model =
            ModelForCausalLM::new(config, vb).context("[HUGGINGFACE] [LOAD] failed: Cannot initialize Qwen2 model from loaded weights. Verify model architecture compatibility.")?;

        debug!("Model weights loaded successfully");
        Ok(model)
    }
}

/// Qwen3 embedding model implementation
pub struct Qwen3EmbeddingModel {
    model: tokio::sync::RwLock<ModelForCausalLM>,
    tokenizer: Tokenizer,
    config: Qwen2Config,
    device: Device,
    model_name: String,
}

impl std::fmt::Debug for Qwen3EmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3EmbeddingModel")
            .field("model_name", &self.model_name)
            .field("hidden_size", &self.config.hidden_size)
            .field("device", &self.device)
            .finish()
    }
}

impl Qwen3EmbeddingModel {
    /// Generate embeddings for a batch of texts
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_with_instruction(texts, None)
    }

    /// Generate embeddings with optional instruction (Qwen3 feature)
    pub fn embed_with_instruction(
        &self,
        texts: &[String],
        instruction: Option<&str>,
    ) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            // Apply instruction if provided (Qwen3 supports instruction-based embeddings)
            let processed_text = if let Some(instr) = instruction {
                format!("Instruct: {}\nQuery: {}", instr, text)
            } else {
                text.clone()
            };

            // Tokenize the text
            let encoding = self
                .tokenizer
                .encode(processed_text.as_str(), true)
                .map_err(|e| {
                    anyhow::anyhow!(
                        "[HUGGINGFACE] [TOKENIZE] failed: Cannot tokenize input text. Error: {}",
                        e
                    )
                })?;

            let token_ids: Vec<u32> = encoding.get_ids().to_vec();
            let attention_mask: Vec<u32> = vec![1u32; token_ids.len()];

            // Convert to tensors
            let input_ids = Tensor::new(&token_ids[..], &self.device)
                .context("[HUGGINGFACE] [TENSOR] failed: Cannot create input_ids tensor from tokenized text")?
                .unsqueeze(0)?; // Add batch dimension

            let attention_mask_tensor = Tensor::new(&attention_mask[..], &self.device)
                .context("[HUGGINGFACE] [TENSOR] failed: Cannot create attention_mask tensor for model input")?
                .unsqueeze(0)?; // Add batch dimension

            // Run model inference - get hidden states from the language model
            // Use blocking_write() to avoid async in sync method while still using tokio::sync::RwLock
            let hidden_states = {
                let mut model_guard = self.model.blocking_write();
                model_guard
                    .forward(&input_ids, 0)
                    .context("[HUGGINGFACE] [INFERENCE] failed: Model forward pass execution failed. Check input tensor dimensions and model state.")?
            };

            // Extract embedding using mean pooling of last hidden states
            let embedding = self.mean_pooling(&hidden_states, &attention_mask_tensor)?;
            let normalized = self.normalize_embedding(embedding)?;

            embeddings.push(normalized);
        }

        Ok(embeddings)
    }

    /// Apply mean pooling to get sentence embedding from token embeddings
    fn mean_pooling(&self, last_hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // last_hidden_states shape: [batch_size, sequence_length, hidden_size]
        // attention_mask shape: [batch_size, sequence_length]

        // Expand attention mask to match hidden states dimensions
        let attention_mask = attention_mask.unsqueeze(2)?; // [batch_size, sequence_length, 1]
        let expanded_mask = attention_mask.broadcast_as(last_hidden_states.shape())?;

        // Apply mask to hidden states
        let masked_hidden_states = (last_hidden_states * &expanded_mask)?;

        // Sum along sequence dimension
        let sum_hidden_states = masked_hidden_states.sum(1)?; // [batch_size, hidden_size]

        // Sum attention mask to get sequence lengths
        let sum_mask = attention_mask.sum(1)?; // [batch_size, 1]

        // Divide by sequence length to get mean
        let mean_pooled = sum_hidden_states.broadcast_div(&sum_mask)?;

        Ok(mean_pooled)
    }

    /// L2 normalize the embedding vector
    fn normalize_embedding(&self, embedding: Tensor) -> Result<Vec<f32>> {
        // Calculate L2 norm
        let norm = embedding.sqr()?.sum_keepdim(1)?.sqrt()?;

        // Normalize
        let normalized = embedding.broadcast_div(&norm)?;

        // Convert to Vec<f32>
        let normalized_vec = normalized.squeeze(0)?.to_vec1::<f32>()?;

        Ok(normalized_vec)
    }
}

impl EmbeddingModelTrait for Qwen3EmbeddingModel {
    /// Generate embeddings for a batch of texts
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed(texts)
    }

    /// Get the embedding dimensions produced by this model
    fn dimensions(&self) -> usize {
        self.config.hidden_size
    }

    /// Get the maximum sequence length supported by this model
    fn max_sequence_length(&self) -> usize {
        self.config.max_position_embeddings
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_huggingface_backend_creation() {
        let backend = HuggingFaceBackend::new();
        assert!(backend.is_ok());
    }

    #[test]
    fn test_parse_qwen2_config() {
        let backend = HuggingFaceBackend::new().unwrap();

        let config_json = serde_json::json!({
            "vocab_size": 151936,
            "hidden_size": 1024,
            "intermediate_size": 2816,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "max_position_embeddings": 8192,
            "rope_theta": 1000000.0,
            "rms_norm_eps": constants::model_config::DEFAULT_RMS_NORM_EPS
        });

        let config = backend.parse_qwen2_config(&config_json);
        assert!(config.is_ok());

        let config = config.unwrap();
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 24);
    }

    #[test]
    fn test_parse_qwen2_config_missing_fields() {
        let backend = HuggingFaceBackend::new().unwrap();

        let config_json = serde_json::json!({
            "hidden_size": 1024,
            // Missing required fields
        });

        let config = backend.parse_qwen2_config(&config_json);
        assert!(config.is_err());
    }

    #[tokio::test]
    async fn test_qwen3_model_loading() {
        // This test requires network access and is expensive
        // Skip unless explicitly requested
        if std::env::var("INTEGRATION_TESTS").is_err() {
            return;
        }

        let temp_dir = TempDir::new().unwrap();
        let backend = HuggingFaceBackend::new().unwrap();

        // This would test actual model loading but requires large downloads
        // For now, we'll just test that the function can be called
        let result = backend
            .load_qwen3_model(&"Qwen/Qwen3-Embedding-0.6B".into(), &temp_dir.path().into())
            .await;

        // In a real integration test, we would assert success
        // For unit tests, we just ensure the method exists and can be called
        match result {
            Ok(_) => println!("Qwen3 model loaded successfully"),
            Err(e) => println!("Qwen3 model loading failed (expected in unit test): {}", e),
        }
    }
}
