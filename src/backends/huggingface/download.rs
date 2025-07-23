//! Model download and loading utilities for HuggingFace backend.
//!
//! This module handles downloading model files from HuggingFace repositories,
//! caching them locally, and loading them into memory for inference.

use anyhow::{Context, Result};
use candle_core::Device;
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM};
use hf_hub::api::tokio::ApiRepo;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

use crate::types::{CachePath, ModelName};
use super::model::Qwen3EmbeddingModel;

/// Download a file from HuggingFace repository, using cache if available
pub async fn download_file(
    repo: &ApiRepo,
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

/// Load Qwen3 model weights using candle
pub fn load_qwen3_weights(
    model_path: &Path,
    config: &Qwen2Config,
    device: &Device,
) -> Result<ModelForCausalLM> {
    debug!("Loading model weights from: {}", model_path.display());

    // Load model weights using candle's safetensors or pytorch loader
    let weights = if model_path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
        candle_core::safetensors::load(model_path, device).with_context(|| {
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

    let vb = candle_nn::VarBuilder::from_tensors(weights, candle_core::DType::F32, device);

    // Create the Qwen2 model from the configuration and weights
    let model = ModelForCausalLM::new(config, vb).with_context(|| {
        format!(
            "[HUGGINGFACE] [LOAD] failed: Cannot initialize model from weights at {}. Verify model architecture compatibility and weight completeness.",
            model_path.display()
        )
    })?;

    debug!("[HUGGINGFACE] [LOAD] Model weights loaded successfully");
    Ok(model)
}

/// Complete model loading pipeline: download files and create Qwen3EmbeddingModel
pub async fn load_qwen3_model(
    api: &hf_hub::api::tokio::Api,
    device: &Device,
    model_name: &ModelName,
    cache_dir: &CachePath,
) -> Result<Qwen3EmbeddingModel> {
    info!("[HUGGINGFACE] [LOAD] Loading Qwen3 model: {}", model_name);

    // Download model files from HuggingFace
    let repo = api.model(model_name.as_str().to_string());

    // Create model-specific cache directory
    let model_cache_dir = cache_dir.join(model_name.as_str().replace('/', "_"));
    if !model_cache_dir.exists() {
        std::fs::create_dir_all(&model_cache_dir).with_context(|| {
            format!(
                "Failed to create model cache directory: {}",
                model_cache_dir.as_path().display()
            )
        })?;
    }

    // Download required files
    let config_path = download_file(&repo, "config.json", model_cache_dir.as_path()).await?;
    let tokenizer_path = download_file(&repo, "tokenizer.json", model_cache_dir.as_path()).await?;

    // Try to download model weights (safetensors preferred, fallback to pytorch)
    let model_path = match download_file(&repo, "model.safetensors", model_cache_dir.as_path()).await {
        Ok(path) => path,
        Err(e) => {
            warn!("[HUGGINGFACE] [DOWNLOAD] safetensors download failed: {}. Trying pytorch_model.bin as fallback", e);
            download_file(&repo, "pytorch_model.bin", model_cache_dir.as_path())
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
    let config_json: serde_json::Value = serde_json::from_str(&config_content)
        .context("[HUGGINGFACE] [CONFIG] failed: Cannot parse model configuration JSON. Verify config file format and completeness.")?;

    // Create Qwen2 config from the JSON
    let qwen_config = super::config::parse_qwen2_config(&config_json)?;

    // Load model weights
    let model = load_qwen3_weights(&model_path, &qwen_config, device)?;

    Ok(Qwen3EmbeddingModel {
        model: tokio::sync::RwLock::new(model),
        tokenizer,
        config: qwen_config,
        device: device.clone(),
        model_name: model_name.to_string(),
    })
}