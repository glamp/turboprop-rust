//! Main HuggingFace backend implementation.
//!
//! This module provides the HuggingFaceBackend struct that orchestrates
//! model loading, validation, and initialization for HuggingFace models.

use anyhow::Result;
use candle_core::Device;
use hf_hub::api::tokio::Api;

use super::{download, model::Qwen3EmbeddingModel, validation};
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

    /// Load a Qwen3 embedding model from HuggingFace
    pub async fn load_qwen3_model(
        &self,
        model_name: &ModelName,
        cache_dir: &CachePath,
    ) -> Result<Qwen3EmbeddingModel> {
        // Validate inputs before proceeding
        validation::validate_model_inputs(model_name, cache_dir)?;

        // Use the download module to handle the complete loading pipeline
        download::load_qwen3_model(&self.api, &self.device, model_name, cache_dir).await
    }
}
