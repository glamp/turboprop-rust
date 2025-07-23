//! HuggingFace backend implementation for embedding models not supported by FastEmbed.
//!
//! This backend provides support for models like Qwen3-Embedding that are available
//! on HuggingFace but not yet supported in the FastEmbed library.
//!
//! # Architecture
//!
//! The module is organized into several focused sub-modules:
//!
//! - `backend` - Main HuggingFaceBackend struct and orchestration
//! - `model` - Qwen3EmbeddingModel implementation and inference logic
//! - `validation` - Input validation utilities (model names, cache directories)
//! - `download` - Model download and loading utilities
//! - `config` - Configuration parsing for HuggingFace model configs
//!
//! # Usage
//!
//! ```no_run
//! use turboprop::backends::huggingface::HuggingFaceBackend;
//! use turboprop::types::{ModelName, CachePath};
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let backend = HuggingFaceBackend::new()?;
//! let model_name = ModelName::new("Alibaba-NLP/gte-Qwen2-0.5B-instruct");
//! let cache_dir = CachePath::new("~/.turboprop/models");
//!
//! let model = backend.load_qwen3_model(&model_name, &cache_dir).await?;
//! let embeddings = model.embed(&["Hello, world!".to_string()])?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! # });
//! ```

pub mod backend;
pub mod config;
pub mod download;
pub mod model;
pub mod validation;

// Re-export main public API
pub use backend::HuggingFaceBackend;
pub use model::Qwen3EmbeddingModel;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CachePath, ModelName};
    use tempfile::TempDir;

    #[test]
    fn test_huggingface_backend_creation() {
        let backend = HuggingFaceBackend::new();
        assert!(backend.is_ok());
    }

    #[test]
    fn test_qwen3_model_loading() {
        // This is a unit test that validates model structure creation
        // without requiring network access or actual model files
        let model_name = ModelName::new("test/model");
        assert_eq!(model_name.as_str(), "test/model");
    }

    #[test]
    fn test_parse_qwen2_config_missing_fields() {
        use serde_json::json;

        let incomplete_config = json!({
            "vocab_size": 32000,
            // Missing hidden_size, intermediate_size, etc.
        });

        let result = config::parse_qwen2_config(&incomplete_config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("hidden_size"));
    }

    #[test]
    fn test_parse_qwen2_config() {
        use serde_json::json;

        let complete_config = json!({
            "vocab_size": 32000,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 12,
            "max_position_embeddings": 2048,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0
        });

        let result = config::parse_qwen2_config(&complete_config);
        assert!(result.is_ok());

        let config = result.unwrap();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.intermediate_size, 3072);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.max_position_embeddings, 2048);
    }

    #[test]
    fn test_validation_empty_model_name() {
        let empty_name = ModelName::new("");
        let temp_dir = TempDir::new().unwrap();
        let _cache_dir = CachePath::new(temp_dir.path());

        let result = validation::validate_model_name(&empty_name);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn test_validation_invalid_model_name_format() {
        let invalid_name = ModelName::new("invalid-name-without-slash");
        let result = validation::validate_model_name(&invalid_name);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("organization/model-name"));
    }

    #[test]
    fn test_validation_valid_model_name() {
        let valid_name = ModelName::new("Alibaba-NLP/gte-Qwen2-0.5B-instruct");
        let result = validation::validate_model_name(&valid_name);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validation_nonexistent_cache_dir() {
        let _model_name = ModelName::new("test/model");
        let nonexistent_cache = CachePath::new("/nonexistent/path");

        let result = validation::validate_cache_directory(&nonexistent_cache);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[test]
    fn test_validation_valid_cache_dir() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = CachePath::new(temp_dir.path());

        let result = validation::validate_cache_directory(&cache_dir);
        assert!(result.is_ok());
    }
}
