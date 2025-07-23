//! Model validation utilities for command execution.
//!
//! This module provides validation functions to ensure that models are available
//! and properly cached before command execution.

use anyhow::Result;
use tracing::info;

use crate::models::{ModelInfo, ModelManager};
use crate::types::ModelBackend;

/// Validate that a model is available and handle download requirements
///
/// This function checks if the specified model is available and cached.
/// For models that require manual download, it will provide appropriate
/// error messages guiding users to download the model first.
///
/// # Arguments
/// * `model_name` - Name of the model to validate
///
/// # Returns
/// * `Result<ModelInfo>` - The validated model information
///
/// # Examples
/// ```no_run
/// # use turboprop::model_validation::validate_model_selection;
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let model_info = validate_model_selection("sentence-transformers/all-MiniLM-L6-v2").await.unwrap();
/// println!("Model: {}", model_info.name.as_str());
/// # });
/// ```
pub async fn validate_model_selection(model_name: &str) -> Result<ModelInfo> {
    let available_models = ModelManager::get_available_models();

    let model_info = available_models
        .iter()
        .find(|m| m.name.as_str() == model_name)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Model '{}' is not available. Use 'tp model list' to see available models.",
                model_name
            )
        })?;

    // Check if model requires download
    let manager = ModelManager::default();
    if !manager.is_model_cached(&model_info.name) {
        match model_info.backend {
            ModelBackend::FastEmbed => {
                info!("Model will be downloaded automatically on first use");
            }
            ModelBackend::Candle => {
                return Err(anyhow::anyhow!(
                    "GGUF model '{}' is not cached. Download it first with: tp model download {}",
                    model_name,
                    model_name
                ));
            }
            ModelBackend::Custom => {
                return Err(anyhow::anyhow!(
                    "HuggingFace model '{}' is not cached. Download it first with: tp model download {}",
                    model_name, model_name
                ));
            }
        }
    }

    Ok(model_info.clone())
}

/// Validate model compatibility with instruction parameter
///
/// Some models support instruction-based embeddings while others don't.
/// This function validates that the instruction parameter is only used
/// with compatible models.
///
/// # Arguments
/// * `model_info` - Information about the model to validate
/// * `instruction` - Optional instruction parameter
///
/// # Returns
/// * `Result<()>` - Ok if the combination is valid
pub fn validate_instruction_compatibility(
    model_info: &ModelInfo,
    instruction: Option<&str>,
) -> Result<()> {
    if let Some(_instr) = instruction {
        // Check if the model supports instructions
        let supports_instructions = model_info.name.as_str().contains("qwen3")
            || model_info.name.as_str().contains("Qwen3");

        if !supports_instructions {
            return Err(anyhow::anyhow!(
                "Model '{}' does not support instruction-based embeddings. \
                The --instruction parameter can only be used with instruction-capable models like Qwen3. \
                Use 'tp model list' to see which models support instructions.",
                model_info.name.as_str()
            ));
        }
    }

    Ok(())
}

/// Get the effective model name, applying precedence rules
///
/// This function resolves the model name by applying the precedence:
/// command-level > global CLI option > configuration default > system default
///
/// # Arguments
/// * `command_model` - Model specified at command level
/// * `global_model` - Model specified as global CLI option
/// * `config_default` - Default model from configuration
///
/// # Returns
/// * `String` - The effective model name to use
pub fn resolve_effective_model(
    command_model: Option<&str>,
    global_model: Option<&str>,
    config_default: Option<&str>,
) -> String {
    command_model
        .or(global_model)
        .or(config_default)
        .unwrap_or_else(|| ModelManager::default_model())
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelName;

    #[tokio::test]
    async fn test_validate_model_selection_existing() {
        let result = validate_model_selection("sentence-transformers/all-MiniLM-L6-v2").await;
        assert!(result.is_ok());

        let model_info = result.unwrap();
        assert_eq!(
            model_info.name.as_str(),
            "sentence-transformers/all-MiniLM-L6-v2"
        );
    }

    #[tokio::test]
    async fn test_validate_model_selection_nonexistent() {
        let result = validate_model_selection("nonexistent-model").await;
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("not available"));
        assert!(error_msg.contains("tp model list"));
    }

    #[test]
    fn test_validate_instruction_compatibility_qwen3() {
        let model_info = ModelInfo::simple(
            ModelName::from("Qwen/Qwen3-Embedding-0.6B"),
            "Test Qwen3 model".to_string(),
            1024,
            1000000,
        );

        // Should work with instruction
        let result = validate_instruction_compatibility(&model_info, Some("test instruction"));
        assert!(result.is_ok());

        // Should work without instruction
        let result = validate_instruction_compatibility(&model_info, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_instruction_compatibility_non_instruction_model() {
        let model_info = ModelInfo::simple(
            ModelName::from("sentence-transformers/all-MiniLM-L6-v2"),
            "Test non-instruction model".to_string(),
            384,
            1000000,
        );

        // Should fail with instruction
        let result = validate_instruction_compatibility(&model_info, Some("test instruction"));
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("does not support instruction-based embeddings"));

        // Should work without instruction
        let result = validate_instruction_compatibility(&model_info, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_resolve_effective_model_precedence() {
        // Command-level has highest precedence
        let result = resolve_effective_model(
            Some("command-model"),
            Some("global-model"),
            Some("config-model"),
        );
        assert_eq!(result, "command-model");

        // Global has second precedence
        let result = resolve_effective_model(None, Some("global-model"), Some("config-model"));
        assert_eq!(result, "global-model");

        // Config has third precedence
        let result = resolve_effective_model(None, None, Some("config-model"));
        assert_eq!(result, "config-model");

        // System default as fallback
        let result = resolve_effective_model(None, None, None);
        assert_eq!(result, ModelManager::default_model());
    }
}
