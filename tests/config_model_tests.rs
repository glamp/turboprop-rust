//! Tests for model-specific configuration functionality.
//!
//! These tests verify that model configuration, instruction handling,
//! and CLI overrides work correctly.

use std::collections::HashMap;
use turboprop::config::{CliConfigOverrides, ModelConfig, TurboPropConfig};

#[test]
fn test_model_config_default() {
    let model_config = ModelConfig::default();
    assert!(model_config.instruction.is_none());
    assert!(model_config.cache_dir.is_none());
    assert!(model_config.download_url.is_none());
}

#[test]
fn test_model_config_serialization() {
    let model_config = ModelConfig {
        instruction: Some("test instruction".to_string()),
        download_url: Some("https://example.com/model.bin".to_string()),
        ..Default::default()
    };

    let serialized = serde_json::to_string_pretty(&model_config).unwrap();
    let deserialized: ModelConfig = serde_json::from_str(&serialized).unwrap();

    assert_eq!(
        deserialized.instruction,
        Some("test instruction".to_string())
    );
    assert_eq!(
        deserialized.download_url,
        Some("https://example.com/model.bin".to_string())
    );
    assert!(deserialized.cache_dir.is_none());
}

#[test]
fn test_turboprop_config_with_models() {
    let mut config = TurboPropConfig::default();

    // Add model-specific configurations
    let mut models = HashMap::new();
    let qwen_config = ModelConfig {
        instruction: Some("Represent this code for search".to_string()),
        ..Default::default()
    };
    models.insert("Qwen/Qwen3-Embedding-0.6B".to_string(), qwen_config);

    config.models = Some(models);
    config.default_model = Some("sentence-transformers/all-MiniLM-L6-v2".to_string());

    // Test serialization
    let serialized = serde_json::to_string_pretty(&config).unwrap();
    let deserialized: TurboPropConfig = serde_json::from_str(&serialized).unwrap();

    assert_eq!(
        deserialized.default_model,
        Some("sentence-transformers/all-MiniLM-L6-v2".to_string())
    );
    assert!(deserialized.models.is_some());

    let models = deserialized.models.unwrap();
    assert!(models.contains_key("Qwen/Qwen3-Embedding-0.6B"));

    let qwen_config = models.get("Qwen/Qwen3-Embedding-0.6B").unwrap();
    assert_eq!(
        qwen_config.instruction,
        Some("Represent this code for search".to_string())
    );
}

#[test]
fn test_cli_config_overrides_with_instruction() {
    let overrides = CliConfigOverrides::new()
        .with_model("test-model")
        .with_instruction("test instruction");

    assert_eq!(overrides.model, Some("test-model".to_string()));
    assert_eq!(overrides.instruction, Some("test instruction".to_string()));
}

#[test]
fn test_merge_cli_args_with_instruction() {
    let config = TurboPropConfig::default();

    let overrides = CliConfigOverrides::new()
        .with_model("Qwen/Qwen3-Embedding-0.6B")
        .with_instruction("Represent this code for search")
        .with_verbose(true);

    let merged = config.merge_cli_args(&overrides);

    assert_eq!(merged.embedding.model_name, "Qwen/Qwen3-Embedding-0.6B");
    assert_eq!(
        merged.current_instruction,
        Some("Represent this code for search".to_string())
    );
    assert!(merged.general.verbose);
}

#[test]
fn test_merge_cli_args_instruction_only() {
    let config = TurboPropConfig::default();

    let overrides =
        CliConfigOverrides::new().with_instruction("Encode this text for similarity search");

    let original_model_name = config.embedding.model_name.clone();
    let merged = config.merge_cli_args(&overrides);

    assert_eq!(
        merged.current_instruction,
        Some("Encode this text for similarity search".to_string())
    );
    // Model should remain unchanged
    assert_eq!(merged.embedding.model_name, original_model_name);
}

#[test]
fn test_configuration_precedence_with_instruction() {
    let config = TurboPropConfig {
        current_instruction: Some("default instruction".to_string()),
        ..Default::default()
    };

    let overrides = CliConfigOverrides::new().with_instruction("override instruction");

    let merged = config.merge_cli_args(&overrides);

    // CLI should override default
    assert_eq!(
        merged.current_instruction,
        Some("override instruction".to_string())
    );
}

#[test]
fn test_model_specific_config_lookup() {
    let mut config = TurboPropConfig::default();

    // Set up model-specific configurations
    let mut models = HashMap::new();

    let qwen_config = ModelConfig {
        instruction: Some("Qwen specific instruction".to_string()),
        ..Default::default()
    };
    models.insert("Qwen/Qwen3-Embedding-0.6B".to_string(), qwen_config);

    let minilm_config = ModelConfig {
        instruction: Some("MiniLM specific instruction".to_string()),
        ..Default::default()
    };
    models.insert(
        "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        minilm_config,
    );

    config.models = Some(models);

    // Test that we can look up model-specific configurations
    let models = config.models.as_ref().unwrap();

    let qwen_config = models.get("Qwen/Qwen3-Embedding-0.6B").unwrap();
    assert_eq!(
        qwen_config.instruction,
        Some("Qwen specific instruction".to_string())
    );

    let minilm_config = models
        .get("sentence-transformers/all-MiniLM-L6-v2")
        .unwrap();
    assert_eq!(
        minilm_config.instruction,
        Some("MiniLM specific instruction".to_string())
    );

    // Non-existent model should not be found
    assert!(models.get("nonexistent-model").is_none());
}
