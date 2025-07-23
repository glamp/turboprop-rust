//! Unit tests for embedding generation functionality.
//!
//! These tests validate the embedding API and configuration without requiring
//! model downloads. For tests with real models, see tests/integration/embedding_tests.rs
//!
//! Run these tests with: `cargo test`

use std::path::PathBuf;
use tempfile::TempDir;
use turboprop::config::TurboPropConfig;
use turboprop::embeddings::{EmbeddingConfig, EmbeddingOptions};
use turboprop::models::{CacheStats, ModelManager};
use turboprop::types::ModelName;

/// Test embedding configuration initialization
#[test]
fn test_embedding_config_initialization() {
    let config = EmbeddingConfig::default();
    assert_eq!(
        config.model_name,
        "sentence-transformers/all-MiniLM-L6-v2"
    );
    assert_eq!(config.embedding_dimensions, 384);
    assert_eq!(config.batch_size, 32);
}

/// Test embedding configuration with custom model
#[test]
fn test_embedding_config_with_model() {
    let config = EmbeddingConfig::with_model("sentence-transformers/all-MiniLM-L12-v2");
    assert_eq!(config.model_name, "sentence-transformers/all-MiniLM-L12-v2");
    assert_eq!(config.batch_size, 32); // Should keep default
}

/// Test embedding configuration with cache directory
#[test]
fn test_embedding_config_with_cache_dir() {
    let config = EmbeddingConfig::default().with_cache_dir("/tmp/models");
    assert_eq!(config.cache_dir, PathBuf::from("/tmp/models"));
}

/// Test embedding configuration with batch size
#[test]
fn test_embedding_config_with_batch_size() {
    let config = EmbeddingConfig::default().with_batch_size(16);
    assert_eq!(config.batch_size, 16);
}

/// Test embedding options with instruction
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

/// Test embedding options without normalization
#[test]
fn test_embedding_options_without_normalization() {
    let options = EmbeddingOptions::without_normalization();
    assert!(options.instruction.is_none());
    assert!(!options.normalize);
    assert!(options.max_length.is_none());
}

/// Test embedding options with max length
#[test]
fn test_embedding_options_with_max_length() {
    let options = EmbeddingOptions::with_max_length(512);
    assert!(options.instruction.is_none());
    assert!(options.normalize);
    assert_eq!(options.max_length, Some(512));
}

/// Test embedding options default values
#[test]
fn test_embedding_options_default() {
    let options = EmbeddingOptions::default();
    assert!(options.instruction.is_none());
    assert!(options.normalize);
    assert!(options.max_length.is_none());
}

/// Test model manager cache functionality (no downloads required)
#[test]
fn test_model_manager_cache_operations() {
    let temp_dir = TempDir::new().unwrap();
    let manager = ModelManager::new(temp_dir.path());

    // Test cache initialization
    assert!(manager.init_cache().is_ok());
    assert!(temp_dir.path().exists());

    // Test model path generation
    let model_path = manager.get_model_path(&ModelName::from("sentence-transformers/all-MiniLM-L6-v2"));
    let expected_name = "sentence-transformers_all-MiniLM-L6-v2";
    assert!(model_path
        .file_name()
        .unwrap()
        .to_string_lossy()
        .contains(expected_name));

    // Test model not cached (initially)
    assert!(!manager.is_model_cached(&ModelName::from("sentence-transformers/all-MiniLM-L6-v2")));

    // Test cache stats
    let stats = manager.get_cache_stats().unwrap();
    assert_eq!(stats.model_count, 0);
    assert_eq!(stats.total_size_bytes, 0);

    // Test cache clearing
    assert!(manager.clear_cache().is_ok());
}

/// Test model information (no downloads required)
#[test]
fn test_model_information() {
    let models = ModelManager::get_available_models();

    assert!(!models.is_empty());

    let default_model = models
        .iter()
        .find(|m| m.name == ModelName::from(ModelManager::default_model()));
    assert!(default_model.is_some());

    let default_model = default_model.unwrap();
    assert_eq!(default_model.dimensions, 384);
    assert!(default_model.size_bytes > 0);
    assert!(!default_model.description.is_empty());
}

/// Test configuration integration with embeddings
#[test]
fn test_config_integration() {
    let config = TurboPropConfig::default();

    // Test default embedding configuration
    assert_eq!(
        config.embedding.model_name,
        "sentence-transformers/all-MiniLM-L6-v2"
    );
    assert_eq!(config.embedding.batch_size, 32);
    assert_eq!(
        config.embedding.cache_dir,
        PathBuf::from(".turboprop/models")
    );

    // Test validation passes
    assert!(config.validate().is_ok());
}

/// Test cache stats formatting
#[test]
fn test_cache_stats_formatting() {
    let stats = CacheStats {
        model_count: 2,
        total_size_bytes: 50 * 1024 * 1024, // 50MB
    };

    let formatted = stats.format_size();
    assert!(formatted.contains("50.00 MB"));
}

/// Test embedding configuration validation
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

/// Test different Qwen3 model backend selection logic
#[test]
fn test_qwen3_model_backend_selection() {
    let models = ModelManager::get_available_models();
    
    // Check if Qwen3 model is available in the model list
    let qwen3_model = models
        .iter()
        .find(|m| m.name.as_str().contains("Qwen3"));
    
    if let Some(model) = qwen3_model {
        // Verify it uses the Custom backend (which maps to HuggingFace)
        assert_eq!(model.backend, turboprop::types::ModelBackend::Custom);
        assert!(model.dimensions > 0);
        assert!(!model.description.is_empty());
    }
}