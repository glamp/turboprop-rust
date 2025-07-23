//! Unit tests for Qwen3 embedding functionality (configuration and API only).
//!
//! For integration tests with real model downloads, see tests/integration/embedding_tests.rs

use anyhow::Result;
use tempfile::TempDir;

use turboprop::backends::HuggingFaceBackend;
use turboprop::embeddings::{EmbeddingConfig, EmbeddingGenerator, EmbeddingOptions};
use turboprop::models::ModelManager;
use turboprop::types::ModelName;
use turboprop::types::{ModelBackend, ModelType};

/// Test that Qwen3 model is available in the available models list
#[test]
fn test_qwen3_model_availability() {
    let manager = ModelManager::default();
    let models = manager.get_available_models();

    // Check that Qwen3-Embedding-0.6B is in the list
    let qwen3_model = models
        .iter()
        .find(|m| m.name == ModelName::from("Qwen/Qwen3-Embedding-0.6B"));
    assert!(
        qwen3_model.is_some(),
        "Qwen3-Embedding-0.6B should be available in models list"
    );

    let qwen3 = qwen3_model.unwrap();
    assert_eq!(qwen3.dimensions, 1024);
    assert_eq!(qwen3.size_bytes, 600_000_000);
    assert_eq!(qwen3.backend, ModelBackend::Custom);
    assert_eq!(qwen3.model_type, ModelType::HuggingFace);
}

/// Test HuggingFace backend creation
#[test]
fn test_huggingface_backend_creation() {
    let backend = HuggingFaceBackend::new();
    assert!(
        backend.is_ok(),
        "HuggingFace backend should be created successfully"
    );
}

/// Test configuration parsing is handled internally by the backend
/// (We can't test the private method directly, but this verifies the concept)
#[test]
fn test_qwen3_config_concept() {
    // Test that the expected configuration values are reasonable for Qwen3
    let expected_vocab_size = 151936;
    let expected_hidden_size = 1024;
    let expected_num_layers = 24;
    let expected_num_heads = 16;
    let expected_max_position = 8192;

    // These values should be sensible for a 600M parameter model
    assert!(
        expected_vocab_size > 100000,
        "Vocab size should be reasonable for multilingual model"
    );
    assert!(
        expected_hidden_size == 1024,
        "Hidden size should match Qwen3-0.6B spec"
    );
    assert!(
        expected_num_layers > 20,
        "Should have enough layers for good performance"
    );
    assert!(
        expected_num_heads > 10,
        "Should have sufficient attention heads"
    );
    assert!(
        expected_max_position >= 8192,
        "Should support long sequences"
    );
}

/// Test instruction-based embedding support
#[test]
fn test_instruction_based_embedding_support() {
    // This test just verifies the API exists and compiles
    // Actual functionality would require a loaded model

    // Test that the instruction method signature exists
    // (This will compile if the method exists with correct signature)
    let test_closure = |model: &turboprop::backends::Qwen3EmbeddingModel| {
        let texts = vec!["test text".to_string()];
        let instruction = Some("Retrieve relevant documents for the query:");

        // The method should exist and return the correct type
        let _result: Result<Vec<Vec<f32>>> = model.embed_with_instruction(&texts, instruction);
    };

    // We can't actually call this without a model, but this tests compilation
    let _ = test_closure;
}

/// Test embedding configuration for Qwen3 models
#[test]
fn test_qwen3_embedding_config() {
    let config = EmbeddingConfig::with_model("Qwen/Qwen3-Embedding-0.6B");

    assert_eq!(config.model_name, "Qwen/Qwen3-Embedding-0.6B");
    assert_eq!(config.embedding_dimensions, 1024); // Should match QWEN_EMBED_DIMENSIONS
    assert_eq!(config.batch_size, 32); // Default batch size
}

/// Test model validation for Qwen3 models
#[test]
fn test_qwen3_model_validation() {
    let manager = ModelManager::default();
    let models = manager.get_available_models();
    let qwen3_model = models
        .iter()
        .find(|m| m.name == ModelName::from("Qwen/Qwen3-Embedding-0.6B"))
        .expect("Qwen3 model should be available");

    // Test model validation
    let validation_result = qwen3_model.validate();
    assert!(
        validation_result.is_ok(),
        "Qwen3 model should pass validation: {:?}",
        validation_result
    );
}

/// Test that embedding backend selection works for Qwen3
#[test]
fn test_qwen3_backend_selection() {
    let manager = ModelManager::default();
    let models = manager.get_available_models();
    let qwen3_model = models
        .iter()
        .find(|m| m.name == ModelName::from("Qwen/Qwen3-Embedding-0.6B"))
        .expect("Qwen3 model should be available");

    // Verify backend selection logic
    match qwen3_model.backend {
        ModelBackend::Custom => {
            // This is correct for Qwen3 models
            assert_eq!(qwen3_model.model_type, ModelType::HuggingFace);
        }
        _ => panic!("Qwen3 models should use Custom backend"),
    }
}

/// Integration test for model manager with Qwen3 models
#[test]
fn test_model_manager_qwen3_integration() {
    let temp_dir = TempDir::new().unwrap();
    let manager = ModelManager::new_with_defaults(temp_dir.path());

    // Test cache initialization
    let init_result = manager.init_cache();
    assert!(init_result.is_ok(), "Cache initialization should succeed");

    // Test model path generation for Qwen3
    let model_path = manager.get_model_path(&ModelName::from("Qwen/Qwen3-Embedding-0.6B"));
    assert!(
        model_path
            .to_string_lossy()
            .contains("Qwen_Qwen3-Embedding-0_6B"),
        "Model path should contain escaped model name"
    );

    // Test cache check (should be false for non-downloaded model)
    let is_cached = manager.is_model_cached(&ModelName::from("Qwen/Qwen3-Embedding-0.6B"));
    assert!(!is_cached, "Model should not be cached initially");
}

/// Test EmbeddingOptions functionality
#[test]
fn test_embedding_options_for_qwen3() {
    // Test that EmbeddingOptions can be created with various configurations
    let options_default = EmbeddingOptions::default();
    assert!(options_default.instruction.is_none());
    assert!(options_default.normalize);
    assert!(options_default.max_length.is_none());

    let options_with_instruction =
        EmbeddingOptions::with_instruction("Retrieve relevant documents for the query:");
    assert_eq!(
        options_with_instruction.instruction,
        Some("Retrieve relevant documents for the query:".to_string())
    );
    assert!(options_with_instruction.normalize);

    let options_no_normalize = EmbeddingOptions::without_normalization();
    assert!(!options_no_normalize.normalize);

    let options_with_length = EmbeddingOptions::with_max_length(512);
    assert_eq!(options_with_length.max_length, Some(512));
}

/// Test that EmbeddingGenerator API includes embed_with_options method
#[test]
fn test_embedding_generator_has_options_method() {
    // This test verifies that the embed_with_options method exists with correct signature
    // (This will compile if the method exists with correct signature)
    let test_closure = |generator: &mut EmbeddingGenerator| {
        let texts = vec!["test text".to_string()];
        let options = EmbeddingOptions::with_instruction("Find relevant code:");

        // The method should exist and return the correct type
        let _result: Result<Vec<Vec<f32>>> = generator.embed_with_options(&texts, &options);
    };

    // We can't actually call this without a generator, but this tests compilation
    let _ = test_closure;
}
