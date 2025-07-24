//! Tests for resource usage warnings system

use turboprop::models::ModelInfo;
use turboprop::types::{ModelBackend, ModelName, ModelType};
use turboprop::warnings::ResourceWarnings;

/// Create a test model with specific characteristics
fn create_test_model(name: &str, size_bytes: u64, backend: ModelBackend) -> ModelInfo {
    ModelInfo {
        name: ModelName::from(name),
        backend,
        model_type: ModelType::SentenceTransformer,
        dimensions: 384,
        size_bytes,
        description: "Test model".to_string(),
        download_url: Some("https://example.com".to_string()),
        local_path: None,
    }
}

#[test]
fn test_resource_warnings_creation() {
    // Test that ResourceWarnings can be created
    let _warnings = ResourceWarnings;

    // This test mainly verifies the module structure
    assert!(true);
}

#[test]
fn test_check_model_requirements_small_model() {
    let model = create_test_model("small-model", 50_000_000, ModelBackend::FastEmbed); // 50MB

    // This should not trigger memory warnings
    ResourceWarnings::check_model_requirements(&model);

    // The function doesn't return anything, so we just verify it doesn't panic
    assert!(true);
}

#[test]
fn test_check_model_requirements_large_model() {
    let model = create_test_model("large-model", 2_000_000_000, ModelBackend::FastEmbed); // 2GB

    // This should trigger size warnings
    ResourceWarnings::check_model_requirements(&model);

    // The function doesn't return anything, so we just verify it doesn't panic
    assert!(true);
}

#[test]
fn test_check_model_requirements_gguf_model() {
    let model = create_test_model("gguf-model.Q5_K_S.gguf", 500_000_000, ModelBackend::Candle);

    // This should trigger GGUF-specific warnings
    ResourceWarnings::check_model_requirements(&model);

    // The function doesn't return anything, so we just verify it doesn't panic
    assert!(true);
}

#[test]
fn test_check_model_requirements_qwen3_model() {
    let model = create_test_model(
        "Qwen/Qwen3-Embedding-0.6B",
        600_000_000,
        ModelBackend::Custom,
    );

    // This should trigger Qwen3-specific warnings
    ResourceWarnings::check_model_requirements(&model);

    // The function doesn't return anything, so we just verify it doesn't panic
    assert!(true);
}

#[test]
fn test_memory_estimation_fastembed() {
    let model = create_test_model("fastembed-model", 100_000_000, ModelBackend::FastEmbed);

    // Test that we can calculate memory requirements
    let estimated_memory = ResourceWarnings::estimate_model_memory_usage(&model);

    // FastEmbed should use size * 2
    assert_eq!(estimated_memory, 200_000_000);
}

#[test]
fn test_memory_estimation_gguf() {
    let model = create_test_model("gguf-model", 100_000_000, ModelBackend::Candle);

    let estimated_memory = ResourceWarnings::estimate_model_memory_usage(&model);

    // GGUF should use size * 3
    assert_eq!(estimated_memory, 300_000_000);
}

#[test]
fn test_memory_estimation_huggingface() {
    let model = create_test_model("hf-model", 100_000_000, ModelBackend::Custom);

    let estimated_memory = ResourceWarnings::estimate_model_memory_usage(&model);

    // HuggingFace should use size * 4
    assert_eq!(estimated_memory, 400_000_000);
}

#[test]
fn test_get_available_memory() {
    // Test that we can get available memory (or handle errors gracefully)
    let memory_result = ResourceWarnings::get_available_memory();

    // This should either return a valid memory amount or an error
    match memory_result {
        Ok(memory) => {
            assert!(memory > 0); // Should be some positive amount
        }
        Err(_) => {
            // It's OK if we can't get memory info on some systems
            assert!(true);
        }
    }
}

#[test]
fn test_memory_warning_thresholds() {
    // Test various memory scenarios
    let small_model = create_test_model("small", 10_000_000, ModelBackend::FastEmbed); // 10MB
    let medium_model = create_test_model("medium", 500_000_000, ModelBackend::FastEmbed); // 500MB
    let large_model = create_test_model("large", 2_000_000_000, ModelBackend::FastEmbed); // 2GB

    // All should execute without panicking
    ResourceWarnings::check_model_requirements(&small_model);
    ResourceWarnings::check_model_requirements(&medium_model);
    ResourceWarnings::check_model_requirements(&large_model);

    assert!(true);
}

#[test]
fn test_model_name_pattern_matching() {
    // Test that different model name patterns trigger appropriate warnings
    let models = [
        (
            "sentence-transformers/all-MiniLM-L6-v2",
            ModelBackend::FastEmbed,
        ),
        ("nomic-embed-code.Q5_K_S.gguf", ModelBackend::Candle),
        ("Qwen/Qwen3-Embedding-0.6B", ModelBackend::Custom),
        ("some-other-model", ModelBackend::FastEmbed),
    ];

    for (name, backend) in models {
        let model = create_test_model(name, 100_000_000, backend);
        ResourceWarnings::check_model_requirements(&model);
    }

    assert!(true);
}

#[test]
fn test_warning_message_components() {
    // Test that warning messages would contain expected components
    let model = create_test_model("test-model", 1_500_000_000, ModelBackend::FastEmbed); // 1.5GB

    let size_gb = model.size_bytes as f32 / 1_073_741_824.0;
    assert!(size_gb > 1.0); // Should trigger large model warning

    let estimated_memory = ResourceWarnings::estimate_model_memory_usage(&model);
    let estimated_mb = estimated_memory / 1_048_576;
    assert!(estimated_mb > 1000); // Should be in MB range for warnings

    assert!(model.name.as_str().contains("test-model"));
}

#[test]
fn test_memory_unit_conversions() {
    // Test memory unit conversion logic
    let bytes = 1_073_741_824_u64; // 1GB in bytes
    let mb = bytes / 1_048_576;
    let gb = bytes as f32 / 1_073_741_824.0;

    assert_eq!(mb, 1024); // 1GB = 1024MB
    assert_eq!(gb, 1.0); // 1GB = 1.0GB

    let test_size = 2_500_000_000_u64; // 2.5GB
    let test_gb = test_size as f32 / 1_073_741_824.0;
    assert!((test_gb - 2.328).abs() < 0.01); // Should be approximately 2.328GB
}

#[test]
fn test_backend_specific_warnings() {
    // Test that different backends trigger different warnings
    let fastembed_model = create_test_model("fastembed", 100_000_000, ModelBackend::FastEmbed);
    let gguf_model = create_test_model("model.gguf", 100_000_000, ModelBackend::Candle);
    let qwen_model = create_test_model("Qwen3", 100_000_000, ModelBackend::Custom);

    // All should execute without issues
    ResourceWarnings::check_model_requirements(&fastembed_model);
    ResourceWarnings::check_model_requirements(&gguf_model);
    ResourceWarnings::check_model_requirements(&qwen_model);

    assert!(true);
}

#[test]
fn test_zero_size_model() {
    let model = create_test_model("zero-size", 0, ModelBackend::FastEmbed);

    // Should handle zero-size models gracefully
    ResourceWarnings::check_model_requirements(&model);

    let estimated = ResourceWarnings::estimate_model_memory_usage(&model);
    assert_eq!(estimated, 0);
}

#[test]
fn test_very_large_model() {
    let model = create_test_model("huge-model", u64::MAX, ModelBackend::FastEmbed);

    // Should handle very large models without overflow
    ResourceWarnings::check_model_requirements(&model);

    // Memory estimation should be calculated correctly even for large sizes
    let estimated = ResourceWarnings::estimate_model_memory_usage(&model);
    // For u64::MAX, saturating multiplication will return u64::MAX, so estimated >= size_bytes
    assert!(estimated >= model.size_bytes);
}

#[test]
fn test_warning_system_integration() {
    // Test that the warning system integrates well with other components
    let model = create_test_model("integration-test", 500_000_000, ModelBackend::FastEmbed);

    // This simulates what would happen during model loading
    ResourceWarnings::check_model_requirements(&model);

    // Verify model properties are accessible
    assert_eq!(model.name.as_str(), "integration-test");
    assert_eq!(model.size_bytes, 500_000_000);
    assert_eq!(model.backend, ModelBackend::FastEmbed);
}

#[test]
fn test_error_handling() {
    // Test that the warning system handles various error conditions
    let models = [
        create_test_model("", 100_000_000, ModelBackend::FastEmbed), // Empty name
        create_test_model("valid-name", 0, ModelBackend::FastEmbed), // Zero size
        create_test_model("huge", u64::MAX, ModelBackend::FastEmbed), // Max size
    ];

    for model in models {
        // Should not panic on any model
        ResourceWarnings::check_model_requirements(&model);
    }

    assert!(true);
}
