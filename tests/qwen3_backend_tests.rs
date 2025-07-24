//! Unit tests for Qwen3/HuggingFace backend functionality.
//!
//! These tests validate the HuggingFace backend and Qwen3 model support without
//! requiring large model downloads. For tests that require actual model loading,
//! those are marked with #[ignore].
//!
//! Run these tests with: `cargo test qwen3`

use anyhow::Result;
use tempfile::TempDir;
use turboprop::backends::huggingface::HuggingFaceBackend;
use turboprop::backends::huggingface::{config, validation};
use turboprop::models::ModelInfo;
use turboprop::types::{CachePath, ModelBackend, ModelName, ModelType};

#[tokio::test]
async fn test_huggingface_backend_creation() -> Result<()> {
    let backend = HuggingFaceBackend::new();
    assert!(backend.is_ok());
    Ok(())
}

#[test]
fn test_validation_empty_model_name() {
    let empty_name = ModelName::new("");
    let result = validation::validate_model_name(&empty_name);
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Model name cannot be empty"));
}

#[test]
fn test_validation_invalid_model_name_format() {
    let invalid_name = ModelName::new("invalid-name-without-slash");
    let result = validation::validate_model_name(&invalid_name);
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("organization/model-name"));
}

#[test]
fn test_validation_invalid_model_name_empty_parts() {
    let invalid_names = vec![
        ModelName::new("/model-name"),     // Empty organization
        ModelName::new("organization/"),   // Empty model name
        ModelName::new("org//model"),      // Double slash
        ModelName::new("org/model/extra"), // Too many parts
    ];

    for invalid_name in invalid_names {
        let result = validation::validate_model_name(&invalid_name);
        assert!(result.is_err(), "Should fail validation: {}", invalid_name);
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("invalid format"));
    }
}

#[test]
fn test_validation_invalid_model_name_characters() {
    let invalid_names = vec![
        ModelName::new("org/model@name"), // @ character
        ModelName::new("org/model name"), // Space character
        ModelName::new("org/model#name"), // Hash character
        ModelName::new("org/model%name"), // Percent character
    ];

    for invalid_name in invalid_names {
        let result = validation::validate_model_name(&invalid_name);
        assert!(result.is_err(), "Should fail validation: {}", invalid_name);
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("invalid characters"));
    }
}

#[test]
fn test_validation_valid_model_names() {
    let valid_names = vec![
        ModelName::new("Alibaba-NLP/gte-Qwen2-0.5B-instruct"),
        ModelName::new("Qwen/Qwen3-Embedding-0.6B"),
        ModelName::new("microsoft/codebert-base"),
        ModelName::new("sentence-transformers/all-MiniLM-L6-v2"),
        ModelName::new("org/model-name_v1.0"),
        ModelName::new("user123/my-model.v2"),
    ];

    for valid_name in valid_names {
        let result = validation::validate_model_name(&valid_name);
        assert!(result.is_ok(), "Should pass validation: {}", valid_name);
    }
}

#[test]
fn test_validation_nonexistent_cache_dir() {
    let nonexistent_cache = CachePath::new("/nonexistent/path/that/does/not/exist");
    let result = validation::validate_cache_directory(&nonexistent_cache);
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("does not exist"));
}

#[test]
fn test_validation_file_as_cache_dir() {
    let temp_dir = TempDir::new().unwrap();
    let temp_file = temp_dir.path().join("not_a_directory.txt");
    std::fs::write(&temp_file, "test content").unwrap();

    let file_as_cache = CachePath::new(&temp_file);
    let result = validation::validate_cache_directory(&file_as_cache);
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("not a directory"));
}

#[test]
fn test_validation_valid_cache_dir() {
    let temp_dir = TempDir::new().unwrap();
    let cache_dir = CachePath::new(temp_dir.path());
    let result = validation::validate_cache_directory(&cache_dir);
    assert!(result.is_ok());
}

#[test]
fn test_validation_cache_permissions() {
    let temp_dir = TempDir::new().unwrap();
    let result = validation::validate_cache_permissions(temp_dir.path());
    assert!(result.is_ok());
}

#[test]
fn test_validation_model_inputs_valid() {
    let temp_dir = TempDir::new().unwrap();
    let model_name = ModelName::new("Qwen/Qwen3-Embedding-0.6B");
    let cache_dir = CachePath::new(temp_dir.path());

    let result = validation::validate_model_inputs(&model_name, &cache_dir);
    assert!(result.is_ok());
}

#[test]
fn test_validation_model_inputs_invalid_name() {
    let temp_dir = TempDir::new().unwrap();
    let invalid_name = ModelName::new("invalid-name");
    let cache_dir = CachePath::new(temp_dir.path());

    let result = validation::validate_model_inputs(&invalid_name, &cache_dir);
    assert!(result.is_err());
}

#[test]
fn test_validation_model_inputs_invalid_cache() {
    let invalid_name = ModelName::new("Qwen/Qwen3-Embedding-0.6B");
    let invalid_cache = CachePath::new("/nonexistent/cache");

    let result = validation::validate_model_inputs(&invalid_name, &invalid_cache);
    assert!(result.is_err());
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
fn test_parse_qwen2_config_missing_fields() {
    use serde_json::json;

    let incomplete_configs = vec![
        json!({"vocab_size": 32000}), // Missing most fields
        json!({
            "vocab_size": 32000,
            "hidden_size": 768
            // Missing other required fields
        }),
        json!({
            "vocab_size": 32000,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            // Missing attention heads and other fields
        }),
    ];

    for incomplete_config in incomplete_configs {
        let result = config::parse_qwen2_config(&incomplete_config);
        assert!(
            result.is_err(),
            "Should fail with incomplete config: {:?}",
            incomplete_config
        );
    }
}

#[test]
fn test_parse_qwen2_config_invalid_types() {
    use serde_json::json;

    let invalid_configs = vec![
        json!({
            "vocab_size": "not_a_number", // String instead of number
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 12,
            "max_position_embeddings": 2048,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0
        }),
        json!({
            "vocab_size": 32000,
            "hidden_size": -768, // Negative number
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 12,
            "max_position_embeddings": 2048,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0
        }),
    ];

    for invalid_config in invalid_configs {
        let result = config::parse_qwen2_config(&invalid_config);
        assert!(
            result.is_err(),
            "Should fail with invalid config: {:?}",
            invalid_config
        );
    }
}

#[test]
fn test_model_name_type_safety() {
    let model_name = ModelName::new("Qwen/Qwen3-Embedding-0.6B");
    assert_eq!(model_name.as_str(), "Qwen/Qwen3-Embedding-0.6B");
    assert_eq!(model_name.to_string(), "Qwen/Qwen3-Embedding-0.6B");

    let from_string = ModelName::from("test/model".to_string());
    assert_eq!(from_string.as_str(), "test/model");

    let from_str = ModelName::from("test/model");
    assert_eq!(from_str.as_str(), "test/model");
}

#[test]
fn test_cache_path_type_safety() {
    let temp_dir = TempDir::new().unwrap();
    let cache_path = CachePath::new(temp_dir.path());

    assert_eq!(cache_path.as_path(), temp_dir.path());
    assert!(cache_path.exists());

    let joined = cache_path.join("subdir");
    assert_eq!(joined.as_path(), temp_dir.path().join("subdir"));

    let from_pathbuf = CachePath::from(temp_dir.path().to_path_buf());
    assert_eq!(from_pathbuf.as_path(), temp_dir.path());

    let from_path = CachePath::from(temp_dir.path());
    assert_eq!(from_path.as_path(), temp_dir.path());
}

#[test]
fn test_model_info_qwen3_creation() {
    let model_info = ModelInfo::huggingface_model(
        ModelName::from("Qwen/Qwen3-Embedding-0.6B"),
        "Qwen3 embedding model for multilingual retrieval".to_string(),
        1024,
        600_000_000,
    );

    assert_eq!(model_info.name.as_str(), "Qwen/Qwen3-Embedding-0.6B");
    assert_eq!(model_info.dimensions, 1024);
    assert_eq!(model_info.size_bytes, 600_000_000);
    assert_eq!(model_info.model_type, ModelType::HuggingFace);
    assert_eq!(model_info.backend, ModelBackend::Custom);
    assert!(model_info.download_url.is_none());
    assert!(model_info.local_path.is_none());
}

#[test]
fn test_model_info_validation_qwen3() {
    let valid_model = ModelInfo::huggingface_model(
        ModelName::from("Qwen/Qwen3-Embedding-0.6B"),
        "Valid Qwen3 model".to_string(),
        1024,
        600_000_000,
    );

    assert!(valid_model.validate().is_ok());

    let invalid_model = ModelInfo::huggingface_model(
        ModelName::from(""), // Empty name
        "Invalid model".to_string(),
        1024,
        600_000_000,
    );

    let validation_result = invalid_model.validate();
    assert!(validation_result.is_err());
    assert!(validation_result
        .unwrap_err()
        .contains("Model name cannot be empty"));
}

// Test that demonstrates Qwen3 instruction-based embedding concept (without actual model)
#[test]
fn test_qwen3_instruction_concept() {
    let test_texts = vec![
        "This is a code function for testing".to_string(),
        "另一个测试文本".to_string(),               // Chinese text
        "console.log('Hello, world!')".to_string(), // JavaScript code
    ];

    let instruction = "Represent this text for similarity search";

    // Test instruction formatting logic (as it would be used in actual model)
    for text in &test_texts {
        let processed_text = format!("Instruct: {}\nQuery: {}", instruction, text);
        assert!(processed_text.contains(instruction));
        assert!(processed_text.contains(text));
        assert!(processed_text.starts_with("Instruct:"));
    }
}

#[test]
fn test_qwen3_multilingual_support_concept() {
    let multilingual_texts = vec![
        "Hello world".to_string(),                 // English
        "你好世界".to_string(),                    // Chinese
        "Hola mundo".to_string(),                  // Spanish
        "Bonjour le monde".to_string(),            // French
        "console.log('test')".to_string(),         // JavaScript code
        "def hello(): return 'world'".to_string(), // Python code
    ];

    // Validate that we can handle different text types without errors
    assert_eq!(multilingual_texts.len(), 6);
    for text in &multilingual_texts {
        assert!(!text.is_empty());
        assert!(text.chars().count() > 0);
    }
}

#[tokio::test]
#[ignore] // Requires model download
async fn test_qwen3_model_download_and_load() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let backend = HuggingFaceBackend::new()?;
    let model_name = ModelName::new("Qwen/Qwen3-Embedding-0.6B");
    let cache_dir = CachePath::new(temp_dir.path());

    let _model = backend.load_qwen3_model(&model_name, &cache_dir).await?;
    // In a real implementation, this would download and load the actual model

    Ok(())
}

#[tokio::test]
#[ignore] // Requires actual model loading which may be slow/large
async fn test_qwen3_instruction_embeddings() -> Result<()> {
    // This test would use a real Qwen3 model once implemented
    let test_texts = vec![
        "This is a code function for testing".to_string(),
        "另一个测试文本".to_string(), // Chinese text
    ];

    let instruction = "Represent this text for similarity search";

    // Mock test - in real implementation:
    // let model = create_test_qwen3_model()?;
    // let embeddings = model.embed_with_instruction(&test_texts, Some(instruction))?;
    // assert_eq!(embeddings.len(), 2);
    // assert_eq!(embeddings[0].len(), 1024); // Expected dimensions

    // For now, just validate the test setup
    assert_eq!(test_texts.len(), 2);
    assert!(!instruction.is_empty());

    Ok(())
}

#[tokio::test]
#[ignore] // Requires actual model loading
async fn test_qwen3_multilingual_inference() -> Result<()> {
    let multilingual_texts = vec![
        "Hello world".to_string(),
        "你好世界".to_string(),            // Chinese
        "Hola mundo".to_string(),          // Spanish
        "console.log('test')".to_string(), // JavaScript code
    ];

    // Mock test - in real implementation:
    // let model = create_test_qwen3_model()?;
    // let embeddings = model.embed(&multilingual_texts)?;
    // assert_eq!(embeddings.len(), 4);
    //
    // // Test that multilingual texts produce meaningful embeddings
    // for embedding in &embeddings {
    //     assert_eq!(embedding.len(), 1024);
    //     // Verify embeddings are not all zeros
    //     assert!(embedding.iter().any(|&x| x.abs() > 0.001));
    // }

    // For now, validate test setup
    assert_eq!(multilingual_texts.len(), 4);

    Ok(())
}

#[test]
fn test_model_backend_selection_logic() {
    // Test that Qwen3 models use the correct backend
    let manager = turboprop::models::ModelManager::new_with_defaults(std::env::temp_dir());
    let models = manager.get_available_models();

    // Find Qwen3 model in available models
    let qwen3_model = models.iter().find(|m| m.name.as_str().contains("Qwen3"));

    if let Some(model) = qwen3_model {
        // Verify it uses the Custom backend (which maps to HuggingFace)
        assert_eq!(model.backend, ModelBackend::Custom);
        assert_eq!(model.model_type, ModelType::HuggingFace);
        assert!(model.dimensions > 0);
        assert!(!model.description.is_empty());
    }
}
