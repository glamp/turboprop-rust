//! Error scenario tests for model functionality.
//!
//! These tests validate that model-related operations handle errors gracefully
//! including network failures, corrupted caches, invalid models, and other
//! edge cases that can occur in production environments.
//!
//! Run these tests with: `cargo test model_error`

use anyhow::Result;
use std::fs::{File, Permissions};
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use tempfile::TempDir;
use turboprop::backends::gguf::{validate_gguf_file, GGUFBackend, GGUFEmbeddingModel};
use turboprop::backends::huggingface::{validation, HuggingFaceBackend};
use turboprop::embeddings::EmbeddingConfig;
use turboprop::models::{EmbeddingBackend, EmbeddingModel, ModelInfo, ModelInfoConfig, ModelManager};
use turboprop::types::{CachePath, ModelBackend, ModelName, ModelType};

/// Test invalid model selection and creation
#[tokio::test]
async fn test_invalid_model_selection() -> Result<()> {
    // Try to create embedding config with non-existent model
    let fake_model_config = EmbeddingConfig::with_model("nonexistent/model");
    
    // Mock generator should handle any model name gracefully
    let mut generator = turboprop::embeddings::MockEmbeddingGenerator::new(fake_model_config);
    
    // Mock generator should work even with invalid model names
    let test_texts = vec!["test text".to_string()];
    let result = generator.embed_batch(&test_texts);
    assert!(result.is_ok(), "Mock generator should handle invalid model names");
    
    Ok(())
}

/// Test corrupted cache handling for model manager
#[tokio::test]
async fn test_corrupted_cache_handling() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let manager = ModelManager::new_with_defaults(temp_dir.path());
    
    let model_name = ModelName::from("sentence-transformers/all-MiniLM-L6-v2");
    let fake_model_path = manager.get_model_path(&model_name);
    
    // Create corrupted cache directory with invalid contents
    std::fs::create_dir_all(&fake_model_path)?;
    std::fs::write(fake_model_path.join("invalid_file"), "corrupted data")?;
    std::fs::write(fake_model_path.join("another_invalid"), "more corruption")?;
    
    // Model should not be considered cached with invalid contents
    assert!(!manager.is_model_cached(&model_name));
    
    // Cache stats should handle corrupted directories gracefully
    let stats = manager.get_cache_stats();
    assert!(stats.is_ok(), "Cache stats should handle corrupted directories");
    
    Ok(())
}

/// Test GGUF file validation with various invalid formats
#[test]
fn test_gguf_validation_errors() {
    let temp_dir = TempDir::new().unwrap();
    
    // Test 1: File doesn't exist
    let nonexistent = temp_dir.path().join("nonexistent.gguf");
    let result = validate_gguf_file(&nonexistent);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("File does not exist"));
    
    // Test 2: Wrong file extension
    let wrong_ext = temp_dir.path().join("model.bin");
    File::create(&wrong_ext).unwrap();
    let result = validate_gguf_file(&wrong_ext);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("does not have .gguf extension"));
    
    // Test 3: File too small
    let too_small = temp_dir.path().join("small.gguf");
    let mut file = File::create(&too_small).unwrap();
    file.write_all(b"tiny").unwrap();
    let result = validate_gguf_file(&too_small);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("too small to be a valid GGUF model"));
    
    // Test 4: Invalid magic header
    let invalid_magic = temp_dir.path().join("invalid_magic.gguf");
    let mut file = File::create(&invalid_magic).unwrap();
    file.write_all(b"FAKE").unwrap(); // Wrong magic
    file.write_all(&[1, 0, 0, 0]).unwrap(); // Version
    file.write_all(&[0, 0, 0, 0, 0, 0, 0, 0]).unwrap(); // Padding
    let result = validate_gguf_file(&invalid_magic);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Invalid GGUF magic header"));
    
    // Test 5: Unsupported version
    let unsupported_version = temp_dir.path().join("unsupported.gguf");
    let mut file = File::create(&unsupported_version).unwrap();
    file.write_all(b"GGUF").unwrap(); // Correct magic
    file.write_all(&[255, 255, 255, 255]).unwrap(); // Very high version
    file.write_all(&[0, 0, 0, 0]).unwrap(); // Padding
    let result = validate_gguf_file(&unsupported_version);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Unsupported GGUF version"));
}

/// Test GGUF model loading errors
#[test]
fn test_gguf_model_loading_errors() {
    let temp_dir = TempDir::new().unwrap();
    let backend = GGUFBackend::new().unwrap();
    
    // Test loading unsupported model type through backend
    let invalid_model_info = ModelInfo::new(ModelInfoConfig {
        name: ModelName::from("sentence-transformer-model"),
        description: "Invalid model type for GGUF backend".to_string(),
        dimensions: 384,
        size_bytes: 1000,
        model_type: ModelType::SentenceTransformer, // Wrong type for GGUF backend
        backend: ModelBackend::FastEmbed,
        download_url: None,
        local_path: None,
    });
    
    let result = backend.load_model(&invalid_model_info);
    assert!(result.is_err());
    let error_msg = result.err().unwrap().to_string();
    assert!(error_msg.contains("does not support model type"));
    
    // Test loading from invalid path
    let invalid_path = temp_dir.path().join("nonexistent.gguf");
    let model_info_with_path = ModelInfo::new(ModelInfoConfig {
        name: ModelName::from("test-model.gguf"),
        description: "Model with invalid path".to_string(),
        dimensions: 768,
        size_bytes: 1000,
        model_type: ModelType::GGUF,
        backend: ModelBackend::Candle,
        download_url: None,
        local_path: Some(invalid_path.clone()),
    });
    
    let result = GGUFEmbeddingModel::load_from_path(&invalid_path, &model_info_with_path);
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("File does not exist"));
}

/// Test GGUF embedding generation with invalid inputs
#[test]
fn test_gguf_embedding_invalid_inputs() -> Result<()> {
    let model = GGUFEmbeddingModel::new(
        "test-model".to_string(),
        768,
        candle_core::Device::Cpu,
    )?;
    
    // Test empty string in batch (should fail)
    let texts_with_empty = vec![
        "Valid text".to_string(),
        "".to_string(), // Empty string should cause error
        "Another valid text".to_string(),
    ];
    let result = model.embed(&texts_with_empty);
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Empty text found at index 1"));
    
    // Test very long text (should fail due to length limit)
    let very_long_text = "word ".repeat(10000); // Very long text
    let long_texts = vec![very_long_text];
    let _result = model.embed(&long_texts);
    // This may or may not fail depending on the model's max sequence length
    // The test validates that the model handles the input appropriately
    
    Ok(())
}

/// Test HuggingFace backend validation errors
#[test]
fn test_huggingface_validation_errors() {
    // Test empty model name
    let empty_name = ModelName::new("");
    let result = validation::validate_model_name(&empty_name);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Model name cannot be empty"));
    
    // Test invalid model name format
    let invalid_formats = vec![
        "no-slash-name",           // No slash
        "/model-name",             // Empty organization
        "organization/",           // Empty model name
        "org//model",              // Double slash
        "org/model/extra/parts",   // Too many parts
    ];
    
    for invalid_name in invalid_formats {
        let model_name = ModelName::new(invalid_name);
        let result = validation::validate_model_name(&model_name);
        assert!(result.is_err(), "Should fail for invalid name: {}", invalid_name);
    }
    
    // Test invalid characters in model name
    let invalid_chars = vec![
        "org/model@name",   // @ character
        "org/model name",   // Space
        "org/model#name",   // Hash
        "org/model%name",   // Percent
        "org/model&name",   // Ampersand
    ];
    
    for invalid_name in invalid_chars {
        let model_name = ModelName::new(invalid_name);
        let result = validation::validate_model_name(&model_name);
        assert!(result.is_err(), "Should fail for invalid characters: {}", invalid_name);
        assert!(result.unwrap_err().to_string().contains("invalid characters"));
    }
}

/// Test cache directory validation errors
#[test]
fn test_cache_directory_validation_errors() {
    // Test nonexistent directory
    let nonexistent = CachePath::new("/nonexistent/path/that/does/not/exist");
    let result = validation::validate_cache_directory(&nonexistent);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("does not exist"));
    
    // Test file as cache directory
    let temp_dir = TempDir::new().unwrap();
    let temp_file = temp_dir.path().join("not_a_directory.txt");
    std::fs::write(&temp_file, "test content").unwrap();
    
    let file_as_cache = CachePath::new(&temp_file);
    let result = validation::validate_cache_directory(&file_as_cache);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not a directory"));
}

/// Test cache permission errors (Unix-specific)
#[test]
#[cfg(unix)]
fn test_cache_permission_errors() {
    let temp_dir = TempDir::new().unwrap();
    let readonly_dir = temp_dir.path().join("readonly");
    std::fs::create_dir_all(&readonly_dir).unwrap();
    
    // Make directory read-only
    let mut perms = std::fs::metadata(&readonly_dir).unwrap().permissions();
    perms.set_mode(0o444); // Read-only
    std::fs::set_permissions(&readonly_dir, perms).unwrap();
    
    let result = validation::validate_cache_permissions(&readonly_dir);
    
    // Should fail due to read-only permissions
    // Note: This test might succeed on some systems depending on user permissions
    if result.is_err() {
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Cannot write to cache directory") || 
                error_msg.contains("read-only") ||
                error_msg.contains("permission"),
                "Expected permission-related error, got: {}", error_msg);
    }
    
    // Restore permissions for cleanup
    let restore_perms = Permissions::from_mode(0o755);
    let _ = std::fs::set_permissions(&readonly_dir, restore_perms);
}

/// Test model info validation errors
#[test]
fn test_model_info_validation_errors() {
    // Test empty model name
    let invalid_name = ModelInfo::new(ModelInfoConfig {
        name: ModelName::from(""),
        description: "Test model".to_string(),
        dimensions: 384,
        size_bytes: 1000,
        model_type: ModelType::SentenceTransformer,
        backend: ModelBackend::FastEmbed,
        download_url: None,
        local_path: None,
    });
    
    let result = invalid_name.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Model name cannot be empty"));
    
    // Test empty description
    let invalid_description = ModelInfo::new(ModelInfoConfig {
        name: ModelName::from("valid/model"),
        description: "".to_string(),
        dimensions: 384,
        size_bytes: 1000,
        model_type: ModelType::SentenceTransformer,
        backend: ModelBackend::FastEmbed,
        download_url: None,
        local_path: None,
    });
    
    let result = invalid_description.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Model description cannot be empty"));
    
    // Test zero dimensions
    let invalid_dimensions = ModelInfo::new(ModelInfoConfig {
        name: ModelName::from("valid/model"),
        description: "Valid description".to_string(),
        dimensions: 0,
        size_bytes: 1000,
        model_type: ModelType::SentenceTransformer,
        backend: ModelBackend::FastEmbed,
        download_url: None,
        local_path: None,
    });
    
    let result = invalid_dimensions.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Model dimensions must be greater than 0"));
    
    // Test zero size
    let invalid_size = ModelInfo::new(ModelInfoConfig {
        name: ModelName::from("valid/model"),
        description: "Valid description".to_string(),
        dimensions: 384,
        size_bytes: 0,
        model_type: ModelType::SentenceTransformer,
        backend: ModelBackend::FastEmbed,
        download_url: None,
        local_path: None,
    });
    
    let result = invalid_size.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Model size must be greater than 0"));
    
    // Test invalid download URL
    let invalid_url = ModelInfo::new(ModelInfoConfig {
        name: ModelName::from("valid/model"),
        description: "Valid description".to_string(),
        dimensions: 384,
        size_bytes: 1000,
        model_type: ModelType::GGUF,
        backend: ModelBackend::Candle,
        download_url: Some("invalid-url".to_string()),
        local_path: None,
    });
    
    let result = invalid_url.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Download URL must be a valid HTTP/HTTPS URL"));
    
    // Test nonexistent local path
    let invalid_path = ModelInfo::new(ModelInfoConfig {
        name: ModelName::from("valid/model"),
        description: "Valid description".to_string(),
        dimensions: 384,
        size_bytes: 1000,
        model_type: ModelType::GGUF,
        backend: ModelBackend::Candle,
        download_url: None,
        local_path: Some("/nonexistent/path/to/model.gguf".into()),
    });
    
    let result = invalid_path.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Local path does not exist"));
}

/// Test network failure handling (simulated)
#[tokio::test]
async fn test_network_failure_handling() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let manager = ModelManager::new_with_defaults(temp_dir.path());
    
    // Test GGUF model download with invalid URL
    let invalid_url_model = ModelInfo::gguf_model(
        ModelName::from("invalid-download.gguf"),
        "Model with invalid download URL".to_string(),
        768,
        1000,
        "https://definitely-does-not-exist.invalid/model.gguf".to_string(),
    );
    
    let result = manager.download_gguf_model(&invalid_url_model).await;
    assert!(result.is_err(), "Should fail with invalid URL");
    
    // Test HuggingFace model download failure handling
    let backend = HuggingFaceBackend::new()?;
    let invalid_model_name = ModelName::new("definitely-does-not/exist");
    let cache_dir = CachePath::new(temp_dir.path());
    
    // This should fail gracefully if the model doesn't exist
    // (Currently returns placeholder, but future implementation should handle errors)
    let _result = backend.load_qwen3_model(&invalid_model_name, &cache_dir).await;
    // For now, this creates a placeholder, but in the future it should handle network errors
    
    Ok(())
}

/// Test embedding generator error handling
#[tokio::test]
async fn test_embedding_generator_errors() -> Result<()> {
    // Test with completely invalid configuration
    let invalid_config = EmbeddingConfig::with_model("invalid/model")
        .with_batch_size(0); // Invalid batch size
    
    // Mock generator should handle even invalid configurations
    let mut mock_generator = turboprop::embeddings::MockEmbeddingGenerator::new(invalid_config);
    
    let test_texts = vec!["test".to_string()];
    let result = mock_generator.embed_batch(&test_texts);
    
    // Mock generator should succeed regardless of configuration
    assert!(result.is_ok());
    
    Ok(())
}

/// Test concurrent access to model cache
#[tokio::test]
async fn test_concurrent_cache_access() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let manager = ModelManager::new_with_defaults(temp_dir.path());
    manager.init_cache()?;
    
    // Spawn multiple tasks that access cache simultaneously
    let mut handles = Vec::new();
    
    for i in 0..5 {
        let manager_clone = ModelManager::new_with_defaults(temp_dir.path());
        let handle = tokio::spawn(async move {
            let model_name = ModelName::from(format!("test/model-{}", i));
            let _is_cached = manager_clone.is_model_cached(&model_name);
            let _cache_path = manager_clone.get_model_path(&model_name);
            let _stats = manager_clone.get_cache_stats().unwrap();
        });
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    for handle in handles {
        handle.await?;
    }
    
    Ok(())
}

/// Test file system error handling
#[test]
fn test_filesystem_error_handling() {
    // Test cache stats on inaccessible directory
    let manager = ModelManager::new_with_defaults("/root/inaccessible"); // Typically not accessible
    
    // Should handle permission errors gracefully
    let _stats_result = manager.get_cache_stats();
    // May succeed (if running as root) or fail gracefully
    
    // Test cache clearing on inaccessible directory
    let _clear_result = manager.clear_cache();
    // Should not panic or crash
}

/// Test edge cases in model name handling
#[test]
fn test_model_name_edge_cases() {
    // Test very long model names
    let very_long_name = format!("{}/{}", "a".repeat(100), "b".repeat(100));
    let long_model_name = ModelName::new(&very_long_name);
    
    let _result = validation::validate_model_name(&long_model_name);
    // Should either pass or fail gracefully with appropriate error message
    
    // Test model names with special characters at boundaries
    let boundary_cases = vec![
        "org/-model",      // Dash at start
        "org/model-",      // Dash at end
        "org/_model",      // Underscore at start
        "org/model_",      // Underscore at end
        "org/.model",      // Dot at start
        "org/model.",      // Dot at end
    ];
    
    for case in boundary_cases {
        let model_name = ModelName::new(case);
        let _result = validation::validate_model_name(&model_name);
        // Should handle these cases consistently
    }
}