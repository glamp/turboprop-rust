//! Unit tests for GGUF backend functionality.
//!
//! These tests validate the GGUF backend without requiring large model downloads.
//! For tests that require actual GGUF model files, those are marked with #[ignore].
//!
//! Run these tests with: `cargo test gguf`

use anyhow::Result;
use std::fs::File;
use std::io::Write;
use tempfile::TempDir;
use turboprop::backends::gguf::{validate_gguf_file, GGUFBackend, GGUFConfig, GGUFDevice, GGUFEmbeddingModel};
use turboprop::models::{EmbeddingBackend, EmbeddingModel, ModelInfo, ModelInfoConfig};
use turboprop::types::{ModelBackend, ModelName, ModelType};

#[test]
fn test_gguf_backend_creation() -> Result<()> {
    let backend = GGUFBackend::new()?;
    assert!(matches!(backend.device(), candle_core::Device::Cpu));
    Ok(())
}

#[test]
fn test_gguf_backend_creation_with_config() -> Result<()> {
    let config = GGUFConfig::new()
        .with_device(GGUFDevice::Cpu)
        .with_context_length(1024)
        .with_batching(false);

    let backend = GGUFBackend::new_with_config(config.clone())?;
    assert_eq!(backend.config().device, GGUFDevice::Cpu);
    assert_eq!(backend.config().context_length, 1024);
    assert!(!backend.config().enable_batching);
    Ok(())
}

#[test]
fn test_gguf_backend_supports_model() -> Result<()> {
    let backend = GGUFBackend::new()?;

    assert!(backend.supports_model(&ModelType::GGUF));
    assert!(!backend.supports_model(&ModelType::SentenceTransformer));
    assert!(!backend.supports_model(&ModelType::HuggingFace));
    Ok(())
}

#[test]
fn test_gguf_backend_load_model_success() -> Result<()> {
    let backend = GGUFBackend::new()?;

    let model_info = ModelInfo::new(ModelInfoConfig {
        name: ModelName::from("nomic-embed-code.Q5_K_S.gguf"),
        description: "Test GGUF model".to_string(),
        dimensions: 768,
        size_bytes: 2_500_000_000,
        model_type: ModelType::GGUF,
        backend: ModelBackend::Candle,
        download_url: None,
        local_path: None,
    });

    let result = backend.load_model(&model_info);
    assert!(result.is_ok());

    let model = result.unwrap();
    assert_eq!(model.dimensions(), 768);
    assert_eq!(model.max_sequence_length(), 512); // Default context length
    Ok(())
}

#[test]
fn test_gguf_backend_load_model_unsupported_type() -> Result<()> {
    let backend = GGUFBackend::new()?;

    let model_info = ModelInfo::new(ModelInfoConfig {
        name: ModelName::from("sentence-transformer-model"),
        description: "Test sentence transformer model".to_string(),
        dimensions: 384,
        size_bytes: 23_000_000,
        model_type: ModelType::SentenceTransformer,
        backend: ModelBackend::FastEmbed,
        download_url: None,
        local_path: None,
    });

    let result = backend.load_model(&model_info);
    assert!(result.is_err());

    let error_message = result.err().unwrap().to_string();
    assert!(error_message.contains("does not support model type"));
    Ok(())
}

#[test]
fn test_gguf_embedding_model_creation() -> Result<()> {
    let model = GGUFEmbeddingModel::new(
        "test-model".to_string(),
        768,
        candle_core::Device::Cpu,
    )?;

    assert_eq!(model.dimensions(), 768);
    assert_eq!(model.max_sequence_length(), 512);
    Ok(())
}

#[test]
fn test_gguf_embedding_model_creation_with_config() -> Result<()> {
    let config = GGUFConfig::new()
        .with_context_length(1024)
        .with_batching(false);

    let model = GGUFEmbeddingModel::new_with_config(
        "test-model".to_string(),
        768,
        candle_core::Device::Cpu,
        config,
    )?;

    assert_eq!(model.dimensions(), 768);
    assert_eq!(model.max_sequence_length(), 1024); // Should use config context_length
    assert!(!model.config().enable_batching);
    Ok(())
}

#[test]
fn test_gguf_embedding_model_embed_single() -> Result<()> {
    let model = GGUFEmbeddingModel::new(
        "test-model".to_string(),
        768,
        candle_core::Device::Cpu,
    )?;

    let texts = vec!["Hello, world!".to_string()];
    let result = model.embed(&texts);

    assert!(result.is_ok());
    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 1);
    assert_eq!(embeddings[0].len(), 768);

    // Verify embeddings contain actual values
    assert!(embeddings[0].iter().any(|&x| x != 0.0));
    Ok(())
}

#[test]
fn test_gguf_embedding_model_embed_batch() -> Result<()> {
    let model = GGUFEmbeddingModel::new(
        "test-model".to_string(),
        768,
        candle_core::Device::Cpu,
    )?;

    let texts = vec![
        "First text".to_string(),
        "Second text".to_string(),
        "Third text".to_string(),
    ];
    let result = model.embed(&texts);

    assert!(result.is_ok());
    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 3);
    
    for (i, embedding) in embeddings.iter().enumerate() {
        assert_eq!(embedding.len(), 768);
        assert!(embedding.iter().any(|&x| x != 0.0), "Embedding {} should not be all zeros", i);
    }

    // Different texts should produce different embeddings (at least slightly)
    assert_ne!(embeddings[0], embeddings[1]);
    assert_ne!(embeddings[1], embeddings[2]);
    Ok(())
}

#[test]
fn test_gguf_embedding_model_embed_empty() -> Result<()> {
    let model = GGUFEmbeddingModel::new(
        "test-model".to_string(),
        768,
        candle_core::Device::Cpu,
    )?;

    let texts: Vec<String> = vec![];
    let result = model.embed(&texts);

    assert!(result.is_ok());
    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 0);
    Ok(())
}

#[test]
fn test_gguf_embedding_model_embed_empty_text_error() -> Result<()> {
    let model = GGUFEmbeddingModel::new(
        "test-model".to_string(),
        768,
        candle_core::Device::Cpu,
    )?;

    let texts = vec![
        "Valid text".to_string(),
        "".to_string(), // Empty string should cause error
        "Another valid text".to_string(),
    ];
    let result = model.embed(&texts);

    assert!(result.is_err());
    let error_message = result.err().unwrap().to_string();
    assert!(error_message.contains("Empty text found at index 1"));
    Ok(())
}

#[test]
fn test_gguf_embedding_model_deterministic() -> Result<()> {
    let model = GGUFEmbeddingModel::new(
        "test-model".to_string(),
        768,
        candle_core::Device::Cpu,
    )?;

    let texts = vec!["Deterministic test text".to_string()];
    
    let result1 = model.embed(&texts)?;
    let result2 = model.embed(&texts)?;

    assert_eq!(result1.len(), result2.len());
    assert_eq!(result1[0], result2[0]); // Should be deterministic
    Ok(())
}

#[test]
fn test_validate_gguf_file_nonexistent() {
    let temp_dir = TempDir::new().unwrap();
    let fake_path = temp_dir.path().join("nonexistent.gguf");

    let result = validate_gguf_file(&fake_path);
    assert!(result.is_err());

    let error_msg = result.err().unwrap().to_string();
    assert!(error_msg.contains("File does not exist"));
}

#[test]
fn test_validate_gguf_file_wrong_extension() {
    let temp_dir = TempDir::new().unwrap();
    let wrong_ext_path = temp_dir.path().join("model.bin");

    // Create a file with wrong extension
    File::create(&wrong_ext_path).unwrap();

    let result = validate_gguf_file(&wrong_ext_path);
    assert!(result.is_err());

    let error_msg = result.err().unwrap().to_string();
    assert!(error_msg.contains("does not have .gguf extension"));
}

#[test]
fn test_validate_gguf_file_too_small() {
    let temp_dir = TempDir::new().unwrap();
    let small_file_path = temp_dir.path().join("small.gguf");

    // Create a file that's too small
    let mut file = File::create(&small_file_path).unwrap();
    file.write_all(b"GGUF").unwrap(); // Only 4 bytes, need at least 12

    let result = validate_gguf_file(&small_file_path);
    assert!(result.is_err());

    let error_msg = result.err().unwrap().to_string();
    assert!(error_msg.contains("too small to be a valid GGUF model"));
}

#[test]
fn test_validate_gguf_file_invalid_magic() {
    let temp_dir = TempDir::new().unwrap();
    let invalid_magic_path = temp_dir.path().join("invalid.gguf");

    // Create a file with invalid magic header
    let mut file = File::create(&invalid_magic_path).unwrap();
    file.write_all(b"FAKE").unwrap(); // Wrong magic
    file.write_all(&[1, 0, 0, 0]).unwrap(); // Version 1
    file.write_all(&[0, 0, 0, 0]).unwrap(); // Extra bytes to meet minimum size

    let result = validate_gguf_file(&invalid_magic_path);
    assert!(result.is_err());

    let error_msg = result.err().unwrap().to_string();
    assert!(error_msg.contains("Invalid GGUF magic header"));
}

#[test]
fn test_validate_gguf_file_unsupported_version() {
    let temp_dir = TempDir::new().unwrap();
    let unsupported_version_path = temp_dir.path().join("unsupported.gguf");

    // Create a file with unsupported version
    let mut file = File::create(&unsupported_version_path).unwrap();
    file.write_all(b"GGUF").unwrap(); // Correct magic
    file.write_all(&[99, 0, 0, 0]).unwrap(); // Version 99 (unsupported)
    file.write_all(&[0, 0, 0, 0]).unwrap(); // Extra bytes

    let result = validate_gguf_file(&unsupported_version_path);
    assert!(result.is_err());

    let error_msg = result.err().unwrap().to_string();
    assert!(error_msg.contains("Unsupported GGUF version: 99"));
}

#[test]
fn test_validate_gguf_file_valid() {
    let temp_dir = TempDir::new().unwrap();
    let valid_gguf_path = temp_dir.path().join("valid.gguf");

    // Create a valid GGUF file header
    let mut file = File::create(&valid_gguf_path).unwrap();
    file.write_all(b"GGUF").unwrap(); // Correct magic
    file.write_all(&[2, 0, 0, 0]).unwrap(); // Version 2 (supported)
    file.write_all(&[0, 0, 0, 0]).unwrap(); // Extra bytes to meet minimum size

    let result = validate_gguf_file(&valid_gguf_path);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_config_default() {
    let config = GGUFConfig::default();
    assert_eq!(config.device, GGUFDevice::Cpu);
    assert_eq!(config.context_length, 512);
    assert!(config.enable_batching);
    assert_eq!(config.gpu_layers, 0);
    assert!(config.memory_limit_bytes.is_none());
    assert!(config.cpu_threads.is_none());
}

#[test]
fn test_gguf_config_builder_pattern() {
    let config = GGUFConfig::new()
        .with_device(GGUFDevice::Cuda)
        .with_memory_limit(2048 * 1024 * 1024) // 2GB
        .with_context_length(1024)
        .with_batching(false)
        .with_gpu_layers(32)
        .with_cpu_threads(8);

    assert_eq!(config.device, GGUFDevice::Cuda);
    assert_eq!(config.memory_limit_bytes, Some(2048 * 1024 * 1024));
    assert_eq!(config.context_length, 1024);
    assert!(!config.enable_batching);
    assert_eq!(config.gpu_layers, 32);
    assert_eq!(config.cpu_threads, Some(8));
}

#[test]
fn test_gguf_config_parse_memory_limit() -> Result<()> {
    assert_eq!(
        GGUFConfig::parse_memory_limit("2GB")?,
        2 * 1024 * 1024 * 1024
    );
    assert_eq!(
        GGUFConfig::parse_memory_limit("512MB")?,
        512 * 1024 * 1024
    );
    assert_eq!(GGUFConfig::parse_memory_limit("1024B")?, 1024);
    assert_eq!(
        GGUFConfig::parse_memory_limit("1.5GB")?,
        (1.5 * 1024.0 * 1024.0 * 1024.0) as u64
    );

    assert!(GGUFConfig::parse_memory_limit("invalid").is_err());
    assert!(GGUFConfig::parse_memory_limit("2TB").is_err()); // Unsupported unit
    Ok(())
}

#[test]
fn test_gguf_config_parse_device() -> Result<()> {
    assert_eq!(GGUFConfig::parse_device("cpu")?, GGUFDevice::Cpu);
    assert_eq!(GGUFConfig::parse_device("GPU")?, GGUFDevice::Gpu);
    assert_eq!(GGUFConfig::parse_device("cuda")?, GGUFDevice::Cuda);
    assert_eq!(GGUFConfig::parse_device("METAL")?, GGUFDevice::Metal);

    assert!(GGUFConfig::parse_device("invalid").is_err());
    Ok(())
}

// Test loading model from local path (requires valid GGUF file)
#[test]
fn test_gguf_embedding_model_load_from_path_invalid() {
    let temp_dir = TempDir::new().unwrap();
    let invalid_path = temp_dir.path().join("invalid.gguf");

    // Create invalid GGUF file
    let mut file = File::create(&invalid_path).unwrap();
    file.write_all(b"INVALID").unwrap();

    let model_info = ModelInfo::new(ModelInfoConfig {
        name: ModelName::from("test-model.gguf"),
        description: "Test GGUF model".to_string(),
        dimensions: 768,
        size_bytes: 1000,
        model_type: ModelType::GGUF,
        backend: ModelBackend::Candle,
        download_url: None,
        local_path: Some(invalid_path.clone()),
    });

    let result = GGUFEmbeddingModel::load_from_path(&invalid_path, &model_info);
    assert!(result.is_err());
}

#[test]
#[ignore] // Requires large model download
fn test_gguf_model_download_and_load() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let model_info = ModelInfo::gguf_model(
        ModelName::from("nomic-embed-code.Q5_K_S.gguf"),
        "Test GGUF model".to_string(),
        768,
        2_500_000_000,
        "https://huggingface.co/nomic-ai/nomic-embed-code-GGUF/resolve/main/nomic-embed-code.Q5_K_S.gguf".to_string(),
    );
    
    let manager = turboprop::models::ModelManager::new_with_defaults(temp_dir.path());
    let model_path = tokio::runtime::Runtime::new()?.block_on(async {
        manager.download_gguf_model(&model_info).await
    })?;
    
    assert!(model_path.exists());
    assert!(model_path.is_file());
    
    let backend = GGUFBackend::new()?;
    let _model = backend.load_model(&model_info)?;
    
    Ok(())
}

#[test]
#[ignore] // Requires actual model loading which may be slow
fn test_gguf_embedding_generation_with_real_model() -> Result<()> {
    // This test would use a small test GGUF model
    // Implementation depends on test data availability
    let _test_texts = vec![
        "function calculateSum(a, b) { return a + b; }".to_string(),
        "def process_data(data): return data.strip()".to_string(),
    ];
    
    // When real model loading is implemented:
    // let model = create_test_gguf_model()?;
    // let embeddings = model.embed(&test_texts)?;
    // assert_eq!(embeddings.len(), 2);
    // assert_eq!(embeddings[0].len(), 768); // Expected dimensions
    
    Ok(())
}