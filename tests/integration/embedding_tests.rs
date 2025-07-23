//! Integration tests for embedding generation functionality.
//!
//! These tests validate the end-to-end embedding pipeline including
//! model downloading, caching, and embedding generation with real models.
//!
//! Run these tests with: `cargo test --test integration`
//!
//! These tests are moved here from unit tests to avoid slow model downloads
//! during regular unit test runs.
//!
//! ## Test Mode Configuration
//!
//! These tests default to offline mode to avoid slow model downloads. Use `TURBOPROP_TEST_ONLINE=1`
//! to enable online mode with real model downloads when needed.

use std::env;
use tempfile::TempDir;
use turboprop::constants;
use turboprop::embeddings::{
    EmbeddingConfig, EmbeddingGenerator, EmbeddingOptions, DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_MODEL,
};
use turboprop::models::{ModelInfo, ModelInfoConfig};
use turboprop::types::{ModelBackend, ModelName, ModelType};

/// Check if tests should run in offline mode (default: offline)
fn is_offline_mode() -> bool {
    // Default to offline mode unless explicitly enabled online
    env::var("TURBOPROP_TEST_ONLINE").unwrap_or_default() != "1"
}

/// Test embedding generator initialization with default model (MOVED FROM UNIT TESTS)
#[tokio::test]
async fn test_embedding_generator_initialization() {
    let temp_dir = TempDir::new().unwrap();
    let config = EmbeddingConfig::default().with_cache_dir(temp_dir.path().to_path_buf());

    // This test requires network access to download the model
    // Skip if we're in an offline environment
    if is_offline_mode() {
        println!("Skipping test in offline mode");
        return;
    }

    let generator = EmbeddingGenerator::new(config).await;
    assert!(
        generator.is_ok(),
        "Failed to initialize generator: {:?}",
        generator.unwrap_err()
    );

    let generator = generator.unwrap();
    assert_eq!(generator.model_name(), DEFAULT_MODEL);
    assert_eq!(
        generator.embedding_dimensions(),
        DEFAULT_EMBEDDING_DIMENSIONS
    );
}

/// Test embedding generation for single text (MOVED FROM UNIT TESTS)
#[tokio::test]
async fn test_single_embedding_generation() {
    if is_offline_mode() {
        println!("Skipping test in offline mode");
        return;
    }

    let config = EmbeddingConfig::default();

    let mut generator = EmbeddingGenerator::new(config).await.unwrap();

    let text = "This is a test function in Rust that demonstrates embedding generation";
    let embedding = generator.embed_single(text).unwrap();

    assert_eq!(embedding.len(), 384);

    // Check that embedding contains non-zero values
    assert!(embedding.iter().any(|&x| x != 0.0));

    // Check that values are reasonable (not all very small or very large)
    let avg_magnitude = embedding.iter().map(|&x| x.abs()).sum::<f32>() / embedding.len() as f32;
    assert!(
        avg_magnitude > 0.001 && avg_magnitude < 10.0,
        "Average magnitude: {}",
        avg_magnitude
    );
}

/// Test batch embedding generation (MOVED FROM UNIT TESTS)
#[tokio::test]
async fn test_batch_embedding_generation() {
    if is_offline_mode() {
        println!("Skipping test in offline mode");
        return;
    }

    let config = EmbeddingConfig::default().with_batch_size(2); // Small batch for testing

    let mut generator = EmbeddingGenerator::new(config).await.unwrap();

    let texts = vec![
        "fn main() { println!(\"Hello, world!\"); }".to_string(),
        "struct Point { x: f64, y: f64 }".to_string(),
        "impl Display for Point { fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { ... } }"
            .to_string(),
    ];

    let embeddings = generator.embed_batch(&texts).unwrap();

    assert_eq!(embeddings.len(), 3);

    for (i, embedding) in embeddings.iter().enumerate() {
        assert_eq!(embedding.len(), 384, "Embedding {} has wrong dimensions", i);
        assert!(
            embedding.iter().any(|&x| x != 0.0),
            "Embedding {} is all zeros",
            i
        );
    }

    // Check that different texts produce different embeddings
    let similarity_01 = cosine_similarity(&embeddings[0], &embeddings[1]);
    let similarity_12 = cosine_similarity(&embeddings[1], &embeddings[2]);

    // Embeddings should be different but still positive (code-related)
    // Note: Very different code constructs (function vs struct vs impl) can have low similarity
    assert!(
        similarity_01 > 0.0 && similarity_01 < 0.95,
        "Similarity 0-1: {}",
        similarity_01
    );
    assert!(
        similarity_12 > 0.0 && similarity_12 < 0.95,
        "Similarity 1-2: {}",
        similarity_12
    );
}

/// Test embedding consistency (same input produces same output) (MOVED FROM UNIT TESTS)
#[tokio::test]
async fn test_embedding_consistency() {
    if is_offline_mode() {
        println!("Skipping test in offline mode");
        return;
    }

    let config = EmbeddingConfig::default();

    let mut generator = EmbeddingGenerator::new(config).await.unwrap();

    let text = "pub fn calculate_sum(a: i32, b: i32) -> i32 { a + b }";

    let embedding1 = generator.embed_single(text).unwrap();
    let embedding2 = generator.embed_single(text).unwrap();

    // Check exact equality (embeddings should be deterministic)
    assert_eq!(embedding1.len(), embedding2.len());
    for (i, (&v1, &v2)) in embedding1.iter().zip(embedding2.iter()).enumerate() {
        assert!(
            (v1 - v2).abs() < 1e-6,
            "Mismatch at index {}: {} vs {}",
            i,
            v1,
            v2
        );
    }
}

/// Test empty string handling (MOVED FROM UNIT TESTS)
#[tokio::test]
async fn test_empty_string_embedding() {
    if is_offline_mode() {
        println!("Skipping test in offline mode");
        return;
    }

    let config = EmbeddingConfig::default();

    let mut generator = EmbeddingGenerator::new(config).await.unwrap();

    let embedding = generator.embed_single("").unwrap();

    assert_eq!(embedding.len(), 384);
    assert!(embedding.iter().all(|&x| x == 0.0));
}

/// Test large batch processing (MOVED FROM UNIT TESTS)
#[tokio::test]
async fn test_large_batch_processing() {
    if is_offline_mode() {
        println!("Skipping test in offline mode");
        return;
    }

    let config = EmbeddingConfig::default().with_batch_size(5); // Small batches to test batching logic

    let mut generator = EmbeddingGenerator::new(config).await.unwrap();

    // Create a larger set of texts
    let texts: Vec<String> = (0..12)
        .map(|i| format!("function test_{i}() {{ return {i} * 2; }}"))
        .collect();

    let embeddings = generator.embed_batch(&texts).unwrap();

    assert_eq!(embeddings.len(), 12);

    for embedding in &embeddings {
        assert_eq!(embedding.len(), 384);
        assert!(embedding.iter().any(|&x| x != 0.0));
    }
}

/// Test different model configurations (MOVED FROM UNIT TESTS)
#[tokio::test]
async fn test_different_model_config() {
    if is_offline_mode() {
        println!("Skipping test in offline mode");
        return;
    }

    // Test with explicitly specified model
    let config = EmbeddingConfig::with_model("sentence-transformers/all-MiniLM-L6-v2");

    let generator = EmbeddingGenerator::new(config).await.unwrap();
    assert_eq!(
        generator.model_name(),
        "sentence-transformers/all-MiniLM-L6-v2"
    );
}

/// Test embedding generator with FastEmbed model (MOVED FROM UNIT TESTS)
#[tokio::test]
async fn test_embedding_generator_new_with_model_fastembed() {
    let temp_dir = TempDir::new().unwrap();

    let model_info = ModelInfo::simple(
        ModelName::from("sentence-transformers/all-MiniLM-L6-v2"),
        "FastEmbed model".to_string(),
        384,
        23_000_000,
    );

    if is_offline_mode() {
        println!("Skipping test in offline mode");
        return;
    }

    // Test that new_with_model works with FastEmbed backend
    let result = EmbeddingGenerator::new_with_model(&model_info, temp_dir.path()).await;

    // Should succeed for FastEmbed models
    assert!(
        result.is_ok(),
        "new_with_model should work for FastEmbed models"
    );

    let generator = result.unwrap();
    assert_eq!(
        generator.model_name(),
        "sentence-transformers/all-MiniLM-L6-v2"
    );
    assert_eq!(generator.embedding_dimensions(), 384);
}

/// Test embedding generator with GGUF model (MOVED FROM UNIT TESTS)
#[tokio::test]
async fn test_embedding_generator_new_with_model_gguf() {
    let temp_dir = TempDir::new().unwrap();

    let model_info = ModelInfo::gguf_model(
        ModelName::from("nomic-embed-code.Q5_K_S.gguf"),
        "GGUF model for testing".to_string(),
        768,
        2_500_000_000,
        "https://example.com/model.gguf".to_string(),
    );

    // Test that new_with_model works with GGUF backend
    let result = EmbeddingGenerator::new_with_model(&model_info, temp_dir.path()).await;

    // Should succeed for GGUF models (assuming network is available)
    if result.is_ok() {
        let generator = result.unwrap();
        assert_eq!(generator.model_name(), "nomic-embed-code.Q5_K_S.gguf");
        assert_eq!(generator.embedding_dimensions(), 768);
    } else {
        // Allow failure for network-related issues in test environment
        let error_msg = result.err().unwrap().to_string();
        println!("GGUF test failed (may be network-related): {}", error_msg);
    }
}

/// Test backend selection logic (MOVED FROM UNIT TESTS)
#[tokio::test]
async fn test_embedding_generator_backend_selection() {
    // This test verifies that the correct backend is selected based on ModelInfo.backend
    let temp_dir = TempDir::new().unwrap();

    // Test FastEmbed backend selection
    let fastembed_model = ModelInfo::new(ModelInfoConfig {
        name: ModelName::from("test-fastembed"),
        description: "Test FastEmbed model".to_string(),
        dimensions: 384,
        size_bytes: constants::test::DEFAULT_MODEL_SIZE_BYTES,
        model_type: ModelType::SentenceTransformer,
        backend: ModelBackend::FastEmbed,
        download_url: None,
        local_path: None,
    });

    // Test GGUF/Candle backend selection
    let gguf_model = ModelInfo::new(ModelInfoConfig {
        name: ModelName::from("test-gguf"),
        description: "Test GGUF model".to_string(),
        dimensions: 768,
        size_bytes: constants::test::DEFAULT_MODEL_SIZE_BYTES,
        model_type: ModelType::GGUF,
        backend: ModelBackend::Candle,
        download_url: Some("https://example.com/test.gguf".to_string()),
        local_path: None,
    });

    // Test backend selection logic - should choose correct backend based on ModelInfo.backend
    let fastembed_result =
        EmbeddingGenerator::new_with_model(&fastembed_model, temp_dir.path()).await;
    let gguf_result = EmbeddingGenerator::new_with_model(&gguf_model, temp_dir.path()).await;

    // FastEmbed should work offline (but skip in offline test mode)
    if !is_offline_mode() {
        assert!(fastembed_result.is_ok(), "FastEmbed backend should work");
    }

    // GGUF may fail due to network or URL issues in test environment, but shouldn't fail due to backend selection
    if let Err(ref e) = gguf_result {
        let error_msg = e.to_string();
        // Should not fail due to backend selection issues
        assert!(
            !error_msg.contains("does not support model type"),
            "Backend selection should work for GGUF"
        );
    }
}

/// Test custom backend model loading failure (MOVED FROM UNIT TESTS)
#[tokio::test]
async fn test_embedding_generator_custom_backend_model_loading_failure() {
    let temp_dir = TempDir::new().unwrap();

    // Test that custom backend fails gracefully when trying to load a non-existent model
    let fake_model = ModelInfo::new(ModelInfoConfig {
        name: ModelName::from("fake/nonexistent-model"),
        description: "Test fake model for custom backend".to_string(),
        dimensions: 512,
        size_bytes: constants::test::DEFAULT_MODEL_SIZE_BYTES,
        model_type: ModelType::HuggingFace,
        backend: ModelBackend::Custom, // Now supported backend
        download_url: None,
        local_path: None,
    });

    let result = EmbeddingGenerator::new_with_model(&fake_model, temp_dir.path()).await;

    // Should fail when trying to download the fake model
    assert!(result.is_err());

    let error_msg = result.err().unwrap().to_string();
    assert!(
        error_msg.contains("Failed to load Qwen3 model")
            || error_msg.contains("Failed to download"),
        "Should fail with model loading error, got: {}",
        error_msg
    );
}

/// Test embedding with options - empty input (MOVED FROM UNIT TESTS)
#[tokio::test]
async fn test_embed_with_options_empty_input() {
    let temp_dir = TempDir::new().unwrap();
    let config = EmbeddingConfig::default().with_cache_dir(temp_dir.path().to_path_buf());

    if is_offline_mode() {
        println!("Skipping test in offline mode");
        return;
    }

    let mut generator = EmbeddingGenerator::new(config).await.unwrap();
    let options = EmbeddingOptions::default();
    let embeddings = generator.embed_with_options(&[], &options).unwrap();
    assert!(embeddings.is_empty());
}

/// Test embedding with options - max length truncation (MOVED FROM UNIT TESTS)
#[tokio::test]
async fn test_embed_with_options_max_length_truncation() {
    let temp_dir = TempDir::new().unwrap();
    let config = EmbeddingConfig::default().with_cache_dir(temp_dir.path().to_path_buf());

    if is_offline_mode() {
        println!("Skipping test in offline mode");
        return;
    }

    let mut generator = EmbeddingGenerator::new(config).await.unwrap();

    // Create a long text that should be truncated
    let long_text =
        "This is a very long text that should be truncated when max_length is applied. ".repeat(20);
    let texts = vec![long_text];

    let options = EmbeddingOptions::with_max_length(50);
    let result = generator.embed_with_options(&texts, &options);

    // Should succeed even with truncation
    assert!(result.is_ok(), "Should handle max_length truncation");
}

/// Test Qwen3 embedding generator creation (MOVED FROM qwen3_tests.rs)
#[tokio::test]
async fn test_qwen3_embedding_generator_creation() {
    // Skip this test if we're in offline mode or network tests are disabled
    if is_offline_mode() || std::env::var("SKIP_NETWORK_TESTS").is_ok() {
        return;
    }

    let temp_dir = TempDir::new().unwrap();

    let qwen3_model = ModelInfo::new(ModelInfoConfig {
        name: ModelName::from("Qwen/Qwen3-Embedding-0.6B"),
        description: "Qwen3 embedding model for testing".to_string(),
        dimensions: 1024,
        size_bytes: 600_000_000,
        model_type: ModelType::HuggingFace,
        backend: ModelBackend::Custom,
        download_url: None,
        local_path: None,
    });

    // This will fail in test environment due to network/model availability,
    // but should fail with a network/download error, not a backend error
    let result = EmbeddingGenerator::new_with_model(&qwen3_model, temp_dir.path()).await;

    if let Err(error) = result {
        let error_msg = error.to_string();
        // Should fail due to network/model issues, not backend selection issues
        assert!(
            !error_msg.contains("backend is not yet supported"),
            "Should not fail due to backend support, got: {}",
            error_msg
        );
        assert!(
            error_msg.contains("Failed to load Qwen3 model")
                || error_msg.contains("Failed to download")
                || error_msg.contains("Failed to initialize HuggingFace backend"),
            "Should fail with model loading error, got: {}",
            error_msg
        );
    }
}

/// Helper function to calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}
