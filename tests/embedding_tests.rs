//! Integration tests for embedding generation functionality.
//!
//! These tests validate the end-to-end embedding pipeline including
//! model downloading, caching, and embedding generation with real models.

use std::path::PathBuf;
use tempfile::TempDir;
use turboprop::config::TurboPropConfig;
use turboprop::embeddings::{EmbeddingConfig, EmbeddingGenerator};
use turboprop::models::{CacheStats, ModelManager};

/// Test embedding generator initialization with default model
#[tokio::test]
async fn test_embedding_generator_initialization() {
    // Skip if we're in offline mode
    if std::env::var("OFFLINE_TESTS").is_ok() {
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let config = EmbeddingConfig::default().with_cache_dir(temp_dir.path().join("models"));

    let result = EmbeddingGenerator::new(config).await;
    assert!(
        result.is_ok(),
        "Failed to initialize embedding generator: {:?}",
        result.unwrap_err()
    );

    let generator = result.unwrap();
    assert_eq!(
        generator.model_name(),
        "sentence-transformers/all-MiniLM-L6-v2"
    );
    assert_eq!(generator.embedding_dimensions(), 384);
}

/// Test embedding generation for single text
#[tokio::test]
async fn test_single_embedding_generation() {
    if std::env::var("OFFLINE_TESTS").is_ok() {
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let config = EmbeddingConfig::default().with_cache_dir(temp_dir.path().join("models"));

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

/// Test batch embedding generation
#[tokio::test]
async fn test_batch_embedding_generation() {
    if std::env::var("OFFLINE_TESTS").is_ok() {
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let config = EmbeddingConfig::default()
        .with_cache_dir(temp_dir.path().join("models"))
        .with_batch_size(2); // Small batch for testing

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

/// Test embedding consistency (same input produces same output)
#[tokio::test]
async fn test_embedding_consistency() {
    if std::env::var("OFFLINE_TESTS").is_ok() {
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let config = EmbeddingConfig::default().with_cache_dir(temp_dir.path().join("models"));

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

/// Test empty string handling
#[tokio::test]
async fn test_empty_string_embedding() {
    if std::env::var("OFFLINE_TESTS").is_ok() {
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let config = EmbeddingConfig::default().with_cache_dir(temp_dir.path().join("models"));

    let mut generator = EmbeddingGenerator::new(config).await.unwrap();

    let embedding = generator.embed_single("").unwrap();

    assert_eq!(embedding.len(), 384);
    assert!(embedding.iter().all(|&x| x == 0.0));
}

/// Test model manager cache functionality
#[test]
fn test_model_manager_cache_operations() {
    let temp_dir = TempDir::new().unwrap();
    let manager = ModelManager::new(temp_dir.path());

    // Test cache initialization
    assert!(manager.init_cache().is_ok());
    assert!(temp_dir.path().exists());

    // Test model path generation
    let model_path = manager.get_model_path("sentence-transformers/all-MiniLM-L6-v2");
    let expected_name = "sentence-transformers_all-MiniLM-L6-v2";
    assert!(model_path
        .file_name()
        .unwrap()
        .to_string_lossy()
        .contains(expected_name));

    // Test model not cached (initially)
    assert!(!manager.is_model_cached("sentence-transformers/all-MiniLM-L6-v2"));

    // Test cache stats
    let stats = manager.get_cache_stats().unwrap();
    assert_eq!(stats.model_count, 0);
    assert_eq!(stats.total_size_bytes, 0);

    // Test cache clearing
    assert!(manager.clear_cache().is_ok());
}

/// Test model information
#[test]
fn test_model_information() {
    let models = ModelManager::get_available_models();

    assert!(!models.is_empty());

    let default_model = models
        .iter()
        .find(|m| m.name == ModelManager::default_model());
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

/// Test large batch processing
#[tokio::test]
async fn test_large_batch_processing() {
    if std::env::var("OFFLINE_TESTS").is_ok() {
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let config = EmbeddingConfig::default()
        .with_cache_dir(temp_dir.path().join("models"))
        .with_batch_size(5); // Small batches to test batching logic

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

/// Test different model configurations
#[tokio::test]
async fn test_different_model_config() {
    if std::env::var("OFFLINE_TESTS").is_ok() {
        return;
    }

    let temp_dir = TempDir::new().unwrap();

    // Test with explicitly specified model
    let config = EmbeddingConfig::with_model("sentence-transformers/all-MiniLM-L6-v2")
        .with_cache_dir(temp_dir.path().join("models"));

    let generator = EmbeddingGenerator::new(config).await.unwrap();
    assert_eq!(
        generator.model_name(),
        "sentence-transformers/all-MiniLM-L6-v2"
    );
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
