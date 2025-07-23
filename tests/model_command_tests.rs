//! Tests for model command functionality.
//!
//! These tests verify the model management commands work correctly
//! including listing, downloading, showing info, and clearing models.

use tempfile::TempDir;
use turboprop::cli::ModelCommands;
use turboprop::commands::handle_model_command;
use turboprop::models::ModelManager;

#[tokio::test]
async fn test_model_list_command() {
    let result = handle_model_command(ModelCommands::List).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_model_info_command() {
    let result = handle_model_command(ModelCommands::Info {
        model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
    })
    .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_model_info_nonexistent() {
    let result = handle_model_command(ModelCommands::Info {
        model: "nonexistent-model".to_string(),
    })
    .await;
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("not found"));
}

#[tokio::test]
async fn test_model_download_fastembed() {
    let result = handle_model_command(ModelCommands::Download {
        model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
    })
    .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_model_download_nonexistent() {
    let result = handle_model_command(ModelCommands::Download {
        model: "nonexistent-model".to_string(),
    })
    .await;
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("not found"));
}

#[tokio::test]
async fn test_model_clear_all() {
    let result = handle_model_command(ModelCommands::Clear { model: None }).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_model_clear_specific() {
    let result = handle_model_command(ModelCommands::Clear {
        model: Some("test-model".to_string()),
    })
    .await;
    assert!(result.is_ok());
}

#[test]
fn test_format_size_utility() {
    // Test the format_size function
    use turboprop::commands::model::format_size;

    assert_eq!(format_size(1024), "1.0 KB");
    assert_eq!(format_size(1024 * 1024), "1.0 MB");
    assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
    assert_eq!(format_size(500), "500.0 B");
    assert_eq!(format_size(1536), "1.5 KB"); // 1.5 KB
}

#[test]
fn test_model_manager_available_models() {
    let manager = ModelManager::default();
    let models = manager.get_available_models();
    assert!(!models.is_empty());

    // Check that we have the expected models
    let model_names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
    assert!(model_names.contains(&"sentence-transformers/all-MiniLM-L6-v2"));
    assert!(model_names.contains(&"Qwen/Qwen3-Embedding-0.6B"));
    assert!(model_names.contains(&"nomic-embed-code.Q5_K_S.gguf"));
}

#[test]
fn test_model_cache_management() {
    let temp_dir = TempDir::new().unwrap();
    let manager = ModelManager::new_with_defaults(temp_dir.path());

    // Test cache initialization
    assert!(manager.init_cache().is_ok());

    // Test cache stats for empty cache
    let stats = manager.get_cache_stats().unwrap();
    assert_eq!(stats.model_count, 0);
    assert_eq!(stats.total_size_bytes, 0);

    // Test clear cache on empty directory
    assert!(manager.clear_cache().is_ok());
}
