//! Model management commands for the TurboProp CLI.
//!
//! This module provides commands for listing, downloading, showing info about,
//! and managing the cache of embedding models.

use anyhow::{Context, Result};
use tracing::{info, warn};

use crate::cli::ModelCommands;
use crate::models::ModelManager;
use crate::types::{ModelBackend, ModelName};

/// Handle model management commands
pub async fn handle_model_command(cmd: ModelCommands) -> Result<()> {
    match cmd {
        ModelCommands::List => list_models().await,
        ModelCommands::Download { model } => download_model(&model).await,
        ModelCommands::Info { model } => show_model_info(&model).await,
        ModelCommands::Clear { model } => clear_model_cache(model.as_deref()).await,
    }
}

/// List all available embedding models
async fn list_models() -> Result<()> {
    let manager = ModelManager::default();
    let models = manager.get_available_models();

    println!("Available Embedding Models:\n");

    for model in models {
        let cached = manager.is_model_cached(&model.name);
        let cache_status = if cached {
            "✓ cached"
        } else {
            "  download required"
        };

        println!("  {} [{}]", model.name.as_str(), cache_status);
        println!("    Description: {}", model.description);
        println!(
            "    Type: {:?}, Backend: {:?}",
            model.model_type, model.backend
        );
        println!(
            "    Dimensions: {}, Size: {}",
            model.dimensions,
            format_size(model.size_bytes)
        );

        if let Some(url) = &model.download_url {
            println!("    Download URL: {}", url);
        }

        println!();
    }

    Ok(())
}

/// Download a specific model
async fn download_model(model_name: &str) -> Result<()> {
    let manager = ModelManager::default();
    let models = manager.get_available_models();
    let model_info = models
        .iter()
        .find(|m| m.name.as_str() == model_name)
        .ok_or_else(|| anyhow::anyhow!("Model '{}' not found", model_name))?;

    match model_info.backend {
        ModelBackend::FastEmbed => {
            info!("FastEmbed models are downloaded automatically on first use");
            println!(
                "✓ FastEmbed model '{}' will be downloaded automatically when first used",
                model_name
            );
        }
        ModelBackend::Candle => {
            info!("Downloading GGUF model: {}", model_name);
            println!("⬇️  Downloading GGUF model: {}", model_name);
            let _path = manager
                .download_gguf_model(model_info)
                .await
                .with_context(|| format!("Failed to download GGUF model '{}'", model_name))?;
            println!("✅ Successfully downloaded GGUF model: {}", model_name);
            info!("Successfully downloaded GGUF model");
        }
        ModelBackend::Custom => {
            info!("Downloading Hugging Face model: {}", model_name);
            println!("⬇️  Downloading Hugging Face model: {}", model_name);
            let _path = manager
                .download_huggingface_model(model_name)
                .await
                .with_context(|| {
                    format!("Failed to download HuggingFace model '{}'", model_name)
                })?;
            println!(
                "⚠️  HuggingFace model download created placeholder (full implementation pending)"
            );
            info!("HuggingFace model placeholder created");
        }
    }

    Ok(())
}

/// Show detailed information about a specific model
async fn show_model_info(model_name: &str) -> Result<()> {
    let manager = ModelManager::default();
    let models = manager.get_available_models();
    let model_info = models
        .iter()
        .find(|m| m.name.as_str() == model_name)
        .ok_or_else(|| anyhow::anyhow!("Model '{}' not found", model_name))?;

    println!("Model Information: {}\n", model_info.name.as_str());
    println!("Description: {}", model_info.description);
    println!("Type: {:?}", model_info.model_type);
    println!("Backend: {:?}", model_info.backend);
    println!("Dimensions: {}", model_info.dimensions);
    println!("Size: {}", format_size(model_info.size_bytes));

    if let Some(url) = &model_info.download_url {
        println!("Download URL: {}", url);
    }

    let manager = ModelManager::default();
    let cached = manager.is_model_cached(&model_info.name);
    println!(
        "Cache Status: {}",
        if cached {
            "Cached locally"
        } else {
            "Not cached"
        }
    );

    // Model-specific information
    match model_info.name.as_str() {
        name if name.contains("qwen3") || name.contains("Qwen3") => {
            println!("\nQwen3 Model Features:");
            println!("- Supports instruction-based embeddings with --instruction flag");
            println!("- Multilingual support (100+ languages)");
            println!("- Optimized for code and text retrieval");
        }
        name if name.contains("nomic-embed-code") => {
            println!("\nNomic Embed Code Features:");
            println!("- Specialized for code search and retrieval");
            println!("- Supports multiple programming languages");
            println!("- GGUF quantized for efficient inference");
        }
        _ => {}
    }

    Ok(())
}

/// Clear model cache (all models or a specific model)
async fn clear_model_cache(model_name: Option<&str>) -> Result<()> {
    let manager = ModelManager::default();

    match model_name {
        Some(name) => {
            info!("Clearing cache for model: {}", name);
            println!("🗑️  Clearing cache for model: {}", name);
            let model_name = ModelName::from(name);
            manager
                .remove_model(&model_name)
                .with_context(|| format!("Failed to remove model '{}'", name))?;
            println!("✅ Successfully cleared cache for model: {}", name);
        }
        None => {
            warn!("Clearing all model caches");
            println!("🗑️  Clearing all model caches...");
            manager
                .clear_cache()
                .context("Failed to clear model cache")?;
            println!("✅ Successfully cleared all model caches");
        }
    }

    info!("Model cache cleared successfully");
    Ok(())
}

/// Format file size in human-readable format
pub fn format_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    format!("{:.1} {}", size, UNITS[unit_index])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(0), "0.0 B");
        assert_eq!(format_size(512), "512.0 B");
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(format_size(2 * 1024 * 1024 * 1024), "2.0 GB");
    }

    #[tokio::test]
    async fn test_list_models_succeeds() {
        let result = list_models().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_show_model_info_existing() {
        let result = show_model_info("sentence-transformers/all-MiniLM-L6-v2").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_show_model_info_nonexistent() {
        let result = show_model_info("nonexistent-model").await;
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("not found"));
    }

    #[tokio::test]
    async fn test_clear_cache_success() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new_with_defaults(temp_dir.path());
        manager.init_cache().unwrap();

        let result = clear_model_cache(None).await;
        assert!(result.is_ok());
    }
}
