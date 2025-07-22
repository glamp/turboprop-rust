//! Model management and caching functionality.
//!
//! This module handles downloading, caching, and managing embedding models
//! for the TurboProp indexing system.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Information about an available embedding model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model identifier (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Embedding dimensions this model produces
    pub dimensions: usize,
    /// Approximate model size in bytes
    pub size_bytes: u64,
}

impl ModelInfo {
    /// Create a new ModelInfo
    pub fn new(name: String, description: String, dimensions: usize, size_bytes: u64) -> Self {
        Self {
            name,
            description,
            dimensions,
            size_bytes,
        }
    }
}

/// Manager for handling embedding model lifecycle
pub struct ModelManager {
    cache_dir: PathBuf,
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new(".turboprop/models")
    }
}

impl ModelManager {
    /// Create a new model manager with the specified cache directory
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
        }
    }

    /// Initialize the cache directory, creating it if it doesn't exist
    pub fn init_cache(&self) -> Result<()> {
        if !self.cache_dir.exists() {
            std::fs::create_dir_all(&self.cache_dir).with_context(|| {
                format!(
                    "Failed to create model cache directory: {:?}",
                    self.cache_dir
                )
            })?;
            info!("Created model cache directory: {:?}", self.cache_dir);
        }
        Ok(())
    }

    /// Check if a model is cached locally
    pub fn is_model_cached(&self, model_name: &str) -> bool {
        let model_path = self.get_model_path(model_name);
        model_path.exists() && self.is_valid_model_cache(&model_path)
    }

    /// Get the local path where a model should be cached
    pub fn get_model_path(&self, model_name: &str) -> PathBuf {
        // Convert model name to filesystem-safe directory name
        let safe_name = model_name.replace(['/', ':'], "_");
        self.cache_dir.join(safe_name)
    }

    /// Get information about available models
    pub fn get_available_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo::new(
                "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                "Fast and lightweight model, good for general use".to_string(),
                384,
                23_000_000, // ~23MB
            ),
            ModelInfo::new(
                "sentence-transformers/all-MiniLM-L12-v2".to_string(),
                "Larger model with better accuracy".to_string(),
                384,
                44_000_000, // ~44MB
            ),
        ]
    }

    /// Get the default model name
    pub fn default_model() -> &'static str {
        "sentence-transformers/all-MiniLM-L6-v2"
    }

    /// Validate that a cached model directory contains the expected files
    fn is_valid_model_cache(&self, model_path: &Path) -> bool {
        if !model_path.is_dir() {
            return false;
        }

        // Check for common model files that fastembed expects
        let expected_files = ["config.json", "tokenizer.json"];
        let has_expected_files = expected_files
            .iter()
            .any(|&filename| model_path.join(filename).exists());

        if !has_expected_files {
            debug!(
                "Model cache directory {:?} missing expected files",
                model_path
            );
            return false;
        }

        true
    }

    /// Clear the model cache (removes all cached models)
    pub fn clear_cache(&self) -> Result<()> {
        if self.cache_dir.exists() {
            info!("Clearing model cache: {:?}", self.cache_dir);
            std::fs::remove_dir_all(&self.cache_dir)
                .with_context(|| format!("Failed to clear model cache: {:?}", self.cache_dir))?;
        }
        Ok(())
    }

    /// Remove a specific model from the cache
    pub fn remove_model(&self, model_name: &str) -> Result<()> {
        let model_path = self.get_model_path(model_name);
        if model_path.exists() {
            info!("Removing cached model: {} at {:?}", model_name, model_path);
            std::fs::remove_dir_all(&model_path)
                .with_context(|| format!("Failed to remove model cache: {:?}", model_path))?;
        } else {
            warn!("Model {} not found in cache", model_name);
        }
        Ok(())
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> Result<CacheStats> {
        let mut stats = CacheStats::default();

        if !self.cache_dir.exists() {
            return Ok(stats);
        }

        for entry in std::fs::read_dir(&self.cache_dir)
            .with_context(|| format!("Failed to read cache directory: {:?}", self.cache_dir))?
        {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                stats.model_count += 1;
                stats.total_size_bytes += Self::calculate_directory_size(&path)?;
            }
        }

        Ok(stats)
    }

    /// Calculate the total size of a directory recursively
    fn calculate_directory_size(dir: &Path) -> Result<u64> {
        let mut total_size = 0;

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                total_size += entry.metadata()?.len();
            } else if path.is_dir() {
                total_size += Self::calculate_directory_size(&path)?;
            }
        }

        Ok(total_size)
    }
}

/// Statistics about the model cache
#[derive(Debug, Default)]
pub struct CacheStats {
    /// Number of models cached
    pub model_count: usize,
    /// Total size of cached models in bytes
    pub total_size_bytes: u64,
}

impl CacheStats {
    /// Format the total size in human-readable format
    pub fn format_size(&self) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
        let mut size = self.total_size_bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        format!("{:.2} {}", size, UNITS[unit_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_model_manager_new() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new(temp_dir.path());
        assert_eq!(manager.cache_dir, temp_dir.path());
    }

    #[test]
    fn test_model_manager_default() {
        let manager = ModelManager::default();
        assert_eq!(manager.cache_dir, PathBuf::from(".turboprop/models"));
    }

    #[test]
    fn test_init_cache() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("models");
        let manager = ModelManager::new(&cache_path);

        assert!(!cache_path.exists());
        manager.init_cache().unwrap();
        assert!(cache_path.exists());
    }

    #[test]
    fn test_get_model_path() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new(temp_dir.path());

        let path = manager.get_model_path("sentence-transformers/all-MiniLM-L6-v2");
        let expected = temp_dir
            .path()
            .join("sentence-transformers_all-MiniLM-L6-v2");
        assert_eq!(path, expected);
    }

    #[test]
    fn test_is_model_cached_not_exists() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new(temp_dir.path());

        assert!(!manager.is_model_cached("nonexistent-model"));
    }

    #[test]
    fn test_get_available_models() {
        let models = ModelManager::get_available_models();
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.name.contains("all-MiniLM-L6-v2")));
    }

    #[test]
    fn test_default_model() {
        let default = ModelManager::default_model();
        assert_eq!(default, "sentence-transformers/all-MiniLM-L6-v2");
    }

    #[test]
    fn test_cache_stats_format_size() {
        let stats = CacheStats {
            model_count: 2,
            total_size_bytes: 1024 * 1024 + 512, // ~1MB + 512B
        };

        let formatted = stats.format_size();
        assert!(formatted.contains("MB"));
    }

    #[test]
    fn test_clear_cache_nonexistent() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("nonexistent");
        let manager = ModelManager::new(&cache_path);

        // Should not error even if cache doesn't exist
        assert!(manager.clear_cache().is_ok());
    }

    #[test]
    fn test_get_cache_stats_empty() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new(temp_dir.path());

        let stats = manager.get_cache_stats().unwrap();
        assert_eq!(stats.model_count, 0);
        assert_eq!(stats.total_size_bytes, 0);
    }
}
