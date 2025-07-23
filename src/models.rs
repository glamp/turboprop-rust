//! Model management and caching functionality.
//!
//! This module handles downloading, caching, and managing embedding models
//! for the TurboProp indexing system.

use anyhow::{Context, Result};
use futures::TryStreamExt;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

use crate::types::{ModelType, ModelBackend};

// Model dimension and size constants
pub const NOMIC_EMBED_DIMENSIONS: usize = 768;
pub const NOMIC_EMBED_SIZE_BYTES: u64 = 2_500_000_000; // ~2.5GB
pub const QWEN_EMBED_DIMENSIONS: usize = 1024;
pub const QWEN_EMBED_SIZE_BYTES: u64 = 600_000_000; // ~600MB

/// Configuration for creating ModelInfo instances
#[derive(Debug, Clone)]
pub struct ModelInfoConfig {
    pub name: String,
    pub description: String,
    pub dimensions: usize,
    pub size_bytes: u64,
    pub model_type: ModelType,
    pub backend: ModelBackend,
    pub download_url: Option<String>,
    pub local_path: Option<PathBuf>,
}

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
    /// Type of embedding model
    pub model_type: ModelType,
    /// Backend used to load and run the model
    pub backend: ModelBackend,
    /// Optional direct download URL for the model
    pub download_url: Option<String>,
    /// Optional local path for models stored locally
    pub local_path: Option<PathBuf>,
}

impl ModelInfo {
    /// Create a new ModelInfo using a configuration struct
    pub fn new(config: ModelInfoConfig) -> Self {
        Self {
            name: config.name,
            description: config.description,
            dimensions: config.dimensions,
            size_bytes: config.size_bytes,
            model_type: config.model_type,
            backend: config.backend,
            download_url: config.download_url,
            local_path: config.local_path,
        }
    }

    /// Create a simple ModelInfo with default values for common cases
    pub fn simple(name: String, description: String, dimensions: usize, size_bytes: u64) -> Self {
        Self::new(ModelInfoConfig {
            name,
            description,
            dimensions,
            size_bytes,
            model_type: ModelType::SentenceTransformer,
            backend: ModelBackend::FastEmbed,
            download_url: None,
            local_path: None,
        })
    }

    /// Create a GGUF model with Candle backend
    pub fn gguf_model(name: String, description: String, dimensions: usize, size_bytes: u64, download_url: String) -> Self {
        Self::new(ModelInfoConfig {
            name,
            description,
            dimensions,
            size_bytes,
            model_type: ModelType::GGUF,
            backend: ModelBackend::Candle,
            download_url: Some(download_url),
            local_path: None,
        })
    }

    /// Create a HuggingFace model with Custom backend
    pub fn huggingface_model(name: String, description: String, dimensions: usize, size_bytes: u64) -> Self {
        Self::new(ModelInfoConfig {
            name,
            description,
            dimensions,
            size_bytes,
            model_type: ModelType::HuggingFace,
            backend: ModelBackend::Custom,
            download_url: None,
            local_path: None,
        })
    }

    /// Validate that this ModelInfo has consistent and valid configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("Model name cannot be empty".to_string());
        }
        
        if self.description.is_empty() {
            return Err("Model description cannot be empty".to_string());
        }
        
        if self.dimensions == 0 {
            return Err("Model dimensions must be greater than 0".to_string());
        }
        
        if self.size_bytes == 0 {
            return Err("Model size must be greater than 0".to_string());
        }
        
        // Validate that download_url is a valid URL if present
        if let Some(ref url) = self.download_url {
            if url.is_empty() {
                return Err("Download URL cannot be empty if specified".to_string());
            }
            if !url.starts_with("http://") && !url.starts_with("https://") {
                return Err("Download URL must be a valid HTTP/HTTPS URL".to_string());
            }
        }
        
        // Validate that local_path exists if specified
        if let Some(ref path) = self.local_path {
            if !path.exists() {
                return Err(format!("Local path does not exist: {}", path.display()));
            }
        }
        
        Ok(())
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
            // Existing sentence-transformer models
            ModelInfo::simple(
                "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                "Fast and lightweight model, good for general use".to_string(),
                384,
                23_000_000, // ~23MB
            ),
            ModelInfo::simple(
                "sentence-transformers/all-MiniLM-L12-v2".to_string(),
                "Larger model with better accuracy".to_string(),
                384,
                44_000_000, // ~44MB
            ),
            // New GGUF model
            ModelInfo::gguf_model(
                "nomic-embed-code.Q5_K_S.gguf".to_string(),
                "Nomic code embedding model optimized for code search".to_string(),
                NOMIC_EMBED_DIMENSIONS,
                NOMIC_EMBED_SIZE_BYTES,
                "https://huggingface.co/nomic-ai/nomic-embed-code-GGUF/resolve/main/nomic-embed-code.Q5_K_S.gguf".to_string(),
            ),
            // New Qwen model
            ModelInfo::huggingface_model(
                "Qwen/Qwen3-Embedding-0.6B".to_string(),
                "Qwen3 embedding model for multilingual and code retrieval".to_string(),
                QWEN_EMBED_DIMENSIONS,
                QWEN_EMBED_SIZE_BYTES,
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

    /// Download a GGUF model file and cache it locally
    ///
    /// This method downloads the model from the provided URL and stores it in the cache directory.
    /// If the model is already cached, it returns the existing path without re-downloading.
    ///
    /// # Arguments
    /// * `model_info` - Model information including download URL
    ///
    /// # Returns
    /// * `Result<PathBuf>` - Path to the cached model file
    ///
    /// # Examples
    /// ```no_run
    /// # use turboprop::models::{ModelManager, ModelInfo};
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let manager = ModelManager::default();
    /// let model_info = ModelInfo::gguf_model(
    ///     "nomic-embed-code.Q5_K_S.gguf".to_string(),
    ///     "Nomic code embedding model".to_string(),
    ///     768,
    ///     2_500_000_000,
    ///     "https://huggingface.co/nomic-ai/nomic-embed-code-GGUF/resolve/main/nomic-embed-code.Q5_K_S.gguf".to_string(),
    /// );
    /// let model_path = manager.download_gguf_model(&model_info).await.unwrap();
    /// # });
    /// ```
    pub async fn download_gguf_model(&self, model_info: &ModelInfo) -> Result<PathBuf> {
        if let Some(url) = &model_info.download_url {
            let model_cache_dir = self.get_model_path(&model_info.name);
            let model_file_path = model_cache_dir.join("model.gguf");
            
            // Check if model is already cached
            if model_file_path.exists() {
                info!("GGUF model already cached: {}", model_info.name);
                return Ok(model_file_path);
            }
            
            // Create cache directory
            if !model_cache_dir.exists() {
                std::fs::create_dir_all(&model_cache_dir).with_context(|| {
                    format!(
                        "Failed to create model cache directory: {}",
                        model_cache_dir.display()
                    )
                })?;
            }
            
            info!("Downloading GGUF model: {} from {}", model_info.name, url);
            
            // Handle different URL schemes
            if url.starts_with("file://") {
                // For file:// URLs (mainly used in tests), copy the file
                let source_path = url.strip_prefix("file://").unwrap();
                tokio::fs::copy(source_path, &model_file_path).await.with_context(|| {
                    format!("Failed to copy local GGUF model file from {}", source_path)
                })?;
            } else {
                // For HTTP/HTTPS URLs, download with streaming
                let response = reqwest::get(url).await.map_err(|e| {
                    crate::error::TurboPropError::gguf_download(&model_info.name, e.to_string())
                })?;
                
                if !response.status().is_success() {
                    return Err(crate::error::TurboPropError::gguf_download(
                        &model_info.name,
                        format!("HTTP error: {}", response.status()),
                    ).into());
                }
                
                // Create the file and stream the response body to it
                let mut file = tokio::fs::File::create(&model_file_path).await.with_context(|| {
                    format!("Failed to create GGUF model file: {}", model_file_path.display())
                })?;
                
                let mut stream = response.bytes_stream();
                while let Some(chunk) = stream.try_next().await.map_err(|e| {
                    crate::error::TurboPropError::gguf_download(&model_info.name, e.to_string())
                })? {
                    tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await.with_context(|| {
                        format!("Failed to write GGUF model data to {}", model_file_path.display())
                    })?;
                }
            }
            
            info!("Downloaded GGUF model to: {}", model_file_path.display());
            
            // Verify the downloaded file exists and has reasonable size
            let metadata = std::fs::metadata(&model_file_path).with_context(|| {
                format!("Failed to read metadata for downloaded GGUF model: {}", model_file_path.display())
            })?;
            
            if metadata.len() == 0 {
                return Err(crate::error::TurboPropError::gguf_format(
                    &model_info.name,
                    "Downloaded file is empty",
                ).into());
            }
            
            info!("GGUF model download completed: {} bytes", metadata.len());
            Ok(model_file_path)
        } else {
            Err(crate::error::TurboPropError::gguf_download(
                &model_info.name,
                "No download URL provided for GGUF model",
            ).into())
        }
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

/// Trait for different embedding backend implementations
pub trait EmbeddingBackend: Send + Sync {
    /// Load a model using this backend
    fn load_model(&self, model_info: &ModelInfo) -> Result<Box<dyn EmbeddingModel>>;
    
    /// Check if this backend supports the given model type
    fn supports_model(&self, model_type: &ModelType) -> bool;
}

/// Trait for embedding model implementations
pub trait EmbeddingModel: Send + Sync {
    /// Generate embeddings for a batch of texts
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    
    /// Get the embedding dimensions produced by this model
    fn dimensions(&self) -> usize;
    
    /// Get the maximum sequence length supported by this model
    fn max_sequence_length(&self) -> usize;
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
    use tokio;
    
    // Mock HTTP server for testing downloads
    async fn setup_mock_model_file(temp_dir: &TempDir) -> (String, PathBuf) {
        let model_content = b"mock GGUF model file content";
        let model_file = temp_dir.path().join("test_model.gguf");
        tokio::fs::write(&model_file, model_content).await.unwrap();
        
        // For actual tests, we would use a real HTTP server, but for unit tests
        // we'll create a file:// URL
        let file_url = format!("file://{}", model_file.display());
        (file_url, model_file)
    }

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

    #[test]
    fn test_model_info_validation_valid() {
        let model = ModelInfo::simple(
            "test-model".to_string(),
            "Test description".to_string(),
            384,
            1000,
        );
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_model_info_validation_empty_name() {
        let model = ModelInfo::simple(
            "".to_string(),
            "Test description".to_string(),
            384,
            1000,
        );
        let result = model.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Model name cannot be empty"));
    }

    #[test]
    fn test_model_info_validation_empty_description() {
        let model = ModelInfo::simple(
            "test-model".to_string(),
            "".to_string(),
            384,
            1000,
        );
        let result = model.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Model description cannot be empty"));
    }

    #[test]
    fn test_model_info_validation_zero_dimensions() {
        let model = ModelInfo::simple(
            "test-model".to_string(),
            "Test description".to_string(),
            0,
            1000,
        );
        let result = model.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Model dimensions must be greater than 0"));
    }

    #[test]
    fn test_model_info_validation_zero_size() {
        let model = ModelInfo::simple(
            "test-model".to_string(),
            "Test description".to_string(),
            384,
            0,
        );
        let result = model.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Model size must be greater than 0"));
    }

    #[test]
    fn test_model_info_validation_invalid_url() {
        let model = ModelInfo::gguf_model(
            "test-model".to_string(),
            "Test description".to_string(),
            384,
            1000,
            "invalid-url".to_string(),
        );
        let result = model.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Download URL must be a valid HTTP/HTTPS URL"));
    }

    #[test]
    fn test_model_info_validation_valid_url() {
        let model = ModelInfo::gguf_model(
            "test-model".to_string(),
            "Test description".to_string(),
            384,
            1000,
            "https://example.com/model.gguf".to_string(),
        );
        assert!(model.validate().is_ok());
    }

    #[tokio::test]
    async fn test_download_gguf_model_no_url() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new(temp_dir.path());
        
        let model_info = ModelInfo::new(ModelInfoConfig {
            name: "test-model.gguf".to_string(),
            description: "Test GGUF model without URL".to_string(),
            dimensions: 768,
            size_bytes: 1000,
            model_type: ModelType::GGUF,
            backend: ModelBackend::Candle,
            download_url: None, // No URL provided
            local_path: None,
        });

        // This should fail because no download URL is provided
        let result = manager.download_gguf_model(&model_info).await;
        assert!(result.is_err());
        
        let error_message = result.err().unwrap().to_string();
        assert!(error_message.contains("No download URL provided"));
    }

    #[tokio::test]
    async fn test_download_gguf_model_success() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new(temp_dir.path());
        manager.init_cache().unwrap();
        
        let (mock_url, _mock_file) = setup_mock_model_file(&temp_dir).await;
        
        let model_info = ModelInfo::gguf_model(
            "test-model.gguf".to_string(),
            "Test GGUF model for download".to_string(),
            768,
            1000,
            mock_url,
        );

        // Test successful download
        let result = manager.download_gguf_model(&model_info).await;
        assert!(result.is_ok());
        
        let downloaded_path = result.unwrap();
        assert!(downloaded_path.exists());
        assert_eq!(downloaded_path.file_name().unwrap(), "model.gguf");
        
        // Verify file content
        let content = tokio::fs::read(&downloaded_path).await.unwrap();
        assert_eq!(content, b"mock GGUF model file content");
    }

    #[tokio::test]
    async fn test_download_gguf_model_already_cached() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new(temp_dir.path());
        manager.init_cache().unwrap();
        
        let (mock_url, _mock_file) = setup_mock_model_file(&temp_dir).await;
        
        let model_info = ModelInfo::gguf_model(
            "cached-model.gguf".to_string(),
            "Test GGUF model for caching".to_string(),
            768,
            1000,
            mock_url,
        );

        // First download
        let result1 = manager.download_gguf_model(&model_info).await;
        assert!(result1.is_ok());
        let path1 = result1.unwrap();
        
        // Get modification time of the first download
        let metadata1 = std::fs::metadata(&path1).unwrap();
        let modified1 = metadata1.modified().unwrap();
        
        // Small delay to ensure modification times would be different if file was re-downloaded
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // Second download - should use cached version
        let result2 = manager.download_gguf_model(&model_info).await;
        assert!(result2.is_ok());
        let path2 = result2.unwrap();
        
        // Paths should be the same
        assert_eq!(path1, path2);
        
        // File should not have been modified (indicating it wasn't re-downloaded)
        let metadata2 = std::fs::metadata(&path2).unwrap();
        let modified2 = metadata2.modified().unwrap();
        assert_eq!(modified1, modified2);
    }

    #[tokio::test]
    async fn test_download_gguf_model_network_error() {
        // Skip this test for now - it will be implemented after download_gguf_model exists  
        if std::env::var("SKIP_UNIMPLEMENTED_TESTS").is_ok() {
            return;
        }
        
        // This test will verify that network errors are properly handled
        // and appropriate GGUF download errors are returned
    }
}
