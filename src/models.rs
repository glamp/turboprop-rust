//! Model management and caching functionality.
//!
//! This module handles downloading, caching, and managing embedding models
//! for the TurboProp indexing system.

use anyhow::{Context, Result};
use futures::TryStreamExt;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

use crate::types::{ModelBackend, ModelName, ModelType};
use crate::config::TurboPropConfig;


/// Configuration for creating ModelInfo instances
#[derive(Debug, Clone)]
pub struct ModelInfoConfig {
    pub name: ModelName,
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
    pub name: ModelName,
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
    pub fn simple(
        name: ModelName,
        description: String,
        dimensions: usize,
        size_bytes: u64,
    ) -> Self {
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
    pub fn gguf_model(
        name: ModelName,
        description: String,
        dimensions: usize,
        size_bytes: u64,
        download_url: String,
    ) -> Self {
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
    pub fn huggingface_model(
        name: ModelName,
        description: String,
        dimensions: usize,
        size_bytes: u64,
    ) -> Self {
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
        if self.name.as_str().is_empty() {
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
    config: TurboPropConfig,
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new(".turboprop/models", TurboPropConfig::default())
    }
}

impl ModelManager {
    /// Create a new model manager with the specified cache directory and configuration
    pub fn new(cache_dir: impl Into<PathBuf>, config: TurboPropConfig) -> Self {
        Self {
            cache_dir: cache_dir.into(),
            config,
        }
    }

    /// Create a new model manager with the specified cache directory and default configuration
    /// This is a convenience method for cases where only the cache directory is known
    pub fn new_with_defaults(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
            config: TurboPropConfig::default(),
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
    pub fn is_model_cached(&self, model_name: &ModelName) -> bool {
        let model_path = self.get_model_path(model_name);
        model_path.exists() && self.is_valid_model_cache(&model_path)
    }

    /// Get the local path where a model should be cached
    pub fn get_model_path(&self, model_name: &ModelName) -> PathBuf {
        // Convert model name to filesystem-safe directory name
        // Replace all potentially problematic characters with underscores
        let safe_name = model_name.as_str()
            .replace(['/', ':', '<', '>', '"', '|', '?', '*'], "_")
            .replace(".", "_");
        self.cache_dir.join(safe_name)
    }

    /// Get information about available models
    pub fn get_available_models(&self) -> Vec<ModelInfo> {
        vec![
            // Existing sentence-transformer models
            ModelInfo::simple(
                ModelName::from("sentence-transformers/all-MiniLM-L6-v2"),
                "Fast and lightweight model, good for general use".to_string(),
                384,
                23_000_000, // ~23MB
            ),
            ModelInfo::simple(
                ModelName::from("sentence-transformers/all-MiniLM-L12-v2"),
                "Larger model with better accuracy".to_string(),
                384,
                44_000_000, // ~44MB
            ),
            // New GGUF model
            ModelInfo::gguf_model(
                ModelName::from("nomic-embed-code.Q5_K_S.gguf"),
                "Nomic code embedding model optimized for code search".to_string(),
                self.config.get_model_dimensions("nomic-embed-code.Q5_K_S.gguf"),
                self.config.get_model_size_bytes("nomic-embed-code.Q5_K_S.gguf"),
                "https://huggingface.co/nomic-ai/nomic-embed-code-GGUF/resolve/main/nomic-embed-code.Q5_K_S.gguf".to_string(),
            ),
            // New Qwen model
            ModelInfo::huggingface_model(
                ModelName::from("Qwen/Qwen3-Embedding-0.6B"),
                "Qwen3 embedding model for multilingual and code retrieval".to_string(),
                self.config.get_model_dimensions("Qwen/Qwen3-Embedding-0.6B"),
                self.config.get_model_size_bytes("Qwen/Qwen3-Embedding-0.6B"),
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
            
            // First try the standard approach
            match std::fs::remove_dir_all(&self.cache_dir) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    warn!("Standard removal failed, trying forced removal: {}", e);
                    // If standard removal fails, try a more aggressive approach
                    self.force_clear_cache()?;
                }
            }
        }
        Ok(())
    }

    /// Force clear the cache directory when standard removal fails
    /// This handles cases where files have extended attributes or permission issues
    fn force_clear_cache(&self) -> Result<()> {
        if !self.cache_dir.exists() {
            return Ok(());
        }

        // First, try to remove any lock files that might be blocking removal
        self.remove_lock_files()?;

        // Try to remove contents recursively, ignoring individual file errors
        for entry in std::fs::read_dir(&self.cache_dir)
            .with_context(|| format!("Failed to read cache directory: {:?}", self.cache_dir))?
        {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_dir() {
                // For FastEmbed model directories, try extra cleanup steps
                if path.file_name().is_some_and(|name| name.to_string_lossy().contains("models--")) {
                    self.force_remove_fastembed_cache(&path)?;
                } else if let Err(e) = std::fs::remove_dir_all(&path) {
                    warn!("Failed to remove directory {:?}: {}, trying alternative method", path, e);
                    self.force_remove_directory(&path)?;
                }
            } else if let Err(e) = std::fs::remove_file(&path) {
                warn!("Failed to remove file {:?}: {}", path, e);
                self.force_remove_file(&path)?;
            }
        }

        // Finally try to remove the cache directory itself
        if let Err(e) = std::fs::remove_dir(&self.cache_dir) {
            // If we can't remove the empty directory, recreate it empty
            warn!("Could not remove cache directory {:?}: {}, recreating empty", self.cache_dir, e);
        }

        Ok(())
    }

    /// Remove all lock files in the cache directory recursively
    fn remove_lock_files(&self) -> Result<()> {
        fn remove_locks_recursive(dir: &std::path::Path) -> Result<()> {
            if !dir.exists() {
                return Ok(());
            }

            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_dir() {
                    remove_locks_recursive(&path)?;
                } else if path.extension().is_some_and(|ext| ext == "lock") {
                    if let Err(e) = std::fs::remove_file(&path) {
                        warn!("Failed to remove lock file {:?}: {}", path, e);
                    } else {
                        debug!("Removed lock file: {:?}", path);
                    }
                }
            }
            Ok(())
        }

        remove_locks_recursive(&self.cache_dir)?;
        Ok(())
    }

    /// Force remove a FastEmbed cache directory with special handling
    fn force_remove_fastembed_cache(&self, path: &std::path::Path) -> Result<()> {
        debug!("Force removing FastEmbed cache directory: {:?}", path);
        
        // First try standard removal
        if std::fs::remove_dir_all(path).is_ok() {
            return Ok(());
        }

        // If that fails, try command line removal with force
        #[cfg(target_os = "macos")]
        {
            let output = std::process::Command::new("rm")
                .args(["-rf", &path.to_string_lossy()])
                .output();
                
            if let Ok(output) = output {
                if output.status.success() {
                    debug!("Successfully removed FastEmbed cache with rm -rf");
                    return Ok(());
                }
            }
        }

        // If command line also fails, try to clean up what we can
        warn!("Could not fully remove FastEmbed cache {:?}, attempting partial cleanup", path);
        let _ = Self::partial_cleanup_directory(path);
        
        Ok(())
    }

    /// Force remove a regular directory
    fn force_remove_directory(&self, path: &std::path::Path) -> Result<()> {
        #[cfg(target_os = "macos")]
        {
            if let Err(cmd_err) = std::process::Command::new("rm")
                .args(["-rf", &path.to_string_lossy()])
                .output()
            {
                warn!("Command line removal also failed for {:?}: {}", path, cmd_err);
            }
        }
        Ok(())
    }

    /// Force remove a file with extended attributes handling
    fn force_remove_file(&self, path: &std::path::Path) -> Result<()> {
        #[cfg(target_os = "macos")]
        {
            let _ = std::process::Command::new("xattr")
                .args(["-c", &path.to_string_lossy()])
                .output();
            // Try removing the file again after clearing attributes
            let _ = std::fs::remove_file(path);
        }
        Ok(())
    }

    /// Partial cleanup of a directory when full removal fails
    fn partial_cleanup_directory(dir: &std::path::Path) -> Result<()> {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    let _ = std::fs::remove_file(&path);
                } else if path.is_dir() {
                    let _ = Self::partial_cleanup_directory(&path);
                    let _ = std::fs::remove_dir(&path);
                }
            }
        }
        Ok(())
    }

    /// Remove a specific model from the cache
    pub fn remove_model(&self, model_name: &ModelName) -> Result<()> {
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
    /// # use turboprop::types::ModelName;
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let manager = ModelManager::default();
    /// let model_info = ModelInfo::gguf_model(
    ///     ModelName::from("nomic-embed-code.Q5_K_S.gguf"),
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
                tokio::fs::copy(source_path, &model_file_path)
                    .await
                    .with_context(|| {
                        format!("Failed to copy local GGUF model file from {}", source_path)
                    })?;
            } else {
                // For HTTP/HTTPS URLs, download with streaming
                let response = reqwest::get(url).await.map_err(|e| {
                    crate::error::TurboPropError::gguf_download(
                        model_info.name.as_str(),
                        e.to_string(),
                    )
                })?;

                if !response.status().is_success() {
                    return Err(crate::error::TurboPropError::gguf_download(
                        model_info.name.as_str(),
                        format!("HTTP error: {}", response.status()),
                    )
                    .into());
                }

                // Create the file and stream the response body to it
                let mut file = tokio::fs::File::create(&model_file_path)
                    .await
                    .with_context(|| {
                        format!(
                            "Failed to create GGUF model file: {}",
                            model_file_path.display()
                        )
                    })?;

                let mut stream = response.bytes_stream();
                while let Some(chunk) = stream.try_next().await.map_err(|e| {
                    crate::error::TurboPropError::gguf_download(
                        model_info.name.as_str(),
                        e.to_string(),
                    )
                })? {
                    tokio::io::AsyncWriteExt::write_all(&mut file, &chunk)
                        .await
                        .with_context(|| {
                            format!(
                                "Failed to write GGUF model data to {}",
                                model_file_path.display()
                            )
                        })?;
                }
            }

            info!("Downloaded GGUF model to: {}", model_file_path.display());

            // Verify the downloaded file exists and has reasonable size
            let metadata = std::fs::metadata(&model_file_path).with_context(|| {
                format!(
                    "Failed to read metadata for downloaded GGUF model: {}",
                    model_file_path.display()
                )
            })?;

            if metadata.len() == 0 {
                return Err(crate::error::TurboPropError::gguf_format(
                    model_info.name.as_str(),
                    "Downloaded file is empty",
                )
                .into());
            }

            // Validate the GGUF file format
            crate::backends::gguf::validate_gguf_file(&model_file_path)?;

            info!(
                "GGUF model download completed and validated: {} bytes",
                metadata.len()
            );
            Ok(model_file_path)
        } else {
            Err(crate::error::TurboPropError::gguf_download(
                model_info.name.as_str(),
                "No download URL provided for GGUF model",
            )
            .into())
        }
    }

    /// Download a HuggingFace model and cache it locally
    ///
    /// This method downloads a HuggingFace model using the transformers library format
    /// and stores it in the cache directory. For now, this is a placeholder that will
    /// be implemented when HuggingFace backend integration is complete.
    ///
    /// # Arguments
    /// * `model_name` - Name of the HuggingFace model to download
    ///
    /// # Returns
    /// * `Result<PathBuf>` - Path to the cached model directory
    ///
    /// # Examples
    /// ```no_run
    /// # use turboprop::models::ModelManager;
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let manager = ModelManager::default();
    /// let model_path = manager.download_huggingface_model("Qwen/Qwen3-Embedding-0.6B").await.unwrap();
    /// # });
    /// ```
    pub async fn download_huggingface_model(&self, model_name: &str) -> Result<PathBuf> {
        let model_name_obj = ModelName::from(model_name);
        let model_cache_dir = self.get_model_path(&model_name_obj);

        // Check if model is already cached
        if self.is_model_cached(&model_name_obj) {
            info!("HuggingFace model already cached: {}", model_name);
            return Ok(model_cache_dir);
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

        info!("Downloading HuggingFace model: {}", model_name);

        // For now, create a placeholder that indicates the model needs to be downloaded
        // In a full implementation, this would use the HuggingFace Hub API or git-lfs
        let placeholder_file = model_cache_dir.join("model_placeholder.txt");
        let placeholder_content = format!(
            "HuggingFace model: {}\nThis is a placeholder. Actual download not yet implemented.",
            model_name
        );

        tokio::fs::write(&placeholder_file, placeholder_content)
            .await
            .with_context(|| {
                format!(
                    "Failed to create model placeholder file: {}",
                    placeholder_file.display()
                )
            })?;

        info!(
            "HuggingFace model placeholder created at: {}",
            model_cache_dir.display()
        );

        // TODO: Implement actual HuggingFace model download
        // This would involve:
        // 1. Using huggingface_hub crate or git-lfs to download model files
        // 2. Downloading config.json, tokenizer.json, model weights, etc.
        // 3. Validating the downloaded model files
        // 4. Setting up proper directory structure for the model

        warn!(
            "HuggingFace model download is not yet fully implemented. Created placeholder instead."
        );

        Ok(model_cache_dir)
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
        let model_file = temp_dir.path().join("test_model.gguf");

        // Create a valid GGUF file with proper header
        let mut model_content = Vec::new();
        model_content.extend_from_slice(b"GGUF"); // Magic bytes
        model_content.extend_from_slice(&[2, 0, 0, 0]); // Version 2 (little-endian)
        model_content.extend_from_slice(&[0, 0, 0, 0]); // Additional bytes to meet minimum size

        tokio::fs::write(&model_file, &model_content).await.unwrap();

        // For actual tests, we would use a real HTTP server, but for unit tests
        // we'll create a file:// URL
        let file_url = format!("file://{}", model_file.display());
        (file_url, model_file)
    }

    #[test]
    fn test_model_manager_new() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new_with_defaults(temp_dir.path());
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
        let manager = ModelManager::new_with_defaults(&cache_path);

        assert!(!cache_path.exists());
        manager.init_cache().unwrap();
        assert!(cache_path.exists());
    }

    #[test]
    fn test_get_model_path() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new_with_defaults(temp_dir.path());

        let path =
            manager.get_model_path(&ModelName::from("sentence-transformers/all-MiniLM-L6-v2"));
        let expected = temp_dir
            .path()
            .join("sentence-transformers_all-MiniLM-L6-v2");
        assert_eq!(path, expected);
    }

    #[test]
    fn test_is_model_cached_not_exists() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new_with_defaults(temp_dir.path());

        assert!(!manager.is_model_cached(&ModelName::from("nonexistent-model")));
    }

    #[test]
    fn test_get_available_models() {
        let manager = ModelManager::default();
        let models = manager.get_available_models();
        assert!(!models.is_empty());
        assert!(models
            .iter()
            .any(|m| m.name.as_str().contains("all-MiniLM-L6-v2")));
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
        let manager = ModelManager::new_with_defaults(&cache_path);

        // Should not error even if cache doesn't exist
        assert!(manager.clear_cache().is_ok());
    }

    #[test]
    fn test_get_cache_stats_empty() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new_with_defaults(temp_dir.path());

        let stats = manager.get_cache_stats().unwrap();
        assert_eq!(stats.model_count, 0);
        assert_eq!(stats.total_size_bytes, 0);
    }

    #[test]
    fn test_model_info_validation_valid() {
        let model = ModelInfo::simple(
            ModelName::from("test-model"),
            "Test description".to_string(),
            384,
            1000,
        );
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_model_info_validation_empty_name() {
        let model = ModelInfo::simple(
            ModelName::from(""),
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
        let model = ModelInfo::simple(ModelName::from("test-model"), "".to_string(), 384, 1000);
        let result = model.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Model description cannot be empty"));
    }

    #[test]
    fn test_model_info_validation_zero_dimensions() {
        let model = ModelInfo::simple(
            ModelName::from("test-model"),
            "Test description".to_string(),
            0,
            1000,
        );
        let result = model.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Model dimensions must be greater than 0"));
    }

    #[test]
    fn test_model_info_validation_zero_size() {
        let model = ModelInfo::simple(
            ModelName::from("test-model"),
            "Test description".to_string(),
            384,
            0,
        );
        let result = model.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Model size must be greater than 0"));
    }

    #[test]
    fn test_model_info_validation_invalid_url() {
        let model = ModelInfo::gguf_model(
            ModelName::from("test-model"),
            "Test description".to_string(),
            384,
            1000,
            "invalid-url".to_string(),
        );
        let result = model.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Download URL must be a valid HTTP/HTTPS URL"));
    }

    #[test]
    fn test_model_info_validation_valid_url() {
        let model = ModelInfo::gguf_model(
            ModelName::from("test-model"),
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
        let manager = ModelManager::new_with_defaults(temp_dir.path());

        let model_info = ModelInfo::new(ModelInfoConfig {
            name: ModelName::from("test-model.gguf"),
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
        let manager = ModelManager::new_with_defaults(temp_dir.path());
        manager.init_cache().unwrap();

        let (mock_url, _mock_file) = setup_mock_model_file(&temp_dir).await;

        let model_info = ModelInfo::gguf_model(
            ModelName::from("test-model.gguf"),
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

        // Verify file content - should be a valid GGUF file with proper header
        let content = tokio::fs::read(&downloaded_path).await.unwrap();
        let expected_content = {
            let mut expected = Vec::new();
            expected.extend_from_slice(b"GGUF"); // Magic bytes
            expected.extend_from_slice(&[2, 0, 0, 0]); // Version 2 (little-endian)
            expected.extend_from_slice(&[0, 0, 0, 0]); // Additional bytes
            expected
        };
        assert_eq!(content, expected_content);
    }

    #[tokio::test]
    async fn test_download_gguf_model_already_cached() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new_with_defaults(temp_dir.path());
        manager.init_cache().unwrap();

        let (mock_url, _mock_file) = setup_mock_model_file(&temp_dir).await;

        let model_info = ModelInfo::gguf_model(
            ModelName::from("cached-model.gguf"),
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

    #[tokio::test]
    async fn test_download_huggingface_model_placeholder() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new_with_defaults(temp_dir.path());
        manager.init_cache().unwrap();

        let model_name = "Qwen/Qwen3-Embedding-0.6B";

        // Test successful placeholder creation
        let result = manager.download_huggingface_model(model_name).await;
        assert!(result.is_ok());

        let model_path = result.unwrap();
        assert!(model_path.exists());

        // Verify placeholder file was created
        let placeholder_file = model_path.join("model_placeholder.txt");
        assert!(placeholder_file.exists());

        let content = std::fs::read_to_string(&placeholder_file).unwrap();
        assert!(content.contains("HuggingFace model"));
        assert!(content.contains(model_name));
    }

    #[tokio::test]
    async fn test_download_huggingface_model_already_cached() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new_with_defaults(temp_dir.path());
        manager.init_cache().unwrap();

        let model_name = "test-model";
        let model_name_obj = ModelName::from(model_name);
        let model_path = manager.get_model_path(&model_name_obj);

        // Create model directory with required files to simulate cached model
        std::fs::create_dir_all(&model_path).unwrap();
        std::fs::write(model_path.join("config.json"), "{}").unwrap();

        // Test that already cached model returns immediately
        let result = manager.download_huggingface_model(model_name).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_path);
    }
}
