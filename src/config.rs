//! Configuration management for TurboProp.
//!
//! This module handles loading and managing configuration from various sources
//! including command line arguments, configuration files, and environment variables.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{debug, info};

use crate::embeddings::EmbeddingConfig;
use crate::models::ModelManager;
use crate::types::{ChunkingConfig, FileDiscoveryConfig};

/// Default configuration file name
pub const CONFIG_FILE_NAME: &str = "turboprop.json";

/// Main configuration structure for TurboProp
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TurboPropConfig {
    /// Embedding generation configuration
    pub embedding: EmbeddingConfig,
    /// File discovery configuration
    pub file_discovery: FileDiscoveryConfig,
    /// Chunking configuration
    pub chunking: ChunkingConfig,
    /// General application settings
    pub general: GeneralConfig,
}

/// General application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Default index path when none specified
    pub default_index_path: PathBuf,
    /// Cache directory for models and other data
    pub cache_dir: PathBuf,
    /// Enable verbose logging
    pub verbose: bool,
    /// Number of parallel workers for processing
    pub worker_threads: Option<usize>,
    /// Storage schema version for index compatibility
    pub storage_version: String,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            default_index_path: PathBuf::from("."),
            cache_dir: PathBuf::from(".turboprop"),
            verbose: false,
            worker_threads: None, // Use system default
            storage_version: "1.0.0".to_string(),
        }
    }
}

impl TurboPropConfig {
    /// Load configuration from the default location
    pub fn load() -> Result<Self> {
        Self::load_from_default_paths()
    }

    /// Load configuration from a specific file
    pub fn load_from_file(config_path: &Path) -> Result<Self> {
        if !config_path.exists() {
            debug!("Config file not found: {:?}, using defaults", config_path);
            return Ok(Self::default());
        }

        info!("Loading configuration from: {:?}", config_path);

        let config_content = std::fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read config file: {:?}", config_path))?;

        let config: TurboPropConfig = serde_json::from_str(&config_content)
            .with_context(|| format!("Failed to parse config file: {:?}", config_path))?;

        debug!("Configuration loaded successfully");
        Ok(config)
    }

    /// Load configuration searching in default locations
    fn load_from_default_paths() -> Result<Self> {
        // Search order:
        // 1. Current directory: ./turboprop.json
        // 2. User config directory: ~/.config/turboprop/turboprop.json
        // 3. User home directory: ~/turboprop.json

        let search_paths = vec![
            PathBuf::from(CONFIG_FILE_NAME),
            dirs::config_dir()
                .map(|p| p.join("turboprop").join(CONFIG_FILE_NAME))
                .unwrap_or_else(|| PathBuf::from("~/.config/turboprop").join(CONFIG_FILE_NAME)),
            dirs::home_dir()
                .map(|p| p.join(CONFIG_FILE_NAME))
                .unwrap_or_else(|| PathBuf::from("~").join(CONFIG_FILE_NAME)),
        ];

        for path in search_paths {
            if path.exists() {
                return Self::load_from_file(&path);
            }
        }

        debug!("No configuration file found, using defaults");
        Ok(Self::default())
    }

    /// Save configuration to a specific file
    pub fn save_to_file(&self, config_path: &Path) -> Result<()> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create config directory: {:?}", parent))?;
        }

        let config_content =
            serde_json::to_string_pretty(self).context("Failed to serialize configuration")?;

        std::fs::write(config_path, config_content)
            .with_context(|| format!("Failed to write config file: {:?}", config_path))?;

        info!("Configuration saved to: {:?}", config_path);
        Ok(())
    }

    /// Save configuration to the default location
    pub fn save(&self) -> Result<()> {
        let config_path = PathBuf::from(CONFIG_FILE_NAME);
        self.save_to_file(&config_path)
    }

    /// Override configuration with command-line arguments
    pub fn merge_cli_args(mut self, args: &CliConfigOverrides) -> Self {
        if let Some(ref model) = args.model {
            self.embedding.model_name = model.clone();
            debug!("Overriding embedding model from CLI: {}", model);
        }

        if let Some(ref cache_dir) = args.cache_dir {
            self.general.cache_dir = cache_dir.clone();
            self.embedding.cache_dir = cache_dir.join("models");
            debug!("Overriding cache directory from CLI: {:?}", cache_dir);
        }

        if args.verbose {
            self.general.verbose = true;
            debug!("Enabling verbose mode from CLI");
        }

        if let Some(max_filesize) = args.max_filesize {
            self.file_discovery = self.file_discovery.with_max_filesize(max_filesize);
            debug!("Overriding max filesize from CLI: {} bytes", max_filesize);
        }

        if let Some(threads) = args.worker_threads {
            self.general.worker_threads = Some(threads);
            debug!("Overriding worker threads from CLI: {}", threads);
        }

        if let Some(batch_size) = args.batch_size {
            self.embedding.batch_size = batch_size;
            debug!("Overriding batch size from CLI: {}", batch_size);
        }

        self
    }

    /// Validate the configuration and return any issues
    pub fn validate(&self) -> Result<()> {
        // Validate embedding model
        let available_models = ModelManager::get_available_models();
        if !available_models
            .iter()
            .any(|m| m.name == self.embedding.model_name)
        {
            tracing::warn!(
                "Model '{}' is not in the list of known models. It may still work if it's a valid fastembed model.",
                self.embedding.model_name
            );
        }

        // Validate paths are reasonable
        if self.general.cache_dir.as_os_str().is_empty() {
            anyhow::bail!("Failed to validate config: cache directory cannot be empty");
        }

        if self.general.default_index_path.as_os_str().is_empty() {
            anyhow::bail!("Failed to validate config: default index path cannot be empty");
        }

        // Validate worker threads
        if let Some(threads) = self.general.worker_threads {
            if threads == 0 {
                anyhow::bail!("Failed to validate config: worker threads must be greater than 0");
            }
            if threads > 1000 {
                tracing::warn!(
                    "Very high number of worker threads ({}), this may cause issues",
                    threads
                );
            }
        }

        Ok(())
    }
}

/// Command-line configuration overrides
#[derive(Debug, Clone, Default)]
pub struct CliConfigOverrides {
    pub model: Option<String>,
    pub cache_dir: Option<PathBuf>,
    pub verbose: bool,
    pub max_filesize: Option<u64>,
    pub worker_threads: Option<usize>,
    pub batch_size: Option<usize>,
}

impl CliConfigOverrides {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(cache_dir.into());
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn with_max_filesize(mut self, max_filesize: u64) -> Self {
        self.max_filesize = Some(max_filesize);
        self
    }

    pub fn with_worker_threads(mut self, worker_threads: usize) -> Self {
        self.worker_threads = Some(worker_threads);
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_default_config() {
        let config = TurboPropConfig::default();
        assert_eq!(config.general.default_index_path, PathBuf::from("."));
        assert!(!config.general.verbose);
        assert_eq!(config.general.cache_dir, PathBuf::from(".turboprop"));
    }

    #[test]
    fn test_config_serialization() {
        let config = TurboPropConfig::default();
        let serialized = serde_json::to_string_pretty(&config).unwrap();
        let deserialized: TurboPropConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            config.general.default_index_path,
            deserialized.general.default_index_path
        );
        assert_eq!(config.general.verbose, deserialized.general.verbose);
    }

    #[test]
    fn test_save_and_load_config() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test_config.json");

        let original_config = TurboPropConfig::default();
        original_config.save_to_file(&config_path).unwrap();

        let loaded_config = TurboPropConfig::load_from_file(&config_path).unwrap();
        assert_eq!(
            original_config.general.verbose,
            loaded_config.general.verbose
        );
    }

    #[test]
    fn test_load_nonexistent_config() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("nonexistent.json");

        let config = TurboPropConfig::load_from_file(&config_path).unwrap();
        // Should return default config
        assert_eq!(config.general.default_index_path, PathBuf::from("."));
    }

    #[test]
    fn test_cli_overrides() {
        let config = TurboPropConfig::default();
        let overrides = CliConfigOverrides::new()
            .with_model("custom-model")
            .with_verbose(true)
            .with_worker_threads(8);

        let merged = config.merge_cli_args(&overrides);
        assert_eq!(merged.embedding.model_name, "custom-model");
        assert!(merged.general.verbose);
        assert_eq!(merged.general.worker_threads, Some(8));
    }

    #[test]
    fn test_config_validation_success() {
        let config = TurboPropConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_empty_cache() {
        let mut config = TurboPropConfig::default();
        config.general.cache_dir = PathBuf::new();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_zero_threads() {
        let mut config = TurboPropConfig::default();
        config.general.worker_threads = Some(0);
        assert!(config.validate().is_err());
    }
}
