//! Configuration management for TurboProp.
//!
//! This module handles loading and managing configuration from various sources
//! including command line arguments, configuration files, and environment variables.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{debug, info};

use crate::embeddings::EmbeddingConfig;
use crate::types::{parse_filesize, ChunkingConfig, FileDiscoveryConfig};
use crate::validation;

/// Default configuration file name
pub const CONFIG_FILE_NAME: &str = "turboprop.json";

/// YAML configuration file name
pub const YAML_CONFIG_FILE_NAME: &str = ".turboprop.yml";

/// A type-safe wrapper for embedding model names
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelName(String);

impl ModelName {
    /// Create a new ModelName
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Get the model name as a string
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convert into the inner String
    pub fn into_string(self) -> String {
        self.0
    }
}

impl From<String> for ModelName {
    fn from(name: String) -> Self {
        Self(name)
    }
}

impl From<&str> for ModelName {
    fn from(name: &str) -> Self {
        Self(name.to_string())
    }
}

/// Search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Default number of results to return
    pub default_limit: usize,
    /// Minimum similarity threshold for results
    pub min_similarity: f32,
    /// Whether to rehydrate actual content from source files in search results
    /// If false, shows placeholder content for better performance
    pub rehydrate_content: bool,
}

/// Filter limits configuration for glob patterns and file extensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterLimitsConfig {
    /// Maximum allowed length for file extensions (including the dot)
    pub max_extension_length: usize,
    /// Maximum allowed length for glob patterns
    pub max_glob_pattern_length: usize,
    /// Maximum number of patterns to cache (0 = unlimited, default: 1000)
    pub max_cache_size: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            default_limit: 10,
            min_similarity: 0.1,
            rehydrate_content: true,
        }
    }
}

impl Default for FilterLimitsConfig {
    fn default() -> Self {
        Self {
            max_extension_length: 10,
            max_glob_pattern_length: 1000,
            max_cache_size: 1000,
        }
    }
}

/// YAML configuration structure that maps to the .turboprop.yml format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YamlConfig {
    /// Embedding configuration
    pub embedding: Option<YamlEmbeddingConfig>,
    /// Indexing configuration (combines chunking and file discovery)
    pub indexing: Option<YamlIndexingConfig>,
    /// Search configuration
    pub search: Option<YamlSearchConfig>,
    /// Directory configuration
    pub directories: Option<YamlDirectoriesConfig>,
    /// Filtering configuration
    pub filtering: Option<YamlFilterConfig>,
}

/// Configuration for embedding generation settings in YAML format
///
/// This struct maps to the `embedding` section in .turboprop.yml files.
/// All fields are optional and will use defaults if not specified.
///
/// # Example
/// ```yaml
/// embedding:
///   model: "sentence-transformers/all-MiniLM-L6-v2"
///   batch_size: 32
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YamlEmbeddingConfig {
    /// Name of the embedding model to use (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    pub model: Option<ModelName>,
    /// Batch size for embedding generation (default: 32)
    pub batch_size: Option<usize>,
}

/// Configuration for indexing behavior in YAML format
///
/// This struct maps to the `indexing` section in .turboprop.yml files.
/// Controls how files are processed and chunked for indexing.
///
/// # Example
/// ```yaml
/// indexing:
///   max_filesize: "10MB"
///   chunk_size: 512
///   chunk_overlap: 50
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YamlIndexingConfig {
    /// Maximum file size to index (e.g., "10MB", "1GB")
    pub max_filesize: Option<String>,
    /// Target size for text chunks in tokens (default: 512)
    pub chunk_size: Option<usize>,
    /// Number of overlapping tokens between chunks (default: 50)
    pub chunk_overlap: Option<usize>,
}

/// Configuration for search behavior in YAML format
///
/// This struct maps to the `search` section in .turboprop.yml files.
/// Controls default search parameters and result filtering.
///
/// # Example
/// ```yaml
/// search:
///   default_limit: 10
///   min_similarity: 0.5
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YamlSearchConfig {
    /// Default maximum number of search results to return (default: 10)
    pub default_limit: Option<usize>,
    /// Minimum similarity threshold for search results (0.0-1.0, default: 0.3)
    pub min_similarity: Option<f32>,
    /// Whether to rehydrate actual content from source files in search results (default: true)
    pub rehydrate_content: Option<bool>,
}

/// Configuration for directory paths in YAML format
///
/// This struct maps to the `directories` section in .turboprop.yml files.
/// Specifies where indexes and models are stored.
///
/// # Example
/// ```yaml
/// directories:
///   index_dir: "./my-index"
///   models_dir: "~/.cache/turboprop/models"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YamlDirectoriesConfig {
    /// Directory where indexes are stored (default: "./turboprop-index")
    pub index_dir: Option<PathBuf>,
    /// Directory where embedding models are cached (default: system cache dir)
    pub models_dir: Option<PathBuf>,
}

/// Configuration for filtering limits in YAML format
///
/// This struct maps to the `filtering` section in .turboprop.yml files.
/// Controls size limits for glob patterns and file extensions.
///
/// # Example
/// ```yaml
/// filtering:
///   max_extension_length: 15
///   max_glob_pattern_length: 2000
///   max_cache_size: 500
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YamlFilterConfig {
    /// Maximum allowed length for file extensions including the dot (default: 10)
    pub max_extension_length: Option<usize>,
    /// Maximum allowed length for glob patterns (default: 1000)
    pub max_glob_pattern_length: Option<usize>,
    /// Maximum number of patterns to cache, 0 = unlimited (default: 1000)
    pub max_cache_size: Option<usize>,
}

/// Main configuration structure for TurboProp
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TurboPropConfig {
    /// Embedding generation configuration
    pub embedding: EmbeddingConfig,
    /// File discovery configuration
    pub file_discovery: FileDiscoveryConfig,
    /// Chunking configuration
    pub chunking: ChunkingConfig,
    /// Search configuration
    pub search: SearchConfig,
    /// Filtering limits configuration
    pub filtering: FilterLimitsConfig,
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

impl YamlConfig {
    /// Convert YamlConfig to TurboPropConfig by merging with defaults
    pub fn to_turboprop_config(self) -> Result<TurboPropConfig> {
        // Validate the YAML configuration first
        validation::validate_yaml_config(&self).context("YAML configuration validation failed")?;

        let mut config = TurboPropConfig::default();

        // Apply each configuration section
        self.apply_embedding_config(&mut config);
        self.apply_indexing_config(&mut config)?;
        self.apply_search_config(&mut config);
        self.apply_directories_config(&mut config);
        self.apply_filtering_config(&mut config);

        // Validate the final configuration
        validation::validate_config(&config).context("Final configuration validation failed")?;

        Ok(config)
    }

    /// Apply embedding configuration to the base config
    fn apply_embedding_config(&self, config: &mut TurboPropConfig) {
        if let Some(embedding) = &self.embedding {
            if let Some(model) = &embedding.model {
                config.embedding.model_name = model.as_str().to_string();
            }
            if let Some(batch_size) = embedding.batch_size {
                config.embedding.batch_size = batch_size;
            }
        }
    }

    /// Apply indexing configuration to the base config
    fn apply_indexing_config(&self, config: &mut TurboPropConfig) -> Result<()> {
        if let Some(indexing) = &self.indexing {
            if let Some(max_filesize_str) = &indexing.max_filesize {
                let max_filesize = parse_filesize(max_filesize_str)
                    .map_err(|e| anyhow::anyhow!("Failed to parse max_filesize: {}", e))?;
                config.file_discovery.max_filesize_bytes = Some(max_filesize);
            }
            if let Some(chunk_size) = indexing.chunk_size {
                config.chunking.target_chunk_size_tokens = chunk_size;
            }
            if let Some(chunk_overlap) = indexing.chunk_overlap {
                config.chunking.overlap_tokens = chunk_overlap;
            }
        }
        Ok(())
    }

    /// Apply search configuration to the base config
    fn apply_search_config(&self, config: &mut TurboPropConfig) {
        if let Some(search) = &self.search {
            if let Some(default_limit) = search.default_limit {
                config.search.default_limit = default_limit;
            }
            if let Some(min_similarity) = search.min_similarity {
                config.search.min_similarity = min_similarity;
            }
            if let Some(rehydrate_content) = search.rehydrate_content {
                config.search.rehydrate_content = rehydrate_content;
            }
        }
    }

    /// Apply directories configuration to the base config
    fn apply_directories_config(&self, config: &mut TurboPropConfig) {
        if let Some(directories) = &self.directories {
            if let Some(index_dir) = &directories.index_dir {
                config.general.default_index_path = index_dir.clone();
            }
            if let Some(models_dir) = &directories.models_dir {
                config.general.cache_dir = models_dir.clone();
                config.embedding.cache_dir = models_dir.clone();
            }
        }
    }

    /// Apply filtering configuration to the base config
    fn apply_filtering_config(&self, config: &mut TurboPropConfig) {
        if let Some(filtering) = &self.filtering {
            if let Some(max_extension_length) = filtering.max_extension_length {
                config.filtering.max_extension_length = max_extension_length;
            }
            if let Some(max_glob_pattern_length) = filtering.max_glob_pattern_length {
                config.filtering.max_glob_pattern_length = max_glob_pattern_length;
            }
            if let Some(max_cache_size) = filtering.max_cache_size {
                config.filtering.max_cache_size = max_cache_size;
            }
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

        let config = if Self::is_yaml_file(config_path) {
            let yaml_config: YamlConfig = serde_yaml::from_str(&config_content)
                .with_context(|| format!("Failed to parse YAML config file: {:?}", config_path))?;
            yaml_config.to_turboprop_config()?
        } else {
            serde_json::from_str(&config_content)
                .with_context(|| format!("Failed to parse JSON config file: {:?}", config_path))?
        };

        debug!("Configuration loaded successfully");
        Ok(config)
    }

    /// Check if a file is a YAML configuration file based on extension or name
    fn is_yaml_file(path: &Path) -> bool {
        if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
            if filename == ".turboprop.yml" {
                return true;
            }
        }

        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            matches!(extension, "yml" | "yaml")
        } else {
            false
        }
    }

    /// Load configuration searching in default locations
    fn load_from_default_paths() -> Result<Self> {
        let search_paths = Self::get_config_search_paths();

        for path in search_paths {
            if path.exists() {
                return Self::load_from_file(&path);
            }
        }

        debug!("No configuration file found, using defaults");
        Ok(Self::default())
    }

    /// Build the list of paths to search for configuration files
    /// Search order (CLI args > .turboprop.yml > turboprop.json > defaults):
    /// 1. Current directory: ./.turboprop.yml
    /// 2. Current directory: ./turboprop.json
    /// 3. User config directory: ~/.config/turboprop/.turboprop.yml
    /// 4. User config directory: ~/.config/turboprop/turboprop.json
    /// 5. User home directory: ~/.turboprop.yml
    /// 6. User home directory: ~/turboprop.json
    fn get_config_search_paths() -> Vec<PathBuf> {
        vec![
            // Current directory - YAML first
            PathBuf::from(YAML_CONFIG_FILE_NAME),
            PathBuf::from(CONFIG_FILE_NAME),
            // User config directory - YAML first
            dirs::config_dir()
                .map(|p| p.join("turboprop").join(YAML_CONFIG_FILE_NAME))
                .unwrap_or_else(|| {
                    PathBuf::from("~/.config/turboprop").join(YAML_CONFIG_FILE_NAME)
                }),
            dirs::config_dir()
                .map(|p| p.join("turboprop").join(CONFIG_FILE_NAME))
                .unwrap_or_else(|| PathBuf::from("~/.config/turboprop").join(CONFIG_FILE_NAME)),
            // User home directory - YAML first
            dirs::home_dir()
                .map(|p| p.join(YAML_CONFIG_FILE_NAME))
                .unwrap_or_else(|| PathBuf::from("~").join(YAML_CONFIG_FILE_NAME)),
            dirs::home_dir()
                .map(|p| p.join(CONFIG_FILE_NAME))
                .unwrap_or_else(|| PathBuf::from("~").join(CONFIG_FILE_NAME)),
        ]
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

        if let Some(search_limit) = args.search_limit {
            self.search.default_limit = search_limit;
            debug!("Overriding search limit from CLI: {}", search_limit);
        }

        if let Some(min_similarity) = args.min_similarity {
            self.search.min_similarity = min_similarity;
            debug!("Overriding min similarity from CLI: {}", min_similarity);
        }

        if let Some(chunk_size) = args.chunk_size {
            self.chunking.target_chunk_size_tokens = chunk_size;
            debug!("Overriding chunk size from CLI: {}", chunk_size);
        }

        if let Some(chunk_overlap) = args.chunk_overlap {
            self.chunking.overlap_tokens = chunk_overlap;
            debug!("Overriding chunk overlap from CLI: {}", chunk_overlap);
        }

        self
    }

    /// Validate the configuration and return any issues
    pub fn validate(&self) -> Result<()> {
        validation::validate_config(self).map_err(anyhow::Error::from)
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
    pub search_limit: Option<usize>,
    pub min_similarity: Option<f32>,
    pub chunk_size: Option<usize>,
    pub chunk_overlap: Option<usize>,
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

    pub fn with_search_limit(mut self, search_limit: usize) -> Self {
        self.search_limit = Some(search_limit);
        self
    }

    pub fn with_min_similarity(mut self, min_similarity: f32) -> Self {
        self.min_similarity = Some(min_similarity);
        self
    }

    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = Some(chunk_size);
        self
    }

    pub fn with_chunk_overlap(mut self, chunk_overlap: usize) -> Self {
        self.chunk_overlap = Some(chunk_overlap);
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

    #[test]
    fn test_load_yaml_config_file() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join(".turboprop.yml");

        let yaml_content = r#"
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 64
  
indexing:
  max_filesize: "5mb"
  chunk_size: 500
  chunk_overlap: 75

search:
  default_limit: 20
  min_similarity: 0.2

directories:
  index_dir: ".turboprop"
  models_dir: ".turboprop/models"
"#;
        std::fs::write(&config_path, yaml_content).unwrap();

        let config = TurboPropConfig::load_from_file(&config_path).unwrap();

        assert_eq!(
            config.embedding.model_name,
            "sentence-transformers/all-MiniLM-L6-v2"
        );
        assert_eq!(config.embedding.batch_size, 64);
        assert_eq!(config.chunking.target_chunk_size_tokens, 500);
        assert_eq!(config.chunking.overlap_tokens, 75);
        assert_eq!(config.search.default_limit, 20);
        assert_eq!(config.search.min_similarity, 0.2);
        assert_eq!(
            config.file_discovery.max_filesize_bytes,
            Some(5 * 1024 * 1024)
        );
    }

    #[test]
    fn test_configuration_precedence() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join(".turboprop.yml");

        // Create a YAML config with specific values
        let yaml_content = r#"
embedding:
  model: "yaml-model"
  batch_size: 32

search:
  default_limit: 15
  min_similarity: 0.3
"#;
        std::fs::write(&config_path, yaml_content).unwrap();

        let base_config = TurboPropConfig::load_from_file(&config_path).unwrap();

        // Verify YAML values are loaded
        assert_eq!(base_config.embedding.model_name, "yaml-model");
        assert_eq!(base_config.embedding.batch_size, 32);
        assert_eq!(base_config.search.default_limit, 15);

        // Create CLI overrides that should take precedence
        let cli_overrides = CliConfigOverrides::new()
            .with_model("cli-model")
            .with_batch_size(128)
            .with_search_limit(50);

        let final_config = base_config.merge_cli_args(&cli_overrides);

        // Verify CLI args override YAML values
        assert_eq!(final_config.embedding.model_name, "cli-model");
        assert_eq!(final_config.embedding.batch_size, 128);
        assert_eq!(final_config.search.default_limit, 50);

        // Verify non-overridden YAML values remain
        assert_eq!(final_config.search.min_similarity, 0.3);
    }
}
