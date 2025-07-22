//! Configuration validation module for TurboProp.
//!
//! This module provides comprehensive validation of configuration values
//! with detailed error messages to help users fix configuration issues.

use std::path::Path;
use tracing::warn;

use crate::config::{SearchConfig, TurboPropConfig, YamlConfig};
use crate::embeddings::EmbeddingConfig;
use crate::error::{TurboPropError, TurboPropResult};
use crate::models::ModelManager;
use crate::types::{ChunkingConfig, FileDiscoveryConfig};

/// Configuration thresholds for validation warnings and limits.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Batch size warning threshold (default: 1000)
    pub batch_size_warning_threshold: usize,
    /// Large file size threshold in bytes (default: 100MB)
    pub large_file_threshold: u64,
    /// Search limit warning threshold (default: 10000)
    pub search_limit_warning_threshold: usize,
    /// Worker threads warning threshold (default: 1000)
    pub worker_threads_warning_threshold: usize,
    /// CPU multiplier for thread count warnings (default: 4)
    pub cpu_multiplier_threshold: usize,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            batch_size_warning_threshold: 1000,
            large_file_threshold: 100 * 1024 * 1024, // 100MB
            search_limit_warning_threshold: 10000,
            worker_threads_warning_threshold: 1000,
            cpu_multiplier_threshold: 4,
        }
    }
}

/// Validate a complete TurboPropConfig
pub fn validate_config(config: &TurboPropConfig) -> TurboPropResult<()> {
    let validation_config = ValidationConfig::default();
    validate_config_with_validation_config(config, &validation_config)
}

/// Validate a complete TurboPropConfig with custom validation thresholds
pub fn validate_config_with_validation_config(
    config: &TurboPropConfig,
    validation_config: &ValidationConfig,
) -> TurboPropResult<()> {
    validate_embedding_config(&config.embedding, validation_config)?;
    validate_chunking_config(&config.chunking)?;
    validate_file_discovery_config(&config.file_discovery, validation_config)?;
    validate_search_config(&config.search, validation_config)?;
    validate_general_config(config, validation_config)?;
    validate_config_consistency(config)?;

    Ok(())
}

/// Validate embedding configuration
pub fn validate_embedding_config(
    config: &EmbeddingConfig,
    validation_config: &ValidationConfig,
) -> TurboPropResult<()> {
    // Validate model name
    let available_models = ModelManager::get_available_models();
    if !available_models.iter().any(|m| m.name == config.model_name) {
        warn!(
            "Model '{}' is not in the list of known models. It may still work if it's a valid fastembed model.",
            config.model_name
        );
    }

    // Validate batch size
    if config.batch_size == 0 {
        return Err(TurboPropError::config_validation(
            "embedding.batch_size",
            config.batch_size.to_string(),
            "positive integer greater than 0"
        ));
    }

    if config.batch_size > validation_config.batch_size_warning_threshold {
        warn!(
            "Large batch size ({}) may cause memory issues. Consider reducing to 32-128.",
            config.batch_size
        );
    }

    // Validate cache directory
    validate_directory_path(&config.cache_dir, "embedding cache directory")?;

    // Validate embedding dimensions
    if config.embedding_dimensions == 0 {
        return Err(TurboPropError::config_validation(
            "embedding.embedding_dimensions",
            config.embedding_dimensions.to_string(),
            "positive integer greater than 0"
        ));
    }

    Ok(())
}

/// Validate chunking configuration
pub fn validate_chunking_config(config: &ChunkingConfig) -> TurboPropResult<()> {
    // Validate chunk sizes
    if config.target_chunk_size_tokens == 0 {
        return Err(TurboPropError::config_validation(
            "chunking.target_chunk_size_tokens",
            config.target_chunk_size_tokens.to_string(),
            "positive integer greater than 0"
        ));
    }

    if config.max_chunk_size_tokens == 0 {
        return Err(TurboPropError::config_validation(
            "chunking.max_chunk_size_tokens",
            config.max_chunk_size_tokens.to_string(),
            "positive integer greater than 0"
        ));
    }

    if config.min_chunk_size_tokens == 0 {
        return Err(TurboPropError::config_validation(
            "chunking.min_chunk_size_tokens",
            config.min_chunk_size_tokens.to_string(),
            "positive integer greater than 0"
        ));
    }

    // Validate chunk size relationships
    if config.target_chunk_size_tokens > config.max_chunk_size_tokens {
        return Err(TurboPropError::config_validation(
            "chunking",
            format!("target_chunk_size_tokens ({}) > max_chunk_size_tokens ({})", 
                config.target_chunk_size_tokens, config.max_chunk_size_tokens),
            "target_chunk_size_tokens must not exceed max_chunk_size_tokens"
        ));
    }

    if config.min_chunk_size_tokens > config.target_chunk_size_tokens {
        return Err(TurboPropError::config_validation(
            "chunking",
            format!("min_chunk_size_tokens ({}) > target_chunk_size_tokens ({})", 
                config.min_chunk_size_tokens, config.target_chunk_size_tokens),
            "min_chunk_size_tokens must not exceed target_chunk_size_tokens"
        ));
    }

    // Validate overlap
    if config.overlap_tokens >= config.target_chunk_size_tokens {
        return Err(TurboPropError::config_validation(
            "chunking.overlap_tokens",
            config.overlap_tokens.to_string(),
            format!("less than target_chunk_size_tokens ({})", config.target_chunk_size_tokens)
        ));
    }

    // Warn about potentially problematic values
    if config.overlap_tokens > config.target_chunk_size_tokens / 2 {
        warn!(
            "Large chunk overlap ({} tokens) relative to chunk size ({} tokens) may cause excessive duplication",
            config.overlap_tokens, config.target_chunk_size_tokens
        );
    }

    Ok(())
}

/// Validate file discovery configuration
pub fn validate_file_discovery_config(
    config: &FileDiscoveryConfig,
    validation_config: &ValidationConfig,
) -> TurboPropResult<()> {
    // Validate max file size
    if let Some(max_size) = config.max_filesize_bytes {
        if max_size == 0 {
            return Err(TurboPropError::config_validation(
                "file_discovery.max_filesize_bytes",
                max_size.to_string(),
                "positive integer greater than 0"
            ));
        }

        // Warn about very large file size limits
        if max_size > validation_config.large_file_threshold {
            warn!(
                "Large maximum file size ({} bytes) may cause memory issues when processing files",
                max_size
            );
        }
    }

    Ok(())
}

/// Validate search configuration
pub fn validate_search_config(
    config: &SearchConfig,
    validation_config: &ValidationConfig,
) -> TurboPropResult<()> {
    // Validate default limit
    if config.default_limit == 0 {
        return Err(TurboPropError::config_validation(
            "search.default_limit",
            config.default_limit.to_string(),
            "positive integer greater than 0"
        ));
    }

    if config.default_limit > validation_config.search_limit_warning_threshold {
        warn!(
            "Very large default search limit ({}) may impact performance",
            config.default_limit
        );
    }

    // Validate similarity threshold
    if !(0.0..=1.0).contains(&config.min_similarity) {
        return Err(TurboPropError::config_validation(
            "search.min_similarity",
            config.min_similarity.to_string(),
            "number between 0.0 and 1.0"
        ));
    }

    Ok(())
}

/// Validate general configuration settings
pub fn validate_general_config(
    config: &TurboPropConfig,
    validation_config: &ValidationConfig,
) -> TurboPropResult<()> {
    // Validate paths
    validate_directory_path(
        &config.general.default_index_path,
        "default index directory",
    )?;
    validate_directory_path(&config.general.cache_dir, "cache directory")?;

    // Validate worker threads
    if let Some(threads) = config.general.worker_threads {
        if threads == 0 {
            return Err(TurboPropError::config_validation(
                "general.worker_threads",
                threads.to_string(),
                "positive integer greater than 0"
            ));
        }

        if threads > validation_config.worker_threads_warning_threshold {
            warn!(
                "Very high number of worker threads ({}) may cause system issues",
                threads
            );
        }

        let cpu_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        if threads > cpu_count * validation_config.cpu_multiplier_threshold {
            warn!(
                "Worker threads ({}) significantly exceed CPU count ({}), which may degrade performance",
                threads, cpu_count
            );
        }
    }

    Ok(())
}

/// Validate configuration consistency across different sections
pub fn validate_config_consistency(config: &TurboPropConfig) -> TurboPropResult<()> {
    // Ensure cache directories are consistent
    let general_cache = &config.general.cache_dir;
    let embedding_cache = &config.embedding.cache_dir;

    // The embedding cache should be a subdirectory of the general cache
    if !embedding_cache.starts_with(general_cache) {
        warn!(
            "Embedding cache directory ({:?}) is not within general cache directory ({:?}). This may cause confusion.",
            embedding_cache, general_cache
        );
    }

    Ok(())
}

/// Validate that a directory path is reasonable
fn validate_directory_path(path: &Path, description: &str) -> TurboPropResult<()> {
    let path_str = path.to_string_lossy();

    // Check for empty paths
    if path_str.is_empty() {
        return Err(TurboPropError::config_validation(
            description,
            "<empty>".to_string(),
            "non-empty directory path"
        ));
    }

    // Check for obviously invalid paths
    if path_str.contains('\0') {
        return Err(TurboPropError::config_validation(
            description,
            path_str.to_string(),
            "path without null characters"
        ));
    }

    // Check if path is absolute and warn if it might be problematic
    if path.is_absolute() {
        let home_dir = dirs::home_dir();
        if let Some(home) = &home_dir {
            if !path.starts_with(home) && !path.starts_with("/tmp") && !path.starts_with("/var/tmp")
            {
                warn!(
                    "{} uses absolute path outside home directory: {:?}. This may not be portable.",
                    description, path
                );
            }
        }
    }

    Ok(())
}

/// Validate a YAML configuration before converting to TurboPropConfig
pub fn validate_yaml_config(yaml_config: &YamlConfig) -> TurboPropResult<()> {
    // Validate embedding section
    if let Some(ref embedding) = yaml_config.embedding {
        if let Some(batch_size) = embedding.batch_size {
            if batch_size == 0 {
                return Err(TurboPropError::config_validation(
                    "embedding.batch_size",
                    batch_size.to_string(),
                    "positive integer greater than 0"
                ));
            }
        }
    }

    // Validate indexing section
    if let Some(ref indexing) = yaml_config.indexing {
        if let Some(chunk_size) = indexing.chunk_size {
            if chunk_size == 0 {
                return Err(TurboPropError::config_validation(
                    "indexing.chunk_size",
                    chunk_size.to_string(),
                    "positive integer greater than 0"
                ));
            }
        }
    }

    // Validate search section
    if let Some(ref search) = yaml_config.search {
        if let Some(limit) = search.default_limit {
            if limit == 0 {
                return Err(TurboPropError::config_validation(
                    "search.default_limit",
                    limit.to_string(),
                    "positive integer greater than 0"
                ));
            }
        }

        if let Some(threshold) = search.min_similarity {
            if !(0.0..=1.0).contains(&threshold) {
                return Err(TurboPropError::config_validation(
                    "search.min_similarity",
                    threshold.to_string(),
                    "number between 0.0 and 1.0"
                ));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TurboPropConfig;

    #[test]
    fn test_valid_default_config() {
        let config = TurboPropConfig::default();
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_invalid_batch_size() {
        let mut config = TurboPropConfig::default();
        config.embedding.batch_size = 0;

        let result = validate_config(&config);
        assert!(
            result.is_err(),
            "Expected validation to fail for zero batch size"
        );
    }

    #[test]
    fn test_invalid_chunk_size_relationship() {
        let mut config = TurboPropConfig::default();
        config.chunking.target_chunk_size_tokens = 500;
        config.chunking.max_chunk_size_tokens = 400; // Less than target

        let result = validate_config(&config);
        assert!(
            result.is_err(),
            "Expected validation to fail when target > max chunk size"
        );
    }

    #[test]
    fn test_invalid_similarity_threshold() {
        let mut config = TurboPropConfig::default();
        config.search.min_similarity = 2.0; // Out of range

        let result = validate_config(&config);
        assert!(
            result.is_err(),
            "Expected validation to fail for similarity threshold > 1.0"
        );
    }

    #[test]
    fn test_invalid_worker_threads() {
        let mut config = TurboPropConfig::default();
        config.general.worker_threads = Some(0);

        let result = validate_config(&config);
        assert!(
            result.is_err(),
            "Expected validation to fail for zero worker threads"
        );
    }
}
