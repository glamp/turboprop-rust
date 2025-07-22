//! Configuration validation module for TurboProp.
//!
//! This module provides comprehensive validation of configuration values
//! with detailed error messages to help users fix configuration issues.

use anyhow::{Context, Result};
use std::path::Path;
use tracing::warn;

use crate::config::{SearchConfig, TurboPropConfig, YamlConfig};
use crate::embeddings::EmbeddingConfig;
use crate::models::ModelManager;
use crate::types::{ChunkingConfig, FileDiscoveryConfig};

/// Validate a complete TurboPropConfig
pub fn validate_config(config: &TurboPropConfig) -> Result<()> {
    validate_embedding_config(&config.embedding).context("Invalid embedding configuration")?;

    validate_chunking_config(&config.chunking).context("Invalid chunking configuration")?;

    validate_file_discovery_config(&config.file_discovery)
        .context("Invalid file discovery configuration")?;

    validate_search_config(&config.search).context("Invalid search configuration")?;

    validate_general_config(config).context("Invalid general configuration")?;

    validate_config_consistency(config).context("Configuration consistency check failed")?;

    Ok(())
}

/// Validate embedding configuration
pub fn validate_embedding_config(config: &EmbeddingConfig) -> Result<()> {
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
        anyhow::bail!(
            "Invalid batch size {}: Batch size must be greater than 0",
            config.batch_size
        );
    }

    if config.batch_size > 1000 {
        warn!(
            "Large batch size ({}) may cause memory issues. Consider reducing to 32-128.",
            config.batch_size
        );
    }

    // Validate cache directory
    validate_directory_path(&config.cache_dir, "embedding cache directory")?;

    // Validate embedding dimensions
    if config.embedding_dimensions == 0 {
        return Err(anyhow::anyhow!(
            "Embedding dimensions must be greater than 0"
        ));
    }

    Ok(())
}

/// Validate chunking configuration
pub fn validate_chunking_config(config: &ChunkingConfig) -> Result<()> {
    // Validate chunk sizes
    if config.target_chunk_size_tokens == 0 {
        anyhow::bail!(
            "Invalid target chunk size {}: Target chunk size must be greater than 0",
            config.target_chunk_size_tokens
        );
    }

    if config.max_chunk_size_tokens == 0 {
        anyhow::bail!(
            "Invalid max chunk size {}: Maximum chunk size must be greater than 0",
            config.max_chunk_size_tokens
        );
    }

    if config.min_chunk_size_tokens == 0 {
        anyhow::bail!(
            "Invalid min chunk size {}: Minimum chunk size must be greater than 0",
            config.min_chunk_size_tokens
        );
    }

    // Validate chunk size relationships
    if config.target_chunk_size_tokens > config.max_chunk_size_tokens {
        anyhow::bail!(
            "Invalid chunk size configuration: Target chunk size ({}) cannot exceed maximum chunk size ({})",
            config.target_chunk_size_tokens, config.max_chunk_size_tokens
        );
    }

    if config.min_chunk_size_tokens > config.target_chunk_size_tokens {
        anyhow::bail!(
            "Invalid chunk size configuration: Minimum chunk size ({}) cannot exceed target chunk size ({})",
            config.min_chunk_size_tokens, config.target_chunk_size_tokens
        );
    }

    // Validate overlap
    if config.overlap_tokens >= config.target_chunk_size_tokens {
        anyhow::bail!(
            "Invalid chunk overlap {}: Chunk overlap must be less than target chunk size ({})",
            config.overlap_tokens,
            config.target_chunk_size_tokens
        );
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
pub fn validate_file_discovery_config(config: &FileDiscoveryConfig) -> Result<()> {
    // Validate max file size
    if let Some(max_size) = config.max_filesize_bytes {
        if max_size == 0 {
            anyhow::bail!(
                "Invalid max file size {}: Maximum file size must be greater than 0",
                max_size
            );
        }

        // Warn about very large file size limits
        const LARGE_FILE_THRESHOLD: u64 = 100 * 1024 * 1024; // 100MB
        if max_size > LARGE_FILE_THRESHOLD {
            warn!(
                "Large maximum file size ({} bytes) may cause memory issues when processing files",
                max_size
            );
        }
    }

    Ok(())
}

/// Validate search configuration
pub fn validate_search_config(config: &SearchConfig) -> Result<()> {
    // Validate default limit
    if config.default_limit == 0 {
        anyhow::bail!(
            "Invalid search limit {}: Default search limit must be greater than 0",
            config.default_limit
        );
    }

    if config.default_limit > 10000 {
        warn!(
            "Very large default search limit ({}) may impact performance",
            config.default_limit
        );
    }

    // Validate similarity threshold
    if !(0.0..=1.0).contains(&config.min_similarity) {
        anyhow::bail!(
            "Invalid similarity threshold {}: Similarity threshold must be between 0.0 and 1.0",
            config.min_similarity
        );
    }

    Ok(())
}

/// Validate general configuration settings
pub fn validate_general_config(config: &TurboPropConfig) -> Result<()> {
    // Validate paths
    validate_directory_path(
        &config.general.default_index_path,
        "default index directory",
    )?;
    validate_directory_path(&config.general.cache_dir, "cache directory")?;

    // Validate worker threads
    if let Some(threads) = config.general.worker_threads {
        if threads == 0 {
            anyhow::bail!(
                "Invalid worker threads {}: Worker threads must be greater than 0",
                threads
            );
        }

        if threads > 1000 {
            warn!(
                "Very high number of worker threads ({}) may cause system issues",
                threads
            );
        }

        let cpu_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        if threads > cpu_count * 4 {
            warn!(
                "Worker threads ({}) significantly exceed CPU count ({}), which may degrade performance",
                threads, cpu_count
            );
        }
    }

    Ok(())
}

/// Validate configuration consistency across different sections
pub fn validate_config_consistency(config: &TurboPropConfig) -> Result<()> {
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
fn validate_directory_path(path: &Path, description: &str) -> Result<()> {
    let path_str = path.to_string_lossy();

    // Check for empty paths
    if path_str.is_empty() {
        anyhow::bail!(
            "Invalid directory path '{}': {} cannot be empty",
            path_str,
            description
        );
    }

    // Check for obviously invalid paths
    if path_str.contains('\0') {
        anyhow::bail!(
            "Invalid directory path '{}': Directory path contains null characters",
            path_str
        );
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
pub fn validate_yaml_config(yaml_config: &YamlConfig) -> Result<()> {
    // Validate embedding section
    if let Some(ref embedding) = yaml_config.embedding {
        if let Some(batch_size) = embedding.batch_size {
            if batch_size == 0 {
                anyhow::bail!(
                    "Invalid batch size {}: Batch size must be greater than 0",
                    batch_size
                );
            }
        }
    }

    // Validate indexing section
    if let Some(ref indexing) = yaml_config.indexing {
        if let Some(chunk_size) = indexing.chunk_size {
            if chunk_size == 0 {
                anyhow::bail!(
                    "Invalid chunk size {}: Chunk size must be greater than 0",
                    chunk_size
                );
            }
        }
    }

    // Validate search section
    if let Some(ref search) = yaml_config.search {
        if let Some(limit) = search.default_limit {
            if limit == 0 {
                anyhow::bail!(
                    "Invalid search limit {}: Search limit must be greater than 0",
                    limit
                );
            }
        }

        if let Some(threshold) = search.min_similarity {
            if !(0.0..=1.0).contains(&threshold) {
                anyhow::bail!("Invalid similarity threshold {}: Similarity threshold must be between 0.0 and 1.0", threshold);
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
