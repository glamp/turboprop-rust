//! Input validation utilities for HuggingFace backend operations.
//!
//! This module provides validation functions for model names, cache directories,
//! and other inputs to ensure they meet requirements before expensive operations.

use anyhow::{Context, Result};
use std::path::Path;

use crate::types::{CachePath, ModelName};

/// Validate model name format and constraints
pub fn validate_model_name(model_name: &ModelName) -> Result<()> {
    let model_str = model_name.as_str();
    
    if model_str.is_empty() {
        return Err(anyhow::anyhow!(
            "[HUGGINGFACE] [VALIDATION] failed: Model name cannot be empty"
        ));
    }

    if !model_str.contains('/') {
        return Err(anyhow::anyhow!(
            "[HUGGINGFACE] [VALIDATION] failed: Model name '{}' must follow HuggingFace format 'organization/model-name'",
            model_str
        ));
    }

    let parts: Vec<&str> = model_str.split('/').collect();
    if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
        return Err(anyhow::anyhow!(
            "[HUGGINGFACE] [VALIDATION] failed: Model name '{}' has invalid format. Expected 'organization/model-name'",
            model_str
        ));
    }

    // Validate model name contains only allowed characters
    let allowed_chars = |c: char| c.is_alphanumeric() || c == '-' || c == '_' || c == '.' || c == '/';
    if !model_str.chars().all(allowed_chars) {
        return Err(anyhow::anyhow!(
            "[HUGGINGFACE] [VALIDATION] failed: Model name '{}' contains invalid characters. Only alphanumeric, '-', '_', '.', and '/' are allowed",
            model_str
        ));
    }

    Ok(())
}

/// Validate cache directory exists and is accessible
pub fn validate_cache_directory(cache_dir: &CachePath) -> Result<()> {
    let cache_path = cache_dir.as_ref();
    
    if !cache_path.exists() {
        return Err(anyhow::anyhow!(
            "[HUGGINGFACE] [VALIDATION] failed: Cache directory does not exist: {}",
            cache_dir
        ));
    }

    if !cache_path.is_dir() {
        return Err(anyhow::anyhow!(
            "[HUGGINGFACE] [VALIDATION] failed: Cache path is not a directory: {}",
            cache_dir
        ));
    }

    Ok(())
}

/// Validate cache directory permissions (read, write, execute)
pub fn validate_cache_permissions(cache_path: &Path) -> Result<()> {
    let metadata = std::fs::metadata(cache_path).with_context(|| {
        format!(
            "[HUGGINGFACE] [VALIDATION] failed: Cannot read cache directory metadata: {}",
            cache_path.display()
        )
    })?;

    if metadata.permissions().readonly() {
        return Err(anyhow::anyhow!(
            "[HUGGINGFACE] [VALIDATION] failed: Cache directory is read-only: {}. Write permissions required for model downloads",
            cache_path.display()
        ));
    }

    // Test write permissions by attempting to create a temporary file
    let test_file = cache_path.join(".turboprop_write_test");
    match std::fs::File::create(&test_file) {
        Ok(_) => {
            // Clean up test file
            let _ = std::fs::remove_file(&test_file);
        }
        Err(e) => {
            return Err(anyhow::anyhow!(
                "[HUGGINGFACE] [VALIDATION] failed: Cannot write to cache directory: {}. Error: {}",
                cache_path.display(),
                e
            ));
        }
    }

    // Note: Skipping disk space check as fs2 crate is not available
    // In a production environment, you might want to add this dependency
    // or implement platform-specific disk space checks

    Ok(())
}

/// Validate inputs for model loading operations
pub fn validate_model_inputs(model_name: &ModelName, cache_dir: &CachePath) -> Result<()> {
    validate_model_name(model_name)?;
    validate_cache_directory(cache_dir)?;
    validate_cache_permissions(cache_dir.as_path())?;
    Ok(())
}