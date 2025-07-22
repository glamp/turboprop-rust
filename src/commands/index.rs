//! Enhanced index command implementation with progress tracking and error handling.
//!
//! This module provides the complete index command that coordinates file discovery,
//! chunking, embedding generation, and persistent storage with rich user feedback.

use anyhow::{Context, Result};
use std::path::Path;
use tracing::{info, warn};

use crate::config::TurboPropConfig;
use crate::pipeline::{IndexingPipeline, PipelineConfig};

// Error message constants for user-friendly error formatting
const ERROR_ICON: &str = "❌";
const HELP_ICON: &str = "💡";

const HELP_NO_FILES_FOUND: &str =
    "Make sure the directory contains files that match the configured file discovery settings.";
const HELP_CHECK_GITIGNORE: &str =
    "Check if your .gitignore is excluding too many files, or try --include-untracked.";

const MSG_EMBEDDING_INIT_FAILED: &str = "Failed to initialize the embedding model";
const HELP_MODEL_DOWNLOAD: &str =
    "This usually means the model needs to be downloaded or there's a network issue.";
const HELP_RETRY_CONNECTION: &str =
    "Try running the command again with a stable internet connection.";
const HELP_TRY_DIFFERENT_MODEL: &str =
    "Consider using a different model with --model if the issue persists.";

const MSG_PERMISSION_DENIED: &str = "Permission denied while accessing files";
const HELP_RUN_WITH_PERMISSIONS: &str =
    "Try running with appropriate permissions or choose a different directory.";

const MSG_DISK_SPACE: &str = "Insufficient disk space for indexing";
const HELP_DISK_SPACE: &str = "The indexing process requires space to store the generated index.";
const HELP_FREE_SPACE: &str = "Free up disk space or try indexing a smaller directory.";
const HELP_USE_MAX_FILESIZE: &str =
    "Consider using --max-filesize to limit the files being processed.";

const MSG_NETWORK_TIMEOUT: &str = "Network or timeout issue";
const HELP_MODEL_DOWNLOAD_ISSUE: &str = "This usually occurs when downloading embedding models.";
const HELP_CHECK_CONNECTION: &str = "Check your internet connection and try again.";
const HELP_USE_CACHE_DIR: &str = "Consider using a local cache directory with --cache-dir.";

const HELP_USE_VERBOSE: &str = "Try running with --verbose for more detailed error information.";
const HELP_EXCLUDE_LARGE_FILES: &str =
    "Consider using --max-filesize to exclude large files that might be causing issues.";
const HELP_CHECK_DIRECTORY: &str =
    "Check that the directory is accessible and contains readable files.";

/// Execute the index command with the provided configuration
///
/// This is the main entry point for the `tp index` command that replaces the
/// simpler `index_files_with_config` function with enhanced progress tracking,
/// error handling, and user feedback.
///
/// # Arguments
///
/// * `path` - The directory path to index
/// * `config` - Complete TurboProp configuration
/// * `show_progress` - Whether to show interactive progress bars (disable for testing)
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if indexing completed successfully, error otherwise
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use tp::{config::TurboPropConfig, commands::index::execute_index_command};
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let config = TurboPropConfig::default();
/// let result = execute_index_command(Path::new("./src"), &config, true).await;
/// assert!(result.is_ok());
/// # });
/// ```
pub async fn execute_index_command(
    path: &Path,
    config: &TurboPropConfig,
    show_progress: bool,
) -> Result<()> {
    info!("Executing index command for path: {}", path.display());

    // Create pipeline configuration
    let pipeline_config = PipelineConfig::default()
        .with_progress(show_progress)
        .with_error_handling(true) // Always continue on individual file errors
        .with_batch_size(config.embedding.batch_size);

    // Create and execute the indexing pipeline
    let mut pipeline = IndexingPipeline::new(config.clone(), pipeline_config);

    match pipeline.execute_persistent(path).await {
        Ok((persistent_index, stats)) => {
            info!("Index command completed successfully");
            info!("Final index size: {} chunks", persistent_index.len());
            info!(
                "Index saved to: {}",
                persistent_index.storage_path().display()
            );

            // Show warnings if there were any failures, but don't fail the command
            if stats.files_failed > 0 {
                warn!(
                    "⚠️  {} out of {} files failed to process, but indexing completed successfully",
                    stats.files_failed, stats.files_discovered
                );
                warn!("Success rate: {:.1}%", stats.success_rate());
            }

            Ok(())
        }
        Err(e) => {
            Err(e).with_context(|| format!("Index command failed for path: {}", path.display()))
        }
    }
}

/// Execute index command with detailed error context for CLI usage
///
/// This wrapper provides additional error context and user-friendly error messages
/// for CLI usage, converting technical errors into actionable user feedback.
pub async fn execute_index_command_cli(
    path: &Path,
    config: &TurboPropConfig,
    show_progress: bool,
) -> Result<()> {
    // Validate inputs before starting
    validate_index_inputs(path, config)?;

    // Execute the command
    match execute_index_command(path, config, show_progress).await {
        Ok(()) => Ok(()),
        Err(e) => {
            // Provide user-friendly error messages
            let user_message = format_user_error(&e, path);
            eprintln!("{}", user_message);
            Err(e)
        }
    }
}

/// Validate inputs for the index command
fn validate_index_inputs(path: &Path, config: &TurboPropConfig) -> Result<()> {
    // Validate path
    if !path.exists() {
        anyhow::bail!("Directory does not exist: {}", path.display());
    }

    if !path.is_dir() {
        anyhow::bail!("Path is not a directory: {}", path.display());
    }

    // Validate configuration
    config
        .validate()
        .with_context(|| "Invalid configuration provided")?;

    // Check if embedding model is reasonable
    if config.embedding.embedding_dimensions == 0 {
        anyhow::bail!("Invalid embedding dimensions: cannot be zero");
    }

    if config.embedding.batch_size == 0 {
        anyhow::bail!("Invalid batch size: cannot be zero");
    }

    // Warn about very large batch sizes that might cause memory issues
    if config.embedding.batch_size > config.embedding.batch_size_warning_threshold {
        warn!(
            "Large batch size ({}) exceeds threshold ({}), this may cause memory issues",
            config.embedding.batch_size, config.embedding.batch_size_warning_threshold
        );
    }

    Ok(())
}

/// Format a user error message with optional help tips and technical details
fn format_error_message(
    main_message: &str,
    help_tips: &[&str],
    technical_details: Option<&anyhow::Error>,
) -> String {
    let mut message = format!("{} {}", ERROR_ICON, main_message);
    
    for tip in help_tips {
        message.push_str(&format!("\n{} {}", HELP_ICON, tip));
    }
    
    if let Some(details) = technical_details {
        message.push_str(&format!("\n\nTechnical details: {}", details));
    }
    
    message
}

/// Format technical errors into user-friendly messages
fn format_user_error(error: &anyhow::Error, path: &Path) -> String {
    let error_str = error.to_string().to_lowercase();

    if error_str.contains("no files found") {
        format_error_message(
            &format!("No files found to index in '{}'", path.display()),
            &[HELP_NO_FILES_FOUND, HELP_CHECK_GITIGNORE],
            None,
        )
    } else if error_str.contains("failed to initialize embedding generator") {
        format_error_message(
            MSG_EMBEDDING_INIT_FAILED,
            &[HELP_MODEL_DOWNLOAD, HELP_RETRY_CONNECTION, HELP_TRY_DIFFERENT_MODEL],
            Some(error),
        )
    } else if error_str.contains("permission denied") {
        format_error_message(
            MSG_PERMISSION_DENIED,
            &[
                &format!("Make sure you have read permissions for the directory: {}", path.display()),
                HELP_RUN_WITH_PERMISSIONS,
            ],
            Some(error),
        )
    } else if error_str.contains("disk") || error_str.contains("space") {
        format_error_message(
            MSG_DISK_SPACE,
            &[HELP_DISK_SPACE, HELP_FREE_SPACE, HELP_USE_MAX_FILESIZE],
            Some(error),
        )
    } else if error_str.contains("timeout") || error_str.contains("network") {
        format_error_message(
            MSG_NETWORK_TIMEOUT,
            &[HELP_MODEL_DOWNLOAD_ISSUE, HELP_CHECK_CONNECTION, HELP_USE_CACHE_DIR],
            Some(error),
        )
    } else {
        format_error_message(
            &format!("Indexing failed for '{}'", path.display()),
            &[HELP_USE_VERBOSE, HELP_EXCLUDE_LARGE_FILES, HELP_CHECK_DIRECTORY],
            Some(error),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TurboPropConfig;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_file(dir: &Path, name: &str, content: &str) -> std::path::PathBuf {
        let file_path = dir.join(name);
        fs::write(&file_path, content).unwrap();
        file_path
    }

    #[test]
    fn test_validate_index_inputs() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();

        // Valid inputs should pass
        assert!(validate_index_inputs(temp_dir.path(), &config).is_ok());

        // Non-existent directory should fail
        let non_existent = temp_dir.path().join("non_existent");
        assert!(validate_index_inputs(&non_existent, &config).is_err());

        // File instead of directory should fail
        let file_path = create_test_file(temp_dir.path(), "test.txt", "content");
        assert!(validate_index_inputs(&file_path, &config).is_err());
    }

    #[test]
    fn test_validate_config() {
        let temp_dir = TempDir::new().unwrap();

        // Config with zero embedding dimensions should fail
        let mut bad_config = TurboPropConfig::default();
        bad_config.embedding.embedding_dimensions = 0;
        assert!(validate_index_inputs(temp_dir.path(), &bad_config).is_err());

        // Config with zero batch size should fail
        let mut bad_config = TurboPropConfig::default();
        bad_config.embedding.batch_size = 0;
        assert!(validate_index_inputs(temp_dir.path(), &bad_config).is_err());
    }

    #[tokio::test]
    async fn test_execute_index_command_with_empty_dir() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();

        // Empty directory should fail with "no files found" error
        let result = execute_index_command(temp_dir.path(), &config, false).await;
        assert!(result.is_err());

        // Check the full error chain, not just the top-level context
        let error = result.unwrap_err();
        let mut error_chain = vec![error.to_string()];
        let mut current = error.source();
        while let Some(source) = current {
            error_chain.push(source.to_string());
            current = source.source();
        }

        // Look for "No files found" in any part of the error chain
        let contains_no_files_found = error_chain.iter().any(|msg| msg.contains("No files found"));
        assert!(
            contains_no_files_found,
            "Expected error chain to contain 'No files found', got: {:?}",
            error_chain
        );
    }

    #[test]
    fn test_format_user_error() {
        let temp_path = Path::new("/test/path");

        // Test no files found error
        let error = anyhow::anyhow!("No files found to index");
        let formatted = format_user_error(&error, temp_path);
        assert!(formatted.contains("No files found"));
        assert!(formatted.contains("💡"));

        // Test permission error
        let error = anyhow::anyhow!("Permission denied accessing files");
        let formatted = format_user_error(&error, temp_path);
        assert!(formatted.contains("Permission denied"));
        assert!(formatted.contains("💡"));

        // Test generic error
        let error = anyhow::anyhow!("Some unknown error");
        let formatted = format_user_error(&error, temp_path);
        assert!(formatted.contains("Indexing failed"));
        assert!(formatted.contains("💡"));
    }

    #[tokio::test]
    async fn test_execute_index_command_with_files() {
        // Skip this test if we're running in an environment without model access
        if std::env::var("OFFLINE_TESTS").is_ok() {
            return;
        }

        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();

        // Create test files
        create_test_file(
            temp_dir.path(),
            "test1.txt",
            "Hello world this is test content",
        );
        create_test_file(
            temp_dir.path(),
            "test2.rs",
            "fn main() { println!(\"Hello\"); }",
        );

        // This test requires network access for embedding model
        // In a real environment, this should work
        let result = execute_index_command(temp_dir.path(), &config, false).await;

        // We expect this might fail due to network/model requirements in test environment
        // So we just verify the error message is reasonable if it fails
        if let Err(e) = result {
            let error_str = e.to_string();
            // Common acceptable errors in test environment
            assert!(
                error_str.contains("embedding")
                    || error_str.contains("model")
                    || error_str.contains("network")
                    || error_str.contains("Failed to initialize")
            );
        }
    }
}
