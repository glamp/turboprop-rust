//! Enhanced index command implementation with progress tracking and error handling.
//!
//! This module provides the complete index command that coordinates file discovery,
//! chunking, embedding generation, and persistent storage with rich user feedback.

use anyhow::{Context, Result};
use std::path::Path;
use tracing::{info, warn};

use crate::config::TurboPropConfig;
use crate::git::GitignoreFilter;
use crate::incremental::{IncrementalStats, IncrementalUpdater};
use crate::pipeline::{IndexingPipeline, PipelineConfig};
use crate::storage::PersistentIndex;
use crate::watcher::{FileWatcher, SignalHandler};

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

const MSG_NETWORK_TIMEOUT: &str = "Network timeout or connection issue";
const HELP_MODEL_DOWNLOAD_ISSUE: &str = "This usually occurs when downloading embedding models.";
const HELP_CHECK_CONNECTION: &str = "Check your internet connection and try again.";
const HELP_USE_CACHE_DIR: &str = "Consider using a local cache directory with --cache-dir.";

const HELP_USE_VERBOSE: &str = "Try running with --verbose for more detailed error information.";
const HELP_EXCLUDE_LARGE_FILES: &str =
    "Consider using --max-filesize to exclude large files that might be causing issues.";
const HELP_CHECK_DIRECTORY: &str =
    "Check that the directory is accessible and contains readable files.";

/// Enum for classifying different error types for structured error handling
#[derive(Debug)]
enum ErrorType {
    NoFilesFound,
    EmbeddingInit,
    PermissionDenied,
    DiskSpace,
    Network,
    Generic,
}

/// Error classification and formatting configuration
struct ErrorInfo {
    message: &'static str,
    help_texts: &'static [&'static str],
}

impl ErrorType {
    /// Get error information for formatting
    fn get_info(&self) -> ErrorInfo {
        match self {
            ErrorType::NoFilesFound => ErrorInfo {
                message: "No files found to index",
                help_texts: &[HELP_NO_FILES_FOUND, HELP_CHECK_GITIGNORE],
            },
            ErrorType::EmbeddingInit => ErrorInfo {
                message: MSG_EMBEDDING_INIT_FAILED,
                help_texts: &[
                    HELP_MODEL_DOWNLOAD,
                    HELP_RETRY_CONNECTION,
                    HELP_TRY_DIFFERENT_MODEL,
                ],
            },
            ErrorType::PermissionDenied => ErrorInfo {
                message: MSG_PERMISSION_DENIED,
                help_texts: &[HELP_RUN_WITH_PERMISSIONS],
            },
            ErrorType::DiskSpace => ErrorInfo {
                message: MSG_DISK_SPACE,
                help_texts: &[HELP_DISK_SPACE, HELP_FREE_SPACE, HELP_USE_MAX_FILESIZE],
            },
            ErrorType::Network => ErrorInfo {
                message: MSG_NETWORK_TIMEOUT,
                help_texts: &[
                    HELP_MODEL_DOWNLOAD_ISSUE,
                    HELP_CHECK_CONNECTION,
                    HELP_USE_CACHE_DIR,
                ],
            },
            ErrorType::Generic => ErrorInfo {
                message: "Indexing failed",
                help_texts: &[
                    HELP_USE_VERBOSE,
                    HELP_EXCLUDE_LARGE_FILES,
                    HELP_CHECK_DIRECTORY,
                ],
            },
        }
    }

    /// Classify an error based on its message content
    fn classify(error: &anyhow::Error) -> Self {
        let error_str = error.to_string().to_lowercase();

        if error_str.contains("no files found") {
            ErrorType::NoFilesFound
        } else if error_str.contains("failed to initialize embedding generator") {
            ErrorType::EmbeddingInit
        } else if error_str.contains("permission denied") {
            ErrorType::PermissionDenied
        } else if error_str.contains("disk") || error_str.contains("space") {
            ErrorType::DiskSpace
        } else if error_str.contains("timeout") || error_str.contains("network") {
            ErrorType::Network
        } else {
            ErrorType::Generic
        }
    }
}

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
/// use turboprop::{config::TurboPropConfig, commands::index::execute_index_command};
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

/// Execute the watch command with file system monitoring
///
/// This function starts the index in watch mode, continuously monitoring
/// for file changes and updating the index incrementally.
///
/// # Arguments
///
/// * `path` - The directory path to watch
/// * `config` - Complete TurboProp configuration
/// * `show_progress` - Whether to show progress output
///
/// # Returns
///
/// * `Result<()>` - Ok(()) when gracefully shut down, error otherwise
pub async fn execute_watch_command(
    path: &Path,
    config: &TurboPropConfig,
    show_progress: bool,
) -> Result<()> {
    info!("Starting watch mode for path: {}", path.display());

    // First, build initial index if it doesn't exist
    let mut persistent_index = if PersistentIndex::exists(path) {
        info!("Loading existing index...");
        PersistentIndex::load(path)
            .with_context(|| format!("Failed to load existing index for: {}", path.display()))?
    } else {
        info!("Building initial index...");
        // Build the initial index using the same logic as the regular index command
        execute_index_command(path, config, show_progress)
            .await
            .context("Failed to build initial index for watch mode")?;

        // Now load the index that was just created
        PersistentIndex::load(path).with_context(|| {
            format!("Failed to load newly created index for: {}", path.display())
        })?
    };

    // Initialize file watcher
    let gitignore_filter = GitignoreFilter::new(path)
        .with_context(|| format!("Failed to create gitignore filter for: {}", path.display()))?;

    let mut file_watcher = FileWatcher::new(path, gitignore_filter)
        .with_context(|| format!("Failed to create file watcher for: {}", path.display()))?;

    // Initialize incremental updater
    let mut incremental_updater = IncrementalUpdater::new(config.clone(), path)
        .await
        .context("Failed to initialize incremental updater")?;

    // Set up signal handler for graceful shutdown
    let signal_handler = SignalHandler::new().context("Failed to set up signal handler")?;

    if show_progress {
        println!(
            "✓ Watching for changes in {} (Press Ctrl+C to stop)",
            path.display()
        );
        println!("  Initial index: {} chunks", persistent_index.len());
    }

    // Main watch loop
    let mut total_stats = IncrementalStats::default();

    tokio::select! {
        _ = signal_handler.wait_for_shutdown() => {
            info!("Received shutdown signal, stopping watch mode");
        }
        _ = async {
            while let Some(batch) = file_watcher.next_batch().await {
                if show_progress {
                    println!("🔄 Processing {} file changes...", batch.events.len());
                }

                match incremental_updater.process_batch(&batch, &mut persistent_index).await {
                    Ok(stats) => {
                        if show_progress && (stats.files_added > 0 || stats.files_modified > 0 || stats.files_removed > 0) {
                            println!(
                                "✓ Updated index - {} added, {} modified, {} removed ({} chunks)",
                                stats.files_added,
                                stats.files_modified,
                                stats.files_removed,
                                persistent_index.len()
                            );
                        }
                        total_stats.merge(stats);
                    }
                    Err(e) => {
                        warn!("Failed to process file changes: {}", e);
                        if show_progress {
                            eprintln!("⚠️  Error processing changes: {}", e);
                        }
                    }
                }
            }
        } => {
            info!("File watcher closed, stopping watch mode");
        }
    }

    info!("Watch mode shutting down...");
    info!(
        "Total changes processed - added: {}, modified: {}, removed: {}",
        total_stats.files_added, total_stats.files_modified, total_stats.files_removed
    );

    if show_progress {
        println!("👋 Watch mode stopped");
        println!("   Total files processed: {}", total_stats.files_processed);
        println!("   Files added: {}", total_stats.files_added);
        println!("   Files modified: {}", total_stats.files_modified);
        println!("   Files removed: {}", total_stats.files_removed);
        println!("   Final index size: {} chunks", persistent_index.len());
    }

    Ok(())
}

/// Execute watch command with detailed error context for CLI usage
///
/// This wrapper provides additional error context and user-friendly error messages
/// for CLI usage of the watch command.
pub async fn execute_watch_command_cli(
    path: &Path,
    config: &TurboPropConfig,
    show_progress: bool,
) -> Result<()> {
    // Validate inputs before starting watch mode
    validate_index_inputs(path, config)?;

    // Execute the watch command
    match execute_watch_command(path, config, show_progress).await {
        Ok(()) => Ok(()),
        Err(e) => {
            // Provide user-friendly error messages
            let user_message = format_watch_error(&e, path);
            eprintln!("{}", user_message);
            Err(e)
        }
    }
}

/// Format technical errors into user-friendly messages for watch mode
fn format_watch_error(error: &anyhow::Error, path: &Path) -> String {
    let error_str = error.to_string().to_lowercase();

    if error_str.contains("permission denied") {
        format_error_message(
            &format!("Cannot watch directory: {}", path.display()),
            &[
                "File system watching requires read permissions for the directory and its contents",
                "Make sure you have appropriate permissions to watch this directory",
                "Try running with elevated permissions if necessary",
            ],
            Some(error),
        )
    } else if error_str.contains("too many open files") {
        format_error_message(
            "File system resource limit reached",
            &[
                "Your system has reached the maximum number of file handles",
                "Try watching a smaller directory or increase the system file handle limit",
                "On macOS/Linux, you can increase limits with 'ulimit -n'",
            ],
            Some(error),
        )
    } else if error_str.contains("no such file") || error_str.contains("not found") {
        format_error_message(
            &format!("Directory not found: {}", path.display()),
            &[
                "The specified directory does not exist",
                "Make sure the path is correct and the directory exists",
                "Check that the directory wasn't moved or deleted",
            ],
            Some(error),
        )
    } else {
        format_error_message(
            &format!("Watch mode failed for '{}'", path.display()),
            &[
                "An unexpected error occurred while setting up file watching",
                "Try running with --verbose for more detailed error information",
                "Make sure the directory is accessible and you have read permissions",
            ],
            Some(error),
        )
    }
}

/// Format technical errors into user-friendly messages using structured error classification
fn format_user_error(error: &anyhow::Error, path: &Path) -> String {
    let error_type = ErrorType::classify(error);
    let error_info = error_type.get_info();

    let message = match error_type {
        ErrorType::NoFilesFound => format!("No files found to index in '{}'", path.display()),
        ErrorType::Generic => format!("Indexing failed for '{}'", path.display()),
        _ => error_info.message.to_string(),
    };

    // For permission errors, we need to add path-specific help
    if matches!(error_type, ErrorType::PermissionDenied) {
        let path_help = format!(
            "Make sure you have read permissions for the directory: {}",
            path.display()
        );
        let mut help_with_path = vec![path_help.as_str()];
        help_with_path.extend_from_slice(error_info.help_texts);
        format_error_message(&message, &help_with_path, Some(error))
    } else {
        let error_ref = if matches!(error_type, ErrorType::NoFilesFound) {
            None
        } else {
            Some(error)
        };
        format_error_message(&message, error_info.help_texts, error_ref)
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
