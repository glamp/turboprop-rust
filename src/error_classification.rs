//! Error classification and user-friendly error handling.
//!
//! This module provides structured error classification and formatting
//! to give users helpful, actionable error messages and recovery suggestions.

/// Enum for classifying different error types for structured error handling
#[derive(Debug)]
pub enum ErrorType {
    NoFilesFound,
    EmbeddingInit,
    PermissionDenied,
    DiskSpace,
    Network,
    Generic,
}

/// Error classification and formatting configuration
pub struct ErrorInfo {
    pub message: &'static str,
    pub help_texts: &'static [&'static str],
}

/// Error messages and help text constants
mod error_messages {
    pub const MSG_EMBEDDING_INIT_FAILED: &str =
        "Failed to initialize embedding generator. This usually means the model couldn't be loaded.";
    pub const MSG_PERMISSION_DENIED: &str =
        "Permission denied while accessing files or directories.";
    pub const MSG_DISK_SPACE: &str =
        "Insufficient disk space or disk-related error occurred during indexing.";
    pub const MSG_NETWORK_TIMEOUT: &str =
        "Network timeout occurred while downloading embedding model.";

    pub const HELP_NO_FILES_FOUND: &str = "Try specifying a directory that contains code files.";
    pub const HELP_CHECK_GITIGNORE: &str = "Check if files are excluded by .gitignore patterns.";
    pub const HELP_MODEL_DOWNLOAD: &str = "The embedding model may need to be downloaded first. This requires an internet connection.";
    pub const HELP_RETRY_CONNECTION: &str = "If this is a transient network issue, try running the command again.";
    pub const HELP_TRY_DIFFERENT_MODEL: &str = "Consider using a different embedding model with --embedding-model.";
    pub const HELP_RUN_WITH_PERMISSIONS: &str = "Try running with appropriate file system permissions, or use sudo if necessary.";
    pub const HELP_DISK_SPACE: &str = "Free up disk space and try again.";
    pub const HELP_FREE_SPACE: &str = "You may need several GB of free space for embeddings and model files.";
    pub const HELP_USE_MAX_FILESIZE: &str = "Use --max-filesize to limit processing to smaller files.";
    pub const HELP_MODEL_DOWNLOAD_ISSUE: &str = "There may be an issue downloading the embedding model.";
    pub const HELP_CHECK_CONNECTION: &str = "Check your internet connection and try again.";
    pub const HELP_USE_CACHE_DIR: &str = "Specify a cache directory with --cache-dir if there are permission issues.";
    pub const HELP_USE_VERBOSE: &str = "Use --verbose to see more detailed error information.";
    pub const HELP_EXCLUDE_LARGE_FILES: &str = "Consider using --max-filesize to exclude large files that might be causing issues.";
    pub const HELP_CHECK_DIRECTORY: &str = "Check that the directory is accessible and contains readable files.";
}

use error_messages::*;

impl ErrorType {
    /// Get error information for formatting
    pub fn get_info(&self) -> ErrorInfo {
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
    pub fn classify(error: &anyhow::Error) -> Self {
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