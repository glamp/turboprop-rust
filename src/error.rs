//! Structured error types for TurboProp application.
//!
//! This module defines comprehensive error types using `thiserror` to provide
//! clear, actionable error messages for all failure modes in the system.

use std::path::PathBuf;
use thiserror::Error;

/// Result type alias for TurboProp operations.
pub type TurboPropResult<T> = Result<T, TurboPropError>;

/// Comprehensive error type for all TurboProp failures.
///
/// Each error variant includes contextual information and user-friendly
/// messages with suggested solutions where applicable.
#[derive(Error, Debug)]
pub enum TurboPropError {
    /// Network-related errors (model downloads, API calls, etc.)
    #[error("Network error: {message}{}",
        url.as_ref().map(|u| format!(" (URL: {})", u)).unwrap_or_default())]
    NetworkError {
        message: String,
        url: Option<String>,
    },

    /// File system permission errors
    #[error("Permission denied for {operation} operation on '{path}'. Check file permissions and ensure the process has necessary access rights.")]
    FilePermissionError { path: PathBuf, operation: String },

    /// Generic file system errors
    #[error("File system error: {message}{}",
        path.as_ref().map(|p| format!(" (Path: {})", p.display())).unwrap_or_default())]
    FileSystemError {
        message: String,
        path: Option<PathBuf>,
    },

    /// Corrupted index files
    #[error("Index is corrupted at '{path}': {reason}. The index will be rebuilt automatically.")]
    CorruptedIndex { path: PathBuf, reason: String },

    /// Insufficient disk space
    #[error("Insufficient disk space at '{path}'. Required: {}, Available: {}. Free up disk space or choose a different location.",
        format_bytes(*required_bytes),
        format_bytes(*available_bytes))]
    InsufficientDiskSpace {
        required_bytes: u64,
        available_bytes: u64,
        path: PathBuf,
    },

    /// Embedding model errors
    #[error("Embedding model error for '{model_name}': {reason}. Verify the model name and check your internet connection.")]
    EmbeddingModelError { model_name: String, reason: String },

    /// Invalid Git repository
    #[error("Invalid Git repository at '{path}'. Ensure the path contains a valid Git repository or use a different directory.")]
    InvalidGitRepository { path: PathBuf },

    /// File encoding issues
    #[error("File encoding error for '{path}'{}: File may be binary or use an unsupported encoding. Skip this file or convert to UTF-8.",
        encoding.as_ref().map(|e| format!(" (detected: {})", e)).unwrap_or_default())]
    FileEncodingError {
        path: PathBuf,
        encoding: Option<String>,
    },

    /// Configuration validation errors
    #[error("Configuration validation error: field '{field}' has invalid value '{value}'. Expected: {expected}.")]
    ConfigurationValidationError {
        field: String,
        value: String,
        expected: String,
    },

    /// Network timeout errors
    #[error("Network timeout: {operation} took longer than {timeout_seconds} seconds. Check your internet connection and try again.")]
    NetworkTimeout {
        operation: String,
        timeout_seconds: u64,
    },

    /// Model loading errors
    #[error(
        "Failed to load model '{model_name}': {reason}. Verify the model exists and is compatible."
    )]
    ModelLoadingError { model_name: String, reason: String },

    /// Index not found errors
    #[error("No index found at '{path}'. Run 'tp index' to create an index first.")]
    IndexNotFound { path: PathBuf },

    /// GGUF model loading errors
    #[error("Failed to load GGUF model '{model_name}': {reason}. Verify the model file exists and is valid.")]
    GGUFModelLoadError { model_name: String, reason: String },

    /// GGUF model inference errors
    #[error("GGUF model inference failed for '{model_name}': {reason}. Check input data and model compatibility.")]
    GGUFInferenceError { model_name: String, reason: String },

    /// GGUF model download errors
    #[error("Failed to download GGUF model '{model_name}': {reason}. Check your internet connection and model URL.")]
    GGUFDownloadError { model_name: String, reason: String },

    /// GGUF model format errors
    #[error("Invalid GGUF model format for '{model_name}': {reason}. The model file may be corrupted or incompatible.")]
    GGUFFormatError { model_name: String, reason: String },

    /// Generic errors for compatibility
    #[error("{message}")]
    Other { message: String },
}

impl TurboPropError {
    /// Create a network error with optional URL context.
    pub fn network(message: impl Into<String>, url: Option<String>) -> Self {
        Self::NetworkError {
            message: message.into(),
            url,
        }
    }

    /// Create a file permission error.
    pub fn file_permission(path: PathBuf, operation: impl Into<String>) -> Self {
        Self::FilePermissionError {
            path,
            operation: operation.into(),
        }
    }

    /// Create a file system error with optional path context.
    pub fn file_system(message: impl Into<String>, path: Option<PathBuf>) -> Self {
        Self::FileSystemError {
            message: message.into(),
            path,
        }
    }

    /// Create a corrupted index error.
    pub fn corrupted_index(path: PathBuf, reason: impl Into<String>) -> Self {
        Self::CorruptedIndex {
            path,
            reason: reason.into(),
        }
    }

    /// Create an insufficient disk space error.
    pub fn insufficient_disk_space(required: u64, available: u64, path: PathBuf) -> Self {
        Self::InsufficientDiskSpace {
            required_bytes: required,
            available_bytes: available,
            path,
        }
    }

    /// Create an embedding model error.
    pub fn embedding_model(model_name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::EmbeddingModelError {
            model_name: model_name.into(),
            reason: reason.into(),
        }
    }

    /// Create an invalid git repository error.
    pub fn invalid_git_repo(path: PathBuf) -> Self {
        Self::InvalidGitRepository { path }
    }

    /// Create a file encoding error.
    pub fn file_encoding(path: PathBuf, encoding: Option<String>) -> Self {
        Self::FileEncodingError { path, encoding }
    }

    /// Create a configuration validation error.
    pub fn config_validation(
        field: impl Into<String>,
        value: impl Into<String>,
        expected: impl Into<String>,
    ) -> Self {
        Self::ConfigurationValidationError {
            field: field.into(),
            value: value.into(),
            expected: expected.into(),
        }
    }

    /// Create a network timeout error.
    pub fn network_timeout(operation: impl Into<String>, timeout_seconds: u64) -> Self {
        Self::NetworkTimeout {
            operation: operation.into(),
            timeout_seconds,
        }
    }

    /// Create a model loading error.
    pub fn model_loading(model_name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ModelLoadingError {
            model_name: model_name.into(),
            reason: reason.into(),
        }
    }

    /// Create an index not found error.
    pub fn index_not_found(path: PathBuf) -> Self {
        Self::IndexNotFound { path }
    }

    /// Create a GGUF model loading error.
    pub fn gguf_model_load(model_name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::GGUFModelLoadError {
            model_name: model_name.into(),
            reason: reason.into(),
        }
    }

    /// Create a GGUF model inference error.
    pub fn gguf_inference(model_name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::GGUFInferenceError {
            model_name: model_name.into(),
            reason: reason.into(),
        }
    }

    /// Create a GGUF model download error.
    pub fn gguf_download(model_name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::GGUFDownloadError {
            model_name: model_name.into(),
            reason: reason.into(),
        }
    }

    /// Create a GGUF model format error.
    pub fn gguf_format(model_name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::GGUFFormatError {
            model_name: model_name.into(),
            reason: reason.into(),
        }
    }

    /// Create a generic error.
    pub fn other(message: impl Into<String>) -> Self {
        Self::Other {
            message: message.into(),
        }
    }
}

// Conversion from anyhow::Error
impl From<anyhow::Error> for TurboPropError {
    fn from(error: anyhow::Error) -> Self {
        Self::Other {
            message: error.to_string(),
        }
    }
}

// Conversion from std::io::Error
impl From<std::io::Error> for TurboPropError {
    fn from(error: std::io::Error) -> Self {
        use std::io::ErrorKind;

        let message = error.to_string();
        match error.kind() {
            ErrorKind::PermissionDenied => Self::FilePermissionError {
                path: PathBuf::from("<unknown>"),
                operation: "access".to_string(),
            },
            ErrorKind::NotFound => Self::FileSystemError {
                message,
                path: None,
            },
            _ => Self::FileSystemError {
                message,
                path: None,
            },
        }
    }
}

/// Format bytes in human-readable form (B, KB, MB, GB).
fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else if size.fract() == 0.0 {
        // Show as integer for whole numbers
        format!("{} {}", size as u64, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1024 * 1024), "1 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1 GB");
    }

    #[test]
    fn test_error_constructors() {
        let error =
            TurboPropError::network("Connection failed", Some("https://example.com".to_string()));
        match error {
            TurboPropError::NetworkError { message, url } => {
                assert_eq!(message, "Connection failed");
                assert_eq!(url, Some("https://example.com".to_string()));
            }
            _ => panic!("Expected NetworkError"),
        }
    }

    #[test]
    fn test_gguf_error_constructors() {
        let model_load_error = TurboPropError::gguf_model_load("test-model", "File not found");
        match model_load_error {
            TurboPropError::GGUFModelLoadError { model_name, reason } => {
                assert_eq!(model_name, "test-model");
                assert_eq!(reason, "File not found");
            }
            _ => panic!("Expected GGUFModelLoadError"),
        }

        let inference_error = TurboPropError::gguf_inference("test-model", "Invalid input");
        match inference_error {
            TurboPropError::GGUFInferenceError { model_name, reason } => {
                assert_eq!(model_name, "test-model");
                assert_eq!(reason, "Invalid input");
            }
            _ => panic!("Expected GGUFInferenceError"),
        }

        let download_error = TurboPropError::gguf_download("test-model", "Network timeout");
        match download_error {
            TurboPropError::GGUFDownloadError { model_name, reason } => {
                assert_eq!(model_name, "test-model");
                assert_eq!(reason, "Network timeout");
            }
            _ => panic!("Expected GGUFDownloadError"),
        }

        let format_error = TurboPropError::gguf_format("test-model", "Invalid header");
        match format_error {
            TurboPropError::GGUFFormatError { model_name, reason } => {
                assert_eq!(model_name, "test-model");
                assert_eq!(reason, "Invalid header");
            }
            _ => panic!("Expected GGUFFormatError"),
        }
    }
}
