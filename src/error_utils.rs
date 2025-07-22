//! Common error handling utilities for consistent error contexts across the codebase.
//!
//! This module provides reusable error handling patterns to reduce duplication
//! and ensure consistent error messages throughout the application.

use anyhow::{Context, Result};
use std::path::Path;

/// File operation error contexts
pub trait FileErrorContext<T> {
    /// Add context for file read operations
    fn with_file_read_context(self, path: &Path) -> Result<T>;

    /// Add context for file write operations
    fn with_file_write_context(self, path: &Path) -> Result<T>;

    /// Add context for file metadata operations
    fn with_file_metadata_context(self, path: &Path) -> Result<T>;

    /// Add context for file creation operations
    fn with_file_create_context(self, path: &Path) -> Result<T>;
}

impl<T> FileErrorContext<T> for Result<T> {
    fn with_file_read_context(self, path: &Path) -> Result<T> {
        self.with_context(|| format!("Failed to read file: {}", path.display()))
    }

    fn with_file_write_context(self, path: &Path) -> Result<T> {
        self.with_context(|| format!("Failed to write file: {}", path.display()))
    }

    fn with_file_metadata_context(self, path: &Path) -> Result<T> {
        self.with_context(|| format!("Failed to get metadata for: {}", path.display()))
    }

    fn with_file_create_context(self, path: &Path) -> Result<T> {
        self.with_context(|| format!("Failed to create file: {}", path.display()))
    }
}

// Also implement for std::io::Result and other common result types
impl<T> FileErrorContext<T> for std::io::Result<T> {
    fn with_file_read_context(self, path: &Path) -> Result<T> {
        self.with_context(|| format!("Failed to read file: {}", path.display()))
    }

    fn with_file_write_context(self, path: &Path) -> Result<T> {
        self.with_context(|| format!("Failed to write file: {}", path.display()))
    }

    fn with_file_metadata_context(self, path: &Path) -> Result<T> {
        self.with_context(|| format!("Failed to get metadata for: {}", path.display()))
    }

    fn with_file_create_context(self, path: &Path) -> Result<T> {
        self.with_context(|| format!("Failed to create file: {}", path.display()))
    }
}

/// Directory operation error contexts
pub trait DirectoryErrorContext<T> {
    /// Add context for directory creation operations
    fn with_dir_create_context(self, path: &Path) -> Result<T>;

    /// Add context for directory read operations
    fn with_dir_read_context(self, path: &Path) -> Result<T>;
}

impl<T> DirectoryErrorContext<T> for Result<T> {
    fn with_dir_create_context(self, path: &Path) -> Result<T> {
        self.with_context(|| format!("Failed to create directory: {}", path.display()))
    }

    fn with_dir_read_context(self, path: &Path) -> Result<T> {
        self.with_context(|| format!("Failed to read directory: {}", path.display()))
    }
}

impl<T> DirectoryErrorContext<T> for std::io::Result<T> {
    fn with_dir_create_context(self, path: &Path) -> Result<T> {
        self.with_context(|| format!("Failed to create directory: {}", path.display()))
    }

    fn with_dir_read_context(self, path: &Path) -> Result<T> {
        self.with_context(|| format!("Failed to read directory: {}", path.display()))
    }
}

/// Serialization error contexts
pub trait SerializationErrorContext<T> {
    /// Add context for serialization operations
    fn with_serialize_context(self) -> Result<T>;

    /// Add context for deserialization operations
    fn with_deserialize_context(self) -> Result<T>;
}

impl<T> SerializationErrorContext<T> for Result<T> {
    fn with_serialize_context(self) -> Result<T> {
        self.with_context(|| "Failed to serialize data")
    }

    fn with_deserialize_context(self) -> Result<T> {
        self.with_context(|| "Failed to deserialize data")
    }
}

/// Processing operation error contexts
pub trait ProcessingErrorContext<T> {
    /// Add context for file processing operations
    fn with_file_processing_context(self, path: &Path) -> Result<T>;

    /// Add context for chunking operations
    fn with_chunking_context(self, path: &Path) -> Result<T>;

    /// Add context for embedding generation operations
    fn with_embedding_context(self, path: &Path) -> Result<T>;
}

impl<T> ProcessingErrorContext<T> for Result<T> {
    fn with_file_processing_context(self, path: &Path) -> Result<T> {
        self.with_context(|| format!("Failed to process file: {}", path.display()))
    }

    fn with_chunking_context(self, path: &Path) -> Result<T> {
        self.with_context(|| format!("Failed to chunk file: {}", path.display()))
    }

    fn with_embedding_context(self, path: &Path) -> Result<T> {
        self.with_context(|| format!("Failed to generate embeddings for: {}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::path::PathBuf;

    #[test]
    fn test_file_error_contexts() {
        let path = PathBuf::from("test.txt");
        let error: Result<()> =
            Err(io::Error::new(io::ErrorKind::NotFound, "file not found").into());

        let result = error.with_file_read_context(&path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Failed to read file: test.txt"));
    }

    #[test]
    fn test_directory_error_contexts() {
        let path = PathBuf::from("test_dir");
        let error: Result<()> =
            Err(io::Error::new(io::ErrorKind::PermissionDenied, "permission denied").into());

        let result = error.with_dir_create_context(&path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Failed to create directory: test_dir"));
    }

    #[test]
    fn test_serialization_error_contexts() {
        let error: Result<()> = Err(anyhow::anyhow!("serialization failed"));

        let result = error.with_serialize_context();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Failed to serialize data"));
    }

    #[test]
    fn test_processing_error_contexts() {
        let path = PathBuf::from("process.txt");
        let error: Result<()> = Err(anyhow::anyhow!("processing failed"));

        let result = error.with_file_processing_context(&path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Failed to process file: process.txt"));
    }
}
