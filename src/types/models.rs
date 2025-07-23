//! Model-related type definitions.
//!
//! This module contains strongly-typed wrappers and enums for model names,
//! backends, types, and cache paths to provide type safety and prevent
//! mixing up different model identifiers.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Enumeration of different embedding model types
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ModelType {
    SentenceTransformer,
    GGUF,
    HuggingFace,
}

/// Enumeration of different embedding backends
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ModelBackend {
    FastEmbed,
    Candle,
    Custom,
}

/// A strongly-typed wrapper for model names to prevent mixing up different model identifiers
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelName(String);

impl ModelName {
    /// Create a new ModelName from a string
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Get the string representation of the model name
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convert to owned String
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

impl AsRef<str> for ModelName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ModelName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A strongly-typed wrapper for cache paths to prevent mixing up different path types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CachePath(PathBuf);

impl CachePath {
    /// Create a new CachePath from a PathBuf
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self(path.into())
    }

    /// Get the PathBuf representation of the cache path
    pub fn as_path(&self) -> &Path {
        &self.0
    }

    /// Convert to owned PathBuf
    pub fn into_pathbuf(self) -> PathBuf {
        self.0
    }

    /// Join with another path component
    pub fn join<P: AsRef<Path>>(&self, path: P) -> Self {
        Self(self.0.join(path))
    }

    /// Check if the cache path exists
    pub fn exists(&self) -> bool {
        self.0.exists()
    }
}

impl From<PathBuf> for CachePath {
    fn from(path: PathBuf) -> Self {
        Self(path)
    }
}

impl From<&Path> for CachePath {
    fn from(path: &Path) -> Self {
        Self(path.to_path_buf())
    }
}

impl AsRef<Path> for CachePath {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl std::fmt::Display for CachePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.display())
    }
}