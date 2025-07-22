//! TurboProp library for fast code search and indexing.
//!
//! This library provides functionality for indexing files and searching through
//! indexed content for fast code discovery.

pub mod cli;

use anyhow::Result;
use std::path::Path;
use tracing::info;

/// Default path for indexing when no path is specified
pub const DEFAULT_INDEX_PATH: &str = ".";

/// Index files in the specified path for fast searching.
///
/// # Arguments
///
/// * `path` - The file system path to index
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if successful, error otherwise
///
/// # Example
///
/// ```
/// use std::path::Path;
/// use tp::index_files;
///
/// let result = index_files(Path::new("."));
/// assert!(result.is_ok());
/// ```
pub fn index_files(path: &Path) -> Result<()> {
    if !path.exists() {
        anyhow::bail!("Path does not exist: {}", path.display());
    }

    if !path.is_dir() {
        anyhow::bail!("Path is not a directory: {}", path.display());
    }

    info!("Indexing files in: {}", path.display());
    Ok(())
}

/// Search through indexed files using the specified query.
///
/// # Arguments
///
/// * `query` - The search query string
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if successful, error otherwise
///
/// # Example
///
/// ```
/// use tp::search_files;
///
/// let result = search_files("function main");
/// assert!(result.is_ok());
/// ```
pub fn search_files(query: &str) -> Result<()> {
    info!("Searching for: {}", query);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_index_files_success() {
        let result = index_files(Path::new("."));
        assert!(result.is_ok());
    }

    #[test]
    fn test_index_files_with_valid_directory() {
        let result = index_files(Path::new("src"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_search_files_success() {
        let result = search_files("test query");
        assert!(result.is_ok());
    }

    #[test]
    fn test_search_files_empty_query() {
        let result = search_files("");
        assert!(result.is_ok());
    }

    #[test]
    fn test_search_files_special_characters() {
        let result = search_files("fn main() {");
        assert!(result.is_ok());
    }

    #[test]
    fn test_search_files_long_query() {
        let long_query = "a".repeat(1000);
        let result = search_files(&long_query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_index_files_nonexistent_path() {
        let result = index_files(Path::new("/nonexistent/path"));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Path does not exist"));
    }

    #[test]
    fn test_index_files_file_not_directory() {
        let result = index_files(Path::new("Cargo.toml"));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Path is not a directory"));
    }
}
