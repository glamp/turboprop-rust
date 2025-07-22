use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub last_modified: std::time::SystemTime,
    pub is_git_tracked: bool,
}

#[derive(Debug, Clone)]
pub struct FileDiscoveryConfig {
    pub max_filesize_bytes: Option<u64>,
    pub respect_gitignore: bool,
    pub include_untracked: bool,
}

impl Default for FileDiscoveryConfig {
    fn default() -> Self {
        Self {
            max_filesize_bytes: None,
            respect_gitignore: true,
            include_untracked: false,
        }
    }
}

impl FileDiscoveryConfig {
    pub fn with_max_filesize(mut self, max_size: u64) -> Self {
        self.max_filesize_bytes = Some(max_size);
        self
    }

    pub fn with_gitignore_respect(mut self, respect: bool) -> Self {
        self.respect_gitignore = respect;
        self
    }

    pub fn with_untracked(mut self, include: bool) -> Self {
        self.include_untracked = include;
        self
    }
}

pub fn parse_filesize(input: &str) -> Result<u64, String> {
    let input = input.to_lowercase();

    if let Some(stripped) = input.strip_suffix("kb") {
        stripped
            .parse::<u64>()
            .map(|n| n * 1024)
            .map_err(|_| format!("Invalid number in filesize: {}", input))
    } else if let Some(stripped) = input.strip_suffix("mb") {
        stripped
            .parse::<u64>()
            .map(|n| n * 1024 * 1024)
            .map_err(|_| format!("Invalid number in filesize: {}", input))
    } else if let Some(stripped) = input.strip_suffix("gb") {
        stripped
            .parse::<u64>()
            .map(|n| n * 1024 * 1024 * 1024)
            .map_err(|_| format!("Invalid number in filesize: {}", input))
    } else if let Some(stripped) = input.strip_suffix("b") {
        stripped
            .parse::<u64>()
            .map_err(|_| format!("Invalid number in filesize: {}", input))
    } else {
        input
            .parse::<u64>()
            .map_err(|_| format!("Invalid filesize format: {}", input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_filesize() {
        assert_eq!(parse_filesize("100"), Ok(100));
        assert_eq!(parse_filesize("100b"), Ok(100));
        assert_eq!(parse_filesize("2kb"), Ok(2048));
        assert_eq!(parse_filesize("2KB"), Ok(2048));
        assert_eq!(parse_filesize("5mb"), Ok(5 * 1024 * 1024));
        assert_eq!(parse_filesize("1gb"), Ok(1024 * 1024 * 1024));

        assert!(parse_filesize("invalid").is_err());
        assert!(parse_filesize("2.5mb").is_err());
    }

    #[test]
    fn test_file_discovery_config() {
        let config = FileDiscoveryConfig::default()
            .with_max_filesize(1024)
            .with_gitignore_respect(false)
            .with_untracked(true);

        assert_eq!(config.max_filesize_bytes, Some(1024));
        assert!(!config.respect_gitignore);
        assert!(config.include_untracked);
    }
}
