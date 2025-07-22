use crate::git::GitRepo;
use crate::types::{FileDiscoveryConfig, FileMetadata};
use anyhow::{Context, Result};
use std::collections::HashSet;
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;
use walkdir::WalkDir;

pub struct FileDiscovery {
    config: FileDiscoveryConfig,
}

impl FileDiscovery {
    pub fn new(config: FileDiscoveryConfig) -> Self {
        Self { config }
    }

    pub fn discover_files(&self, path: &Path) -> Result<Vec<FileMetadata>> {
        if GitRepo::is_git_repo(path) {
            self.discover_git_files(path)
        } else {
            self.discover_filesystem_files(path)
        }
    }

    fn discover_git_files(&self, path: &Path) -> Result<Vec<FileMetadata>> {
        let git_repo = GitRepo::discover(path).context("Failed to open git repository")?;

        let git_files = git_repo
            .get_all_files(self.config.include_untracked)
            .context("Failed to get files from git repository")?;

        let mut file_metadata = Vec::new();

        for file_path in git_files {
            if let Some(metadata) = self.get_file_metadata(&file_path, true)? {
                file_metadata.push(metadata);
            }
        }

        Ok(file_metadata)
    }

    fn discover_filesystem_files(&self, path: &Path) -> Result<Vec<FileMetadata>> {
        let mut file_metadata = Vec::new();

        for entry in WalkDir::new(path).follow_links(false) {
            let entry = entry.context("Failed to read directory entry")?;

            if entry.file_type().is_file() {
                let file_path = entry.path().to_path_buf();
                if let Some(metadata) = self.get_file_metadata(&file_path, false)? {
                    file_metadata.push(metadata);
                }
            }
        }

        Ok(file_metadata)
    }

    fn get_file_metadata(&self, path: &Path, is_git_tracked: bool) -> Result<Option<FileMetadata>> {
        let metadata = std::fs::metadata(path)
            .with_context(|| format!("Failed to get metadata for file: {}", path.display()))?;

        if let Some(max_size) = self.config.max_filesize_bytes {
            if metadata.len() > max_size {
                return Ok(None);
            }
        }

        let last_modified = metadata
            .modified()
            .with_context(|| format!("Failed to get modification time for: {}", path.display()))?;

        Ok(Some(FileMetadata {
            path: path.to_path_buf(),
            size_bytes: metadata.len(),
            last_modified,
            is_git_tracked,
        }))
    }

    pub fn filter_files_by_patterns(
        &self,
        files: Vec<FileMetadata>,
        patterns: &[String],
    ) -> Vec<FileMetadata> {
        if patterns.is_empty() {
            return files;
        }

        files
            .into_iter()
            .filter(|file| {
                let path_str = file.path.to_string_lossy().to_lowercase();
                patterns.iter().any(|pattern| {
                    let pattern = pattern.to_lowercase();
                    if pattern.starts_with('*') && pattern.ends_with('*') {
                        let inner = &pattern[1..pattern.len() - 1];
                        path_str.contains(inner)
                    } else if let Some(stripped) = pattern.strip_prefix('*') {
                        path_str.ends_with(stripped)
                    } else if pattern.ends_with('*') {
                        path_str.starts_with(&pattern[..pattern.len() - 1])
                    } else {
                        path_str.contains(&pattern)
                    }
                })
            })
            .collect()
    }
}

/// Discover files in the given directory using default configuration
pub fn discover_files(
    path: &Path,
    config: &FileDiscoveryConfig,
) -> Result<Vec<std::path::PathBuf>> {
    let discovery = FileDiscovery::new(config.clone());
    let file_metadata = discovery.discover_files(path)?;

    Ok(file_metadata
        .into_iter()
        .map(|metadata| metadata.path)
        .collect())
}

pub fn get_common_file_extensions() -> HashSet<&'static str> {
    let mut extensions = HashSet::new();

    // Code files
    extensions.insert("rs");
    extensions.insert("py");
    extensions.insert("js");
    extensions.insert("ts");
    extensions.insert("jsx");
    extensions.insert("tsx");
    extensions.insert("java");
    extensions.insert("cpp");
    extensions.insert("c");
    extensions.insert("h");
    extensions.insert("hpp");
    extensions.insert("cs");
    extensions.insert("go");
    extensions.insert("php");
    extensions.insert("rb");
    extensions.insert("swift");
    extensions.insert("kt");
    extensions.insert("scala");
    extensions.insert("clj");
    extensions.insert("hs");
    extensions.insert("elm");

    // Web files
    extensions.insert("html");
    extensions.insert("css");
    extensions.insert("scss");
    extensions.insert("sass");
    extensions.insert("less");
    extensions.insert("vue");
    extensions.insert("svelte");

    // Config files
    extensions.insert("json");
    extensions.insert("yaml");
    extensions.insert("yml");
    extensions.insert("toml");
    extensions.insert("xml");
    extensions.insert("ini");
    extensions.insert("cfg");
    extensions.insert("conf");

    // Documentation
    extensions.insert("md");
    extensions.insert("rst");
    extensions.insert("txt");

    extensions
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_file_discovery_creation() {
        let config = FileDiscoveryConfig::default();
        let discovery = FileDiscovery::new(config);
        assert!(discovery.config.respect_gitignore);
    }

    #[test]
    fn test_discover_files_in_current_dir() {
        let config = FileDiscoveryConfig::default();
        let discovery = FileDiscovery::new(config);
        let current_dir = env::current_dir().unwrap();
        let result = discovery.discover_files(&current_dir);
        assert!(result.is_ok());

        let files = result.unwrap();
        assert!(!files.is_empty());
    }

    #[test]
    fn test_filter_files_by_patterns() {
        let config = FileDiscoveryConfig::default();
        let discovery = FileDiscovery::new(config);

        let test_files = vec![
            FileMetadata {
                path: PathBuf::from("test.rs"),
                size_bytes: 100,
                last_modified: std::time::SystemTime::now(),
                is_git_tracked: true,
            },
            FileMetadata {
                path: PathBuf::from("main.py"),
                size_bytes: 200,
                last_modified: std::time::SystemTime::now(),
                is_git_tracked: true,
            },
        ];

        let patterns = vec!["*.rs".to_string()];
        let filtered = discovery.filter_files_by_patterns(test_files, &patterns);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].path, PathBuf::from("test.rs"));
    }

    #[test]
    fn test_common_file_extensions() {
        let extensions = get_common_file_extensions();
        assert!(extensions.contains("rs"));
        assert!(extensions.contains("py"));
        assert!(extensions.contains("js"));
        assert!(extensions.contains("md"));
    }
}
