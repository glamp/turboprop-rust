use anyhow::{Context, Result};
use git2::{Repository, Status, StatusOptions};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub struct GitRepo {
    repo: Repository,
    root_path: PathBuf,
}

impl GitRepo {
    pub fn discover(path: &Path) -> Result<Self> {
        let repo = Repository::discover(path).with_context(|| {
            format!(
                "Failed to discover git repository from path: {}",
                path.display()
            )
        })?;

        let root_path = repo
            .workdir()
            .ok_or_else(|| anyhow::anyhow!("Failed to get git working directory: repository has no working directory"))?
            .to_path_buf();

        Ok(Self { repo, root_path })
    }

    pub fn is_git_repo(path: &Path) -> bool {
        Repository::discover(path).is_ok()
    }

    pub fn root_path(&self) -> &Path {
        &self.root_path
    }

    pub fn get_tracked_files(&self) -> Result<HashSet<PathBuf>> {
        let mut files = HashSet::new();
        let mut index = self.repo.index()?;
        index.update_all(std::iter::empty::<&str>(), None)?;

        for entry in index.iter() {
            let path = PathBuf::from(String::from_utf8_lossy(&entry.path).to_string());
            let full_path = self.root_path.join(&path);
            if full_path.is_file() {
                files.insert(full_path);
            }
        }

        Ok(files)
    }

    pub fn get_untracked_files(&self) -> Result<HashSet<PathBuf>> {
        let mut files = HashSet::new();
        let mut opts = StatusOptions::new();
        opts.include_untracked(true).include_ignored(false);

        let statuses = self.repo.statuses(Some(&mut opts))?;

        for entry in statuses.iter() {
            let flags = entry.status();
            if flags.contains(Status::WT_NEW) {
                let path = PathBuf::from(entry.path().unwrap_or(""));
                let full_path = self.root_path.join(&path);
                if full_path.is_file() {
                    files.insert(full_path);
                }
            }
        }

        Ok(files)
    }

    pub fn get_all_files(&self, include_untracked: bool) -> Result<HashSet<PathBuf>> {
        let mut all_files = self.get_tracked_files()?;

        if include_untracked {
            let untracked = self.get_untracked_files()?;
            all_files.extend(untracked);
        }

        Ok(all_files)
    }

    pub fn is_ignored(&self, path: &Path) -> Result<bool> {
        let relative_path = path.strip_prefix(&self.root_path).unwrap_or(path);

        let status = self.repo.status_file(relative_path)?;
        Ok(status.contains(Status::IGNORED))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_is_git_repo() {
        let current_dir = env::current_dir().unwrap();
        assert!(GitRepo::is_git_repo(&current_dir));
    }

    #[test]
    fn test_discover_git_repo() {
        let current_dir = env::current_dir().unwrap();
        let git_repo = GitRepo::discover(&current_dir);
        assert!(git_repo.is_ok());
    }

    #[test]
    fn test_non_git_directory() {
        let temp_dir = std::env::temp_dir();
        assert!(!GitRepo::is_git_repo(&temp_dir));
    }
}
