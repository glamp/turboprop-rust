//! Index recovery and validation functionality.
//!
//! This module provides utilities for detecting corrupted indices, cleaning them up,
//! and rebuilding them automatically when needed.

use crate::config::TurboPropConfig;
use crate::error::{TurboPropError, TurboPropResult};
use crate::index::PersistentChunkIndex;
use serde_json;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Strategy for recovering from index corruption.
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryStrategy {
    /// Clean up the corrupted index without rebuilding.
    Clean,
    /// Clean up and rebuild the index from source files.
    Rebuild,
    /// Try to repair in-place if possible, otherwise rebuild.
    Repair,
}

/// Result of index validation.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationResult {
    /// Index is healthy and ready to use.
    Healthy,
    /// Index is missing (needs to be created).
    Missing,
    /// Index is corrupted and needs recovery.
    Corrupted { issues: Vec<String> },
}

/// Index recovery coordinator.
pub struct IndexRecovery {
    index_path: PathBuf,
}

impl IndexRecovery {
    /// Create a new index recovery instance.
    pub fn new(index_path: &Path) -> Self {
        Self {
            index_path: index_path.to_path_buf(),
        }
    }

    /// Validate the index and detect any issues.
    pub async fn validate_index(&self) -> TurboPropResult<ValidationResult> {
        let turboprop_dir = self.index_path.join(".turboprop");

        // Check if index directory exists
        if !turboprop_dir.exists() {
            debug!("Index directory not found: {}", turboprop_dir.display());
            return Ok(ValidationResult::Missing);
        }

        let mut issues = Vec::new();

        // Check for required files
        let metadata_file = turboprop_dir.join("metadata.json");
        let chunks_file = turboprop_dir.join("chunks.db");
        let embeddings_file = turboprop_dir.join("embeddings.bin");

        if !metadata_file.exists() {
            issues.push("Missing metadata.json file".to_string());
        } else {
            // Validate metadata file
            if let Err(e) = self.validate_metadata_file(&metadata_file).await {
                issues.push(format!("Invalid metadata.json: {}", e));
            }
        }

        if !chunks_file.exists() {
            issues.push("Missing chunks.db file".to_string());
        }

        if !embeddings_file.exists() {
            issues.push("Missing embeddings.bin file".to_string());
        }

        // Check file permissions
        for file in &[&metadata_file, &chunks_file, &embeddings_file] {
            if file.exists() {
                match fs::metadata(file) {
                    Ok(metadata) => {
                        if metadata.len() == 0 {
                            issues.push(format!("Empty file: {}", file.display()));
                        }
                    }
                    Err(e) => {
                        issues.push(format!("Cannot access {}: {}", file.display(), e));
                    }
                }
            }
        }

        if issues.is_empty() {
            Ok(ValidationResult::Healthy)
        } else {
            Ok(ValidationResult::Corrupted { issues })
        }
    }

    /// Validate metadata file structure and content.
    async fn validate_metadata_file(&self, metadata_file: &Path) -> TurboPropResult<()> {
        let content = fs::read_to_string(metadata_file)
            .map_err(|e| TurboPropError::file_system(
                format!("Cannot read metadata file: {}", e),
                Some(metadata_file.to_path_buf())
            ))?;

        let metadata: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| TurboPropError::corrupted_index(
                metadata_file.to_path_buf(),
                format!("Invalid JSON: {}", e)
            ))?;

        // Check required fields
        let required_fields = ["version", "created_at", "chunk_count", "embedding_dimensions"];
        for field in &required_fields {
            if !metadata.get(field).is_some() {
                return Err(TurboPropError::corrupted_index(
                    metadata_file.to_path_buf(),
                    format!("Missing required field: {}", field)
                ));
            }
        }

        Ok(())
    }

    /// Clean up the corrupted index.
    pub async fn cleanup_index(&self) -> TurboPropResult<()> {
        let turboprop_dir = self.index_path.join(".turboprop");

        if !turboprop_dir.exists() {
            debug!("Index directory already clean: {}", turboprop_dir.display());
            return Ok(());
        }

        info!("Cleaning up corrupted index at: {}", turboprop_dir.display());

        fs::remove_dir_all(&turboprop_dir)
            .map_err(|e| TurboPropError::file_system(
                format!("Failed to remove index directory: {}", e),
                Some(turboprop_dir)
            ))?;

        info!("Index cleanup completed");
        Ok(())
    }

    /// Create a backup of the corrupted index before cleaning.
    pub async fn backup_corrupted_index(&self) -> TurboPropResult<PathBuf> {
        let turboprop_dir = self.index_path.join(".turboprop");
        
        if !turboprop_dir.exists() {
            return Err(TurboPropError::index_not_found(turboprop_dir));
        }

        // Create backup directory name with timestamp
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let backup_name = format!(".turboprop_backup_{}", timestamp);
        let backup_path = self.index_path.join(&backup_name);

        info!("Backing up corrupted index to: {}", backup_path.display());

        // Copy the corrupted index to backup location
        self.copy_dir_recursive(&turboprop_dir, &backup_path).await?;

        // Remove the original corrupted index
        fs::remove_dir_all(&turboprop_dir)
            .map_err(|e| TurboPropError::file_system(
                format!("Failed to remove original index: {}", e),
                Some(turboprop_dir)
            ))?;

        info!("Backup created successfully at: {}", backup_path.display());
        Ok(backup_path)
    }

    /// Recursively copy a directory.
    async fn copy_dir_recursive(&self, src: &Path, dst: &Path) -> TurboPropResult<()> {
        fs::create_dir_all(dst)
            .map_err(|e| TurboPropError::file_system(
                format!("Failed to create backup directory: {}", e),
                Some(dst.to_path_buf())
            ))?;

        for entry in fs::read_dir(src)
            .map_err(|e| TurboPropError::file_system(
                format!("Failed to read directory: {}", e),
                Some(src.to_path_buf())
            ))? 
        {
            let entry = entry
                .map_err(|e| TurboPropError::file_system(
                    format!("Failed to read directory entry: {}", e),
                    Some(src.to_path_buf())
                ))?;

            let src_path = entry.path();
            let dst_path = dst.join(entry.file_name());

            if src_path.is_dir() {
                Box::pin(self.copy_dir_recursive(&src_path, &dst_path)).await?;
            } else {
                fs::copy(&src_path, &dst_path)
                    .map_err(|e| TurboPropError::file_system(
                        format!("Failed to copy file: {}", e),
                        Some(src_path)
                    ))?;
            }
        }

        Ok(())
    }

    /// Check if there's sufficient disk space for index operations.
    pub async fn check_disk_space(&self, required_bytes: u64) -> TurboPropResult<bool> {
        match fs::metadata(&self.index_path) {
            Ok(_) => {
                // For simplicity, we'll assume there's enough space
                // In a real implementation, you'd use platform-specific APIs
                // to check available disk space
                debug!("Disk space check passed (simplified implementation)");
                Ok(true)
            }
            Err(e) => Err(TurboPropError::file_system(
                format!("Cannot check disk space: {}", e),
                Some(self.index_path.clone())
            )),
        }
    }

    /// Validate that we have the necessary permissions for index operations.
    pub async fn validate_permissions(&self) -> TurboPropResult<bool> {
        // Check if we can create the .turboprop directory
        let turboprop_dir = self.index_path.join(".turboprop");
        
        match fs::create_dir_all(&turboprop_dir) {
            Ok(_) => {
                // Clean up test directory if it was created
                if turboprop_dir.exists() && fs::read_dir(&turboprop_dir).unwrap().next().is_none() {
                    let _ = fs::remove_dir(&turboprop_dir);
                }
                Ok(true)
            }
            Err(e) => {
                warn!("Permission check failed: {}", e);
                Ok(false)
            }
        }
    }

    /// Recover the index using the specified strategy.
    pub async fn recover(&self, strategy: RecoveryStrategy, config: &TurboPropConfig) -> TurboPropResult<()> {
        info!("Starting index recovery with strategy: {:?}", strategy);

        match strategy {
            RecoveryStrategy::Clean => {
                self.cleanup_index().await?;
                info!("Index cleaned successfully");
            }
            RecoveryStrategy::Rebuild => {
                self.cleanup_index().await?;
                info!("Rebuilding index...");
                
                // Rebuild the index
                let _index = PersistentChunkIndex::build(&self.index_path, config).await?;
                info!("Index rebuilt successfully");
            }
            RecoveryStrategy::Repair => {
                // For now, repair just does a rebuild
                // In the future, this could try more sophisticated repair strategies
                warn!("Repair strategy not fully implemented, falling back to rebuild");
                self.cleanup_index().await?;
                let _index = PersistentChunkIndex::build(&self.index_path, config).await?;
                info!("Index repaired successfully");
            }
        }

        Ok(())
    }

    /// Automatically determine the best recovery strategy and execute it.
    pub async fn recover_automatically(&self, config: &TurboPropConfig) -> TurboPropResult<()> {
        let validation_result = self.validate_index().await?;

        match validation_result {
            ValidationResult::Healthy => {
                info!("Index is healthy, no recovery needed");
                Ok(())
            }
            ValidationResult::Missing => {
                info!("Index is missing, creating new index");
                let _index = PersistentChunkIndex::build(&self.index_path, config).await?;
                info!("New index created successfully");
                Ok(())
            }
            ValidationResult::Corrupted { issues } => {
                warn!("Index is corrupted: {:?}", issues);
                
                // Check permissions and disk space
                let has_permissions = self.validate_permissions().await?;
                if !has_permissions {
                    return Err(TurboPropError::file_permission(
                        self.index_path.clone(),
                        "write".to_string()
                    ));
                }

                let has_space = self.check_disk_space(100 * 1024 * 1024).await?; // 100MB
                if !has_space {
                    return Err(TurboPropError::insufficient_disk_space(
                        100 * 1024 * 1024,
                        0, // We don't know the actual available space
                        self.index_path.clone()
                    ));
                }

                // Create backup before recovery
                let _backup_path = self.backup_corrupted_index().await?;

                // Rebuild the index
                let _index = PersistentChunkIndex::build(&self.index_path, config).await?;
                info!("Index recovered successfully");
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_missing_index_detection() {
        let temp_dir = TempDir::new().unwrap();
        let recovery = IndexRecovery::new(temp_dir.path());
        
        let result = recovery.validate_index().await.unwrap();
        assert_eq!(result, ValidationResult::Missing);
    }

    #[tokio::test]
    async fn test_healthy_index_detection() {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path();
        
        // Create a valid index structure
        let turboprop_dir = index_path.join(".turboprop");
        fs::create_dir_all(&turboprop_dir).unwrap();
        
        let metadata = serde_json::json!({
            "version": "1.0",
            "created_at": "2024-01-01T00:00:00Z",
            "chunk_count": 5,
            "embedding_dimensions": 384
        });
        
        fs::write(
            turboprop_dir.join("metadata.json"), 
            serde_json::to_string_pretty(&metadata).unwrap()
        ).unwrap();
        fs::write(turboprop_dir.join("chunks.db"), b"chunk data").unwrap();
        fs::write(turboprop_dir.join("embeddings.bin"), b"embedding data").unwrap();
        
        let recovery = IndexRecovery::new(index_path);
        let result = recovery.validate_index().await.unwrap();
        
        assert_eq!(result, ValidationResult::Healthy);
    }

    #[tokio::test]
    async fn test_corrupted_index_detection() {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path();
        
        // Create corrupted index structure
        let turboprop_dir = index_path.join(".turboprop");
        fs::create_dir_all(&turboprop_dir).unwrap();
        
        // Invalid JSON metadata
        fs::write(turboprop_dir.join("metadata.json"), b"invalid json").unwrap();
        
        let recovery = IndexRecovery::new(index_path);
        let result = recovery.validate_index().await.unwrap();
        
        match result {
            ValidationResult::Corrupted { issues } => {
                assert!(!issues.is_empty());
            }
            _ => panic!("Expected corrupted index"),
        }
    }
}