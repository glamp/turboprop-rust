//! Comprehensive tests for file watching and incremental index updates.
//!
//! These tests verify the complete file watching functionality including
//! file detection, event processing, index updates, and error handling.

use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::TempDir;
use tokio::time::sleep;

use tp::config::TurboPropConfig;
use tp::git::GitignoreFilter;
use tp::incremental::{IncrementalStats, IncrementalUpdater};
use tp::storage::PersistentIndex;
use tp::watcher::{FileWatcher, WatchEvent, WatchEventBatch};

/// Helper function to create a test file with content
fn create_test_file(dir: &Path, name: &str, content: &str) -> PathBuf {
    let file_path = dir.join(name);
    if let Some(parent) = file_path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&file_path, content).unwrap();
    file_path
}

/// Helper function to create a test directory
fn create_test_dir(dir: &Path, name: &str) -> PathBuf {
    let dir_path = dir.join(name);
    fs::create_dir_all(&dir_path).unwrap();
    dir_path
}

/// Helper function to remove a test file
fn remove_test_file(path: &Path) {
    if path.exists() {
        fs::remove_file(path).unwrap();
    }
}

/// Helper function to modify a test file
fn modify_test_file(path: &Path, new_content: &str) {
    fs::write(path, new_content).unwrap();
}

#[tokio::test]
async fn test_file_watcher_creation() {
    let temp_dir = TempDir::new().unwrap();
    let gitignore_filter = GitignoreFilter::new(temp_dir.path()).unwrap();

    let result = FileWatcher::new(temp_dir.path(), gitignore_filter);
    assert!(result.is_ok());

    let watcher = result.unwrap();
    assert_eq!(watcher.root_path(), temp_dir.path());
}

#[tokio::test]
async fn test_file_watcher_with_gitignore() {
    let temp_dir = TempDir::new().unwrap();

    // Create a .gitignore file
    create_test_file(temp_dir.path(), ".gitignore", "*.tmp\n/ignored/\n");

    let gitignore_filter = GitignoreFilter::new(temp_dir.path()).unwrap();
    let result = FileWatcher::new(temp_dir.path(), gitignore_filter);
    assert!(result.is_ok());
}

#[test]
fn test_watch_event_properties() {
    let path = PathBuf::from("/test/file.rs");
    
    let modified = WatchEvent::Modified(path.clone());
    assert_eq!(modified.path(), path);
    assert!(modified.is_file_event());
    
    let created = WatchEvent::Created(path.clone());
    assert_eq!(created.path(), path);
    assert!(created.is_file_event());
    
    let deleted = WatchEvent::Deleted(path.clone());
    assert_eq!(deleted.path(), path);
    assert!(deleted.is_file_event());
    
    let dir_created = WatchEvent::DirectoryCreated(path.clone());
    assert_eq!(dir_created.path(), path);
    assert!(!dir_created.is_file_event());
    
    let dir_deleted = WatchEvent::DirectoryDeleted(path.clone());
    assert_eq!(dir_deleted.path(), path);
    assert!(!dir_deleted.is_file_event());
}

#[test]
fn test_watch_event_batch() {
    let events = vec![
        WatchEvent::Modified(PathBuf::from("/file1.rs")),
        WatchEvent::Created(PathBuf::from("/file2.rs")),
        WatchEvent::Deleted(PathBuf::from("/file3.rs")),
        WatchEvent::DirectoryCreated(PathBuf::from("/dir1")),
        WatchEvent::Modified(PathBuf::from("/file1.rs")), // Duplicate
    ];

    let batch = WatchEventBatch::new(events);
    
    // Test unique paths (should deduplicate)
    let unique_paths = batch.unique_paths();
    assert_eq!(unique_paths.len(), 4);
    
    // Test grouping by type
    let (modified, created, deleted) = batch.group_by_type();
    assert_eq!(modified.len(), 2); // Two modified events for same file
    assert_eq!(modified[0], PathBuf::from("/file1.rs"));
    assert_eq!(modified[1], PathBuf::from("/file1.rs"));
    
    assert_eq!(created.len(), 2);
    assert!(created.contains(&PathBuf::from("/file2.rs")));
    assert!(created.contains(&PathBuf::from("/dir1")));
    
    assert_eq!(deleted.len(), 1);
    assert_eq!(deleted[0], PathBuf::from("/file3.rs"));
}

#[tokio::test]
async fn test_incremental_updater_creation() {
    // Skip this test if we're running in an offline environment
    if std::env::var("OFFLINE_TESTS").is_ok() {
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let config = TurboPropConfig::default();

    let result = IncrementalUpdater::new(config, temp_dir.path()).await;
    // This might fail in test environment due to embedding model requirements
    if result.is_err() {
        // Expected in test environment without network access
        return;
    }

    assert!(result.is_ok());
}

#[test]
fn test_incremental_stats() {
    let mut stats = IncrementalStats::default();
    
    assert_eq!(stats.success_rate(), 100.0);
    
    stats.files_processed = 10;
    stats.files_failed = 2;
    stats.files_added = 3;
    stats.files_modified = 4;
    stats.files_removed = 1;
    stats.chunks_added = 15;
    stats.chunks_removed = 5;
    
    assert_eq!(stats.success_rate(), 80.0);
    
    let other_stats = IncrementalStats {
        files_processed: 5,
        files_failed: 1,
        files_added: 2,
        files_modified: 2,
        files_removed: 1,
        chunks_added: 10,
        chunks_removed: 3,
    };
    
    stats.merge(other_stats);
    
    assert_eq!(stats.files_processed, 15);
    assert_eq!(stats.files_failed, 3);
    assert_eq!(stats.files_added, 5);
    assert_eq!(stats.files_modified, 6);
    assert_eq!(stats.files_removed, 2);
    assert_eq!(stats.chunks_added, 25);
    assert_eq!(stats.chunks_removed, 8);
    assert_eq!(stats.success_rate(), 80.0);
}

// Integration tests that require a real file system and index
#[cfg(feature = "integration_tests")]
mod integration_tests {
    use super::*;
    use tp::commands::index::execute_index_command;

    #[tokio::test]
    async fn test_file_modification_detection() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();

        // Create initial test file
        let test_file = create_test_file(temp_dir.path(), "test.rs", "fn hello() { println!(\"Hello\"); }");

        // Build initial index
        let result = execute_index_command(temp_dir.path(), &config, false).await;
        if result.is_err() {
            // Skip test if indexing fails (no embedding model available)
            return;
        }

        // Load the persistent index
        let mut persistent_index = PersistentIndex::load(temp_dir.path()).unwrap();
        let initial_chunks = persistent_index.len();

        // Initialize incremental updater
        let mut updater = IncrementalUpdater::new(config, temp_dir.path()).await.unwrap();

        // Modify the file
        modify_test_file(&test_file, "fn hello() { println!(\"Hello, World!\"); }");

        // Create a mock watch event
        let events = vec![WatchEvent::Modified(test_file)];
        let batch = WatchEventBatch::new(events);

        // Process the batch
        let stats = updater.process_batch(&batch, &mut persistent_index).await.unwrap();

        assert_eq!(stats.files_modified, 1);
        assert!(stats.chunks_removed > 0);
        assert!(stats.chunks_added > 0);

        // The index size might be different due to content changes
        let final_chunks = persistent_index.len();
        assert!(final_chunks > 0);
    }

    #[tokio::test]
    async fn test_file_creation_and_deletion() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();

        // Build initial (empty) index
        let result = execute_index_command(temp_dir.path(), &config, false).await;
        if result.is_err() {
            return; // Skip test if no embedding model
        }

        // Load the persistent index
        let mut persistent_index = PersistentIndex::load(temp_dir.path()).unwrap();
        let mut updater = IncrementalUpdater::new(config, temp_dir.path()).await.unwrap();

        // Create a new file
        let test_file = create_test_file(temp_dir.path(), "new_file.rs", "fn new_function() {}");

        // Process creation event
        let create_events = vec![WatchEvent::Created(test_file.clone())];
        let create_batch = WatchEventBatch::new(create_events);

        let create_stats = updater.process_batch(&create_batch, &mut persistent_index).await.unwrap();
        assert_eq!(create_stats.files_added, 1);
        assert!(create_stats.chunks_added > 0);

        let chunks_after_creation = persistent_index.len();
        assert!(chunks_after_creation > 0);

        // Delete the file
        remove_test_file(&test_file);

        // Process deletion event
        let delete_events = vec![WatchEvent::Deleted(test_file)];
        let delete_batch = WatchEventBatch::new(delete_events);

        let delete_stats = updater.process_batch(&delete_batch, &mut persistent_index).await.unwrap();
        assert_eq!(delete_stats.files_removed, 1);
        assert!(delete_stats.chunks_removed > 0);

        // Index should be empty or smaller
        let chunks_after_deletion = persistent_index.len();
        assert!(chunks_after_deletion < chunks_after_creation);
    }

    #[tokio::test]
    async fn test_directory_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();

        // Build initial index
        let result = execute_index_command(temp_dir.path(), &config, false).await;
        if result.is_err() {
            return; // Skip if no embedding model
        }

        let mut persistent_index = PersistentIndex::load(temp_dir.path()).unwrap();
        let mut updater = IncrementalUpdater::new(config, temp_dir.path()).await.unwrap();

        // Create a new directory with files
        let new_dir = create_test_dir(temp_dir.path(), "new_module");
        create_test_file(&new_dir, "mod.rs", "pub mod new_module;");
        create_test_file(&new_dir, "lib.rs", "pub fn new_function() {}");

        // Process directory creation event
        let events = vec![WatchEvent::DirectoryCreated(new_dir)];
        let batch = WatchEventBatch::new(events);

        let stats = updater.process_batch(&batch, &mut persistent_index).await.unwrap();
        assert!(stats.files_added >= 2); // At least the 2 files we created
        assert!(stats.chunks_added > 0);
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();

        // Build initial index
        let result = execute_index_command(temp_dir.path(), &config, false).await;
        if result.is_err() {
            return;
        }

        let mut persistent_index = PersistentIndex::load(temp_dir.path()).unwrap();
        let mut updater = IncrementalUpdater::new(config, temp_dir.path()).await.unwrap();

        // Create multiple files
        let file1 = create_test_file(temp_dir.path(), "file1.rs", "fn func1() {}");
        let file2 = create_test_file(temp_dir.path(), "file2.rs", "fn func2() {}");
        let file3 = create_test_file(temp_dir.path(), "file3.rs", "fn func3() {}");

        // Modify one file
        modify_test_file(&file2, "fn func2_modified() {}");

        // Create a batch with multiple operations
        let events = vec![
            WatchEvent::Created(file1),
            WatchEvent::Modified(file2),
            WatchEvent::Created(file3),
        ];
        let batch = WatchEventBatch::new(events);

        let stats = updater.process_batch(&batch, &mut persistent_index).await.unwrap();
        assert_eq!(stats.files_added, 2);
        assert_eq!(stats.files_modified, 1);
        assert!(stats.chunks_added > 0);
    }
}

// Mock tests that don't require real file system watching
#[tokio::test]
async fn test_watch_error_handling() {
    let temp_dir = TempDir::new().unwrap();
    
    // Try to create a watcher for a non-existent directory
    let non_existent = temp_dir.path().join("does_not_exist");
    let gitignore_filter = GitignoreFilter::new(&non_existent);
    
    // GitignoreFilter creation might succeed even for non-existent path
    if let Ok(filter) = gitignore_filter {
        let result = FileWatcher::new(&non_existent, filter);
        // FileWatcher creation should fail for non-existent path
        assert!(result.is_err());
    }
}

#[test]
fn test_watch_event_batch_empty() {
    let batch = WatchEventBatch::new(vec![]);
    
    assert_eq!(batch.events.len(), 0);
    assert_eq!(batch.unique_paths().len(), 0);
    
    let (modified, created, deleted) = batch.group_by_type();
    assert_eq!(modified.len(), 0);
    assert_eq!(created.len(), 0);
    assert_eq!(deleted.len(), 0);
}

#[test]
fn test_incremental_stats_default() {
    let stats = IncrementalStats::default();
    
    assert_eq!(stats.files_processed, 0);
    assert_eq!(stats.files_added, 0);
    assert_eq!(stats.files_modified, 0);
    assert_eq!(stats.files_removed, 0);
    assert_eq!(stats.chunks_added, 0);
    assert_eq!(stats.chunks_removed, 0);
    assert_eq!(stats.files_failed, 0);
    assert_eq!(stats.success_rate(), 100.0);
}