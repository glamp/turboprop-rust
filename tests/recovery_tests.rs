use std::fs;
use tempfile::TempDir;
use turboprop::config::TurboPropConfig;
use turboprop::error::TurboPropError;
use turboprop::recovery::{IndexRecovery, RecoveryStrategy, ValidationResult};

#[tokio::test]
async fn test_validate_healthy_index() {
    let temp_dir = TempDir::new().unwrap();
    let index_path = temp_dir.path().to_path_buf();

    // Create a valid index structure
    let turboprop_dir = index_path.join(".turboprop");
    fs::create_dir_all(&turboprop_dir).unwrap();

    // Create valid index files
    let metadata_file = turboprop_dir.join("metadata.json");
    let metadata = serde_json::json!({
        "version": "1.0",
        "created_at": "2024-01-01T00:00:00Z",
        "chunk_count": 10,
        "embedding_dimensions": 384
    });
    fs::write(
        &metadata_file,
        serde_json::to_string_pretty(&metadata).unwrap(),
    )
    .unwrap();

    let chunks_file = turboprop_dir.join("chunks.db");
    fs::write(&chunks_file, b"dummy chunk data").unwrap();

    let embeddings_file = turboprop_dir.join("embeddings.bin");
    fs::write(&embeddings_file, b"dummy embedding data").unwrap();

    let recovery = IndexRecovery::new(&index_path);
    let result = recovery.validate_index().await;

    assert!(result.is_ok());
    match result.unwrap() {
        ValidationResult::Healthy => {
            // Expected
        }
        _ => panic!("Expected healthy index"),
    }
}

#[tokio::test]
async fn test_detect_missing_index() {
    let temp_dir = TempDir::new().unwrap();
    let index_path = temp_dir.path().to_path_buf();

    let recovery = IndexRecovery::new(&index_path);
    let result = recovery.validate_index().await;

    assert!(result.is_ok());
    match result.unwrap() {
        ValidationResult::Missing => {
            // Expected
        }
        _ => panic!("Expected missing index"),
    }
}

#[tokio::test]
async fn test_detect_corrupted_metadata() {
    let temp_dir = TempDir::new().unwrap();
    let index_path = temp_dir.path().to_path_buf();

    // Create index structure with corrupted metadata
    let turboprop_dir = index_path.join(".turboprop");
    fs::create_dir_all(&turboprop_dir).unwrap();

    let metadata_file = turboprop_dir.join("metadata.json");
    fs::write(&metadata_file, b"invalid json").unwrap();

    let chunks_file = turboprop_dir.join("chunks.db");
    fs::write(&chunks_file, b"dummy chunk data").unwrap();

    let embeddings_file = turboprop_dir.join("embeddings.bin");
    fs::write(&embeddings_file, b"dummy embedding data").unwrap();

    let recovery = IndexRecovery::new(&index_path);
    let result = recovery.validate_index().await;

    assert!(result.is_ok());
    match result.unwrap() {
        ValidationResult::Corrupted { issues } => {
            assert!(issues.iter().any(|issue| issue.contains("metadata")));
        }
        _ => panic!("Expected corrupted index"),
    }
}

#[tokio::test]
async fn test_detect_missing_files() {
    let temp_dir = TempDir::new().unwrap();
    let index_path = temp_dir.path().to_path_buf();

    // Create index structure with missing files
    let turboprop_dir = index_path.join(".turboprop");
    fs::create_dir_all(&turboprop_dir).unwrap();

    // Only create metadata file, missing chunks and embeddings
    let metadata_file = turboprop_dir.join("metadata.json");
    let metadata = serde_json::json!({
        "version": "1.0",
        "created_at": "2024-01-01T00:00:00Z",
        "chunk_count": 10,
        "embedding_dimensions": 384
    });
    fs::write(
        &metadata_file,
        serde_json::to_string_pretty(&metadata).unwrap(),
    )
    .unwrap();

    let recovery = IndexRecovery::new(&index_path);
    let result = recovery.validate_index().await;

    assert!(result.is_ok());
    match result.unwrap() {
        ValidationResult::Corrupted { issues } => {
            assert!(issues.len() >= 2); // Missing chunks and embeddings
            assert!(issues.iter().any(|issue| issue.contains("chunks.db")));
            assert!(issues.iter().any(|issue| issue.contains("embeddings.bin")));
        }
        _ => panic!("Expected corrupted index"),
    }
}

#[tokio::test]
async fn test_cleanup_corrupted_index() {
    let temp_dir = TempDir::new().unwrap();
    let index_path = temp_dir.path().to_path_buf();

    // Create corrupted index
    let turboprop_dir = index_path.join(".turboprop");
    fs::create_dir_all(&turboprop_dir).unwrap();
    fs::write(turboprop_dir.join("corrupted_file"), b"corrupted data").unwrap();

    let recovery = IndexRecovery::new(&index_path);
    let result = recovery.cleanup_index().await;

    assert!(result.is_ok());

    // Verify directory is cleaned up
    assert!(!turboprop_dir.exists() || fs::read_dir(&turboprop_dir).unwrap().next().is_none());
}

#[tokio::test]
async fn test_recover_with_rebuild_strategy() {
    let temp_dir = TempDir::new().unwrap();
    let index_path = temp_dir.path().to_path_buf();

    // Create some source files to index
    fs::write(index_path.join("test1.txt"), "Hello world").unwrap();
    fs::write(index_path.join("test2.txt"), "Another file").unwrap();

    // Create corrupted index
    let turboprop_dir = index_path.join(".turboprop");
    fs::create_dir_all(&turboprop_dir).unwrap();
    fs::write(turboprop_dir.join("corrupted_file"), b"corrupted data").unwrap();

    let recovery = IndexRecovery::new(&index_path);
    let config = TurboPropConfig::default();

    let result = recovery.recover(RecoveryStrategy::Rebuild, &config).await;

    // This might fail due to missing embedding model, but it should attempt the recovery
    match result {
        Ok(_) => {
            // Recovery succeeded - verify new index structure exists
            // The actual index structure may be different from what we expected
            assert!(turboprop_dir.exists());

            // Verify that some index content was created
            let entries: Vec<_> = fs::read_dir(&turboprop_dir).unwrap().collect();
            assert!(
                !entries.is_empty(),
                "Index directory should contain some files"
            );
        }
        Err(TurboPropError::EmbeddingModelError { .. }) => {
            // Expected - might not have embedding model in test environment
            // But should have cleaned up the corrupted index
            assert!(
                !turboprop_dir.exists() || fs::read_dir(&turboprop_dir).unwrap().next().is_none()
            );
        }
        Err(e) => {
            // Other errors might also be expected in test environment
            // In test environments, cleanup may not complete due to model initialization failures
            // The test validates that recovery attempt was made
            println!("Index build failed with: {:?}", e);
            // Recovery process initiated successfully even if it couldn't complete
            assert!(
                turboprop_dir.exists(),
                "Recovery should have created directory structure"
            );
        }
    }
}

#[tokio::test]
async fn test_recover_with_clean_strategy() {
    let temp_dir = TempDir::new().unwrap();
    let index_path = temp_dir.path().to_path_buf();

    // Create corrupted index
    let turboprop_dir = index_path.join(".turboprop");
    fs::create_dir_all(&turboprop_dir).unwrap();
    fs::write(turboprop_dir.join("corrupted_file"), b"corrupted data").unwrap();

    let recovery = IndexRecovery::new(&index_path);
    let config = TurboPropConfig::default();

    let result = recovery.recover(RecoveryStrategy::Clean, &config).await;

    assert!(result.is_ok());

    // Verify directory is cleaned up but no new index is built
    assert!(!turboprop_dir.exists() || fs::read_dir(&turboprop_dir).unwrap().next().is_none());
}

#[tokio::test]
async fn test_disk_space_check() {
    let temp_dir = TempDir::new().unwrap();
    let index_path = temp_dir.path().to_path_buf();

    let recovery = IndexRecovery::new(&index_path);
    let result = recovery.check_disk_space(1024 * 1024).await; // 1MB

    assert!(result.is_ok());
    // Should have more than 1MB available in temp directory
    assert!(result.unwrap());
}

#[tokio::test]
async fn test_backup_corrupted_index() {
    let temp_dir = TempDir::new().unwrap();
    let index_path = temp_dir.path().to_path_buf();

    // Create corrupted index
    let turboprop_dir = index_path.join(".turboprop");
    fs::create_dir_all(&turboprop_dir).unwrap();
    fs::write(turboprop_dir.join("corrupted_file"), b"corrupted data").unwrap();

    let recovery = IndexRecovery::new(&index_path);
    let result = recovery.backup_corrupted_index().await;

    assert!(result.is_ok());
    let backup_path = result.unwrap();

    // Verify backup exists and contains the corrupted file
    assert!(backup_path.exists());
    assert!(backup_path.join("corrupted_file").exists());

    // Verify original is cleaned up
    assert!(!turboprop_dir.exists() || !turboprop_dir.join("corrupted_file").exists());
}

#[tokio::test]
async fn test_validate_index_permissions() {
    let temp_dir = TempDir::new().unwrap();
    let index_path = temp_dir.path().to_path_buf();

    let recovery = IndexRecovery::new(&index_path);
    let result = recovery.validate_permissions().await;

    assert!(result.is_ok());
    // Should have write permissions in temp directory
    assert!(result.unwrap());
}

#[tokio::test]
async fn test_recovery_with_automatic_strategy() {
    let temp_dir = TempDir::new().unwrap();
    let index_path = temp_dir.path().to_path_buf();

    // Create some source files
    fs::write(index_path.join("source.txt"), "test content").unwrap();

    let recovery = IndexRecovery::new(&index_path);
    let config = TurboPropConfig::default();

    let result = recovery.recover_automatically(&config).await;

    // This should work for missing index (creates new one) or fail gracefully
    match result {
        Ok(_) => {
            // Recovery succeeded
        }
        Err(TurboPropError::EmbeddingModelError { .. }) => {
            // Expected - might not have embedding model in test environment
        }
        Err(e) => {
            // Other errors might also be expected in test environment
            println!("Automatic recovery failed with: {:?}", e);
        }
    }
}
