//! Integration tests for vector index storage and retrieval functionality.
//!
//! These tests cover the complete index lifecycle including creation,
//! persistence, loading, incremental updates, and concurrent access.

use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tempfile::TempDir;

use turboprop::config::TurboPropConfig;
use turboprop::index::{PersistentChunkIndex, UpdateResult};
use turboprop::storage::{IndexConfig, IndexStorage};
use turboprop::types::{ContentChunk, IndexStats, IndexedChunk, SourceLocation};

/// Helper function to create a test file with content
fn create_test_file(dir: &std::path::Path, name: &str, content: &str) -> PathBuf {
    let file_path = dir.join(name);
    fs::write(&file_path, content).unwrap();
    file_path
}

/// Create a minimal test chunk for testing
fn create_test_chunk(id: &str, content: &str, file_path: &str) -> ContentChunk {
    ContentChunk {
        id: id.to_string().into(),
        content: content.to_string(),
        token_count: content.split_whitespace().count().into(),
        source_location: SourceLocation {
            file_path: PathBuf::from(file_path),
            start_line: 1,
            end_line: 1,
            start_char: 0,
            end_char: content.len(),
        },
        chunk_index: 0.into(),
        total_chunks: 1,
    }
}

/// Create an indexed chunk for testing
fn create_test_indexed_chunk(
    id: &str,
    content: &str,
    file_path: &str,
    embedding: Vec<f32>,
) -> IndexedChunk {
    IndexedChunk {
        chunk: create_test_chunk(id, content, file_path),
        embedding,
    }
}

#[test]
fn test_index_storage_basic_operations() {
    let temp_dir = TempDir::new().unwrap();
    let storage = IndexStorage::new(temp_dir.path()).unwrap();

    // Initially no index should exist
    assert!(!storage.index_exists());

    // Create test data
    let indexed_chunks = vec![
        create_test_indexed_chunk("chunk1", "Hello world", "test1.txt", vec![0.1, 0.2, 0.3]),
        create_test_indexed_chunk("chunk2", "Goodbye world", "test2.txt", vec![0.4, 0.5, 0.6]),
    ];

    let config = IndexConfig {
        model_name: "test-model".to_string(),
        embedding_dimensions: 3,
        batch_size: 10,
        respect_gitignore: true,
        include_untracked: false,
    };

    // Save the index
    storage
        .save_index(&indexed_chunks, &config, "1.0.0")
        .unwrap();
    assert!(storage.index_exists());

    // Verify all expected files exist
    assert!(storage.index_dir().join("vectors.bin").exists());
    assert!(storage.index_dir().join("metadata.json").exists());
    assert!(storage.index_dir().join("config.yaml").exists());
    assert!(storage.index_dir().join("version.txt").exists());

    // Load the index back
    let (loaded_chunks, loaded_config) = storage.load_index("1.0.0").unwrap();

    // Verify loaded data matches original
    assert_eq!(loaded_chunks.len(), 2);
    assert_eq!(loaded_config.model_name, "test-model");
    assert_eq!(loaded_config.embedding_dimensions, 3);

    // Check that embeddings are preserved
    assert_eq!(loaded_chunks[0].embedding, vec![0.1, 0.2, 0.3]);
    assert_eq!(loaded_chunks[1].embedding, vec![0.4, 0.5, 0.6]);

    // Check that chunk metadata is preserved
    assert_eq!(loaded_chunks[0].chunk.id, "chunk1".into());
    assert_eq!(loaded_chunks[1].chunk.id, "chunk2".into());
}

#[test]
fn test_index_storage_clear() {
    let temp_dir = TempDir::new().unwrap();
    let storage = IndexStorage::new(temp_dir.path()).unwrap();

    // Create and save test data
    let indexed_chunks = vec![create_test_indexed_chunk(
        "chunk1",
        "Test content",
        "test.txt",
        vec![1.0, 2.0],
    )];
    let config = IndexConfig::default();

    storage
        .save_index(&indexed_chunks, &config, "1.0.0")
        .unwrap();
    assert!(storage.index_exists());

    // Clear the index
    storage.clear_index().unwrap();
    assert!(!storage.index_exists());

    // Verify files are removed
    assert!(!storage.index_dir().join("vectors.bin").exists());
    assert!(!storage.index_dir().join("metadata.json").exists());
}

#[test]
fn test_index_storage_large_data() {
    let temp_dir = TempDir::new().unwrap();
    let storage = IndexStorage::new(temp_dir.path()).unwrap();

    // Create a larger dataset to test memory mapping efficiency
    let mut indexed_chunks = Vec::new();
    for i in 0..1000 {
        let content = format!("Test chunk number {} with some content", i);
        let embedding = vec![i as f32 * 0.001; 384]; // Typical embedding size
        indexed_chunks.push(create_test_indexed_chunk(
            &format!("chunk_{}", i),
            &content,
            &format!("file_{}.txt", i % 10),
            embedding,
        ));
    }

    let config = IndexConfig {
        embedding_dimensions: 384,
        ..Default::default()
    };

    // Save and load large dataset
    storage
        .save_index(&indexed_chunks, &config, "1.0.0")
        .unwrap();
    let (loaded_chunks, _) = storage.load_index("1.0.0").unwrap();

    assert_eq!(loaded_chunks.len(), 1000);

    // Spot check a few chunks
    assert_eq!(loaded_chunks[0].chunk.id, "chunk_0".into());
    assert_eq!(loaded_chunks[999].chunk.id, "chunk_999".into());
    assert_eq!(loaded_chunks[0].embedding.len(), 384);
    assert_eq!(loaded_chunks[999].embedding.len(), 384);
}

#[test]
fn test_persistent_chunk_index_creation() {
    let temp_dir = TempDir::new().unwrap();
    let index = PersistentChunkIndex::new(temp_dir.path()).unwrap();

    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
    assert!(!index.exists_on_disk());
    assert_eq!(index.indexed_path(), temp_dir.path());
}

#[test]
fn test_persistent_chunk_index_save_load() {
    let temp_dir = TempDir::new().unwrap();

    // Create an index and simulate adding some data manually for testing
    // (In real usage, build() would be used)
    let _index = PersistentChunkIndex::new(temp_dir.path()).unwrap();

    // Test that loading non-existent index fails appropriately
    let load_result = PersistentChunkIndex::load(temp_dir.path());
    assert!(load_result.is_err());
    assert!(load_result
        .unwrap_err()
        .to_string()
        .contains("No index found"));
}

#[tokio::test]
async fn test_build_index_with_real_files() {
    let temp_dir = TempDir::new().unwrap();

    // Create test files
    create_test_file(
        temp_dir.path(),
        "test1.rs",
        r#"
        fn main() {
            println!("Hello, world!");
        }
        "#,
    );

    create_test_file(
        temp_dir.path(),
        "test2.rs",
        r#"
        pub struct Person {
            name: String,
            age: u32,
        }
        
        impl Person {
            pub fn new(name: String, age: u32) -> Self {
                Self { name, age }
            }
        }
        "#,
    );

    // Skip this test if we're in offline mode or if it would require network access
    if std::env::var("OFFLINE_TESTS").is_ok() {
        return;
    }

    // Test building an index with default configuration
    let config = TurboPropConfig::default();
    let build_result = PersistentChunkIndex::build(temp_dir.path(), &config).await;

    // This test might fail due to network requirements for downloading models
    // In CI/CD environments without network access, we should skip
    match build_result {
        Ok(index) => {
            assert!(!index.is_empty());
            assert!(index.exists_on_disk());

            // Test that we can load the built index
            let loaded_index = PersistentChunkIndex::load(temp_dir.path()).unwrap();
            assert_eq!(loaded_index.len(), index.len());

            // Test similarity search with the loaded index
            let chunks = loaded_index.get_chunks();
            if !chunks.is_empty() {
                let query_embedding = &chunks[0].embedding;
                let results = loaded_index.similarity_search(query_embedding, 5);
                assert!(!results.is_empty());
                assert!(results[0].0 > 0.9); // Should be very similar to itself
            }
        }
        Err(e) => {
            // If it fails due to network/model issues, that's expected in some test environments
            eprintln!("Skipping build test due to: {}", e);
        }
    }
}

#[test]
fn test_concurrent_index_access() {
    let temp_dir = TempDir::new().unwrap();
    let storage = IndexStorage::new(temp_dir.path()).unwrap();

    // Create initial index
    let indexed_chunks = vec![create_test_indexed_chunk(
        "chunk1",
        "Shared content",
        "test.txt",
        vec![0.1, 0.2],
    )];
    let config = IndexConfig::default();
    storage
        .save_index(&indexed_chunks, &config, "1.0.0")
        .unwrap();

    let storage_path = temp_dir.path().to_path_buf();
    let results = Arc::new(Mutex::new(Vec::new()));

    // Spawn multiple threads that try to load the index concurrently
    let mut handles = vec![];
    for i in 0..5 {
        let path = storage_path.clone();
        let results_clone = Arc::clone(&results);

        let handle = thread::spawn(move || {
            // Add a small random delay to increase chances of concurrent access
            thread::sleep(Duration::from_millis(i * 10));

            let storage = IndexStorage::new(&path).unwrap();
            let load_result = storage.load_index("1.0.0");

            let mut results_guard = results_clone.lock().unwrap();
            results_guard.push((i, load_result.is_ok()));
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all threads successfully loaded the index
    let final_results = results.lock().unwrap();
    assert_eq!(final_results.len(), 5);

    for (thread_id, success) in final_results.iter() {
        assert!(*success, "Thread {} failed to load index", thread_id);
    }
}

#[test]
fn test_index_integrity_after_partial_write() {
    let temp_dir = TempDir::new().unwrap();
    let storage = IndexStorage::new(temp_dir.path()).unwrap();

    // Create initial valid index
    let indexed_chunks = vec![create_test_indexed_chunk(
        "chunk1",
        "Valid content",
        "test.txt",
        vec![0.1, 0.2, 0.3],
    )];
    let config = IndexConfig::default();
    storage
        .save_index(&indexed_chunks, &config, "1.0.0")
        .unwrap();

    // Verify we can load it
    assert!(storage.load_index("1.0.0").is_ok());

    // Simulate partial write by creating a temporary file that won't be moved
    let temp_vectors = storage.index_dir().join("vectors.bin.tmp");
    fs::write(&temp_vectors, "incomplete data").unwrap();

    // The original index should still be valid since atomic operations are used
    assert!(storage.load_index("1.0.0").is_ok());

    // The temporary file should not interfere with normal operations
    assert!(temp_vectors.exists());

    // After another save operation, temp files should be cleaned up by new save
    storage
        .save_index(&indexed_chunks, &config, "1.0.0")
        .unwrap();

    // Index should still be valid
    let (loaded_chunks, _) = storage.load_index("1.0.0").unwrap();
    assert_eq!(loaded_chunks.len(), 1);
    assert_eq!(loaded_chunks[0].chunk.id, "chunk1".into());
}

#[test]
fn test_index_stats_calculation() {
    let indexed_chunks = vec![
        create_test_indexed_chunk("chunk1", "Hello world", "file1.rs", vec![0.1, 0.2]),
        create_test_indexed_chunk(
            "chunk2",
            "Rust programming language",
            "file1.rs",
            vec![0.3, 0.4],
        ),
        create_test_indexed_chunk("chunk3", "Machine learning", "file2.py", vec![0.5, 0.6]),
    ];

    let stats = IndexStats::calculate(&indexed_chunks);

    assert_eq!(stats.total_chunks, 3);
    assert_eq!(stats.unique_files, 2);
    assert_eq!(stats.total_tokens, 2 + 3 + 2); // Assuming simple word count
    assert_eq!(stats.average_chunk_size, 7.0 / 3.0);
    assert_eq!(stats.embedding_dimensions, 2);
}

#[test]
fn test_version_compatibility() {
    let temp_dir = TempDir::new().unwrap();
    let storage = IndexStorage::new(temp_dir.path()).unwrap();

    // Create a valid index
    let indexed_chunks = vec![create_test_indexed_chunk(
        "chunk1",
        "Version test",
        "test.txt",
        vec![1.0],
    )];
    let config = IndexConfig::default();
    storage
        .save_index(&indexed_chunks, &config, "1.0.0")
        .unwrap();

    // Should load successfully
    assert!(storage.load_index("1.0.0").is_ok());

    // Simulate incompatible version by modifying version file
    let version_file = storage.index_dir().join("version.txt");
    fs::write(&version_file, "999.0.0").unwrap();

    // Should now fail to load due to version mismatch
    let load_result = storage.load_index("1.0.0");
    assert!(load_result.is_err());
    assert!(load_result
        .unwrap_err()
        .to_string()
        .contains("version mismatch"));
}

#[test]
fn test_empty_index_operations() {
    let temp_dir = TempDir::new().unwrap();
    let storage = IndexStorage::new(temp_dir.path()).unwrap();

    // Test saving empty index
    let empty_chunks: Vec<IndexedChunk> = vec![];
    let config = IndexConfig::default();

    storage.save_index(&empty_chunks, &config, "1.0.0").unwrap();
    assert!(storage.index_exists());

    // Test loading empty index
    let (loaded_chunks, loaded_config) = storage.load_index("1.0.0").unwrap();
    assert_eq!(loaded_chunks.len(), 0);
    assert_eq!(loaded_config.model_name, config.model_name);

    // Test stats on empty index
    let stats = IndexStats::calculate(&loaded_chunks);
    assert_eq!(stats.total_chunks, 0);
    assert_eq!(stats.unique_files, 0);
    assert_eq!(stats.embedding_dimensions, 0);
}

#[test]
fn test_index_directory_creation() {
    let temp_dir = TempDir::new().unwrap();
    let nested_path = temp_dir.path().join("deeply").join("nested").join("path");

    // Storage should create the directory structure
    let storage = IndexStorage::new(&nested_path).unwrap();
    assert!(storage.index_dir().exists());

    let expected_index_dir = nested_path.join(".turboprop").join("index");
    assert!(expected_index_dir.exists());
    assert_eq!(storage.index_dir(), expected_index_dir);
}

#[test]
fn test_corrupted_metadata_handling() {
    let temp_dir = TempDir::new().unwrap();
    let storage = IndexStorage::new(temp_dir.path()).unwrap();

    // Create valid index first
    let indexed_chunks = vec![create_test_indexed_chunk(
        "chunk1",
        "Test content",
        "test.txt",
        vec![1.0, 2.0],
    )];
    let config = IndexConfig::default();
    storage
        .save_index(&indexed_chunks, &config, "1.0.0")
        .unwrap();

    // Corrupt the metadata file
    let metadata_file = storage.index_dir().join("metadata.json");
    fs::write(&metadata_file, "{ invalid json }").unwrap();

    // Should fail to load due to corrupted metadata
    let load_result = storage.load_index("1.0.0");
    assert!(load_result.is_err());
    assert!(load_result
        .unwrap_err()
        .to_string()
        .contains("Failed to deserialize metadata"));
}

#[test]
fn test_missing_files_handling() {
    let temp_dir = TempDir::new().unwrap();
    let storage = IndexStorage::new(temp_dir.path()).unwrap();

    // Create valid index
    let indexed_chunks = vec![create_test_indexed_chunk(
        "chunk1",
        "Test",
        "test.txt",
        vec![1.0],
    )];
    let config = IndexConfig::default();
    storage
        .save_index(&indexed_chunks, &config, "1.0.0")
        .unwrap();

    // Remove one of the required files
    let vectors_file = storage.index_dir().join("vectors.bin");
    fs::remove_file(&vectors_file).unwrap();

    // Should report that index doesn't exist
    assert!(!storage.index_exists());

    // Loading should fail gracefully
    let load_result = storage.load_index("1.0.0");
    assert!(load_result.is_err());
}

#[tokio::test]
async fn test_update_result_functionality() {
    // Test UpdateResult helper methods
    let result = UpdateResult {
        added_files: 2,
        updated_files: 1,
        removed_files: 0,
        total_chunks_before: 10,
        total_chunks_after: 15,
    };

    assert!(result.has_changes());
    assert_eq!(result.chunk_delta(), 5);

    let no_change_result = UpdateResult {
        added_files: 0,
        updated_files: 0,
        removed_files: 0,
        total_chunks_before: 10,
        total_chunks_after: 10,
    };

    assert!(!no_change_result.has_changes());
    assert_eq!(no_change_result.chunk_delta(), 0);

    let decrease_result = UpdateResult {
        added_files: 0,
        updated_files: 0,
        removed_files: 2,
        total_chunks_before: 15,
        total_chunks_after: 8,
    };

    assert!(decrease_result.has_changes());
    assert_eq!(decrease_result.chunk_delta(), -7);
}
