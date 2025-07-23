use std::path::PathBuf;
use tempfile::TempDir;
use turboprop::config::TurboPropConfig;
use turboprop::error::{TurboPropError, TurboPropResult};
use turboprop::files::FileDiscovery;
use turboprop::chunking::ChunkingStrategy;

#[test]
fn test_network_error_display() {
    let error = TurboPropError::NetworkError {
        message: "Connection timeout".to_string(),
        url: Some("https://huggingface.co/model".to_string()),
    };

    let error_str = error.to_string();
    assert!(error_str.contains("Network error"));
    assert!(error_str.contains("Connection timeout"));
    assert!(error_str.contains("https://huggingface.co/model"));
}

#[test]
fn test_file_permission_error_display() {
    let path = PathBuf::from("/restricted/file.txt");
    let error = TurboPropError::FilePermissionError {
        path: path.clone(),
        operation: "read".to_string(),
    };

    let error_str = error.to_string();
    assert!(error_str.contains("Permission denied"));
    assert!(error_str.contains("/restricted/file.txt"));
    assert!(error_str.contains("read"));
}

#[test]
fn test_corrupted_index_error_display() {
    let path = PathBuf::from("/project/.turboprop/index.db");
    let error = TurboPropError::CorruptedIndex {
        path: path.clone(),
        reason: "Invalid header".to_string(),
    };

    let error_str = error.to_string();
    assert!(error_str.contains("Index is corrupted"));
    assert!(error_str.contains("/project/.turboprop/index.db"));
    assert!(error_str.contains("Invalid header"));
}

#[test]
fn test_insufficient_disk_space_error_display() {
    let error = TurboPropError::InsufficientDiskSpace {
        required_bytes: 1024 * 1024 * 100, // 100MB
        available_bytes: 1024 * 1024 * 50, // 50MB
        path: PathBuf::from("/project/.turboprop"),
    };

    let error_str = error.to_string();
    assert!(error_str.contains("Insufficient disk space"));
    assert!(error_str.contains("100 MB"));
    assert!(error_str.contains("50 MB"));
}

#[test]
fn test_embedding_model_error_display() {
    let error = TurboPropError::EmbeddingModelError {
        model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        reason: "Model download failed".to_string(),
    };

    let error_str = error.to_string();
    assert!(error_str.contains("Embedding model error"));
    assert!(error_str.contains("sentence-transformers/all-MiniLM-L6-v2"));
    assert!(error_str.contains("Model download failed"));
}

#[test]
fn test_invalid_git_repository_error_display() {
    let path = PathBuf::from("/not-a-git-repo");
    let error = TurboPropError::InvalidGitRepository { path: path.clone() };

    let error_str = error.to_string();
    assert!(error_str.contains("Invalid Git repository"));
    assert!(error_str.contains("/not-a-git-repo"));
}

#[test]
fn test_file_encoding_error_display() {
    let path = PathBuf::from("binary-file.exe");
    let error = TurboPropError::FileEncodingError {
        path: path.clone(),
        encoding: Some("binary".to_string()),
    };

    let error_str = error.to_string();
    assert!(error_str.contains("File encoding error"));
    assert!(error_str.contains("binary-file.exe"));
    assert!(error_str.contains("binary"));
}

#[test]
fn test_configuration_validation_error_display() {
    let error = TurboPropError::ConfigurationValidationError {
        field: "embedding.batch_size".to_string(),
        value: "invalid".to_string(),
        expected: "positive integer".to_string(),
    };

    let error_str = error.to_string();
    assert!(error_str.contains("Configuration validation error"));
    assert!(error_str.contains("embedding.batch_size"));
    assert!(error_str.contains("invalid"));
    assert!(error_str.contains("positive integer"));
}

#[test]
fn test_error_conversion_from_anyhow() {
    let anyhow_error = anyhow::anyhow!("Some generic error");
    let turboprop_error: TurboPropError = anyhow_error.into();

    match turboprop_error {
        TurboPropError::Other { message } => {
            assert!(message.contains("Some generic error"));
        }
        _ => panic!("Expected Other error variant"),
    }
}

#[test]
fn test_error_conversion_from_io_error() {
    let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
    let turboprop_error: TurboPropError = io_error.into();

    match turboprop_error {
        TurboPropError::FileSystemError { message, path: _ } => {
            assert!(message.contains("File not found"));
        }
        _ => panic!("Expected FileSystemError variant"),
    }
}

#[test]
fn test_turboprop_result_type() {
    // Test that our Result type alias works correctly
    let success: TurboPropResult<String> = Ok("success".to_string());
    assert!(success.is_ok());

    let failure: TurboPropResult<String> = Err(TurboPropError::Other {
        message: "failure".to_string(),
    });
    assert!(failure.is_err());
}

// ================================
// EDGE CASE TESTS - Empty Input
// ================================

#[test]
fn test_file_discovery_empty_directory() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let config = TurboPropConfig::default();
    let discovery = FileDiscovery::new(config.file_discovery.clone());
    
    // Empty directory should return empty result, not an error
    let result = discovery.discover_files(temp_dir.path());
    assert!(result.is_ok());
    let files = result.unwrap();
    assert!(files.is_empty(), "Empty directory should return no files");
}

#[test]
fn test_chunking_empty_file_content() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let empty_file_path = temp_dir.path().join("empty.rs");
    std::fs::write(&empty_file_path, "").expect("Failed to create empty file");
    
    let config = TurboPropConfig::default();
    let chunker = ChunkingStrategy::new(config.chunking.clone());
    
    // Empty file should be handled gracefully
    let result = chunker.chunk_file(&empty_file_path);
    assert!(result.is_ok());
    let chunks = result.unwrap();
    assert!(chunks.is_empty() || chunks.len() == 1, "Empty file should produce no chunks or one empty chunk");
}

#[test]
fn test_chunking_whitespace_only_file() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let whitespace_file_path = temp_dir.path().join("whitespace.rs");
    std::fs::write(&whitespace_file_path, "   \n\t\n   \n").expect("Failed to create whitespace file");
    
    let config = TurboPropConfig::default();
    let chunker = ChunkingStrategy::new(config.chunking.clone());
    
    // Whitespace-only file should be handled gracefully
    let result = chunker.chunk_file(&whitespace_file_path);
    assert!(result.is_ok());
    let chunks = result.unwrap();
    assert!(chunks.is_empty() || chunks.iter().all(|c| c.content.trim().is_empty()), 
            "Whitespace-only file should produce no meaningful chunks");
}

// ================================
// EDGE CASE TESTS - Malformed Data  
// ================================

#[test]
fn test_chunking_binary_file_rejection() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let binary_file_path = temp_dir.path().join("binary.exe");
    
    // Create a file with binary content (non-UTF8)
    let binary_data: Vec<u8> = vec![0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE, 0xFD];
    std::fs::write(&binary_file_path, binary_data).expect("Failed to create binary file");
    
    let config = TurboPropConfig::default();
    let chunker = ChunkingStrategy::new(config.chunking.clone());
    
    // Binary file should be rejected or handled gracefully
    let result = chunker.chunk_file(&binary_file_path);
    // Should either fail with encoding error or succeed with no/minimal chunks
    match result {
        Ok(chunks) => {
            // If it succeeds, chunks should be empty or minimal
            assert!(chunks.is_empty() || chunks.iter().all(|c| c.content.len() < 50), 
                    "Binary file should not produce meaningful chunks");
        }
        Err(_) => {
            // If it fails, that's also acceptable for binary files
            // The error should be related to file encoding
        }
    }
}

#[test] 
fn test_chunking_extremely_large_line() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let large_line_file_path = temp_dir.path().join("large_line.rs");
    
    // Create a file with one extremely long line (potential memory issue)
    // Use many words instead of one huge token to make it splittable
    let large_line = "word ".repeat(20_000); // 20k words, about 100KB
    let content = format!("// This is a huge comment: {}\nfn test() {{}}", large_line);
    std::fs::write(&large_line_file_path, content).expect("Failed to create large line file");
    
    let config = TurboPropConfig::default();
    let chunker = ChunkingStrategy::new(config.chunking.clone());
    
    // Extremely large lines should be handled without crashing
    let result = chunker.chunk_file(&large_line_file_path);
    assert!(result.is_ok(), "Large line file should be processed without errors");
    let chunks = result.unwrap();
    
    // Should produce some chunks, possibly truncated or split
    assert!(!chunks.is_empty(), "Large line file should produce some chunks");
    
    // No individual chunk should be excessively large
    // With max 500 tokens and "word " is 5 chars each, max should be around 2500 chars
    for chunk in chunks {
        assert!(chunk.content.len() < 3_000, "Individual chunks should be reasonably sized, got: {} chars", chunk.content.len());
    }
}

#[test]
fn test_file_discovery_with_invalid_unicode_filename() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    
    // On Unix-like systems, create a file with invalid UTF-8 in filename
    #[cfg(unix)]
    {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;
        
        let invalid_filename = OsString::from_vec(vec![b'i', b'n', b'v', b'a', b'l', b'i', b'd', 0xFF, 0xFE]);
        let invalid_file_path = temp_dir.path().join(invalid_filename);
        
        // Try to create the file - this might fail on some systems
        if std::fs::write(&invalid_file_path, "some content").is_ok() {
            let config = TurboPropConfig::default();
            let discovery = FileDiscovery::new(config.file_discovery.clone());
            
            // File discovery should handle invalid unicode filenames gracefully
            let result = discovery.discover_files(temp_dir.path());
            assert!(result.is_ok(), "File discovery should handle invalid unicode filenames");
        }
    }
}

// ================================
// EDGE CASE TESTS - Permission Errors
// ================================

#[test]
#[cfg(unix)] // Permission tests are most relevant on Unix-like systems
fn test_file_discovery_permission_denied_directory() {
    use std::os::unix::fs::PermissionsExt;
    
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let restricted_dir = temp_dir.path().join("restricted");
    std::fs::create_dir(&restricted_dir).expect("Failed to create directory");
    
    // Create a file in the directory first
    let file_path = restricted_dir.join("test.rs");
    std::fs::write(&file_path, "fn test() {}").expect("Failed to create test file");
    
    // Remove read permissions from directory
    let mut perms = std::fs::metadata(&restricted_dir).unwrap().permissions();
    perms.set_mode(0o000); // No permissions
    std::fs::set_permissions(&restricted_dir, perms).expect("Failed to set permissions");
    
    let config = TurboPropConfig::default();
    let discovery = FileDiscovery::new(config.file_discovery.clone());
    
    // File discovery should handle permission denied gracefully
    let result = discovery.discover_files(&restricted_dir);
    
    // Restore permissions for cleanup
    let mut perms = std::fs::metadata(&restricted_dir).unwrap().permissions();
    perms.set_mode(0o755);
    let _ = std::fs::set_permissions(&restricted_dir, perms);
    
    // Should either fail with permission error or succeed with empty results
    match result {
        Ok(files) => {
            // If it succeeds, should find no files due to permission restriction
            assert!(files.is_empty(), "Should not find files in restricted directory");
        }
        Err(_) => {
            // Permission error is also acceptable
        }
    }
}

#[test]
#[cfg(unix)]
fn test_chunking_permission_denied_file() {
    use std::os::unix::fs::PermissionsExt;
    
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let restricted_file_path = temp_dir.path().join("restricted.rs");
    std::fs::write(&restricted_file_path, "fn test() {}").expect("Failed to create test file");
    
    // Remove read permissions from file
    let mut perms = std::fs::metadata(&restricted_file_path).unwrap().permissions();
    perms.set_mode(0o000); // No permissions
    std::fs::set_permissions(&restricted_file_path, perms).expect("Failed to set permissions");
    
    let config = TurboPropConfig::default();
    let chunker = ChunkingStrategy::new(config.chunking.clone());
    
    // Chunking should handle permission denied gracefully
    let result = chunker.chunk_file(&restricted_file_path);
    
    // Restore permissions for cleanup
    let mut perms = std::fs::metadata(&restricted_file_path).unwrap().permissions();
    perms.set_mode(0o644);
    let _ = std::fs::set_permissions(&restricted_file_path, perms);
    
    // Should fail with a clear permission error
    assert!(result.is_err(), "Should fail to chunk file without read permissions");
    
    // Error should indicate permission issue
    let error = result.unwrap_err();
    let error_str = error.to_string().to_lowercase();
    assert!(error_str.contains("permission") || error_str.contains("access") || error_str.contains("denied") || error_str.contains("forbidden") || error_str.contains("failed to read"), 
            "Error should indicate permission issue: {}", error);
}

// ================================
// EDGE CASE TESTS - System Resource Limits
// ================================

#[test]
fn test_handling_many_small_files() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    
    // Create many small files to test file handle limits
    for i in 0..100 {
        let file_path = temp_dir.path().join(format!("file_{}.rs", i));
        std::fs::write(&file_path, format!("// File {}\nfn test_{}() {{}}", i, i))
            .expect("Failed to create test file");
    }
    
    let config = TurboPropConfig::default();
    let discovery = FileDiscovery::new(config.file_discovery.clone());
    
    // Should handle many files without running out of file handles
    let result = discovery.discover_files(temp_dir.path());
    assert!(result.is_ok(), "Should handle many small files");
    
    let files = result.unwrap();
    assert_eq!(files.len(), 100, "Should find all created files");
}

#[test] 
fn test_deeply_nested_directory_structure() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    
    // Create deeply nested directory structure
    let mut current_path = temp_dir.path().to_path_buf();
    for i in 0..50 {  // 50 levels deep
        current_path = current_path.join(format!("level_{}", i));
        std::fs::create_dir(&current_path).expect("Failed to create nested directory");
    }
    
    // Add a file at the deepest level
    let deep_file = current_path.join("deep.rs");
    std::fs::write(&deep_file, "fn deep_function() {}").expect("Failed to create deep file");
    
    let config = TurboPropConfig::default();
    let discovery = FileDiscovery::new(config.file_discovery.clone());
    
    // Should handle deeply nested structures without stack overflow
    let result = discovery.discover_files(temp_dir.path());
    assert!(result.is_ok(), "Should handle deeply nested directories");
    
    let files = result.unwrap();
    assert_eq!(files.len(), 1, "Should find the deeply nested file");
    assert!(files[0].path.ends_with("deep.rs"), "Should find the correct file");
}
