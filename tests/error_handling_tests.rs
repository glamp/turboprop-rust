use std::path::PathBuf;
use tp::error::{TurboPropError, TurboPropResult};

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
        available_bytes: 1024 * 1024 * 50,  // 50MB
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
    let error = TurboPropError::InvalidGitRepository {
        path: path.clone(),
    };
    
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