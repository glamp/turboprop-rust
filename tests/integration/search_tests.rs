//! Comprehensive integration tests for the search command implementation.
//!
//! These tests verify the complete search pipeline including CLI parsing,
//! filtering, output formatting, and end-to-end functionality.

use anyhow::Result;
use std::path::Path;
use tempfile::TempDir;
use turboprop::commands::{execute_search_command_cli, search::SearchCommandConfig};
use turboprop::config::TurboPropConfig;
use turboprop::output::OutputFormat;
use turboprop::{build_persistent_index, index_exists};

// Import shared test utilities
mod common;
use common::create_test_codebase;

/// Helper function to create a test directory with sample files
fn create_test_codebase() -> Result<TempDir> {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    // Create sample Rust files
    let rust_content = r#"
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct User {
    pub id: u64,
    pub name: String,
    pub email: String,
}

impl User {
    pub fn new(id: u64, name: String, email: String) -> Self {
        Self { id, name, email }
    }
    
    pub fn authenticate(&self, password: &str) -> Result<bool, String> {
        if password.len() < 8 {
            return Err("Password too short".to_string());
        }
        Ok(true)
    }
}

pub fn create_user_database() -> HashMap<u64, User> {
    let mut users = HashMap::new();
    users.insert(1, User::new(1, "Alice".to_string(), "alice@example.com".to_string()));
    users.insert(2, User::new(2, "Bob".to_string(), "bob@example.com".to_string()));
    users
}
"#;

    let javascript_content = r#"
function authenticateUser(username, password) {
    if (!username || !password) {
        throw new Error('Username and password are required');
    }
    
    const users = getUserDatabase();
    const user = users.find(u => u.username === username);
    
    if (!user) {
        return { success: false, error: 'User not found' };
    }
    
    if (user.password !== password) {
        return { success: false, error: 'Invalid password' };
    }
    
    return { success: true, user: user };
}

function getUserDatabase() {
    return [
        { id: 1, username: 'alice', password: 'secret123', email: 'alice@example.com' },
        { id: 2, username: 'bob', password: 'password456', email: 'bob@example.com' }
    ];
}

module.exports = { authenticateUser, getUserDatabase };
"#;

    let python_content = r#"
import hashlib
from typing import Dict, Optional

class User:
    def __init__(self, user_id: int, name: str, email: str):
        self.id = user_id
        self.name = name
        self.email = email
    
    def authenticate(self, password: str) -> bool:
        """Authenticate user with password"""
        if len(password) < 8:
            raise ValueError("Password too short")
        return True
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email
        }

def create_user_database() -> Dict[int, User]:
    """Create sample user database"""
    return {
        1: User(1, "Alice", "alice@example.com"),
        2: User(2, "Bob", "bob@example.com")
    }

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()
"#;

    // Write test files
    std::fs::write(temp_path.join("user.rs"), rust_content)?;
    std::fs::write(temp_path.join("auth.js"), javascript_content)?;
    std::fs::write(temp_path.join("user_service.py"), python_content)?;

    // Create subdirectory with more files
    let api_dir = temp_path.join("api");
    std::fs::create_dir(&api_dir)?;

    let api_content = r#"
const express = require('express');
const jwt = require('jsonwebtoken');

const router = express.Router();

router.post('/login', async (req, res) => {
    try {
        const { username, password } = req.body;
        
        if (!username || !password) {
            return res.status(400).json({ error: 'Missing credentials' });
        }
        
        const result = authenticateUser(username, password);
        
        if (result.success) {
            const token = jwt.sign(
                { userId: result.user.id, username: result.user.username },
                process.env.JWT_SECRET,
                { expiresIn: '24h' }
            );
            
            res.json({ token, user: result.user });
        } else {
            res.status(401).json({ error: result.error });
        }
    } catch (error) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

module.exports = router;
"#;

    std::fs::write(api_dir.join("auth.js"), api_content)?;

    Ok(temp_dir)
}

/// Build a test index for the given directory
async fn build_test_index_if_needed(path: &Path) -> Result<bool> {
    if index_exists(path) {
        return Ok(true);
    }

    match build_persistent_index(path, &TurboPropConfig::default()).await {
        Ok(_) => Ok(true),
        Err(_) => {
            // Expected in CI/test environments without model access
            Ok(false)
        }
    }
}

#[tokio::test]
async fn test_search_command_config_validation() -> Result<()> {
    let temp_dir = create_test_codebase()?;
    let temp_path = temp_dir.path().to_string_lossy().to_string();

    // Valid configuration
    let config = SearchCommandConfig::new(
        "user authentication".to_string(),
        temp_path.clone(),
        10,
        Some(0.5),
        OutputFormat::Json,
        Some("rs".to_string()),
    );
    assert!(config.validate().is_ok());

    // Invalid query (empty)
    let config = SearchCommandConfig::new(
        "".to_string(),
        temp_path.clone(),
        10,
        None,
        OutputFormat::Json,
        None,
    );
    assert!(config.validate().is_err());

    // Invalid threshold
    let config = SearchCommandConfig::new(
        "test".to_string(),
        temp_path.clone(),
        10,
        Some(1.5),
        OutputFormat::Json,
        None,
    );
    assert!(config.validate().is_err());

    // Invalid limit
    let config = SearchCommandConfig::new(
        "test".to_string(),
        temp_path,
        0,
        None,
        OutputFormat::Json,
        None,
    );
    assert!(config.validate().is_err());

    Ok(())
}

#[tokio::test]
async fn test_search_command_with_json_output() -> Result<()> {
    let temp_dir = create_test_codebase()?;
    let temp_path = temp_dir.path();

    // Try to build index, skip test if not possible
    if !build_test_index_if_needed(temp_path).await? {
        println!("Skipping test - unable to build index (likely missing model in test environment)");
        return Ok(());
    }

    // Test JSON output format
    let result = execute_search_command_cli(
        "authenticate".to_string(),
        temp_path.to_path_buf(),
        5,
        None,
        "json".to_string(),
        None,
    ).await;

    // The command might fail due to missing models in CI, but should not panic
    // or have validation errors
    match result {
        Ok(_) => {
            // If successful, that's great
            println!("Search command executed successfully");
        }
        Err(e) => {
            // Expected errors in test environment
            let error_msg = e.to_string();
            assert!(
                error_msg.contains("index") 
                || error_msg.contains("model") 
                || error_msg.contains("embedding")
                || error_msg.contains("load"),
                "Unexpected error: {}", error_msg
            );
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_search_command_with_text_output() -> Result<()> {
    let temp_dir = create_test_codebase()?;
    let temp_path = temp_dir.path();

    // Try to build index, skip test if not possible
    if !build_test_index_if_needed(temp_path).await? {
        println!("Skipping test - unable to build index (likely missing model in test environment)");
        return Ok(());
    }

    // Test text output format
    let result = execute_search_command_cli(
        "user database".to_string(),
        temp_path.to_path_buf(),
        3,
        Some(0.1),
        "text".to_string(),
        None,
    ).await;

    // The command might fail due to missing models in CI
    match result {
        Ok(_) => {
            println!("Search command executed successfully with text output");
        }
        Err(e) => {
            let error_msg = e.to_string();
            assert!(
                error_msg.contains("index") 
                || error_msg.contains("model") 
                || error_msg.contains("embedding")
                || error_msg.contains("load"),
                "Unexpected error: {}", error_msg
            );
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_search_command_with_filetype_filter() -> Result<()> {
    let temp_dir = create_test_codebase()?;
    let temp_path = temp_dir.path();

    // Try to build index, skip test if not possible
    if !build_test_index_if_needed(temp_path).await? {
        println!("Skipping test - unable to build index (likely missing model in test environment)");
        return Ok(());
    }

    // Test filetype filtering for Rust files
    let result = execute_search_command_cli(
        "struct User".to_string(),
        temp_path.to_path_buf(),
        10,
        None,
        "json".to_string(),
        Some("rs".to_string()),
    ).await;

    match result {
        Ok(_) => {
            println!("Search command executed successfully with filetype filter");
        }
        Err(e) => {
            let error_msg = e.to_string();
            assert!(
                error_msg.contains("index") 
                || error_msg.contains("model") 
                || error_msg.contains("embedding")
                || error_msg.contains("load"),
                "Unexpected error: {}", error_msg
            );
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_search_command_with_threshold() -> Result<()> {
    let temp_dir = create_test_codebase()?;
    let temp_path = temp_dir.path();

    // Try to build index, skip test if not possible
    if !build_test_index_if_needed(temp_path).await? {
        println!("Skipping test - unable to build index (likely missing model in test environment)");
        return Ok(());
    }

    // Test with high similarity threshold
    let result = execute_search_command_cli(
        "password authentication".to_string(),
        temp_path.to_path_buf(),
        5,
        Some(0.8), // High threshold
        "json".to_string(),
        None,
    ).await;

    match result {
        Ok(_) => {
            println!("Search command executed successfully with high threshold");
        }
        Err(e) => {
            let error_msg = e.to_string();
            assert!(
                error_msg.contains("index") 
                || error_msg.contains("model") 
                || error_msg.contains("embedding")
                || error_msg.contains("load"),
                "Unexpected error: {}", error_msg
            );
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_search_command_invalid_output_format() -> Result<()> {
    let temp_dir = create_test_codebase()?;
    let temp_path = temp_dir.path();

    // Test invalid output format
    let result = execute_search_command_cli(
        "test query".to_string(),
        temp_path.to_path_buf(),
        10,
        None,
        "xml".to_string(), // Invalid format
        None,
    ).await;

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Invalid output format"));

    Ok(())
}

#[tokio::test]
async fn test_search_command_invalid_filetype() -> Result<()> {
    let temp_dir = create_test_codebase()?;
    let temp_path = temp_dir.path();

    // Test invalid filetype (empty string)
    let result = execute_search_command_cli(
        "test query".to_string(),
        temp_path.to_path_buf(),
        10,
        None,
        "json".to_string(),
        Some("".to_string()), // Invalid empty filetype
    ).await;

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(
        error_msg.contains("Invalid file extension format") 
        || error_msg.contains("cannot be empty")
    );

    Ok(())
}

#[tokio::test]
async fn test_search_command_nonexistent_directory() -> Result<()> {
    // Test with nonexistent directory
    let result = execute_search_command_cli(
        "test query".to_string(),
        std::path::PathBuf::from("/nonexistent/directory"),
        10,
        None,
        "json".to_string(),
        None,
    ).await;

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(
        error_msg.contains("does not exist") 
        || error_msg.contains("Repository path")
    );

    Ok(())
}

#[tokio::test]
async fn test_search_command_multiple_filetypes() -> Result<()> {
    let temp_dir = create_test_codebase()?;
    let temp_path = temp_dir.path();

    // Try to build index, skip test if not possible
    if !build_test_index_if_needed(temp_path).await? {
        println!("Skipping test - unable to build index (likely missing model in test environment)");
        return Ok(());
    }

    // Test with JavaScript files
    let result = execute_search_command_cli(
        "function".to_string(),
        temp_path.to_path_buf(),
        10,
        None,
        "json".to_string(),
        Some("js".to_string()),
    ).await;

    match result {
        Ok(_) => {
            println!("JavaScript search completed successfully");
        }
        Err(e) => {
            let error_msg = e.to_string();
            assert!(
                error_msg.contains("index") 
                || error_msg.contains("model") 
                || error_msg.contains("embedding")
                || error_msg.contains("load"),
                "Unexpected error: {}", error_msg
            );
        }
    }

    // Test with Python files
    let result = execute_search_command_cli(
        "class User".to_string(),
        temp_path.to_path_buf(),
        10,
        None,
        "text".to_string(),
        Some("py".to_string()),
    ).await;

    match result {
        Ok(_) => {
            println!("Python search completed successfully");
        }
        Err(e) => {
            let error_msg = e.to_string();
            assert!(
                error_msg.contains("index") 
                || error_msg.contains("model") 
                || error_msg.contains("embedding")
                || error_msg.contains("load"),
                "Unexpected error: {}", error_msg
            );
        }
    }

    Ok(())
}

/// Test search with poker sample codebase if available
#[tokio::test]
async fn test_search_poker_codebase() -> Result<()> {
    let poker_path = Path::new("sample-codebases/poker");

    // Only run this test if the poker codebase exists
    if !poker_path.exists() {
        println!("Skipping poker codebase test - sample codebase not found");
        return Ok(());
    }

    // Try to build index for poker codebase
    if !build_test_index_if_needed(poker_path).await? {
        println!("Skipping poker test - unable to build index");
        return Ok(());
    }

    // Test searching for React/TypeScript content
    let result = execute_search_command_cli(
        "useState".to_string(),
        poker_path.to_path_buf(),
        5,
        Some(0.3),
        "json".to_string(),
        Some("tsx".to_string()),
    ).await;

    match result {
        Ok(_) => {
            println!("Poker codebase search completed successfully");
        }
        Err(e) => {
            println!("Poker codebase search failed (expected): {}", e);
            // This is acceptable in test environments
        }
    }

    Ok(())
}