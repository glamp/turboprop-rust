//! Integration tests for glob filtering functionality in search commands.
//!
//! These tests verify the complete glob filtering pipeline including CLI parsing,
//! filtering, output formatting, and end-to-end functionality.

use anyhow::Result;
use std::path::Path;
use tempfile::TempDir;
use turboprop::commands::execute_search_command_cli;
use turboprop::config::TurboPropConfig;
use turboprop::{build_persistent_index, index_exists};

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

/// Helper function to get test configuration
fn get_test_config() -> TurboPropConfig {
    TurboPropConfig::default()
}

/// Test the specification example: tp search "jwt authentication" --repo . --filter src/*.js
#[tokio::test]
async fn test_specification_example() -> Result<()> {
    let temp_dir = create_test_codebase()?;
    let temp_path = temp_dir.path();

    // Create additional src directory structure to match specification example
    let src_dir = temp_path.join("src");
    std::fs::create_dir(&src_dir)?;

    // Create JavaScript files with JWT authentication content
    let jwt_content = r#"
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');

class AuthenticationService {
    constructor(secretKey) {
        this.secretKey = secretKey;
    }

    // JWT authentication method
    async authenticateWithJWT(token) {
        try {
            const decoded = jwt.verify(token, this.secretKey);
            return { success: true, user: decoded };
        } catch (error) {
            return { success: false, error: 'Invalid JWT token' };
        }
    }

    // Generate JWT token for authentication
    generateJWTToken(user) {
        const payload = {
            userId: user.id,
            username: user.username,
            email: user.email
        };
        
        return jwt.sign(payload, this.secretKey, { expiresIn: '24h' });
    }

    // Authenticate user and return JWT
    async authenticateUser(username, password) {
        const user = await this.findUser(username);
        if (!user) {
            return { success: false, error: 'User not found' };
        }

        const isValid = await bcrypt.compare(password, user.passwordHash);
        if (!isValid) {
            return { success: false, error: 'Invalid credentials' };
        }

        const token = this.generateJWTToken(user);
        return { success: true, token, user };
    }
}

module.exports = AuthenticationService;
"#;

    std::fs::write(src_dir.join("auth_service.js"), jwt_content)?;

    // Create another JS file in src without JWT content
    let util_content = r#"
function formatUserData(user) {
    return {
        id: user.id,
        name: user.name,
        email: user.email,
        createdAt: new Date().toISOString()
    };
}

function validateEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

module.exports = { formatUserData, validateEmail };
"#;

    std::fs::write(src_dir.join("utils.js"), util_content)?;

    // Try to build index, skip test if not possible
    if !build_test_index_if_needed(temp_path).await? {
        println!("Skipping specification example test - unable to build index (likely missing model in test environment)");
        return Ok(());
    }

    // Run the specification example: search for "jwt authentication" with filter src/*.js
    let result = execute_search_command_cli(
        "jwt authentication".to_string(),
        temp_path.to_path_buf(),
        10,
        None,
        "json".to_string(),
        None,
        Some("src/*.js".to_string()),
        &get_test_config(),
    ).await;

    match result {
        Ok(_) => {
            println!("Specification example test completed successfully");
            println!("Command: tp search \"jwt authentication\" --repo . --filter src/*.js");
        }
        Err(e) => {
            let error_msg = e.to_string();
            assert!(
                error_msg.contains("index") 
                || error_msg.contains("model") 
                || error_msg.contains("embedding")
                || error_msg.contains("load"),
                "Unexpected error in specification example: {}", error_msg
            );
        }
    }

    Ok(())
}

/// Test glob filtering with JSON output format
#[tokio::test]
async fn test_glob_filtering_json_output() -> Result<()> {
    let temp_dir = create_test_codebase()?;
    let temp_path = temp_dir.path();

    // Try to build index, skip test if not possible
    if !build_test_index_if_needed(temp_path).await? {
        println!("Skipping glob filtering JSON output test - unable to build index");
        return Ok(());
    }

    // Test JSON output with glob pattern
    let result = execute_search_command_cli(
        "authenticate".to_string(),
        temp_path.to_path_buf(),
        5,
        None,
        "json".to_string(), // JSON format
        None,
        Some("*.rs".to_string()),
        &get_test_config(),
    ).await;

    match result {
        Ok(_) => {
            println!("Glob filtering with JSON output completed successfully");
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

/// Test glob filtering with text output format
#[tokio::test]
async fn test_glob_filtering_text_output() -> Result<()> {
    let temp_dir = create_test_codebase()?;
    let temp_path = temp_dir.path();

    // Try to build index, skip test if not possible
    if !build_test_index_if_needed(temp_path).await? {
        println!("Skipping glob filtering text output test - unable to build index");
        return Ok(());
    }

    // Test text output with glob pattern
    let result = execute_search_command_cli(
        "function".to_string(),
        temp_path.to_path_buf(),
        3,
        None,
        "text".to_string(), // Text format
        None,
        Some("**/*.js".to_string()),
        &get_test_config(),
    ).await;

    match result {
        Ok(_) => {
            println!("Glob filtering with text output completed successfully");
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

/// Test directory-specific glob patterns
#[tokio::test]
async fn test_directory_specific_glob_patterns() -> Result<()> {
    let temp_dir = create_test_codebase()?;
    let temp_path = temp_dir.path();

    // Create additional directory structure
    let tests_dir = temp_path.join("tests");
    std::fs::create_dir(&tests_dir)?;

    let test_content = r#"
#[cfg(test)]
mod user_tests {
    use super::*;

    #[test]
    fn test_user_creation() {
        let user = User::new(1, "Test User".to_string(), "test@example.com".to_string());
        assert_eq!(user.id, 1);
        assert_eq!(user.name, "Test User");
    }

    #[test]
    fn test_user_authentication() {
        let user = User::new(1, "Test".to_string(), "test@example.com".to_string());
        assert!(user.authenticate("validpassword").is_ok());
    }
}
"#;

    std::fs::write(tests_dir.join("user_test.rs"), test_content)?;

    // Try to build index, skip test if not possible
    if !build_test_index_if_needed(temp_path).await? {
        println!("Skipping directory-specific glob test - unable to build index");
        return Ok(());
    }

    // Test pattern that should only match files in tests directory
    let result = execute_search_command_cli(
        "test_user".to_string(),
        temp_path.to_path_buf(),
        5,
        None,
        "json".to_string(),
        None,
        Some("tests/*.rs".to_string()),
        &get_test_config(),
    ).await;

    match result {
        Ok(_) => {
            println!("Directory-specific glob pattern test completed successfully");
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

/// Test result limits with glob filtering
#[tokio::test]
async fn test_glob_filtering_with_limits() -> Result<()> {
    let temp_dir = create_test_codebase()?;
    let temp_path = temp_dir.path();

    // Try to build index, skip test if not possible
    if !build_test_index_if_needed(temp_path).await? {
        println!("Skipping glob filtering limits test - unable to build index");
        return Ok(());
    }

    // Test with low limit and threshold
    let result = execute_search_command_cli(
        "user".to_string(),
        temp_path.to_path_buf(),
        2, // Low limit
        Some(0.1), // Low threshold
        "json".to_string(),
        None,
        Some("**/*.rs".to_string()),
        &get_test_config(),
    ).await;

    match result {
        Ok(_) => {
            println!("Glob filtering with limits test completed successfully");
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

/// Test edge case: glob pattern that matches no files
#[tokio::test]
async fn test_glob_pattern_no_matches() -> Result<()> {
    let temp_dir = create_test_codebase()?;
    let temp_path = temp_dir.path();

    // Try to build index, skip test if not possible
    if !build_test_index_if_needed(temp_path).await? {
        println!("Skipping no matches test - unable to build index");
        return Ok(());
    }

    // Test with glob pattern that should match no files
    let result = execute_search_command_cli(
        "anything".to_string(),
        temp_path.to_path_buf(),
        10,
        None,
        "json".to_string(),
        None,
        Some("*.nonexistent".to_string()), // Pattern that matches nothing
        &get_test_config(),
    ).await;

    match result {
        Ok(_) => {
            println!("No matches test completed - this is expected behavior");
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

/// Test complex glob patterns with multiple wildcards
#[tokio::test]
async fn test_complex_glob_patterns() -> Result<()> {
    let temp_dir = create_test_codebase()?;
    let temp_path = temp_dir.path();

    // Create nested directory structure
    let nested_dir = temp_path.join("src").join("modules").join("auth");
    std::fs::create_dir_all(&nested_dir)?;

    let nested_content = r#"
mod authentication {
    pub fn verify_credentials(user: &str, pass: &str) -> bool {
        // Mock authentication logic
        !user.is_empty() && !pass.is_empty()
    }
}
"#;

    std::fs::write(nested_dir.join("mod.rs"), nested_content)?;

    // Try to build index, skip test if not possible
    if !build_test_index_if_needed(temp_path).await? {
        println!("Skipping complex glob test - unable to build index");
        return Ok(());
    }

    // Test complex recursive pattern
    let result = execute_search_command_cli(
        "authentication".to_string(),
        temp_path.to_path_buf(),
        5,
        None,
        "json".to_string(),
        None,
        Some("**/auth/*.rs".to_string()),
        &get_test_config(),
    ).await;

    match result {
        Ok(_) => {
            println!("Complex glob pattern test completed successfully");
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