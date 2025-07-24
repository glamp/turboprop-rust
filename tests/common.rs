//! Common test utilities and helpers.
//!
//! This module provides shared functionality for test setup, test data creation,
//! and common test operations to reduce duplication across test files.

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::{env, fs};
use tempfile::TempDir;
use turboprop::embeddings::EmbeddingConfig;

/// Get the path to the poker test fixture
#[allow(dead_code)]
pub fn get_poker_fixture_path() -> &'static Path {
    Path::new("tests/fixtures/poker")
}

/// Create a temporary directory with sample Rust codebase for testing
pub fn create_test_codebase() -> Result<TempDir> {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    // Create the main library file
    let rust_content_main = r#"
fn main() {
    println!("Hello, world!");
    let result = calculate_total(vec![1, 2, 3]);
    println!("Total: {}", result);
}

fn calculate_total(numbers: Vec<i32>) -> i32 {
    numbers.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_calculate_total() {
        assert_eq!(calculate_total(vec![1, 2, 3]), 6);
    }
}
"#;

    // Create a user management module
    let rust_content_user = r#"
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

    // Create authentication module
    let rust_content_auth = r#"
use crate::user::{User, create_user_database};
use std::collections::HashMap;

pub struct AuthService {
    users: HashMap<u64, User>,
}

impl AuthService {
    pub fn new() -> Self {
        Self {
            users: create_user_database(),
        }
    }
    
    pub fn login(&self, user_id: u64, password: &str) -> Result<bool, String> {
        match self.users.get(&user_id) {
            Some(user) => user.authenticate(password),
            None => Err("User not found".to_string()),
        }
    }
    
    pub fn validate_password_strength(password: &str) -> Result<(), String> {
        if password.len() < 8 {
            return Err("Password must be at least 8 characters long".to_string());
        }
        if !password.chars().any(|c| c.is_uppercase()) {
            return Err("Password must contain at least one uppercase letter".to_string());
        }
        if !password.chars().any(|c| c.is_lowercase()) {
            return Err("Password must contain at least one lowercase letter".to_string());
        }
        if !password.chars().any(|c| c.is_numeric()) {
            return Err("Password must contain at least one number".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_login_success() {
        let auth_service = AuthService::new();
        assert!(auth_service.login(1, "password123").is_ok());
    }
    
    #[test]
    fn test_login_invalid_user() {
        let auth_service = AuthService::new();
        assert!(auth_service.login(999, "password123").is_err());
    }
}
"#;

    // Write the files
    create_test_file(temp_path, "main.rs", rust_content_main);
    create_test_file(temp_path, "user.rs", rust_content_user);
    create_test_file(temp_path, "auth.rs", rust_content_auth);

    // Create a lib.rs file
    let lib_content = r#"
pub mod user;
pub mod auth;

pub use user::{User, create_user_database};
pub use auth::AuthService;
"#;
    create_test_file(temp_path, "lib.rs", lib_content);

    Ok(temp_dir)
}

/// Create a test file with the given content in the specified directory
pub fn create_test_file(dir: &Path, name: &str, content: &str) -> PathBuf {
    let file_path = dir.join(name);
    fs::write(&file_path, content).expect("Failed to write test file");
    file_path
}

/// Get a persistent test cache directory that survives across test runs
///
/// This prevents model downloads on every test run by using a stable cache location.
/// Supports TURBOPROP_TEST_CACHE_DIR environment variable for customization.
pub fn get_persistent_test_cache_dir() -> PathBuf {
    if let Ok(cache_dir) = env::var("TURBOPROP_TEST_CACHE_DIR") {
        return PathBuf::from(cache_dir);
    }

    // Use a stable directory in the project for test caching
    let cache_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("test_cache")
        .join("models");

    // Ensure the directory exists
    if let Err(e) = fs::create_dir_all(&cache_dir) {
        eprintln!(
            "Warning: Failed to create test cache directory {:?}: {}",
            cache_dir, e
        );
        // Fallback to temp directory if we can't create the persistent one
        return std::env::temp_dir().join("turboprop_test_cache");
    }

    cache_dir
}

/// Create an EmbeddingConfig with persistent cache for tests
///
/// This uses a stable cache directory to avoid re-downloading models on every test run.
pub fn create_persistent_embedding_config() -> EmbeddingConfig {
    EmbeddingConfig::default().with_cache_dir(get_persistent_test_cache_dir())
}
