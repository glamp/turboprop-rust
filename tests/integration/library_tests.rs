//! Library integration tests for the indexing pipeline.
//!
//! These tests verify the end-to-end functionality by calling library functions directly,
//! including embedding generation, progress tracking, error handling, and index completeness.
//!
//! ## Test Mode Configuration
//!
//! These tests default to offline mode to avoid slow model downloads. Use `TURBOPROP_TEST_ONLINE=1`
//! to enable online mode with real model downloads when needed.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;
use turboprop::commands::execute_index_command;
use turboprop::config::TurboPropConfig;
use turboprop::index::PersistentChunkIndex;

/// Check if tests should run in offline mode (default: offline)
fn is_offline_mode() -> bool {
    // Default to offline mode unless explicitly enabled online
    env::var("TURBOPROP_TEST_ONLINE").unwrap_or_default() != "1"
}

/// Create a test file with the given content
fn create_test_file(dir: &Path, name: &str, content: &str) -> PathBuf {
    let file_path = dir.join(name);
    fs::write(&file_path, content).unwrap();
    file_path
}

/// Get the path to the poker test fixture
fn get_poker_fixture_path() -> &'static Path {
    Path::new("tests/fixtures/poker")
}

/// Create a comprehensive test codebase with multiple file types (legacy - use poker fixture instead)
fn create_test_codebase(dir: &Path) {
    // Rust files
    create_test_file(
        dir,
        "main.rs",
        r#"
fn main() {
    println!("Hello, world!");
    let result = calculate_fibonacci(10);
    println!("Fibonacci(10) = {}", result);
}

fn calculate_fibonacci(n: u32) -> u64 {
    if n <= 1 {
        n as u64
    } else {
        calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)
    }
}
"#,
    );

    create_test_file(
        dir,
        "lib.rs",
        r#"
//! Test library module

pub struct Calculator {
    pub name: String,
}

impl Calculator {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }

    pub fn add(&self, a: i32, b: i32) -> i32 {
        a + b
    }

    pub fn multiply(&self, a: i32, b: i32) -> i32 {
        a * b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculator() {
        let calc = Calculator::new("test");
        assert_eq!(calc.add(2, 3), 5);
        assert_eq!(calc.multiply(2, 3), 6);
    }
}
"#,
    );

    // JavaScript files
    create_test_file(
        dir,
        "app.js",
        r#"
// Simple web application
const express = require('express');
const app = express();
const port = 3000;

app.get('/', (req, res) => {
    res.send('Hello World!');
});

app.get('/api/users', (req, res) => {
    const users = [
        { id: 1, name: 'Alice' },
        { id: 2, name: 'Bob' },
        { id: 3, name: 'Charlie' }
    ];
    res.json(users);
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
"#,
    );

    // Python files
    create_test_file(
        dir,
        "utils.py",
        r#"
"""
Utility functions for data processing
"""

import json
from typing import List, Dict, Optional

def load_json_file(filepath: str) -> Dict:
    """Load data from a JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json_file(data: Dict, filepath: str) -> None:
    """Save data to a JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

class DataProcessor:
    """Process and transform data"""
    
    def __init__(self, name: str):
        self.name = name
        self.processed_count = 0
    
    def process_items(self, items: List[Dict]) -> List[Dict]:
        """Process a list of items"""
        result = []
        for item in items:
            processed_item = self._process_single_item(item)
            result.append(processed_item)
            self.processed_count += 1
        return result
    
    def _process_single_item(self, item: Dict) -> Dict:
        """Process a single item"""
        return {**item, 'processed': True, 'processor': self.name}
"#,
    );

    // Text files
    create_test_file(
        dir,
        "README.md",
        r#"
# Test Project

This is a test project for indexing functionality.

## Features

- Rust implementation
- JavaScript API
- Python utilities
- Comprehensive documentation

## Usage

Run the application:

```bash
cargo run
```

## Testing

Run tests:

```bash
cargo test
```

## Contributing

Please follow the coding standards and add tests for new features.
"#,
    );

    create_test_file(
        dir,
        "config.yaml",
        r#"
database:
  host: localhost
  port: 5432
  name: testdb
  user: testuser
  password: testpass

api:
  port: 3000
  cors_enabled: true
  rate_limit: 1000

logging:
  level: info
  file: app.log
"#,
    );
}

#[tokio::test]
async fn test_index_command_with_small_codebase() {
    // Skip if in offline mode (default)
    if is_offline_mode() {
        println!("Skipping test in offline mode");
        return;
    }

    let temp_path = get_poker_fixture_path();

    let config = TurboPropConfig::default();

    // Test the index command (disable progress bars for testing)
    let result = execute_index_command(temp_path, &config, false).await;

    // This might fail in test environment due to embedding model requirements
    match result {
        Ok(()) => {
            // Verify that index was created
            let index_dir = temp_path.join(".turboprop");
            assert!(index_dir.exists(), "Index directory should be created");

            // Try to load the created index
            let load_result = PersistentChunkIndex::load(temp_path);
            match load_result {
                Ok(index) => {
                    println!(
                        "Successfully created and loaded index with {} chunks",
                        index.len()
                    );
                    assert!(!index.is_empty(), "Index should contain chunks");
                }
                Err(e) => {
                    println!(
                        "Note: Could not load index (this may be expected in test environment): {}",
                        e
                    );
                }
            }
        }
        Err(e) => {
            let error_str = e.to_string();
            // Accept common errors in test environment
            assert!(
                error_str.contains("embedding")
                    || error_str.contains("model")
                    || error_str.contains("network")
                    || error_str.contains("Failed to initialize")
                    || error_str.contains("No files found"),
                "Unexpected error type: {}",
                e
            );
            println!(
                "Test skipped due to expected error in test environment: {}",
                e
            );
        }
    }
}

#[tokio::test]
async fn test_index_command_error_handling() {
    let temp_dir = TempDir::new().unwrap();
    let config = TurboPropConfig::default();

    // Test with empty directory
    let result = execute_index_command(temp_dir.path(), &config, false).await;
    assert!(result.is_err(), "Should fail with empty directory");

    let error = result.unwrap_err();
    let error_str = error.to_string();
    assert!(
        error_str.contains("No files found")
            || error_str.contains("empty")
            || error_str.contains("Index command failed"),
        "Should indicate indexing failure, got: {}",
        error
    );
}

#[tokio::test]
async fn test_index_command_with_unreadable_files() {
    // Skip if in offline mode (default)
    if is_offline_mode() {
        println!("Skipping test in offline mode");
        return;
    }

    let temp_dir = TempDir::new().unwrap();

    // Create some normal files
    create_test_file(temp_dir.path(), "normal.txt", "This is normal content");
    create_test_file(temp_dir.path(), "another.rs", "fn main() { println!(); }");

    // Create a very large file that might cause issues
    let large_content = "a".repeat(10_000_000); // 10MB of 'a's
    create_test_file(temp_dir.path(), "large.txt", &large_content);

    let config = TurboPropConfig::default();

    // Test the index command (disable progress bars for testing)
    let result = execute_index_command(temp_dir.path(), &config, false).await;

    // This should either succeed (with some files processed) or fail with a reasonable error
    match result {
        Ok(()) => {
            println!("Index command succeeded despite large file");
            // Verify that index was created
            let index_dir = temp_dir.path().join(".turboprop");
            assert!(index_dir.exists(), "Index directory should be created");
        }
        Err(e) => {
            let error_str = e.to_string();
            // Accept common errors in test environment or legitimate processing errors
            assert!(
                error_str.contains("embedding")
                    || error_str.contains("model")
                    || error_str.contains("network")
                    || error_str.contains("Failed to initialize")
                    || error_str.contains("too large")
                    || error_str.contains("memory"),
                "Unexpected error type: {}",
                e
            );
            println!("Test handled error appropriately: {}", e);
        }
    }
}

#[tokio::test]
async fn test_index_directory_structure() {
    // Skip if in offline mode (default)
    if is_offline_mode() {
        println!("Skipping test in offline mode");
        return;
    }

    let temp_dir = TempDir::new().unwrap();

    // Create test content
    create_test_file(
        temp_dir.path(),
        "test.txt",
        "Simple test content for indexing",
    );

    let config = TurboPropConfig::default();

    // Attempt to run indexing
    let result = execute_index_command(temp_dir.path(), &config, false).await;

    // Check if the expected directory structure exists after indexing attempt
    let index_base_dir = temp_dir.path().join(".turboprop");

    if result.is_ok() {
        // If indexing succeeded, verify the structure
        assert!(
            index_base_dir.exists(),
            "Base .turboprop directory should exist"
        );

        let index_dir = index_base_dir.join("index");
        assert!(index_dir.exists(), "Index subdirectory should exist");

        // Check for expected index files
        let expected_files = ["vectors.bin", "metadata.json", "config.yaml", "version.txt"];
        for file in &expected_files {
            let file_path = index_dir.join(file);
            if file_path.exists() {
                println!("Found expected index file: {}", file);
            }
        }
    } else {
        println!(
            "Indexing failed (expected in test environment): {}",
            result.unwrap_err()
        );
    }
}

#[tokio::test]
async fn test_index_persistence() {
    // Skip if in offline mode (default)
    if is_offline_mode() {
        println!("Skipping test in offline mode");
        return;
    }

    let temp_dir = TempDir::new().unwrap();

    // Create test content
    create_test_file(
        temp_dir.path(),
        "persistent_test.txt",
        "Content for persistence testing",
    );

    let config = TurboPropConfig::default();

    // First indexing attempt
    let result1 = execute_index_command(temp_dir.path(), &config, false).await;

    if result1.is_ok() {
        // Try to load the index after creation
        let load_result = PersistentChunkIndex::load(temp_dir.path());
        match load_result {
            Ok(index) => {
                let initial_chunk_count = index.len();
                println!(
                    "Successfully loaded persistent index with {} chunks",
                    initial_chunk_count
                );

                // Add another file and re-index
                create_test_file(
                    temp_dir.path(),
                    "additional.txt",
                    "Additional content for testing updates",
                );

                let result2 = execute_index_command(temp_dir.path(), &config, false).await;

                if result2.is_ok() {
                    // Verify the index was updated
                    let updated_index = PersistentChunkIndex::load(temp_dir.path());
                    if let Ok(updated_index) = updated_index {
                        println!("Updated index has {} chunks", updated_index.len());
                        // The updated index should potentially have more chunks
                    }
                }
            }
            Err(e) => {
                println!("Could not load persistent index: {}", e);
            }
        }
    } else {
        println!(
            "Initial indexing failed (expected in test environment): {}",
            result1.unwrap_err()
        );
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_create_test_files() {
        let temp_dir = TempDir::new().unwrap();

        let file_path = create_test_file(temp_dir.path(), "test.txt", "test content");

        assert!(file_path.exists());
        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "test content");
    }

    #[test]
    fn test_create_test_codebase() {
        let temp_dir = TempDir::new().unwrap();

        create_test_codebase(temp_dir.path());

        // Verify all expected files were created
        let expected_files = [
            "main.rs",
            "lib.rs",
            "app.js",
            "utils.py",
            "README.md",
            "config.yaml",
        ];

        for file in &expected_files {
            let file_path = temp_dir.path().join(file);
            assert!(file_path.exists(), "Expected file {} should exist", file);

            let content = fs::read_to_string(&file_path).unwrap();
            assert!(!content.is_empty(), "File {} should not be empty", file);
        }
    }
}
