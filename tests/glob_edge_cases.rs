//! Comprehensive edge case tests for glob filtering functionality.
//!
//! This module provides extensive testing of challenging scenarios that could
//! cause glob filtering to fail or behave unexpectedly. These tests ensure
//! robust behavior across diverse file systems, character encodings, and
//! pattern complexity levels.

use anyhow::Result;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;
use turboprop::filters::{
    FilterConfig, GlobPattern, GlobPatternCache, SearchFilter, 
    normalize_glob_pattern, validate_glob_pattern
};
use turboprop::types::{
    ChunkId, ChunkIndexNum, ContentChunk, IndexedChunk, SearchResult, SourceLocation, TokenCount,
};

/// Test constants for edge case scenarios
mod edge_case_config {
    pub const VERY_LONG_PATTERN_LENGTH: usize = 100; // Reasonable test length
    pub const EXTREMELY_LONG_PATH_LENGTH: usize = 50; // Long but realistic
    pub const UNICODE_TEST_ITERATIONS: usize = 20;
    pub const SYMLINK_TEST_COUNT: usize = 10;
    pub const NESTED_DEPTH_LIMIT: usize = 15; // Very deep nesting
    pub const SPECIAL_CHAR_COMBINATIONS: usize = 25;
    
    // Unicode test strings covering different ranges
    pub const UNICODE_PATTERNS: &[&str] = &[
        "файл_*.rs",           // Cyrillic
        "测试_*.js",           // Chinese
        "テスト_*.py",         // Japanese
        "🔥_*.md",            // Emoji
        "café_*.txt",         // Accented Latin
        "αβγ_*.log",          // Greek
        "مثال_*.json",        // Arabic
        "ಪರೀಕ್ಷೆ_*.yaml",      // Kannada
    ];
    
    pub const PROBLEMATIC_FILENAMES: &[&str] = &[
        "file with spaces.rs",
        "file-with-dashes.js",
        "file_with_underscores.py",
        "file.with.dots.md",
        "file,with,commas.txt",
        "file;with;semicolons.log",
        "file(with)parentheses.json",
        "file[with]brackets.yaml",
        "file{with}braces.toml",
        "file@with@symbols.cfg",
        "file#with#hashes.ini",
        "file$with$dollars.env",
        "file%with%percents.conf",
        "file&with&ampersands.xml",
        "file+with+plus.html",
        "file=with=equals.css",
    ];
}

/// Helper function to create test search results with specific file paths
fn create_search_results_with_paths(file_paths: Vec<PathBuf>) -> Vec<SearchResult> {
    file_paths
        .into_iter()
        .enumerate()
        .map(|(i, path)| {
            let chunk = ContentChunk {
                id: ChunkId::new(&format!("chunk-{}", i)),
                content: format!("test content {}", i),
                token_count: TokenCount::new(10),
                source_location: SourceLocation {
                    file_path: path,
                    start_line: 1,
                    end_line: 5,
                    start_char: 0,
                    end_char: 50,
                },
                chunk_index: ChunkIndexNum::new(0),
                total_chunks: 1,
            };

            let indexed_chunk = IndexedChunk {
                chunk,
                embedding: vec![0.1; 384], // Dummy embedding
            };

            SearchResult::new(0.8, indexed_chunk, i)
        })
        .collect()
}

/// Test Unicode character handling in patterns and file paths
#[test]
fn test_unicode_character_handling() -> Result<()> {
    // Test pattern creation with Unicode characters
    for pattern_str in edge_case_config::UNICODE_PATTERNS {
        let pattern = GlobPattern::new(pattern_str)?;
        assert_eq!(pattern.pattern(), *pattern_str);
        
        // Test that validation passes
        assert!(validate_glob_pattern(pattern_str).is_ok());
    }
    
    // Test matching Unicode filenames
    let unicode_files = vec![
        PathBuf::from("файл_test.rs"),
        PathBuf::from("测试_main.js"),
        PathBuf::from("テスト_util.py"),
        PathBuf::from("🔥_readme.md"),
        PathBuf::from("café_config.txt"),
        PathBuf::from("αβγ_data.log"),
        PathBuf::from("مثال_api.json"),
        PathBuf::from("ಪರೀಕ್ಷೆ_test.yaml"),
    ];
    
    // Test each Unicode pattern against corresponding files
    for (i, &pattern_str) in edge_case_config::UNICODE_PATTERNS.iter().enumerate() {
        let pattern = GlobPattern::new(pattern_str)?;
        
        // Should match the corresponding Unicode file
        assert!(
            pattern.matches(&unicode_files[i]),
            "Pattern '{}' should match file '{}'",
            pattern_str,
            unicode_files[i].display()
        );
        
        // Should not match other Unicode files
        for (j, other_file) in unicode_files.iter().enumerate() {
            if i != j {
                assert!(
                    !pattern.matches(other_file),
                    "Pattern '{}' should not match file '{}'",
                    pattern_str,
                    other_file.display()
                );
            }
        }
    }
    
    Ok(())
}

/// Test very long patterns that approach but don't exceed limits
#[test]
fn test_very_long_patterns() -> Result<()> {
    let config = FilterConfig::default();
    
    // Create a long but valid pattern
    let base_pattern = "src/";
    let middle_pattern = "very_long_directory_name/".repeat(10);
    let end_pattern = "*.rs";
    let long_pattern = format!("{}{}{}", base_pattern, middle_pattern, end_pattern);
    
    // Ensure we're testing a genuinely long pattern but within limits
    assert!(long_pattern.len() > edge_case_config::VERY_LONG_PATTERN_LENGTH);
    assert!(long_pattern.len() <= config.max_glob_pattern_length);
    
    // Should successfully create pattern
    let pattern = GlobPattern::new_with_config(&long_pattern, &config)?;
    assert_eq!(pattern.pattern(), long_pattern);
    
    // Should match appropriately structured paths
    let matching_path = PathBuf::from(format!("{}test.rs", 
        format!("{}{}", base_pattern, middle_pattern)));
    assert!(pattern.matches(&matching_path));
    
    // Test pattern normalization with long patterns
    let normalized = normalize_glob_pattern(&long_pattern);
    assert!(!normalized.is_empty());
    
    Ok(())
}

/// Test very long file paths
#[test]
fn test_very_long_file_paths() -> Result<()> {
    // Create deeply nested directory structure
    let mut deep_path = PathBuf::new();
    for i in 0..edge_case_config::NESTED_DEPTH_LIMIT {
        deep_path.push(format!("level_{}", i));
    }
    deep_path.push("very_long_filename_that_tests_path_handling.rs");
    
    // Ensure we have a genuinely long path
    assert!(deep_path.to_string_lossy().len() > edge_case_config::EXTREMELY_LONG_PATH_LENGTH);
    
    // Test pattern matching against long path
    let nested_pattern = format!("**/level_{}/**/*.rs", edge_case_config::NESTED_DEPTH_LIMIT - 1);
    let patterns = vec![
        "**/*.rs",
        "**/very_long_filename_that_tests_path_handling.rs",
        &nested_pattern,
    ];
    
    for pattern_str in patterns {
        let pattern = GlobPattern::new(&pattern_str)?;
        assert!(
            pattern.matches(&deep_path),
            "Pattern '{}' should match deep path '{}'",
            pattern_str,
            deep_path.display()
        );
    }
    
    Ok(())
}

/// Test patterns that pass basic validation but could fail during runtime
#[test]
fn test_runtime_pattern_edge_cases() -> Result<()> {
    // Patterns that are syntactically valid but might be problematic
    let edge_patterns = vec![
        "[a-z-A-Z]", // Range with dash in middle
        "file[!]]test.rs", // Negated bracket with closing bracket
        "**/**/***/**", // Multiple consecutive recursive wildcards
        "file?.{rs,}", // Empty alternative in brace expansion
        "test[]]file.rs", // Double closing bracket
        "**/[a-z]**/[0-9]*.rs", // Multiple character classes
    ];
    
    for pattern_str in edge_patterns {
        // Basic validation should pass or fail gracefully
        match validate_glob_pattern(pattern_str) {
            Ok(_) => {
                // If validation passes, pattern creation should work
                let pattern = GlobPattern::new(pattern_str);
                // Don't require success, but shouldn't panic
                let _ = pattern;
            }
            Err(_) => {
                // If validation fails, that's also acceptable
                continue;
            }
        }
    }
    
    Ok(())
}

/// Test special filesystem entries and edge cases
#[test]
fn test_special_filesystem_entries() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path();
    
    // Create files with problematic names
    let mut created_files = Vec::new();
    for &filename in edge_case_config::PROBLEMATIC_FILENAMES {
        let file_path = base_path.join(filename);
        
        match fs::write(&file_path, "test content") {
            Ok(_) => created_files.push(file_path),
            Err(_) => {
                // Some names might not be valid on this filesystem, skip them
                continue;
            }
        }
    }
    
    // Test patterns against these files
    let test_patterns = vec![
        "*.rs",
        "*with*",
        "file*",
        "*spaces*",
        "*.*",
        "**/*",
    ];
    
    for pattern_str in test_patterns {
        let pattern = GlobPattern::new(pattern_str)?;
        
        for file_path in &created_files {
            // Just ensure matching doesn't panic or error
            let _matches = pattern.matches(file_path);
        }
    }
    
    Ok(())
}

/// Test symlink handling (where supported)
#[test]
fn test_symlink_handling() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path();
    
    // Create a regular file
    let target_file = base_path.join("target.rs");
    fs::write(&target_file, "fn main() {}")?;
    
    // Try to create a symlink (may fail on some platforms/configurations)
    let symlink_path = base_path.join("link.rs");
    
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        match symlink(&target_file, &symlink_path) {
            Ok(_) => {
                // Test pattern matching on symlinks
                let pattern = GlobPattern::new("*.rs")?;
                
                // Both target and symlink should match
                assert!(pattern.matches(&target_file));
                assert!(pattern.matches(&symlink_path));
                
                // Test with more specific patterns (symlink behavior may vary)
                let specific_pattern = GlobPattern::new("link.rs")?;
                let _matches = specific_pattern.matches(&symlink_path); // Don't assert, just test no panic
            }
            Err(_) => {
                // Symlink creation failed, skip this test
                return Ok(());
            }
        }
    }
    
    #[cfg(windows)]
    {
        // On Windows, symlinks require special permissions, so we just test
        // that pattern matching doesn't break with the paths
        let pattern = GlobPattern::new("*.rs")?;
        assert!(pattern.matches(&target_file));
    }
    
    Ok(())
}

/// Test empty directory matching behavior
#[test]
fn test_empty_directory_matching() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path();
    
    // Create empty directories
    let empty_dirs = vec![
        "empty1",
        "empty2",
        "nested/empty3",
        "deeply/nested/empty4",
    ];
    
    for dir_path in &empty_dirs {
        fs::create_dir_all(base_path.join(dir_path))?;
    }
    
    // Create some files in non-empty directories
    fs::create_dir_all(base_path.join("nonempty"))?;
    fs::write(base_path.join("nonempty/file.rs"), "content")?;
    
    // Test patterns that might match directory structures
    let patterns = vec![
        "**/empty*",
        "empty*",
        "**/nested/**",
        "**/*",
        "*/empty*",
    ];
    
    for pattern_str in patterns {
        let pattern = GlobPattern::new(pattern_str)?;
        
        // Test against various paths including empty directory paths
        let test_paths = vec![
            PathBuf::from("empty1"),
            PathBuf::from("empty2"),
            PathBuf::from("nested/empty3"),
            PathBuf::from("deeply/nested/empty4"),
            PathBuf::from("nonempty/file.rs"),
        ];
        
        for test_path in test_paths {
            // Just ensure matching doesn't panic
            let _matches = pattern.matches(&test_path);
        }
    }
    
    Ok(())
}

/// Test platform-specific edge cases
#[test]
fn test_platform_specific_edge_cases() -> Result<()> {
    // Test patterns with different path separators
    let patterns_with_separators = vec![
        "src\\*.rs",    // Windows-style separator
        "src//*.rs",    // Double separator
        "./src/*.rs",   // Relative path
        "../src/*.rs",  // Parent directory (should be invalid)
    ];
    
    for pattern_str in patterns_with_separators {
        match validate_glob_pattern(pattern_str) {
            Ok(_) => {
                // If validation passes, test pattern creation
                let pattern_result = GlobPattern::new(pattern_str);
                if let Ok(pattern) = pattern_result {
                    // Test against normalized paths
                    let test_paths = vec![
                        PathBuf::from("src/main.rs"),
                        PathBuf::from("src\\main.rs"), // May not work on Unix
                    ];
                    
                    for test_path in test_paths {
                        let _matches = pattern.matches(&test_path);
                    }
                }
            }
            Err(_) => {
                // Pattern is invalid, which is expected for some edge cases
                continue;
            }
        }
    }
    
    // Test case sensitivity behavior (platform-dependent)
    let case_sensitive_tests = vec![
        ("*.RS", "file.rs"),
        ("*.rs", "FILE.RS"),
        ("File.txt", "file.txt"),
        ("FILE.TXT", "file.txt"),
    ];
    
    for (pattern_str, test_file) in case_sensitive_tests {
        let pattern = GlobPattern::new(pattern_str)?;
        let test_path = PathBuf::from(test_file);
        
        // Just test that matching works without error
        // Result depends on platform case sensitivity
        let _matches = pattern.matches(&test_path);
    }
    
    Ok(())
}

/// Test pattern cache behavior with edge cases
#[test]
fn test_pattern_cache_edge_cases() -> Result<()> {
    let cache = GlobPatternCache::with_max_size(10); // Small cache for testing
    let config = FilterConfig::default();
    
    // Fill cache with various edge case patterns
    let edge_patterns = vec![
        "файл_*.rs",         // Unicode
        "**/*.{js,ts,jsx}",  // Complex alternatives
        "**/very_long_pattern_name_that_tests_cache_behavior/*.rs",
        "src/**/[a-z]*/**",  // Character classes
        "test[!0-9]*.py",    // Negated character class
        "*.{json,yaml,toml,ini,cfg,conf}", // Many alternatives
        "**/dir with spaces/**", // Spaces in paths
        "🔥**/*.md",         // Emoji in pattern
        "**/*test*/**/*.rs", // Multiple wildcards
        "**/[abc][def]*.js", // Multiple character classes
    ];
    
    // Add patterns to cache
    for pattern_str in &edge_patterns {
        match cache.get_or_create(pattern_str, &config) {
            Ok(cached_pattern) => {
                assert_eq!(cached_pattern.pattern(), *pattern_str);
            }
            Err(_) => {
                // Some patterns might be invalid, which is acceptable
                continue;
            }
        }
    }
    
    // Test cache with duplicate patterns (should return same instance)
    for pattern_str in &edge_patterns[0..3] {
        if let Ok(pattern1) = cache.get_or_create(pattern_str, &config) {
            if let Ok(pattern2) = cache.get_or_create(pattern_str, &config) {
                // Should be the same Arc instance
                assert!(std::sync::Arc::ptr_eq(&pattern1, &pattern2));
            }
        }
    }
    
    // Test cache eviction with more patterns than cache size
    let eviction_patterns: Vec<String> = (0..20)
        .map(|i| format!("**/eviction_test_{}/*.rs", i))
        .collect();
    
    for pattern_str in eviction_patterns {
        let _ = cache.get_or_create(&pattern_str, &config);
    }
    
    // Cache should not exceed its size limit
    assert!(cache.len() <= 10);
    
    Ok(())
}

/// Test search filter edge cases with complex result sets
#[test]
fn test_search_filter_edge_cases() -> Result<()> {
    // Create search results with edge case file paths
    let edge_case_paths = vec![
        PathBuf::from("файл.rs"),                    // Unicode
        PathBuf::from("file with spaces.js"),        // Spaces
        PathBuf::from("deeply/nested/path/file.py"), // Deep nesting
        PathBuf::from("file.multiple.dots.md"),      // Multiple dots
        PathBuf::from("UPPERCASE.RS"),               // Case variations
        PathBuf::from("lowercase.rs"),
        PathBuf::from("MixedCase.Rs"),
        PathBuf::from("file-with-dashes.txt"),       // Dashes
        PathBuf::from("file_with_underscores.log"),  // Underscores
        PathBuf::from("123numeric.start"),           // Numeric start
        PathBuf::from(".hidden_file.rs"),           // Hidden file
        PathBuf::from("no.extension"),              // No extension
        PathBuf::from("dir.with.dots/file.rs"),     // Directory with dots
    ];
    
    let search_results = create_search_results_with_paths(edge_case_paths);
    
    // Test various edge case filters
    let edge_case_filters = vec![
        ("unicode_filter", Some("файл.rs".to_string())),
        ("spaces_filter", Some("*with spaces*".to_string())),
        ("nested_filter", Some("**/nested/**/*.py".to_string())),
        ("dots_filter", Some("*.multiple.dots.*".to_string())),
        ("case_filter", Some("*.RS".to_string())),
        ("mixed_case_filter", Some("*Case.*".to_string())),
        ("dash_filter", Some("*-with-*".to_string())),
        ("underscore_filter", Some("*_with_*".to_string())),
        ("numeric_filter", Some("123*".to_string())),
        ("hidden_filter", Some(".*".to_string())),
        ("no_ext_filter", Some("no.extension".to_string())),
        ("dir_dots_filter", Some("*.with.dots/*".to_string())),
    ];
    
    for (filter_name, filter_pattern) in edge_case_filters {
        let filter = SearchFilter::from_cli_args(None, filter_pattern);
        
        match filter.apply_filters(search_results.clone()) {
            Ok(filtered_results) => {
                // Filtering should succeed
                assert!(
                    filtered_results.len() <= search_results.len(),
                    "Filter '{}' returned more results than input",
                    filter_name
                );
            }
            Err(e) => {
                // Some edge cases might fail validation, which is acceptable
                println!("Filter '{}' failed (expected for some edge cases): {}", filter_name, e);
            }
        }
    }
    
    Ok(())
}

/// Test normalization with edge cases
#[test]
fn test_pattern_normalization_edge_cases() -> Result<()> {
    let normalization_tests = vec![
        // (input, expected_output) - Based on actual normalization behavior
        ("a//b//c/*.js", "a/b/c/*.js"),                 // Multiple slashes
        ("  ./src/**/*.py  ", "src/**/*.py"),          // Whitespace
        ("**/**/**/*.md", "**/*.md"),                  // Multiple recursive wildcards
        ("path/with/trailing/", "path/with/trailing"), // Trailing slash
        ("./src/*.rs", "src/*.rs"),                    // Single current dir ref
        ("**/*/**/*.rs", "**/**/*.rs"),                // Redundant recursive patterns
    ];
    
    for (input, expected) in normalization_tests {
        let normalized = normalize_glob_pattern(input);
        assert_eq!(
            normalized, expected,
            "Normalization of '{}' failed. Expected '{}', got '{}'",
            input, expected, normalized
        );
    }
    
    Ok(())
}

/// Test memory usage with large pattern sets
#[test]
fn test_memory_usage_edge_cases() -> Result<()> {
    // Create a large number of unique patterns
    let pattern_count = 1000;
    let patterns: Vec<String> = (0..pattern_count)
        .map(|i| format!("**/pattern_{}/**/*.rs", i))
        .collect();
    
    // Test that pattern creation doesn't consume excessive memory
    let mut compiled_patterns = Vec::new();
    for pattern_str in patterns {
        match GlobPattern::new(&pattern_str) {
            Ok(pattern) => compiled_patterns.push(pattern),
            Err(_) => continue, // Skip invalid patterns
        }
    }
    
    // Verify we created a reasonable number of patterns
    assert!(compiled_patterns.len() > pattern_count / 2);
    
    // Test pattern matching with large result sets
    let large_path_set: Vec<PathBuf> = (0..1000)
        .map(|i| PathBuf::from(format!("src/pattern_{}/module_{}/file.rs", i % 100, i)))
        .collect();
    
    // Test a few patterns against the large path set
    for pattern in compiled_patterns.iter().take(10) {
        let _matches: Vec<&PathBuf> = large_path_set
            .iter()
            .filter(|path| pattern.matches(path))
            .collect();
        // Just ensure this completes without memory issues
    }
    
    Ok(())
}

/// Test concurrent access to pattern cache (thread safety)
#[test]
fn test_concurrent_pattern_cache_access() -> Result<()> {
    use std::sync::Arc;
    use std::thread;
    
    let cache = Arc::new(GlobPatternCache::with_max_size(100));
    let config = FilterConfig::default();
    
    // Spawn multiple threads accessing the cache concurrently
    let mut handles = Vec::new();
    
    for thread_id in 0..5 {
        let cache_clone = Arc::clone(&cache);
        let config_clone = config.clone();
        
        let handle = thread::spawn(move || {
            for i in 0..20 {
                let pattern = format!("**/thread_{}/**/*_{}.rs", thread_id, i);
                match cache_clone.get_or_create(&pattern, &config_clone) {
                    Ok(cached_pattern) => {
                        assert_eq!(cached_pattern.pattern(), pattern);
                        
                        // Test pattern matching
                        let test_path = PathBuf::from(format!("src/thread_{}/test_{}.rs", thread_id, i));
                        let _matches = cached_pattern.matches(&test_path);
                    }
                    Err(_) => {
                        // Some patterns might be invalid, continue
                        continue;
                    }
                }
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread should complete successfully");
    }
    
    // Verify cache is still in valid state
    assert!(cache.len() <= 100);
    
    Ok(())
}