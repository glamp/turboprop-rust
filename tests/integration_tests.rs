use std::path::Path;
use tp::{files::FileDiscovery, git::GitRepo, index_files, types::FileDiscoveryConfig};

#[test]
fn test_discover_poker_codebase_files() {
    let poker_path = Path::new("sample-codebases/poker");
    if !poker_path.exists() {
        return; // Skip if sample not available
    }

    let config = FileDiscoveryConfig::default();
    let discovery = FileDiscovery::new(config);
    let files = discovery.discover_files(poker_path).unwrap();

    // Should find TypeScript, JavaScript, CSS, and JSON files
    assert!(!files.is_empty(), "Should find files in poker codebase");

    // Check for specific known files
    let file_paths: Vec<String> = files
        .iter()
        .map(|f| f.path.to_string_lossy().to_string())
        .collect();

    let has_tsx = file_paths.iter().any(|p| p.contains("index.tsx"));
    let has_package_json = file_paths.iter().any(|p| p.contains("package.json"));
    let has_next_config = file_paths.iter().any(|p| p.contains("next.config.js"));

    assert!(has_tsx, "Should find index.tsx file");
    assert!(has_package_json, "Should find package.json file");
    assert!(has_next_config, "Should find next.config.js file");
}

#[test]
fn test_index_poker_codebase() {
    let poker_path = Path::new("sample-codebases/poker");
    if !poker_path.exists() {
        return; // Skip if sample not available
    }

    let result = index_files(poker_path, None);
    assert!(result.is_ok(), "Should successfully index poker codebase");
}

#[test]
fn test_index_poker_with_max_filesize() {
    let poker_path = Path::new("sample-codebases/poker");
    if !poker_path.exists() {
        return; // Skip if sample not available
    }

    // Test with a reasonable max file size
    let result = index_files(poker_path, Some("1mb"));
    assert!(result.is_ok(), "Should successfully index with 1mb limit");

    // Test with a very small max file size
    let result = index_files(poker_path, Some("1kb"));
    assert!(
        result.is_ok(),
        "Should successfully index with 1kb limit (may filter many files)"
    );
}

#[test]
fn test_git_operations_on_poker_codebase() {
    let poker_path = Path::new("sample-codebases/poker");
    if !poker_path.exists() {
        return; // Skip if sample not available
    }

    // Test git repository detection
    let is_git_repo = GitRepo::is_git_repo(poker_path);

    // The poker codebase might or might not be a git repo
    // Just ensure the function doesn't crash
    println!("Poker codebase is git repo: {}", is_git_repo);
}

#[test]
fn test_react_use_codebase() {
    let react_use_path = Path::new("sample-codebases/react-use");
    if !react_use_path.exists() {
        return; // Skip if sample not available
    }

    let config = FileDiscoveryConfig::default();
    let discovery = FileDiscovery::new(config);
    let files = discovery.discover_files(react_use_path).unwrap();

    assert!(!files.is_empty(), "Should find files in react-use codebase");

    // Should find TypeScript files
    let has_ts_files = files
        .iter()
        .any(|f| f.path.extension().and_then(|s| s.to_str()) == Some("ts"));

    assert!(has_ts_files, "Should find TypeScript files in react-use");
}

#[test]
fn test_file_filtering_with_max_size() {
    let poker_path = Path::new("sample-codebases/poker");
    if !poker_path.exists() {
        return;
    }

    // Test with different file size limits
    let small_config = FileDiscoveryConfig::default().with_max_filesize(1024); // 1KB
    let large_config = FileDiscoveryConfig::default().with_max_filesize(1024 * 1024); // 1MB

    let small_discovery = FileDiscovery::new(small_config);
    let large_discovery = FileDiscovery::new(large_config);

    let small_files = small_discovery.discover_files(poker_path).unwrap();
    let large_files = large_discovery.discover_files(poker_path).unwrap();

    // Should have fewer files with smaller max size
    assert!(
        small_files.len() <= large_files.len(),
        "Smaller max size should result in fewer or equal files"
    );
}

#[test]
fn test_invalid_directory() {
    let invalid_path = Path::new("nonexistent-directory");

    let config = FileDiscoveryConfig::default();
    let discovery = FileDiscovery::new(config);
    let result = discovery.discover_files(invalid_path);

    assert!(result.is_err(), "Should fail when directory doesn't exist");
}

#[test]
fn test_filesize_parsing() {
    use tp::types::parse_filesize;

    assert_eq!(parse_filesize("100"), Ok(100));
    assert_eq!(parse_filesize("2kb"), Ok(2048));
    assert_eq!(parse_filesize("5mb"), Ok(5 * 1024 * 1024));
    assert_eq!(parse_filesize("1gb"), Ok(1024 * 1024 * 1024));

    assert!(parse_filesize("invalid").is_err());
    assert!(parse_filesize("2.5mb").is_err());
}

#[test]
fn test_cli_integration() {
    let poker_path = Path::new("sample-codebases/poker");
    if !poker_path.exists() {
        return;
    }

    // Test the full CLI integration
    let result = index_files(poker_path, Some("2mb"));
    assert!(
        result.is_ok(),
        "CLI integration should work with poker codebase"
    );
}
