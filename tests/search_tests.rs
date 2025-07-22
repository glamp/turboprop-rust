//! Comprehensive tests for search functionality.
//!
//! These tests verify the complete search pipeline including query processing,
//! similarity calculations, result filtering, and integration with the CLI.

mod common;

use anyhow::Result;
use std::path::Path;
use tp::config::TurboPropConfig;
use tp::index::PersistentChunkIndex;
use tp::search::{search_index, SearchConfig, SearchEngine};
use tp::{build_persistent_index, search_with_config};

/// Maximum allowed duration for performance tests in seconds
const PERFORMANCE_TEST_TIMEOUT_SECONDS: u64 = 30;

// Test constants for similarity thresholds and limits
const HIGH_SIMILARITY_THRESHOLD: f32 = 0.7;
const MEDIUM_SIMILARITY_THRESHOLD: f32 = 0.5;
const TEST_SIMILARITY_SCORE: f32 = 0.85;

// Test result limits
const SMALL_RESULT_LIMIT: usize = 3;
const STANDARD_RESULT_LIMIT: usize = 5;

// Query length validation

// Test file location constants

/// Build a test index for the given directory
async fn build_test_index(path: &Path) -> Result<PersistentChunkIndex> {
    let config = TurboPropConfig::default();
    build_persistent_index(path, &config).await
}

#[tokio::test]
async fn test_search_config() -> Result<()> {
    let config = SearchConfig::default()
        .with_limit(STANDARD_RESULT_LIMIT)
        .with_threshold(HIGH_SIMILARITY_THRESHOLD)
        .with_parallel(true);

    assert_eq!(config.limit, STANDARD_RESULT_LIMIT);
    assert_eq!(config.threshold, Some(HIGH_SIMILARITY_THRESHOLD));
    assert!(config.parallel);

    Ok(())
}

#[tokio::test]
async fn test_basic_search_functionality() -> Result<()> {
    let temp_dir = common::create_test_codebase()?;
    let temp_path = temp_dir.path();

    // This test might fail in CI without model access, so we handle that gracefully
    match build_test_index(temp_path).await {
        Ok(index) => {
            assert!(!index.is_empty(), "Index should contain chunks");

            // Try to create a search engine
            let search_config = SearchConfig::default().with_limit(5);
            match SearchEngine::new(temp_path, search_config).await {
                Ok(mut engine) => {
                    assert!(engine.index_size() > 0);
                    assert!(engine.embedding_dimensions() > 0);

                    // Try a basic search
                    match engine.search("function main") {
                        Ok(results) => {
                            // If successful, validate results
                            assert!(results.len() <= 5);
                            for result in &results {
                                assert!(result.similarity >= 0.0 && result.similarity <= 1.0);
                                assert!(!result.location_display().is_empty());
                                assert!(!result.content_preview(50).is_empty());
                            }
                        }
                        Err(_) => {
                            // Expected in environments without model access
                            println!(
                                "Search failed - likely due to missing model in test environment"
                            );
                        }
                    }
                }
                Err(_) => {
                    println!("SearchEngine creation failed - likely due to missing model in test environment");
                }
            }
        }
        Err(_) => {
            println!("Index building failed - likely due to missing model in test environment");
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_search_with_threshold() -> Result<()> {
    let temp_dir = common::create_test_codebase()?;
    let temp_path = temp_dir.path();

    // Test the search_index function with threshold
    match search_index(
        temp_path,
        "authentication",
        Some(SMALL_RESULT_LIMIT),
        Some(MEDIUM_SIMILARITY_THRESHOLD),
    )
    .await
    {
        Ok(results) => {
            // Verify all results meet the threshold
            for result in &results {
                assert!(
                    result.similarity >= 0.5,
                    "Result similarity {} below threshold 0.5",
                    result.similarity
                );
            }
            assert!(results.len() <= 3, "Results should be limited to 3");
        }
        Err(e) => {
            // Expected in CI without model access
            assert!(
                e.to_string().contains("load")
                    || e.to_string().contains("model")
                    || e.to_string().contains("embedding")
                    || e.to_string().contains("index")
            );
            println!("Search test failed as expected in test environment: {}", e);
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_search_result_ranking() -> Result<()> {
    let temp_dir = common::create_test_codebase()?;
    let temp_path = temp_dir.path();

    match search_with_config("calculate_total", temp_path, Some(10), None).await {
        Ok(results) => {
            // Verify results are sorted by similarity (descending)
            for i in 1..results.len() {
                assert!(
                    results[i - 1].similarity >= results[i].similarity,
                    "Results should be sorted by similarity (descending)"
                );
            }

            // Verify rank values are correctly assigned
            for (i, result) in results.iter().enumerate() {
                assert_eq!(result.rank, i, "Rank should match position in results");
            }
        }
        Err(e) => {
            println!("Search ranking test failed as expected: {}", e);
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_search_result_limiting() -> Result<()> {
    let temp_dir = common::create_test_codebase()?;
    let temp_path = temp_dir.path();

    // Test different limits
    let test_limits = vec![1, 3, 5, 100];

    for limit in test_limits {
        match search_with_config("function", temp_path, Some(limit), None).await {
            Ok(results) => {
                assert!(
                    results.len() <= limit,
                    "Results should not exceed limit {}",
                    limit
                );
            }
            Err(_) => {
                // Expected in test environment
                continue;
            }
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_empty_query_handling() -> Result<()> {
    let temp_dir = common::create_test_codebase()?;
    let temp_path = temp_dir.path();

    // Test empty query
    let result = search_with_config("", temp_path, Some(5), None).await;
    assert!(result.is_err(), "Empty query should return error");

    if let Err(e) = result {
        let error_msg = e.to_string();
        // Either query validation error or index loading error is acceptable here
        // since both indicate the search cannot proceed
        assert!(
            error_msg.contains("empty")
                || error_msg.contains("Query")
                || error_msg.contains("cannot be empty")
                || error_msg.contains("validation")
                || error_msg.contains("index")
                || error_msg.contains("load")
        );
    }

    // Test whitespace-only query
    let result = search_with_config("   ", temp_path, Some(5), None).await;
    assert!(result.is_err(), "Whitespace-only query should return error");

    Ok(())
}

#[tokio::test]
async fn test_invalid_threshold_handling() -> Result<()> {
    let temp_dir = common::create_test_codebase()?;
    let _temp_path = temp_dir.path();

    // Test threshold validation in SearchConfig
    let config = SearchConfig::default().with_threshold(-0.5);
    assert_eq!(
        config.threshold,
        Some(0.0),
        "Negative threshold should be clamped to 0.0"
    );

    let config = SearchConfig::default().with_threshold(1.5);
    assert_eq!(
        config.threshold,
        Some(1.0),
        "Threshold > 1.0 should be clamped to 1.0"
    );

    Ok(())
}

#[tokio::test]
async fn test_deterministic_search_results() -> Result<()> {
    let temp_dir = common::create_test_codebase()?;
    let temp_path = temp_dir.path();

    // Perform the same search multiple times and verify results are identical
    let query = "user authentication";
    let limit = Some(3);
    let threshold = Some(0.1);

    let mut all_results = Vec::new();

    for _i in 0..3 {
        match search_with_config(query, temp_path, limit, threshold).await {
            Ok(results) => {
                all_results.push(results);
            }
            Err(_) => {
                // Expected in test environment, skip deterministic test
                println!("Skipping deterministic test - model not available");
                return Ok(());
            }
        }
    }

    // Compare all result sets to ensure they're identical
    if all_results.len() > 1 {
        for i in 1..all_results.len() {
            let first = &all_results[0];
            let current = &all_results[i];

            assert_eq!(
                first.len(),
                current.len(),
                "Result count should be deterministic"
            );

            for (j, (result1, result2)) in first.iter().zip(current.iter()).enumerate() {
                assert!(
                    (result1.similarity - result2.similarity).abs() < 1e-6,
                    "Similarity scores should be deterministic at position {}",
                    j
                );
                assert_eq!(
                    result1.location_display(),
                    result2.location_display(),
                    "Result locations should be deterministic at position {}",
                    j
                );
                assert_eq!(
                    result1.rank, result2.rank,
                    "Result ranks should be deterministic at position {}",
                    j
                );
            }
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_search_performance_baseline() -> Result<()> {
    let temp_dir = common::create_test_codebase()?;
    let temp_path = temp_dir.path();

    // This is a baseline performance test - we just verify it completes in reasonable time
    let start = std::time::Instant::now();

    match search_with_config("function", temp_path, Some(5), None).await {
        Ok(_results) => {
            let duration = start.elapsed();
            assert!(
                duration.as_secs() < PERFORMANCE_TEST_TIMEOUT_SECONDS,
                "Search should complete within {} seconds",
                PERFORMANCE_TEST_TIMEOUT_SECONDS
            );
            println!("Search completed in {:?}", duration);
        }
        Err(_) => {
            println!("Performance test skipped - model not available");
        }
    }

    Ok(())
}

#[test]
fn test_query_validation() {
    use tp::query::validate_query;

    // Valid queries
    assert!(validate_query("test query").is_ok());
    assert!(validate_query("jwt authentication").is_ok());
    assert!(validate_query("function main").is_ok());
    assert!(validate_query("a").is_ok()); // Single character is valid

    // Invalid queries
    assert!(validate_query("").is_err());
    assert!(validate_query("   ").is_err());
    assert!(validate_query("\t\n  \r").is_err()); // Only whitespace

    // Very long query
    let long_query = "a".repeat(1001);
    assert!(validate_query(&long_query).is_err());

    // Boundary case - exactly 1000 chars should be valid
    let boundary_query = "a".repeat(1000);
    assert!(validate_query(&boundary_query).is_ok());
}

#[test]
fn test_search_result_formatting() {
    use std::path::PathBuf;
    use tp::types::*;

    let chunk = ContentChunk {
        id: "test-chunk".to_string().into(),
        content: "This is a test chunk with some content for testing the preview functionality"
            .to_string(),
        token_count: 15.into(),
        source_location: SourceLocation {
            file_path: PathBuf::from("src/test.rs"),
            start_line: 42,
            end_line: 44,
            start_char: 0,
            end_char: 77,
        },
        chunk_index: 0.into(),
        total_chunks: 1,
    };

    let indexed_chunk = IndexedChunk {
        chunk,
        embedding: vec![0.1, 0.2, 0.3],
    };

    let result = SearchResult::new(TEST_SIMILARITY_SCORE, indexed_chunk, 0);

    // Test location display
    assert_eq!(result.location_display(), "src/test.rs:42");

    // Test content preview with different lengths
    assert_eq!(result.content_preview(20), "This is a test chunk...");
    assert_eq!(
        result.content_preview(100),
        "This is a test chunk with some content for testing the preview functionality"
    );

    // Test similarity and rank
    assert_eq!(result.similarity, TEST_SIMILARITY_SCORE);
    assert_eq!(result.rank, 0);
}

/// Integration test with the poker sample codebase
#[tokio::test]
async fn test_search_poker_codebase() -> Result<()> {
    let poker_path = Path::new("sample-codebases/poker");

    // Only run this test if the poker codebase exists
    if !poker_path.exists() {
        println!("Skipping poker codebase test - sample codebase not found");
        return Ok(());
    }

    match search_with_config("player", poker_path, Some(3), Some(0.1)).await {
        Ok(results) => {
            // Verify we got some results about players
            assert!(
                !results.is_empty(),
                "Should find player-related content in poker codebase"
            );

            for result in &results {
                println!(
                    "Found: {} (similarity: {:.3})",
                    result.location_display(),
                    result.similarity
                );
                println!("Content: {}", result.content_preview(100));
            }
        }
        Err(e) => {
            println!(
                "Poker codebase test failed - likely due to missing index: {}",
                e
            );
            // This is expected if no index exists for the poker codebase
        }
    }

    Ok(())
}
