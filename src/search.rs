//! Main similarity search engine implementation.
//!
//! This module provides the primary search functionality, coordinating query processing,
//! similarity calculations, and result filtering.

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::path::Path;
use tracing::{debug, info};

use crate::config::TurboPropConfig;
use crate::index::PersistentChunkIndex;
use crate::query::QueryProcessor;
use crate::types::{cosine_similarity, IndexedChunk, SearchResult};

/// Default number of search results to return
pub const DEFAULT_SEARCH_LIMIT: usize = 10;

/// Configuration for search operations
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Maximum number of results to return
    pub limit: usize,
    /// Minimum similarity threshold (0.0 to 1.0)
    pub threshold: Option<f32>,
    /// Enable parallel processing for similarity calculations
    pub parallel: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            limit: DEFAULT_SEARCH_LIMIT,
            threshold: None,
            parallel: true,
        }
    }
}

impl SearchConfig {
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold.clamp(0.0, 1.0));
        self
    }

    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }
}

/// Search engine for performing similarity searches against vector indices
pub struct SearchEngine {
    index: PersistentChunkIndex,
    query_processor: QueryProcessor,
    config: SearchConfig,
}

impl SearchEngine {
    /// Create a new search engine from an index path
    pub async fn new<P: AsRef<Path>>(index_path: P, config: SearchConfig) -> Result<Self> {
        let index = PersistentChunkIndex::load(index_path.as_ref())
            .context("Failed to load index for search")?;

        let query_processor = QueryProcessor::from_index_config(&index)
            .await
            .context("Failed to create query processor")?;

        info!(
            "Search engine initialized with {} chunks, embedding dimensions: {}",
            index.len(),
            query_processor.embedding_dimensions()
        );

        Ok(Self {
            index,
            query_processor,
            config,
        })
    }

    /// Create a search engine using explicit configuration
    pub async fn from_config<P: AsRef<Path>>(
        index_path: P,
        search_config: SearchConfig,
        turboprop_config: &TurboPropConfig,
    ) -> Result<Self> {
        let index = PersistentChunkIndex::load(index_path.as_ref())
            .context("Failed to load index for search")?;

        let query_processor = QueryProcessor::from_config(turboprop_config)
            .await
            .context("Failed to create query processor from config")?;

        Ok(Self {
            index,
            query_processor,
            config: search_config,
        })
    }

    /// Perform a similarity search for the given query
    pub fn search(&mut self, query: &str) -> Result<Vec<SearchResult>> {
        // Validate query
        crate::query::validate_query(query).context("Query validation failed")?;

        info!("Performing search for query: '{}'", query);

        // Generate query embedding
        let query_embedding = self
            .query_processor
            .embed_query(query)
            .context("Failed to generate query embedding")?;

        debug!(
            "Generated query embedding with {} dimensions",
            query_embedding.len()
        );

        // Perform similarity search
        let results = if self.config.parallel {
            self.search_parallel(&query_embedding)
        } else {
            self.search_sequential(&query_embedding)
        };

        info!("Search completed, found {} results", results.len());
        Ok(results)
    }

    /// Perform parallel similarity search with optimizations
    fn search_parallel(&self, query_embedding: &[f32]) -> Vec<SearchResult> {
        let chunks = self.index.get_chunks();
        let start_time = std::time::Instant::now();

        // Parallel similarity calculation with chunked processing
        let chunk_size = (chunks.len() / rayon::current_num_threads()).max(100);

        let results: Vec<(f32, &IndexedChunk)> = chunks
            .par_chunks(chunk_size)
            .flat_map_iter(|chunk_batch| {
                // Process batch with SIMD optimizations when possible
                chunk_batch.iter().filter_map(|chunk| {
                    let similarity =
                        self.calculate_similarity_optimized(query_embedding, &chunk.embedding);

                    // Early filtering to reduce memory pressure
                    if let Some(threshold) = self.config.threshold {
                        if similarity < threshold {
                            return None;
                        }
                    }

                    Some((similarity, chunk))
                })
            })
            .collect();

        let search_time = start_time.elapsed();
        debug!(
            "Parallel search completed in {:.2}ms with {} results",
            search_time.as_secs_f64() * 1000.0,
            results.len()
        );

        self.process_results_optimized(results)
    }

    /// Perform sequential similarity search
    fn search_sequential(&self, query_embedding: &[f32]) -> Vec<SearchResult> {
        let chunks = self.index.get_chunks();

        // Sequential similarity calculation with early termination
        let results: Vec<(f32, &IndexedChunk)> = chunks
            .iter()
            .filter_map(|chunk| {
                let similarity =
                    self.calculate_similarity_optimized(query_embedding, &chunk.embedding);

                // Early filtering
                if let Some(threshold) = self.config.threshold {
                    if similarity < threshold {
                        return None;
                    }
                }

                Some((similarity, chunk))
            })
            .collect();

        self.process_results_optimized(results)
    }

    /// Optimized similarity calculation with potential SIMD operations
    fn calculate_similarity_optimized(&self, query: &[f32], embedding: &[f32]) -> f32 {
        // Use the existing cosine_similarity function, but this could be enhanced
        // with SIMD operations for better performance on large vectors
        cosine_similarity(query, embedding)
    }

    /// Optimized result processing with better memory management
    fn process_results_optimized(
        &self,
        mut results: Vec<(f32, &IndexedChunk)>,
    ) -> Vec<SearchResult> {
        if results.is_empty() {
            return Vec::new();
        }

        // Use partial sort for better performance when we only need top k results
        if results.len() > self.config.limit * 2 {
            // For large result sets, use select_nth for O(n) performance
            results.select_nth_unstable_by(self.config.limit, |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(self.config.limit);

            // Sort only the top results
            results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            // For smaller result sets, use full sort
            results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(self.config.limit);
        }

        // Convert to SearchResult with minimal allocations
        results
            .into_iter()
            .enumerate()
            .map(|(rank, (similarity, chunk))| SearchResult::new(similarity, chunk.clone(), rank))
            .collect()
    }

    /// Get the number of chunks in the index
    pub fn index_size(&self) -> usize {
        self.index.len()
    }

    /// Get the embedding dimensions
    pub fn embedding_dimensions(&self) -> usize {
        self.query_processor.embedding_dimensions()
    }
}

/// Convenience function to perform a simple search
pub async fn search_index<P: AsRef<Path>>(
    index_path: P,
    query: &str,
    limit: Option<usize>,
    threshold: Option<f32>,
) -> Result<Vec<SearchResult>> {
    let mut config = SearchConfig::default();

    if let Some(limit) = limit {
        config = config.with_limit(limit);
    }

    if let Some(threshold) = threshold {
        config = config.with_threshold(threshold);
    }

    let mut engine = SearchEngine::new(index_path, config).await?;
    engine.search(query)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Configurable floating-point comparison tolerance for tests
    const FLOAT_COMPARISON_TOLERANCE: f32 = 1e-6;

    #[test]
    fn test_search_config() {
        let config = SearchConfig::default()
            .with_limit(20)
            .with_threshold(0.5)
            .with_parallel(false);

        assert_eq!(config.limit, 20);
        assert_eq!(config.threshold, Some(0.5));
        assert!(!config.parallel);
    }

    #[test]
    fn test_search_config_threshold_clamping() {
        let config = SearchConfig::default()
            .with_threshold(-0.5) // Should be clamped to 0.0
            .with_threshold(1.5); // Should be clamped to 1.0

        assert_eq!(config.threshold, Some(1.0));
    }

    #[test]
    fn test_cosine_similarity() {
        // Test identical vectors
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v1, &v2) - 1.0).abs() < FLOAT_COMPARISON_TOLERANCE);

        // Test orthogonal vectors
        let v1 = vec![1.0, 0.0];
        let v2 = vec![0.0, 1.0];
        assert!((cosine_similarity(&v1, &v2) - 0.0).abs() < FLOAT_COMPARISON_TOLERANCE);

        // Test opposite vectors
        let v1 = vec![1.0, 0.0];
        let v2 = vec![-1.0, 0.0];
        assert!((cosine_similarity(&v1, &v2) - (-1.0)).abs() < FLOAT_COMPARISON_TOLERANCE);

        // Test different magnitudes (should be normalized)
        let v1 = vec![2.0, 0.0];
        let v2 = vec![3.0, 0.0];
        assert!((cosine_similarity(&v1, &v2) - 1.0).abs() < FLOAT_COMPARISON_TOLERANCE);
    }

    #[test]
    fn test_cosine_similarity_edge_cases() {
        // Empty vectors
        assert_eq!(cosine_similarity(&[], &[]), 0.0);

        // Different lengths
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);

        // Zero vectors
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert_eq!(cosine_similarity(&[1.0, 1.0], &[0.0, 0.0]), 0.0);
    }

    // Integration tests would require actual index files, so we'll create unit tests
    // that test the core functionality with mock data

    #[test]
    fn test_process_results_threshold() {
        // This test would be used in an integration test with actual SearchEngine
        // For now, we test the similarity calculation function
        let high_sim = cosine_similarity(&[1.0, 0.0], &[0.9, 0.1]);
        let low_sim = cosine_similarity(&[1.0, 0.0], &[0.1, 0.9]);

        assert!(high_sim > 0.8);
        assert!(low_sim < 0.2);
    }
}
