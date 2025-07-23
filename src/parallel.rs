//! Parallel processing utilities for high-performance file operations.
//!
//! This module provides optimized parallel processing capabilities for large-scale
//! indexing operations, utilizing all available CPU cores for maximum performance.

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::chunking::ChunkingStrategy;
use crate::config::TurboPropConfig;
use crate::embeddings::EmbeddingGenerator;
use crate::error_utils::ProcessingErrorContext;
use crate::files::FileDiscovery;
use crate::types::{ChunkingConfig, ContentChunk, FileMetadata, IndexedChunk};

/// Configuration for parallel processing operations.
///
/// This structure controls how files are processed in parallel, including
/// concurrency limits, batch sizes, and work distribution strategies.
/// Proper configuration is crucial for optimal performance on different
/// hardware configurations.
///
/// # Examples
///
/// ```
/// use turboprop::parallel::ParallelConfig;
///
/// // Create configuration optimized for high-memory systems
/// let config = ParallelConfig {
///     max_concurrent_files: 16,
///     embedding_batch_size: 128,
///     chunk_buffer_size: 512,
///     enable_work_stealing: true,
/// };
/// 
/// // Or use the system-optimized configuration
/// let optimized = turboprop::parallel::config::optimize_for_system();
/// ```
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Maximum number of files to process concurrently
    pub max_concurrent_files: usize,
    /// Batch size for embedding generation across files
    pub embedding_batch_size: usize,
    /// Number of chunks to collect before sending to embedding generation
    pub chunk_buffer_size: usize,
    /// Whether to enable work-stealing for load balancing
    pub enable_work_stealing: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        let num_cpus = num_cpus::get();
        Self {
            max_concurrent_files: num_cpus * 2,
            embedding_batch_size: 64,
            chunk_buffer_size: 256,
            enable_work_stealing: true,
        }
    }
}

/// Statistics from parallel processing operations
#[derive(Debug, Clone, Default)]
pub struct ParallelStats {
    pub files_processed: usize,
    pub chunks_generated: usize,
    pub embeddings_created: usize,
    pub failed_files: usize,
    pub total_processing_time_ms: u64,
    pub avg_file_processing_time_ms: f64,
}

/// Result of a parallel file processing operation
#[derive(Debug)]
pub struct ParallelProcessingResult {
    pub indexed_chunks: Vec<IndexedChunk>,
    pub stats: ParallelStats,
    pub errors: Vec<(std::path::PathBuf, anyhow::Error)>,
}

/// High-performance parallel file processor
pub struct ParallelFileProcessor {
    config: ParallelConfig,
    chunking_strategy: Arc<ChunkingStrategy>,
}

impl ParallelFileProcessor {
    /// Create a new parallel file processor
    pub fn new(config: ParallelConfig, chunking_config: ChunkingConfig) -> Self {
        Self {
            config,
            chunking_strategy: Arc::new(ChunkingStrategy::new(chunking_config)),
        }
    }

    /// Discover files in parallel using work-stealing threads
    pub fn discover_files_parallel(
        &self,
        path: &Path,
        turboprop_config: &TurboPropConfig,
    ) -> Result<Vec<FileMetadata>> {
        info!("Starting parallel file discovery for: {}", path.display());

        let discovery = FileDiscovery::new(turboprop_config.file_discovery.clone());
        let files = discovery.discover_files(path)?;

        info!("Discovered {} files for parallel processing", files.len());
        Ok(files)
    }

    /// Process files in parallel with optimized chunking and batched embedding generation
    pub async fn process_files_parallel(
        &self,
        files: &[FileMetadata],
        embedding_generator: &mut EmbeddingGenerator,
    ) -> Result<ParallelProcessingResult> {
        let start_time = std::time::Instant::now();

        info!("Processing {} files in parallel", files.len());

        // Process files in parallel chunks to generate content chunks
        let chunk_results: Vec<Result<Vec<ContentChunk>, _>> = files
            .par_chunks(self.config.max_concurrent_files)
            .flat_map(|file_batch| {
                file_batch
                    .par_iter()
                    .map(|file| self.process_single_file_chunks(file))
            })
            .collect();

        // Collect all successful chunks and track errors
        let mut all_chunks = Vec::new();
        let mut errors = Vec::new();
        let mut files_processed = 0;

        for (idx, result) in chunk_results.into_iter().enumerate() {
            match result {
                Ok(chunks) => {
                    all_chunks.extend(chunks);
                    files_processed += 1;
                }
                Err(e) => {
                    let file_path = files
                        .get(idx)
                        .map(|f| f.path.clone())
                        .unwrap_or_else(|| std::path::PathBuf::from("unknown"));
                    warn!("Failed to process file {}: {}", file_path.display(), e);
                    errors.push((file_path, e));
                }
            }
        }

        info!(
            "Generated {} chunks from {} files",
            all_chunks.len(),
            files_processed
        );

        // Generate embeddings in optimized batches
        let indexed_chunks = self
            .generate_embeddings_batch(&all_chunks, embedding_generator)
            .await?;

        let processing_time = start_time.elapsed();
        let stats = ParallelStats {
            files_processed,
            chunks_generated: all_chunks.len(),
            embeddings_created: indexed_chunks.len(),
            failed_files: errors.len(),
            total_processing_time_ms: processing_time.as_millis() as u64,
            avg_file_processing_time_ms: if files_processed > 0 {
                processing_time.as_millis() as f64 / files_processed as f64
            } else {
                0.0
            },
        };

        info!(
            "Parallel processing completed: {} indexed chunks, {} failures, {:.2}ms avg per file",
            indexed_chunks.len(),
            errors.len(),
            stats.avg_file_processing_time_ms
        );

        Ok(ParallelProcessingResult {
            indexed_chunks,
            stats,
            errors,
        })
    }

    /// Process a single file to generate content chunks
    fn process_single_file_chunks(&self, file: &FileMetadata) -> Result<Vec<ContentChunk>> {
        debug!("Processing chunks for file: {}", file.path.display());

        let chunks = self
            .chunking_strategy
            .chunk_file(&file.path)
            .with_chunking_context(&file.path)?;

        debug!(
            "Generated {} chunks for {}",
            chunks.len(),
            file.path.display()
        );
        Ok(chunks)
    }

    /// Generate embeddings for chunks in optimized batches
    async fn generate_embeddings_batch(
        &self,
        chunks: &[ContentChunk],
        embedding_generator: &mut EmbeddingGenerator,
    ) -> Result<Vec<IndexedChunk>> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        info!(
            "Generating embeddings for {} chunks in batches of {}",
            chunks.len(),
            self.config.embedding_batch_size
        );

        let mut indexed_chunks = Vec::with_capacity(chunks.len());

        // Process chunks in batches for memory efficiency
        for chunk_batch in chunks.chunks(self.config.embedding_batch_size) {
            let chunk_texts: Vec<String> = chunk_batch
                .iter()
                .map(|chunk| chunk.content.clone())
                .collect();

            // Generate embeddings for this batch
            let embeddings = embedding_generator
                .embed_batch(&chunk_texts)
                .context("Failed to generate embeddings for batch")?;

            // Create indexed chunks by pairing chunks with their embeddings
            for (chunk, embedding) in chunk_batch.iter().zip(embeddings.into_iter()) {
                indexed_chunks.push(IndexedChunk {
                    chunk: chunk.clone(),
                    embedding,
                });
            }

            debug!(
                "Generated embeddings for batch of {} chunks",
                chunk_batch.len()
            );
        }

        info!(
            "Completed embedding generation: {} indexed chunks",
            indexed_chunks.len()
        );
        Ok(indexed_chunks)
    }
}

/// Parallel search operations for high-performance similarity search
pub struct ParallelSearchProcessor {
    config: ParallelConfig,
}

impl ParallelSearchProcessor {
    pub fn new(config: ParallelConfig) -> Self {
        Self { config }
    }

    /// Perform parallel similarity search across index segments
    pub fn search_parallel(
        &self,
        query_embedding: &[f32],
        indexed_chunks: &[IndexedChunk],
        limit: usize,
        threshold: Option<f32>,
    ) -> Vec<crate::types::SearchResult> {
        use crate::types::cosine_similarity;
        use rayon::ThreadPoolBuilder;

        info!(
            "Performing parallel search across {} chunks",
            indexed_chunks.len()
        );

        // Create a thread pool with configuration from self.config
        let pool = ThreadPoolBuilder::new()
            .num_threads(self.config.max_concurrent_files)
            .thread_name(|index| format!("search-worker-{}", index))
            .build()
            .context("Failed to create thread pool for parallel search")
            .unwrap_or_else(|e| {
                warn!("Using default thread pool due to error: {}", e);
                rayon::ThreadPoolBuilder::new()
                    .build()
                    .expect("Default thread pool failed")
            });

        // Process in chunks if we have many items to better utilize the config
        let chunk_size = self.config.embedding_batch_size.max(1);
        let mut all_results = Vec::new();

        for chunk_batch in indexed_chunks.chunks(chunk_size) {
            let batch_results: Vec<(f32, &IndexedChunk)> = pool.install(|| {
                chunk_batch
                    .par_iter()
                    .map(|indexed_chunk| {
                        let similarity =
                            cosine_similarity(query_embedding, &indexed_chunk.embedding);
                        (similarity, indexed_chunk)
                    })
                    .filter(|(similarity, _)| threshold.is_none_or(|t| *similarity >= t))
                    .collect()
            });

            all_results.extend(batch_results);
        }

        // Sort by similarity (descending) and take top results
        all_results.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(limit);

        // Convert to SearchResult format
        all_results
            .into_iter()
            .enumerate()
            .map(|(rank, (similarity, indexed_chunk))| {
                crate::types::SearchResult::new(similarity, indexed_chunk.clone(), rank)
            })
            .collect()
    }
}

/// Utility functions for configuring parallel processing
pub mod config {
    use super::*;

    /// Optimize parallel configuration based on system resources
    pub fn optimize_for_system() -> ParallelConfig {
        let num_cpus = num_cpus::get();
        let total_memory_mb = sys_info::mem_info()
            .map(|info| info.total / 1024) // Convert KB to MB
            .unwrap_or(8192); // Default to 8GB if detection fails

        let max_concurrent_files = if total_memory_mb >= 16384 {
            // 16GB+ RAM: Use more concurrent files
            num_cpus * 4
        } else if total_memory_mb >= 8192 {
            // 8GB+ RAM: Moderate concurrency
            num_cpus * 2
        } else {
            // Less RAM: Conservative approach
            num_cpus
        };

        let embedding_batch_size = if total_memory_mb >= 16384 {
            128 // Larger batches for more RAM
        } else if total_memory_mb >= 8192 {
            64 // Medium batches
        } else {
            32 // Smaller batches for limited RAM
        };

        ParallelConfig {
            max_concurrent_files,
            embedding_batch_size,
            chunk_buffer_size: embedding_batch_size * 4,
            enable_work_stealing: true,
        }
    }

    /// Create configuration optimized for indexing large codebases
    pub fn for_large_codebases() -> ParallelConfig {
        let base_config = optimize_for_system();
        ParallelConfig {
            embedding_batch_size: base_config.embedding_batch_size * 2,
            chunk_buffer_size: base_config.chunk_buffer_size * 2,
            ..base_config
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TurboPropConfig;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_file(dir: &Path, name: &str, content: &str) -> std::path::PathBuf {
        let file_path = dir.join(name);
        fs::write(&file_path, content).unwrap();
        file_path
    }

    #[test]
    fn test_parallel_config_optimization() {
        let config = config::optimize_for_system();
        assert!(config.max_concurrent_files > 0);
        assert!(config.embedding_batch_size > 0);
        assert!(config.chunk_buffer_size >= config.embedding_batch_size);
    }

    #[test]
    fn test_parallel_file_discovery() {
        let temp_dir = TempDir::new().unwrap();
        create_test_file(temp_dir.path(), "test1.rs", "fn main() {}");
        create_test_file(temp_dir.path(), "test2.rs", "fn hello() {}");

        let config = ParallelConfig::default();
        let processor = ParallelFileProcessor::new(config, Default::default());
        let turboprop_config = TurboPropConfig::default();

        let files = processor
            .discover_files_parallel(temp_dir.path(), &turboprop_config)
            .unwrap();
        assert!(files.len() >= 2);
    }

    #[test]
    fn test_parallel_stats() {
        let stats = ParallelStats {
            files_processed: 10,
            chunks_generated: 100,
            embeddings_created: 100,
            failed_files: 1,
            total_processing_time_ms: 1000,
            avg_file_processing_time_ms: 100.0,
        };

        assert_eq!(stats.files_processed, 10);
        assert_eq!(stats.avg_file_processing_time_ms, 100.0);
    }
}
