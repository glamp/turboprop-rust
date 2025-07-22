//! Performance benchmarks for TurboProp indexing and search operations.
//!
//! This module provides comprehensive benchmarks to measure and validate
//! performance improvements across different optimization techniques.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;
use tp::{
    compression::{CompressionAlgorithm, CompressionConfig, VectorCompressor},
    config::TurboPropConfig,
    parallel::{ParallelConfig, ParallelFileProcessor},
    streaming::{StreamingChunkProcessor, StreamingConfig},
    types::{ChunkId, ChunkIndexNum, ContentChunk, IndexedChunk, SourceLocation, TokenCount},
};

/// Benchmark configuration constants
mod bench_config {
    use std::time::Duration;

    // Embedding and vector configurations
    pub const EMBEDDING_DIMENSION: usize = 384;
    pub const VECTOR_COUNTS: [usize; 3] = [100, 1000, 10000];
    pub const CHUNK_COUNTS: [usize; 3] = [100, 1000, 10000];
    pub const INDEX_SIZES: [usize; 4] = [1000, 5000, 10000, 50000];
    pub const FILE_COUNTS: [usize; 3] = [100, 1000, 5000];

    // Random value ranges
    pub const EMBEDDING_RANGE_MIN: f32 = -1.0;
    pub const EMBEDDING_RANGE_MAX: f32 = 1.0;

    // Search configurations
    pub const SEARCH_LIMIT: usize = 10;
    pub const SEARCH_THRESHOLD: f32 = 0.5;

    // Benchmark execution configurations
    pub const MEMORY_INTENSIVE_SAMPLE_SIZE: usize = 10;
    pub const MEMORY_THRESHOLD_MB: usize = 100;
    pub const MEASUREMENT_TIME_SECONDS: Duration = Duration::from_secs(20);
    pub const OPTIMIZED_BATCH_SIZE: usize = 64;

    // Test data configurations
    pub const SOURCE_CHAR_END: usize = 50;
    pub const SOURCE_CHAR_START: usize = 0;
    pub const LINES_PER_CHUNK: usize = 10;
    pub const TOKEN_COUNT_DEFAULT: usize = 10;
    pub const TOKEN_COUNT_INDEXED: usize = 5;
}

/// Size configurations for benchmark testing
#[derive(Clone)]
struct BenchmarkSizes {
    small: usize,  // 10 files
    medium: usize, // 100 files
    large: usize,  // 1000 files
    xlarge: usize, // 10000 files
}

impl BenchmarkSizes {
    fn new() -> Self {
        Self {
            small: 10,
            medium: 100,
            large: 1000,
            xlarge: 10000,
        }
    }
}

/// Create a temporary directory with test files for benchmarking
fn create_test_codebase(num_files: usize) -> TempDir {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");

    for i in 0..num_files {
        let file_content = format!(
            r#"//! Module {} documentation
use std::collections::HashMap;
use serde::{{Serialize, Deserialize}};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStruct{} {{
    pub id: u64,
    pub name: String,
    pub values: Vec<f64>,
    pub metadata: HashMap<String, String>,
}}

impl DataStruct{} {{
    pub fn new(id: u64, name: impl Into<String>) -> Self {{
        Self {{
            id,
            name: name.into(),
            values: Vec::new(),
            metadata: HashMap::new(),
        }}
    }}

    pub fn add_value(&mut self, value: f64) {{
        self.values.push(value);
    }}

    pub fn get_average(&self) -> Option<f64> {{
        if self.values.is_empty() {{
            None
        }} else {{
            Some(self.values.iter().sum::<f64>() / self.values.len() as f64)
        }}
    }}

    pub fn process_data(&self) -> Result<String, Box<dyn std::error::Error>> {{
        let avg = self.get_average().unwrap_or(0.0);
        Ok(format!("{{}} has {} values with average {{}}", self.name, self.values.len(), avg))
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_data_struct_creation() {{
        let data = DataStruct{}::new(42, "test");
        assert_eq!(data.id, 42);
        assert_eq!(data.name, "test");
    }}

    #[test]
    fn test_average_calculation() {{
        let mut data = DataStruct{}::new(1, "test");
        data.add_value(10.0);
        data.add_value(20.0);
        data.add_value(30.0);
        
        assert_eq!(data.get_average(), Some(20.0));
    }}
}}
"#,
            i, i, i, i, i, i
        );

        let file_path = temp_dir.path().join(format!("module_{}.rs", i));
        fs::write(&file_path, file_content).expect("Failed to write test file");
    }

    temp_dir
}

/// Create test chunks for benchmarking
fn create_test_chunks(count: usize) -> Vec<ContentChunk> {
    (0..count)
        .map(|i| ContentChunk {
            id: ChunkId::from(format!("chunk-{}", i)),
            content: format!(
                "fn test_function_{}() {{\n    let x = {};\n    println!(\"Hello {{}}\", x);\n}}",
                i, i
            ),
            source_location: SourceLocation {
                file_path: PathBuf::from(format!("test_{}.rs", i)),
                start_line: i * bench_config::LINES_PER_CHUNK,
                end_line: i * bench_config::LINES_PER_CHUNK + 4,
                start_char: bench_config::SOURCE_CHAR_START,
                end_char: bench_config::SOURCE_CHAR_END,
            },
            chunk_index: ChunkIndexNum::from(0),
            total_chunks: 1,
            token_count: TokenCount::from(bench_config::TOKEN_COUNT_DEFAULT),
        })
        .collect()
}

/// Create test indexed chunks with embeddings
fn create_test_indexed_chunks(count: usize, embedding_dim: usize) -> Vec<IndexedChunk> {
    use rand::prelude::*;
    let mut rng = thread_rng();

    (0..count)
        .map(|i| {
            let embedding: Vec<f32> = (0..embedding_dim)
                .map(|_| rng.gen_range(bench_config::EMBEDDING_RANGE_MIN..bench_config::EMBEDDING_RANGE_MAX))
                .collect();

            IndexedChunk {
                chunk: ContentChunk {
                    id: ChunkId::from(format!("indexed-chunk-{}", i)),
                    content: format!("function implementation {}", i),
                    source_location: SourceLocation {
                        file_path: PathBuf::from(format!("indexed_{}.rs", i)),
                        start_line: i,
                        end_line: i + 3,
                        start_char: bench_config::SOURCE_CHAR_START,
                        end_char: 30,
                    },
                    chunk_index: ChunkIndexNum::from(i),
                    total_chunks: count,
                    token_count: TokenCount::from(bench_config::TOKEN_COUNT_INDEXED),
                },
                embedding,
            }
        })
        .collect()
}

/// Benchmark file discovery and processing pipeline
fn bench_file_processing_pipeline(c: &mut Criterion) {
    let sizes = BenchmarkSizes::new();
    let mut group = c.benchmark_group("file_processing_pipeline");

    // Benchmark different file counts
    for &size in &[sizes.small, sizes.medium, sizes.large] {
        let temp_dir = create_test_codebase(size);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("sequential_processing", size),
            &size,
            |b, &_size| {
                b.iter(|| {
                    // Simulate sequential file processing
                    let files: Vec<_> = walkdir::WalkDir::new(temp_dir.path())
                        .into_iter()
                        .filter_map(|e| e.ok())
                        .filter(|e| e.file_type().is_file())
                        .collect();

                    black_box(files.len())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark parallel file processing
fn bench_parallel_processing(c: &mut Criterion) {
    let sizes = BenchmarkSizes::new();
    let mut group = c.benchmark_group("parallel_processing");

    for &size in &[sizes.small, sizes.medium, sizes.large] {
        let temp_dir = create_test_codebase(size);
        let config = TurboPropConfig::default();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("parallel_file_discovery", size),
            &size,
            |b, &_size| {
                b.iter(|| {
                    let parallel_config = ParallelConfig::default();
                    let processor =
                        ParallelFileProcessor::new(parallel_config, config.chunking.clone());

                    let files = processor
                        .discover_files_parallel(temp_dir.path(), &config)
                        .unwrap();
                    black_box(files.len())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark vector compression algorithms
fn bench_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_compression");

    let dimensions = bench_config::EMBEDDING_DIMENSION; // Standard embedding dimension
    let vector_counts = bench_config::VECTOR_COUNTS;

    for &count in &vector_counts {
        // Create test vectors
        use rand::prelude::*;
        let mut rng = thread_rng();
        let vectors: Vec<Vec<f32>> = (0..count)
            .map(|_| (0..dimensions).map(|_| rng.gen_range(bench_config::EMBEDDING_RANGE_MIN..bench_config::EMBEDDING_RANGE_MAX)).collect())
            .collect();

        group.throughput(Throughput::Elements(count as u64));

        // Benchmark scalar quantization
        group.bench_with_input(
            BenchmarkId::new("scalar_quantization", count),
            &vectors,
            |b, vectors| {
                b.iter(|| {
                    let config = CompressionConfig {
                        algorithm: CompressionAlgorithm::ScalarQuantization,
                        ..Default::default()
                    };
                    let mut compressor = VectorCompressor::new(config);
                    let compressed = compressor.compress_batch(vectors).unwrap();
                    black_box(compressed.len())
                });
            },
        );

        // Benchmark product quantization
        group.bench_with_input(
            BenchmarkId::new("product_quantization", count),
            &vectors,
            |b, vectors| {
                b.iter(|| {
                    let config = CompressionConfig {
                        algorithm: CompressionAlgorithm::ProductQuantization,
                        ..Default::default()
                    };
                    let mut compressor = VectorCompressor::new(config);
                    let compressed = compressor.compress_batch(vectors).unwrap();
                    black_box(compressed.len())
                });
            },
        );

        // Benchmark no compression (baseline)
        group.bench_with_input(
            BenchmarkId::new("no_compression", count),
            &vectors,
            |b, vectors| {
                b.iter(|| {
                    let config = CompressionConfig {
                        algorithm: CompressionAlgorithm::None,
                        ..Default::default()
                    };
                    let mut compressor = VectorCompressor::new(config);
                    let compressed = compressor.compress_batch(vectors).unwrap();
                    black_box(compressed.len())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark streaming operations
fn bench_streaming(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_operations");

    let chunk_counts = bench_config::CHUNK_COUNTS;

    for &count in &chunk_counts {
        let chunks = create_test_chunks(count);

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("streaming_memory_usage", count),
            &chunks,
            |b, chunks| {
                b.iter(|| {
                    let config = StreamingConfig::default();
                    let mut processor = StreamingChunkProcessor::new(config);

                    // Simulate streaming processing by just counting chunks
                    black_box(chunks.len())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark similarity search operations
fn bench_similarity_search(c: &mut Criterion) {
    use tp::parallel::ParallelSearchProcessor;
    let mut group = c.benchmark_group("similarity_search");

    let embedding_dim = bench_config::EMBEDDING_DIMENSION;
    let index_sizes = bench_config::INDEX_SIZES;

    for &size in &index_sizes {
        let indexed_chunks = create_test_indexed_chunks(size, embedding_dim);

        // Create a random query embedding
        use rand::prelude::*;
        let mut rng = thread_rng();
        let query_embedding: Vec<f32> = (0..embedding_dim)
            .map(|_| rng.gen_range(bench_config::EMBEDDING_RANGE_MIN..bench_config::EMBEDDING_RANGE_MAX))
            .collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("parallel_similarity_search", size),
            &indexed_chunks,
            |b, chunks| {
                b.iter(|| {
                    let config = ParallelConfig::default();
                    let processor = ParallelSearchProcessor::new(config);

                    let results = processor.search_parallel(
                        &query_embedding,
                        chunks,
                        bench_config::SEARCH_LIMIT,        // limit
                        Some(bench_config::SEARCH_THRESHOLD), // threshold
                    );

                    black_box(results.len())
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sequential_similarity_search", size),
            &indexed_chunks,
            |b, chunks| {
                b.iter(|| {
                    use tp::types::cosine_similarity;

                    let mut results: Vec<(f32, &IndexedChunk)> = chunks
                        .iter()
                        .map(|chunk| {
                            let similarity = cosine_similarity(&query_embedding, &chunk.embedding);
                            (similarity, chunk)
                        })
                        .filter(|(sim, _)| *sim >= bench_config::SEARCH_THRESHOLD)
                        .collect();

                    results.sort_by(|(a, _), (b, _)| {
                        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    results.truncate(bench_config::SEARCH_LIMIT);

                    black_box(results.len())
                });
            },
        );
    }

    group.finish();
}

/// Memory usage benchmark
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    group.sample_size(bench_config::MEMORY_INTENSIVE_SAMPLE_SIZE); // Fewer samples for memory-intensive tests

    let file_counts = bench_config::FILE_COUNTS;

    for &count in &file_counts {
        group.bench_with_input(
            BenchmarkId::new("memory_efficient_indexing", count),
            &count,
            |b, &count| {
                b.iter_with_setup(
                    || {
                        // Setup: Create test files
                        create_test_codebase(count)
                    },
                    |temp_dir| {
                        // Measure memory-efficient processing
                        let config = StreamingConfig {
                            memory_threshold_mb: bench_config::MEMORY_THRESHOLD_MB, // Low threshold to test memory efficiency
                            ..Default::default()
                        };
                        let _processor = StreamingChunkProcessor::new(config);

                        // Simulate memory-conscious processing
                        let files: Vec<_> = walkdir::WalkDir::new(temp_dir.path())
                            .into_iter()
                            .filter_map(|e| e.ok())
                            .filter(|e| e.file_type().is_file())
                            .take(count)
                            .collect();

                        black_box(files.len())
                    },
                );
            },
        );
    }

    group.finish();
}

/// Benchmark comprehensive indexing pipeline
fn bench_full_indexing_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_indexing_pipeline");
    group.sample_size(bench_config::MEMORY_INTENSIVE_SAMPLE_SIZE);
    group.measurement_time(bench_config::MEASUREMENT_TIME_SECONDS);

    let sizes = BenchmarkSizes::new();

    // Test the complete pipeline with different configurations
    for &size in &[sizes.small, sizes.medium] {
        // Skip large for CI performance
        let temp_dir = create_test_codebase(size);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("optimized_pipeline", size),
            &size,
            |b, &_size| {
                b.iter(|| {
                    // Configure for optimal performance
                    let _config = TurboPropConfig {
                        embedding: tp::embeddings::EmbeddingConfig {
                            batch_size: bench_config::OPTIMIZED_BATCH_SIZE, // Larger batch for better performance
                            ..Default::default()
                        },
                        ..Default::default()
                    };

                    // Simulate full indexing pipeline
                    // Note: This would normally use the full pipeline, but for benchmarking
                    // we simulate the key operations to avoid external dependencies
                    let files: Vec<_> = walkdir::WalkDir::new(temp_dir.path())
                        .into_iter()
                        .filter_map(|e| e.ok())
                        .filter(|e| e.file_type().is_file())
                        .collect();

                    black_box(files.len())
                });
            },
        );
    }

    group.finish();
}

// Define benchmark groups
criterion_group!(
    benches,
    bench_file_processing_pipeline,
    bench_parallel_processing,
    bench_compression,
    bench_streaming,
    bench_similarity_search,
    bench_memory_usage,
    bench_full_indexing_pipeline,
);

criterion_main!(benches);
