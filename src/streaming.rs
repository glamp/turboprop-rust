//! Streaming operations for memory-efficient processing of large datasets.
//!
//! This module provides streaming implementations that minimize memory usage
//! by processing data in chunks and avoiding loading entire datasets into memory.

use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::io::{BufReader, BufWriter, Read, Write};
use tracing::{debug, info, warn};

use crate::compression::{CompressedVector, VectorCompressor};
use crate::error::TurboPropError;
use crate::types::{ContentChunk, IndexedChunk};

/// Configuration for streaming operations
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Buffer size for I/O operations (in bytes)
    pub buffer_size: usize,
    /// Maximum number of chunks to buffer in memory
    pub max_chunk_buffer: usize,
    /// Batch size for processing operations
    pub batch_size: usize,
    /// Enable compression for streaming data
    pub enable_compression: bool,
    /// Memory threshold to trigger streaming mode (in MB)
    pub memory_threshold_mb: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1024 * 1024, // 1MB
            max_chunk_buffer: 1000,
            batch_size: 100,
            enable_compression: true,
            memory_threshold_mb: 512, // 512MB
        }
    }
}

/// Statistics for streaming operations
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    pub bytes_processed: u64,
    pub chunks_streamed: usize,
    pub batches_processed: usize,
    pub memory_peaks_mb: f64,
    pub io_time_ms: u64,
    pub processing_time_ms: u64,
}

/// Streaming chunk processor for memory-efficient indexing
pub struct StreamingChunkProcessor {
    config: StreamingConfig,
    stats: StreamingStats,
    compressor: Option<VectorCompressor>,
}

impl StreamingChunkProcessor {
    /// Create a new streaming chunk processor
    pub fn new(config: StreamingConfig) -> Self {
        let compressor = if config.enable_compression {
            Some(VectorCompressor::new(Default::default()))
        } else {
            None
        };

        Self {
            config,
            stats: StreamingStats::default(),
            compressor,
        }
    }

    /// Stream process a large collection of chunks
    pub async fn process_chunks_streaming<F, Fut>(
        &mut self,
        chunks: impl IntoIterator<Item = ContentChunk>,
        mut processor: F,
    ) -> Result<Vec<IndexedChunk>>
    where
        F: FnMut(Vec<ContentChunk>) -> Fut,
        Fut: std::future::Future<Output = Result<Vec<IndexedChunk>>>,
    {
        let start_time = std::time::Instant::now();
        info!("Starting streaming chunk processing");

        let mut results = Vec::new();
        let mut current_batch = Vec::new();
        let mut batch_count = 0;

        for chunk in chunks {
            current_batch.push(chunk);
            self.stats.chunks_streamed += 1;

            // Process batch when it reaches the configured size
            if current_batch.len() >= self.config.batch_size {
                let batch_results = processor(current_batch)
                    .await
                    .context("Failed to process chunk batch")?;

                results.extend(batch_results);
                current_batch = Vec::new();
                batch_count += 1;
                self.stats.batches_processed = batch_count;

                debug!(
                    "Processed batch {} with {} total results",
                    batch_count,
                    results.len()
                );

                // Check memory usage and apply backpressure if needed
                self.check_memory_pressure().await?;
            }
        }

        // Process remaining chunks
        if !current_batch.is_empty() {
            let batch_results = processor(current_batch)
                .await
                .context("Failed to process final chunk batch")?;
            results.extend(batch_results);
            self.stats.batches_processed = batch_count + 1;
        }

        let elapsed = start_time.elapsed();
        self.stats.processing_time_ms = elapsed.as_millis() as u64;

        info!(
            "Streaming processing completed: {} chunks, {} batches, {:.2}s",
            self.stats.chunks_streamed,
            self.stats.batches_processed,
            elapsed.as_secs_f64()
        );

        Ok(results)
    }

    /// Stream indexed chunks to storage with compression
    pub async fn stream_to_storage<W: Write + Send>(
        &mut self,
        chunks: impl Iterator<Item = IndexedChunk>,
        writer: W,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();
        let mut buffered_writer = BufWriter::with_capacity(self.config.buffer_size, writer);

        info!("Streaming indexed chunks to storage");
        let mut chunk_count = 0;

        // Write header information
        self.write_stream_header(&mut buffered_writer)?;

        for chunk in chunks {
            let serialized = if let Some(compressor) = &mut self.compressor {
                // Compress the embedding vector
                let compressed_vectors = compressor.compress_batch(&[chunk.embedding.clone()])?;
                let compressed_chunk = CompressedIndexedChunk {
                    chunk: chunk.chunk,
                    compressed_embedding: compressed_vectors.into_iter().next().unwrap(),
                };
                bincode::serialize(&compressed_chunk)?
            } else {
                bincode::serialize(&chunk)?
            };

            // Write chunk size followed by chunk data
            let size = serialized.len() as u32;
            buffered_writer.write_all(&size.to_le_bytes())?;
            buffered_writer.write_all(&serialized)?;

            chunk_count += 1;
            self.stats.bytes_processed += serialized.len() as u64;

            if chunk_count % 1000 == 0 {
                debug!("Streamed {} chunks to storage", chunk_count);
                buffered_writer.flush()?;
            }
        }

        buffered_writer.flush()?;

        let elapsed = start_time.elapsed();
        self.stats.io_time_ms = elapsed.as_millis() as u64;

        info!(
            "Completed streaming {} chunks to storage in {:.2}s",
            chunk_count,
            elapsed.as_secs_f64()
        );

        Ok(())
    }

    /// Stream indexed chunks from storage with decompression
    pub async fn stream_from_storage<R: Read + Send>(
        &mut self,
        reader: R,
    ) -> Result<Vec<IndexedChunk>> {
        let start_time = std::time::Instant::now();
        let mut buffered_reader = BufReader::with_capacity(self.config.buffer_size, reader);

        info!("Streaming indexed chunks from storage");

        // Read and validate header
        self.read_stream_header(&mut buffered_reader)?;

        let mut chunks = Vec::new();
        let mut buffer = vec![0u8; 4];

        loop {
            // Read chunk size
            if buffered_reader.read_exact(&mut buffer).is_err() {
                break; // End of stream
            }

            let chunk_size =
                u32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]) as usize;

            // Read chunk data
            let mut chunk_data = vec![0u8; chunk_size];
            buffered_reader
                .read_exact(&mut chunk_data)
                .context("Failed to read chunk data from stream")?;

            // Deserialize chunk
            let indexed_chunk = if self.compressor.is_some() {
                let compressed: CompressedIndexedChunk = bincode::deserialize(&chunk_data)?;
                // Decompress embedding
                let decompressed_embeddings = self
                    .compressor
                    .as_ref()
                    .unwrap()
                    .decompress_batch(&[compressed.compressed_embedding])?;

                IndexedChunk {
                    chunk: compressed.chunk,
                    embedding: decompressed_embeddings.into_iter().next().unwrap(),
                }
            } else {
                bincode::deserialize(&chunk_data)?
            };

            chunks.push(indexed_chunk);
            self.stats.bytes_processed += chunk_data.len() as u64;

            if chunks.len() % 1000 == 0 {
                debug!("Loaded {} chunks from storage", chunks.len());
            }
        }

        let elapsed = start_time.elapsed();
        self.stats.io_time_ms = elapsed.as_millis() as u64;

        info!(
            "Completed loading {} chunks from storage in {:.2}s",
            chunks.len(),
            elapsed.as_secs_f64()
        );

        Ok(chunks)
    }

    /// Check memory pressure and apply backpressure if needed
    async fn check_memory_pressure(&mut self) -> Result<()> {
        // Simple memory monitoring (could use more sophisticated techniques)
        if let Ok(mem_info) = sys_info::mem_info() {
            let used_mb = (mem_info.total - mem_info.avail) / 1024;
            let used_percentage = (used_mb as f64 / mem_info.total as f64) * 100.0;

            self.stats.memory_peaks_mb = self.stats.memory_peaks_mb.max(used_mb as f64);

            if used_percentage > 85.0 {
                warn!(
                    "High memory usage detected: {:.1}%, applying backpressure",
                    used_percentage
                );
                // Small delay to allow memory to be freed
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }

        Ok(())
    }

    /// Write streaming format header
    fn write_stream_header<W: Write>(&self, writer: &mut W) -> Result<()> {
        let header = StreamHeader {
            version: 1,
            compression_enabled: self.compressor.is_some(),
            batch_size: self.config.batch_size,
        };

        let header_bytes = bincode::serialize(&header)?;
        let header_size = header_bytes.len() as u32;

        writer.write_all(b"TPSTREAM")?; // Magic number
        writer.write_all(&header_size.to_le_bytes())?;
        writer.write_all(&header_bytes)?;

        Ok(())
    }

    /// Read and validate streaming format header
    fn read_stream_header<R: Read>(&self, reader: &mut R) -> Result<StreamHeader> {
        let mut magic = vec![0u8; 8];
        reader.read_exact(&mut magic)?;

        if &magic != b"TPSTREAM" {
            return Err(TurboPropError::other("Invalid stream format: missing magic number - file may be corrupted or incompatible").into());
        }

        let mut size_bytes = vec![0u8; 4];
        reader.read_exact(&mut size_bytes)?;
        let header_size =
            u32::from_le_bytes([size_bytes[0], size_bytes[1], size_bytes[2], size_bytes[3]])
                as usize;

        let mut header_bytes = vec![0u8; header_size];
        reader.read_exact(&mut header_bytes)?;

        let header: StreamHeader = bincode::deserialize(&header_bytes)?;

        if header.version != 1 {
            return Err(TurboPropError::other(format!(
                "Unsupported stream version: {} - this file was created with a newer or incompatible version", 
                header.version
            )).into());
        }

        Ok(header)
    }

    /// Get streaming statistics
    pub fn stats(&self) -> &StreamingStats {
        &self.stats
    }
}

/// Header for streaming file format
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct StreamHeader {
    version: u32,
    compression_enabled: bool,
    batch_size: usize,
}

/// Compressed version of IndexedChunk for streaming storage
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct CompressedIndexedChunk {
    chunk: ContentChunk,
    compressed_embedding: CompressedVector,
}

/// Streaming index builder that processes large datasets incrementally
pub struct StreamingIndexBuilder {
    config: StreamingConfig,
    processing_queue: VecDeque<ContentChunk>,
    stats: StreamingStats,
}

impl StreamingIndexBuilder {
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            processing_queue: VecDeque::new(),
            stats: StreamingStats::default(),
        }
    }

    /// Add chunk to the processing queue
    pub fn add_chunk(&mut self, chunk: ContentChunk) -> Result<()> {
        if self.processing_queue.len() >= self.config.max_chunk_buffer {
            warn!("Chunk buffer is full, applying backpressure");
            return Err(TurboPropError::other("Chunk buffer overflow - too many chunks queued for processing").into());
        }

        self.processing_queue.push_back(chunk);
        Ok(())
    }

    /// Process queued chunks in batches
    pub async fn process_batch<F, Fut>(&mut self, mut processor: F) -> Result<Vec<IndexedChunk>>
    where
        F: FnMut(Vec<ContentChunk>) -> Fut,
        Fut: std::future::Future<Output = Result<Vec<IndexedChunk>>>,
    {
        let batch_size = self.config.batch_size.min(self.processing_queue.len());
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            if let Some(chunk) = self.processing_queue.pop_front() {
                batch.push(chunk);
            }
        }

        let results = processor(batch).await?;
        self.stats.chunks_streamed += results.len();
        self.stats.batches_processed += 1;

        Ok(results)
    }

    /// Get number of queued chunks
    pub fn queued_chunks(&self) -> usize {
        self.processing_queue.len()
    }

    /// Check if buffer has space for more chunks
    pub fn has_capacity(&self) -> bool {
        self.processing_queue.len() < self.config.max_chunk_buffer
    }

    /// Get statistics
    pub fn stats(&self) -> &StreamingStats {
        &self.stats
    }
}

/// Optimized streaming configuration for different scenarios
pub mod presets {
    use super::*;

    /// Configuration optimized for large codebase indexing
    pub fn large_codebase() -> StreamingConfig {
        StreamingConfig {
            buffer_size: 4 * 1024 * 1024, // 4MB buffer
            max_chunk_buffer: 2000,
            batch_size: 200,
            enable_compression: true,
            memory_threshold_mb: 1024, // 1GB threshold
        }
    }

    /// Configuration optimized for memory-constrained environments
    pub fn memory_constrained() -> StreamingConfig {
        StreamingConfig {
            buffer_size: 512 * 1024, // 512KB buffer
            max_chunk_buffer: 100,
            batch_size: 25,
            enable_compression: true,
            memory_threshold_mb: 256, // 256MB threshold
        }
    }

    /// Configuration optimized for high-performance processing
    pub fn high_performance() -> StreamingConfig {
        StreamingConfig {
            buffer_size: 8 * 1024 * 1024, // 8MB buffer
            max_chunk_buffer: 5000,
            batch_size: 500,
            enable_compression: false, // Disable compression for speed
            memory_threshold_mb: 2048, // 2GB threshold
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChunkId, SourceLocation, TokenCount};
    use std::io::Cursor;

    fn create_test_chunk(id: usize, content: &str) -> ContentChunk {
        ContentChunk {
            id: ChunkId::from(format!("test-{}", id)),
            content: content.to_string(),
            source_location: SourceLocation {
                file_path: format!("test{}.rs", id).into(),
                start_line: id,
                end_line: id + 1,
                start_char: 0,
                end_char: content.len(),
            },
            chunk_index: crate::types::ChunkIndexNum::from(0),
            total_chunks: 1,
            token_count: TokenCount::from(content.split_whitespace().count()),
        }
    }

    fn create_test_indexed_chunk(id: usize, content: &str) -> IndexedChunk {
        IndexedChunk {
            chunk: create_test_chunk(id, content),
            embedding: vec![0.1 * id as f32; 384], // Simple test embedding
        }
    }

    #[tokio::test]
    async fn test_streaming_chunk_processing() {
        let config = StreamingConfig::default();
        let mut processor = StreamingChunkProcessor::new(config);

        let chunks = vec![
            create_test_chunk(1, "fn main() {}"),
            create_test_chunk(2, "let x = 42;"),
            create_test_chunk(3, "println!(\"Hello\");"),
        ];

        let results = processor
            .process_chunks_streaming(chunks, |batch| async move {
                // Mock processor that creates indexed chunks
                Ok(batch
                    .into_iter()
                    .enumerate()
                    .map(|(_i, chunk)| IndexedChunk {
                        chunk,
                        embedding: vec![0.1; 384],
                    })
                    .collect())
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
        let stats = processor.stats();
        assert_eq!(stats.chunks_streamed, 3);
        assert!(stats.batches_processed > 0);
    }

    #[tokio::test]
    async fn test_stream_to_and_from_storage() {
        let config = StreamingConfig::default();
        let mut processor = StreamingChunkProcessor::new(config);

        let chunks = vec![
            create_test_indexed_chunk(1, "fn test1() {}"),
            create_test_indexed_chunk(2, "let y = 24;"),
        ];

        // Stream to storage
        let mut storage = Vec::new();
        processor
            .stream_to_storage(chunks.clone().into_iter(), &mut storage)
            .await
            .unwrap();

        assert!(!storage.is_empty());

        // Stream from storage
        let mut new_processor = StreamingChunkProcessor::new(StreamingConfig::default());
        let loaded_chunks = new_processor
            .stream_from_storage(Cursor::new(storage))
            .await
            .unwrap();

        assert_eq!(loaded_chunks.len(), 2);
        // Note: Exact equality check would depend on compression settings
    }

    #[test]
    fn test_streaming_index_builder() {
        let config = StreamingConfig::default();
        let mut builder = StreamingIndexBuilder::new(config);

        assert!(builder.has_capacity());
        assert_eq!(builder.queued_chunks(), 0);

        let chunk = create_test_chunk(1, "fn test() {}");
        builder.add_chunk(chunk).unwrap();

        assert_eq!(builder.queued_chunks(), 1);
    }

    #[test]
    fn test_streaming_presets() {
        let large_config = presets::large_codebase();
        assert!(large_config.buffer_size > StreamingConfig::default().buffer_size);

        let constrained_config = presets::memory_constrained();
        assert!(constrained_config.buffer_size < StreamingConfig::default().buffer_size);

        let performance_config = presets::high_performance();
        assert!(!performance_config.enable_compression);
        assert!(performance_config.batch_size > StreamingConfig::default().batch_size);
    }
}
