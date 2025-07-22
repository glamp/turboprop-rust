//! Vector compression algorithms for efficient index storage.
//!
//! This module provides various compression techniques to reduce memory usage
//! and improve I/O performance for large vector indices.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::error::TurboPropError;

/// Configuration for vector compression algorithms.
///
/// This structure defines the parameters for compressing vector embeddings
/// to reduce storage requirements and improve I/O performance.
///
/// # Examples
///
/// ```
/// use tp::compression::{CompressionConfig, CompressionAlgorithm};
///
/// // Create a configuration for scalar quantization
/// let config = CompressionConfig {
///     algorithm: CompressionAlgorithm::ScalarQuantization,
///     quantization_bits: 8,
///     enable_delta_compression: false,
///     clustering_threshold: 0.95,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Type of compression algorithm to use
    pub algorithm: CompressionAlgorithm,
    /// Quantization bits for scalar quantization (1-8)
    pub quantization_bits: u8,
    /// Whether to enable delta compression for similar vectors
    pub enable_delta_compression: bool,
    /// Clustering threshold for similar vectors (0.0 to 1.0)
    pub clustering_threshold: f32,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::ScalarQuantization,
            quantization_bits: 8,
            enable_delta_compression: true,
            clustering_threshold: 0.9,
        }
    }
}

/// Available compression algorithms for vector data.
///
/// Each algorithm provides different trade-offs between compression ratio,
/// decompression speed, and quality loss.
///
/// # Algorithms
///
/// - **None**: No compression applied - preserves original data exactly
/// - **ScalarQuantization**: Reduces precision by quantizing values to fixed levels
/// - **ProductQuantization**: Divides vectors into subspaces for higher compression ratios
///
/// # Examples
///
/// ```
/// use tp::compression::CompressionAlgorithm;
///
/// // For maximum accuracy, use no compression
/// let no_compression = CompressionAlgorithm::None;
///
/// // For balanced compression and speed
/// let scalar = CompressionAlgorithm::ScalarQuantization;
///
/// // For maximum compression with some quality loss
/// let product = CompressionAlgorithm::ProductQuantization;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression (full precision)
    None,
    /// Scalar quantization to reduce precision
    ScalarQuantization,
    /// Product quantization for high compression
    ProductQuantization,
    /// Learned compression using clustering
    LearnedCompression,
}

/// Compressed vector representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedVector {
    /// Compressed data
    pub data: Vec<u8>,
    /// Compression metadata
    pub metadata: CompressionMetadata,
    /// Original vector length
    pub original_length: usize,
}

/// Metadata for decompression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionMetadata {
    /// Algorithm used for compression
    pub algorithm: CompressionAlgorithm,
    /// Quantization parameters
    pub quantization_params: QuantizationParams,
    /// Codebook for product quantization (flattened structure)
    pub codebook: Option<Vec<Vec<Vec<f32>>>>,
}

/// Parameters for quantization algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    /// Minimum value in the original data
    pub min_value: f32,
    /// Maximum value in the original data  
    pub max_value: f32,
    /// Number of quantization levels
    pub num_levels: u32,
    /// Scale factor for quantization
    pub scale: f32,
}

/// Statistics about compression performance
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Original size in bytes
    pub original_size_bytes: usize,
    /// Compressed size in bytes
    pub compressed_size_bytes: usize,
    /// Compression ratio (original / compressed)
    pub compression_ratio: f64,
    /// Number of vectors processed
    pub vectors_processed: usize,
    /// Average compression time per vector in microseconds
    pub avg_compression_time_us: f64,
}

impl CompressionStats {
    pub fn compression_percentage(&self) -> f64 {
        if self.original_size_bytes == 0 {
            0.0
        } else {
            100.0 * (1.0 - (self.compressed_size_bytes as f64 / self.original_size_bytes as f64))
        }
    }
}

/// High-performance vector compressor for embedding data.
///
/// This struct provides methods to compress and decompress vectors using
/// various algorithms to reduce storage space and improve I/O performance.
/// It maintains statistics about compression ratios and processing performance.
///
/// # Examples
///
/// ```no_run
/// use tp::compression::{VectorCompressor, CompressionConfig, CompressionAlgorithm};
///
/// // Create a compressor with scalar quantization
/// let config = CompressionConfig {
///     algorithm: CompressionAlgorithm::ScalarQuantization,
///     quantization_bits: 8,
///     enable_delta_compression: false,
///     clustering_threshold: 0.95,
/// };
/// let mut compressor = VectorCompressor::new(config);
///
/// // Compress a batch of vectors
/// let vectors = vec![
///     vec![1.0, 2.0, 3.0],
///     vec![4.0, 5.0, 6.0],
/// ];
/// let compressed = compressor.compress_batch(&vectors).unwrap();
///
/// // Decompress back to original form
/// let decompressed = compressor.decompress_batch(&compressed).unwrap();
///
/// println!("Compression ratio: {:.2}%", compressor.stats().compression_percentage());
/// ```
pub struct VectorCompressor {
    config: CompressionConfig,
    stats: CompressionStats,
}

impl VectorCompressor {
    /// Create a new vector compressor with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration specifying the compression algorithm and parameters
    ///
    /// # Examples
    ///
    /// ```
    /// use tp::compression::{VectorCompressor, CompressionConfig, CompressionAlgorithm};
    ///
    /// // Create compressor for scalar quantization
    /// let config = CompressionConfig {
    ///     algorithm: CompressionAlgorithm::ScalarQuantization,
    ///     quantization_bits: 8,
    ///     enable_delta_compression: false,
    ///     clustering_threshold: 0.95,
    /// };
    /// let compressor = VectorCompressor::new(config);
    /// ```
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            config,
            stats: CompressionStats::default(),
        }
    }

    /// Compress a batch of vectors using the configured algorithm.
    ///
    /// This method processes multiple vectors simultaneously for better performance.
    /// The compression algorithm and parameters are determined by the configuration
    /// provided during construction.
    ///
    /// # Arguments
    ///
    /// * `vectors` - Slice of vectors to compress (each vector should have the same dimension)
    ///
    /// # Returns
    ///
    /// * `Result<Vec<CompressedVector>>` - Vector of compressed representations
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Vectors have inconsistent dimensions
    /// - Compression algorithm fails due to invalid parameters
    /// - Insufficient memory for processing
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use tp::compression::{VectorCompressor, CompressionConfig, CompressionAlgorithm};
    ///
    /// let config = CompressionConfig::default();
    /// let mut compressor = VectorCompressor::new(config);
    ///
    /// let vectors = vec![
    ///     vec![1.0, 2.0, 3.0, 4.0],
    ///     vec![5.0, 6.0, 7.0, 8.0],
    /// ];
    ///
    /// match compressor.compress_batch(&vectors) {
    ///     Ok(compressed) => {
    ///         println!("Compressed {} vectors", compressed.len());
    ///     },
    ///     Err(e) => eprintln!("Compression failed: {}", e),
    /// }
    /// ```
    pub fn compress_batch(&mut self, vectors: &[Vec<f32>]) -> Result<Vec<CompressedVector>> {
        let start_time = std::time::Instant::now();

        info!(
            "Compressing {} vectors using {:?}",
            vectors.len(),
            self.config.algorithm
        );

        let compressed = match self.config.algorithm {
            CompressionAlgorithm::None => self.compress_none(vectors)?,
            CompressionAlgorithm::ScalarQuantization => {
                self.compress_scalar_quantization(vectors)?
            }
            CompressionAlgorithm::ProductQuantization => {
                self.compress_product_quantization(vectors)?
            }
            CompressionAlgorithm::LearnedCompression => self.compress_learned(vectors)?,
        };

        let elapsed = start_time.elapsed();
        self.update_stats(vectors, &compressed, elapsed);

        info!(
            "Compression completed: {:.1}% size reduction, {:.2}ms total",
            self.stats.compression_percentage(),
            elapsed.as_secs_f64() * 1000.0
        );

        Ok(compressed)
    }

    /// Decompress a batch of vectors
    pub fn decompress_batch(&self, compressed: &[CompressedVector]) -> Result<Vec<Vec<f32>>> {
        info!("Decompressing {} vectors", compressed.len());

        let mut decompressed = Vec::with_capacity(compressed.len());

        for compressed_vector in compressed {
            let vector = self.decompress_single(compressed_vector)?;
            decompressed.push(vector);
        }

        info!("Decompression completed: {} vectors", decompressed.len());
        Ok(decompressed)
    }

    /// No compression - pass through with metadata
    fn compress_none(&mut self, vectors: &[Vec<f32>]) -> Result<Vec<CompressedVector>> {
        vectors
            .iter()
            .map(|vector| {
                let data = vector.iter().flat_map(|&f| f.to_le_bytes()).collect();

                Ok(CompressedVector {
                    data,
                    metadata: CompressionMetadata {
                        algorithm: CompressionAlgorithm::None,
                        quantization_params: QuantizationParams {
                            min_value: 0.0,
                            max_value: 0.0,
                            num_levels: 0,
                            scale: 1.0,
                        },
                        codebook: None,
                    },
                    original_length: vector.len(),
                })
            })
            .collect()
    }

    /// Scalar quantization compression
    fn compress_scalar_quantization(
        &mut self,
        vectors: &[Vec<f32>],
    ) -> Result<Vec<CompressedVector>> {
        // Calculate global min/max for quantization
        let (global_min, global_max) = self.calculate_global_range(vectors);
        let num_levels = (1u32 << self.config.quantization_bits) - 1;
        let scale = (global_max - global_min) / num_levels as f32;

        let quantization_params = QuantizationParams {
            min_value: global_min,
            max_value: global_max,
            num_levels,
            scale,
        };

        vectors
            .iter()
            .map(|vector| {
                let quantized_data = vector
                    .iter()
                    .map(|&value| {
                        let normalized = (value - global_min) / scale;
                        normalized.round().min(num_levels as f32).max(0.0) as u8
                    })
                    .collect();

                Ok(CompressedVector {
                    data: quantized_data,
                    metadata: CompressionMetadata {
                        algorithm: CompressionAlgorithm::ScalarQuantization,
                        quantization_params: quantization_params.clone(),
                        codebook: None,
                    },
                    original_length: vector.len(),
                })
            })
            .collect()
    }

    /// Product quantization compression
    fn compress_product_quantization(
        &mut self,
        vectors: &[Vec<f32>],
    ) -> Result<Vec<CompressedVector>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        let vector_dim = vectors[0].len();
        let subvector_size = 8; // Split into 8-dimensional subvectors
        let num_subvectors = vector_dim.div_ceil(subvector_size);
        let codebook_size = 256; // 256 centroids per subspace

        info!(
            "Building product quantization codebook: {} subvectors of size {}",
            num_subvectors, subvector_size
        );

        // Build codebook for each subspace
        let codebook =
            self.build_product_quantization_codebook(vectors, subvector_size, codebook_size)?;

        // Quantize vectors using the codebook
        let compressed = vectors
            .iter()
            .map(|vector| {
                let mut codes = Vec::new();

                for (i, centroid) in codebook.iter().enumerate() {
                    let start = i * subvector_size;
                    let end = (start + subvector_size).min(vector.len());
                    let subvector = &vector[start..end];

                    // Find closest centroid in this subspace
                    let best_code = self.find_closest_centroid(subvector, centroid);
                    codes.push(best_code);
                }

                CompressedVector {
                    data: codes,
                    metadata: CompressionMetadata {
                        algorithm: CompressionAlgorithm::ProductQuantization,
                        quantization_params: QuantizationParams {
                            min_value: 0.0,
                            max_value: 0.0,
                            num_levels: codebook_size as u32,
                            scale: 1.0,
                        },
                        codebook: Some(codebook.clone()),
                    },
                    original_length: vector.len(),
                }
            })
            .collect::<Vec<_>>();

        Ok(compressed)
    }

    /// Learned compression using clustering
    fn compress_learned(&mut self, vectors: &[Vec<f32>]) -> Result<Vec<CompressedVector>> {
        // For now, fall back to scalar quantization
        // In a full implementation, this would use learned compression techniques
        warn!("Learned compression not fully implemented, using scalar quantization");
        self.compress_scalar_quantization(vectors)
    }

    /// Decompress a single vector
    fn decompress_single(&self, compressed: &CompressedVector) -> Result<Vec<f32>> {
        match compressed.metadata.algorithm {
            CompressionAlgorithm::None => self.decompress_none(compressed),
            CompressionAlgorithm::ScalarQuantization => {
                self.decompress_scalar_quantization(compressed)
            }
            CompressionAlgorithm::ProductQuantization => {
                self.decompress_product_quantization(compressed)
            }
            CompressionAlgorithm::LearnedCompression => self.decompress_learned(compressed),
        }
    }

    fn decompress_none(&self, compressed: &CompressedVector) -> Result<Vec<f32>> {
        let floats = compressed
            .data
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect();
        Ok(floats)
    }

    fn decompress_scalar_quantization(&self, compressed: &CompressedVector) -> Result<Vec<f32>> {
        let params = &compressed.metadata.quantization_params;

        let decompressed = compressed
            .data
            .iter()
            .map(|&quantized| params.min_value + (quantized as f32 * params.scale))
            .collect();

        Ok(decompressed)
    }

    fn decompress_product_quantization(&self, compressed: &CompressedVector) -> Result<Vec<f32>> {
        let codebook = compressed
            .metadata
            .codebook
            .as_ref()
            .context("Product quantization codebook missing")?;

        let mut decompressed = Vec::new();

        for (subvector_idx, &code) in compressed.data.iter().enumerate() {
            if let Some(subspace_codebook) = codebook.get(subvector_idx) {
                if let Some(centroid) = subspace_codebook.get(code as usize) {
                    decompressed.extend_from_slice(centroid);
                } else {
                    return Err(TurboPropError::other(format!(
                        "Invalid code {} for subvector {}: code exceeds codebook size",
                        code,
                        subvector_idx
                    )).into());
                }
            } else {
                return Err(TurboPropError::other(format!(
                    "Missing codebook for subvector {}: compression data is corrupted",
                    subvector_idx
                )).into());
            }
        }

        // Truncate to original length
        decompressed.truncate(compressed.original_length);
        Ok(decompressed)
    }

    fn decompress_learned(&self, compressed: &CompressedVector) -> Result<Vec<f32>> {
        // Fallback to scalar quantization decompression
        self.decompress_scalar_quantization(compressed)
    }

    /// Calculate global min/max for quantization
    fn calculate_global_range(&self, vectors: &[Vec<f32>]) -> (f32, f32) {
        let mut global_min = f32::INFINITY;
        let mut global_max = f32::NEG_INFINITY;

        for vector in vectors {
            for &value in vector {
                global_min = global_min.min(value);
                global_max = global_max.max(value);
            }
        }

        (global_min, global_max)
    }

    /// Build codebook for product quantization
    fn build_product_quantization_codebook(
        &self,
        vectors: &[Vec<f32>],
        subvector_size: usize,
        codebook_size: usize,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let vector_dim = vectors[0].len();
        let num_subvectors = vector_dim.div_ceil(subvector_size);
        let mut codebook = Vec::with_capacity(num_subvectors);

        for i in 0..num_subvectors {
            let start = i * subvector_size;
            let end = (start + subvector_size).min(vector_dim);
            let actual_size = end - start;

            // Extract subvectors for this subspace
            let subvectors: Vec<Vec<f32>> = vectors
                .iter()
                .map(|v| {
                    let mut subvec = v[start..end].to_vec();
                    // Pad with zeros if needed
                    subvec.resize(actual_size, 0.0);
                    subvec
                })
                .collect();

            // Build centroids using k-means (simplified version)
            let centroids =
                self.kmeans_centroids(&subvectors, codebook_size.min(subvectors.len()))?;
            codebook.push(centroids);
        }

        Ok(codebook)
    }

    /// Simple k-means clustering for centroid generation
    fn kmeans_centroids(&self, vectors: &[Vec<f32>], k: usize) -> Result<Vec<Vec<f32>>> {
        if vectors.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let dim = vectors[0].len();
        let mut centroids = Vec::with_capacity(k);

        // Initialize centroids randomly
        for i in 0..k {
            let idx = (i * vectors.len()) / k; // Simple initialization
            centroids.push(vectors[idx].clone());
        }

        // Simple k-means iteration (could be improved)
        for _iteration in 0..10 {
            let mut new_centroids = vec![vec![0.0; dim]; k];
            let mut counts = vec![0; k];

            // Assign points to centroids
            for vector in vectors {
                let closest = self.find_closest_centroid(vector, &centroids) as usize;
                for (j, &val) in vector.iter().enumerate() {
                    new_centroids[closest][j] += val;
                }
                counts[closest] += 1;
            }

            // Update centroids
            for (i, centroid) in new_centroids.iter_mut().enumerate() {
                if counts[i] > 0 {
                    for val in centroid.iter_mut() {
                        *val /= counts[i] as f32;
                    }
                }
            }

            centroids = new_centroids;
        }

        Ok(centroids)
    }

    /// Find closest centroid to a vector
    fn find_closest_centroid(&self, vector: &[f32], centroids: &[Vec<f32>]) -> u8 {
        let mut best_distance = f32::INFINITY;
        let mut best_idx = 0;

        for (i, centroid) in centroids.iter().enumerate() {
            let distance = self.euclidean_distance(vector, centroid);
            if distance < best_distance {
                best_distance = distance;
                best_idx = i;
            }
        }

        best_idx as u8
    }

    /// Calculate Euclidean distance between two vectors
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Update compression statistics
    fn update_stats(
        &mut self,
        original: &[Vec<f32>],
        compressed: &[CompressedVector],
        elapsed: std::time::Duration,
    ) {
        let original_bytes: usize = original
            .iter()
            .map(|v| v.len() * std::mem::size_of::<f32>())
            .sum();

        let compressed_bytes: usize = compressed
            .iter()
            .map(|c| c.data.len() + std::mem::size_of::<CompressionMetadata>())
            .sum();

        self.stats.original_size_bytes += original_bytes;
        self.stats.compressed_size_bytes += compressed_bytes;
        self.stats.vectors_processed += original.len();

        if self.stats.original_size_bytes > 0 {
            self.stats.compression_ratio =
                self.stats.original_size_bytes as f64 / self.stats.compressed_size_bytes as f64;
        }

        self.stats.avg_compression_time_us = elapsed.as_micros() as f64 / original.len() as f64;
    }

    /// Get compression statistics
    pub fn stats(&self) -> &CompressionStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vectors(count: usize, dimensions: usize) -> Vec<Vec<f32>> {
        use rand::prelude::*;
        let mut rng = thread_rng();

        (0..count)
            .map(|_| (0..dimensions).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    #[test]
    fn test_scalar_quantization() {
        let config = CompressionConfig::default();
        let mut compressor = VectorCompressor::new(config);

        let vectors = create_test_vectors(10, 384);
        let compressed = compressor.compress_batch(&vectors).unwrap();
        let decompressed = compressor.decompress_batch(&compressed).unwrap();

        assert_eq!(vectors.len(), decompressed.len());
        assert_eq!(vectors[0].len(), decompressed[0].len());

        // Check compression ratio
        let stats = compressor.stats();
        assert!(stats.compression_ratio > 1.0);
    }

    #[test]
    fn test_no_compression() {
        let config = CompressionConfig {
            algorithm: CompressionAlgorithm::None,
            ..Default::default()
        };
        let mut compressor = VectorCompressor::new(config);

        let vectors = create_test_vectors(5, 100);
        let compressed = compressor.compress_batch(&vectors).unwrap();
        let decompressed = compressor.decompress_batch(&compressed).unwrap();

        // Should be exactly equal for no compression
        for (orig, decomp) in vectors.iter().zip(decompressed.iter()) {
            assert_eq!(orig.len(), decomp.len());
            for (a, b) in orig.iter().zip(decomp.iter()) {
                assert!((a - b).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_compression_stats() {
        let config = CompressionConfig::default();
        let compressor = VectorCompressor::new(config);
        let stats = compressor.stats();

        assert_eq!(stats.vectors_processed, 0);
        assert_eq!(stats.compression_ratio, 0.0);
    }
}
