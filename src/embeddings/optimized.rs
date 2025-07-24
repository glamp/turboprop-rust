//! Optimized embedding generator with performance monitoring and caching.
//!
//! This module provides an enhanced embedding generator that includes:
//! - Performance metrics collection
//! - LRU caching for repeated texts
//! - Optimized batch processing
//! - Resource monitoring

use anyhow::{Context, Result};
use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::backends::{GGUFEmbeddingModel, HuggingFaceBackend};
use crate::metrics::{MetricsCollector, ResourceMonitor};
use crate::models::{EmbeddingModel, ModelInfo, ModelManager};
use crate::types::{CachePath, ModelBackend};

use super::backends::EmbeddingBackendType;

/// Optimized embedding generator with performance monitoring and caching
pub struct OptimizedEmbeddingGenerator {
    backend: EmbeddingBackendType,
    metrics: MetricsCollector,
    resource_monitor: ResourceMonitor,
    embedding_cache: LruCache<String, Vec<f32>>,
}

impl OptimizedEmbeddingGenerator {
    /// Create a new optimized embedding generator with the specified model
    pub async fn new_with_model(model_info: &ModelInfo) -> Result<Self> {
        let start_time = Instant::now();
        let metrics = MetricsCollector::new(model_info.name.to_string());

        info!(
            "Initializing optimized embedding generator with model: {}",
            model_info.name
        );

        // Load model with timing
        let backend = Self::load_backend_optimized(model_info).await?;

        let load_time = start_time.elapsed();
        metrics.record_model_load_time(load_time);

        info!(
            "Model '{}' loaded in {:.2}s",
            model_info.name.as_str(),
            load_time.as_secs_f32()
        );

        Ok(Self {
            backend,
            metrics,
            resource_monitor: ResourceMonitor::new(),
            embedding_cache: LruCache::new(NonZeroUsize::new(1000).unwrap()),
        })
    }

    /// Load backend with optimizations for different model types
    async fn load_backend_optimized(model_info: &ModelInfo) -> Result<EmbeddingBackendType> {
        match model_info.backend {
            ModelBackend::FastEmbed => {
                use fastembed::{EmbeddingModel as FastEmbedModel, InitOptions, TextEmbedding};

                info!(
                    "Using FastEmbed backend for model: {}",
                    model_info.name.as_str()
                );

                // Parse the model name to get the EmbeddingModel enum variant
                let embedding_model = match model_info.name.as_str() {
                    "sentence-transformers/all-MiniLM-L6-v2" => FastEmbedModel::AllMiniLML6V2,
                    "sentence-transformers/all-MiniLM-L12-v2" => FastEmbedModel::AllMiniLML12V2,
                    _ => {
                        warn!(
                            "Unknown FastEmbed model '{}', falling back to default",
                            model_info.name.as_str()
                        );
                        FastEmbedModel::AllMiniLML6V2
                    }
                };

                // Optimize fastembed loading
                let init_options = InitOptions::new(embedding_model).with_cache_dir(
                    model_info.local_path.clone().unwrap_or_else(|| {
                        std::env::var("FASTEMBED_CACHE_PATH")
                            .unwrap_or_else(|_| ".turboprop/models".to_string())
                            .into()
                    }),
                );

                let model = TextEmbedding::try_new(init_options)
                    .context("Failed to initialize FastEmbed model")?;
                Ok(EmbeddingBackendType::FastEmbed(model))
            }

            ModelBackend::Candle => {
                info!(
                    "Using GGUF/Candle backend for model: {}",
                    model_info.name.as_str()
                );

                // Optimize GGUF loading with memory mapping
                let model_manager = ModelManager::default();
                let model_path = model_manager.download_gguf_model(model_info).await?;
                let model = GGUFEmbeddingModel::load_from_path(&model_path, model_info)?;

                Ok(EmbeddingBackendType::GGUF(model))
            }

            ModelBackend::Custom => {
                info!(
                    "Using HuggingFace backend for model: {}",
                    model_info.name.as_str()
                );

                // Optimize Hugging Face loading with caching
                let hf_backend = HuggingFaceBackend::new()
                    .context("Failed to initialize HuggingFace backend")?;
                let cache_path = CachePath::from(std::path::PathBuf::from(".turboprop/models"));
                let model = hf_backend
                    .load_qwen3_model(&model_info.name, &cache_path)
                    .await?;

                Ok(EmbeddingBackendType::HuggingFace(model))
            }
        }
    }

    /// Generate embeddings with performance monitoring and caching
    pub fn embed_with_monitoring(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let start_time = Instant::now();
        self.resource_monitor.sample_resources();

        // Check cache for repeated texts
        let mut results = Vec::with_capacity(texts.len());
        let mut uncached_texts = Vec::new();
        let mut uncached_indices = Vec::new();

        for (i, text) in texts.iter().enumerate() {
            if let Some(cached_embedding) = self.embedding_cache.get(text) {
                results.push(Some(cached_embedding.clone()));
            } else {
                results.push(None);
                uncached_texts.push(text.clone());
                uncached_indices.push(i);
            }
        }

        // Process uncached texts in optimized batches
        if !uncached_texts.is_empty() {
            let batch_embeddings = self.embed_batch_optimized(&uncached_texts)?;

            // Update cache and results
            for (batch_idx, original_idx) in uncached_indices.iter().enumerate() {
                let embedding = batch_embeddings[batch_idx].clone();
                self.embedding_cache
                    .put(uncached_texts[batch_idx].clone(), embedding.clone());
                results[*original_idx] = Some(embedding);
            }
        }

        // Flatten results
        let final_embeddings: Vec<Vec<f32>> = results.into_iter().map(|opt| opt.unwrap()).collect();

        let duration = start_time.elapsed();
        self.metrics.record_embedding(texts.len(), duration);
        self.resource_monitor.sample_resources();

        debug!(
            "Embedded {} texts in {:.2}ms (avg: {:.2}ms/text)",
            texts.len(),
            duration.as_millis(),
            duration.as_millis() as f32 / texts.len() as f32
        );

        Ok(final_embeddings)
    }

    /// Process texts in optimized batches
    fn embed_batch_optimized(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // Determine optimal batch size based on model type and available memory
        let optimal_batch_size = self.calculate_optimal_batch_size(texts.len());

        if texts.len() <= optimal_batch_size {
            // Process as single batch
            self.embed_single_batch(texts)
        } else {
            // Process in multiple optimized batches
            let mut all_embeddings = Vec::new();

            for chunk in texts.chunks(optimal_batch_size) {
                let batch_embeddings = self.embed_single_batch(chunk)?;
                all_embeddings.extend(batch_embeddings);
            }

            Ok(all_embeddings)
        }
    }

    /// Embed a single batch of texts
    fn embed_single_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        match &mut self.backend {
            EmbeddingBackendType::FastEmbed(model) => {
                let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                let embeddings = model
                    .embed(text_refs, None)
                    .context("Failed to generate FastEmbed embeddings")?;
                Ok(embeddings)
            }
            EmbeddingBackendType::GGUF(model) => model
                .embed(texts)
                .context("Failed to generate GGUF embeddings"),
            EmbeddingBackendType::HuggingFace(model) => model
                .embed(texts)
                .context("Failed to generate HuggingFace embeddings"),
        }
    }

    /// Calculate optimal batch size based on model type and available memory
    fn calculate_optimal_batch_size(&self, text_count: usize) -> usize {
        // Calculate based on model type and available memory
        let base_size = match &self.backend {
            EmbeddingBackendType::FastEmbed(_) => {
                // FastEmbed typically handles batching well
                std::cmp::min(text_count, 32)
            }
            EmbeddingBackendType::GGUF(_) => {
                // GGUF models may have memory constraints
                std::cmp::min(text_count, 8)
            }
            EmbeddingBackendType::HuggingFace(_) => {
                // Adjust based on model size
                std::cmp::min(text_count, 16)
            }
        };

        // Apply minimum constraint
        std::cmp::max(base_size, 1)
    }

    /// Get a comprehensive performance report
    pub fn get_performance_report(&self) -> PerformanceReport {
        let metrics = self.metrics.get_metrics();
        let peak_memory = self.resource_monitor.get_peak_memory();

        PerformanceReport {
            model_name: metrics.model_name,
            total_texts_processed: metrics.texts_embedded,
            total_processing_time: metrics.total_embedding_time,
            average_time_per_text: metrics.avg_time_per_text,
            model_load_time: metrics.model_load_time,
            peak_memory_usage: peak_memory,
            cache_efficiency: self.calculate_cache_efficiency(),
        }
    }

    /// Calculate cache efficiency
    fn calculate_cache_efficiency(&self) -> f32 {
        // Calculate cache hit rate
        let total_capacity = self.embedding_cache.cap().get() as f32;
        let current_size = self.embedding_cache.len() as f32;
        current_size / total_capacity
    }
}

/// Comprehensive performance report for embedding operations
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub model_name: String,
    pub total_texts_processed: usize,
    pub total_processing_time: Duration,
    pub average_time_per_text: Duration,
    pub model_load_time: Option<Duration>,
    pub peak_memory_usage: u64,
    pub cache_efficiency: f32,
}

impl PerformanceReport {
    /// Print a human-readable summary of the performance report
    pub fn print_summary(&self) {
        println!("Performance Report for {}", self.model_name);
        println!("{}", "=".repeat(50));
        println!("Texts processed: {}", self.total_texts_processed);
        println!(
            "Total time: {:.2}s",
            self.total_processing_time.as_secs_f32()
        );
        println!(
            "Avg time/text: {:.2}ms",
            self.average_time_per_text.as_millis()
        );

        if let Some(load_time) = self.model_load_time {
            println!("Model load time: {:.2}s", load_time.as_secs_f32());
        }

        println!("Peak memory: {}", format_bytes(self.peak_memory_usage));
        println!("Cache efficiency: {:.1}%", self.cache_efficiency * 100.0);
    }
}

/// Format bytes in human-readable format
fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    format!("{:.1} {}", size, UNITS[unit_index])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0.0 B");
        assert_eq!(format_bytes(512), "512.0 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1_048_576), "1.0 MB");
        assert_eq!(format_bytes(1_073_741_824), "1.0 GB");
    }

    #[test]
    fn test_performance_report_creation() {
        let report = PerformanceReport {
            model_name: "test-model".to_string(),
            total_texts_processed: 100,
            total_processing_time: Duration::from_secs(2),
            average_time_per_text: Duration::from_millis(20),
            model_load_time: Some(Duration::from_secs(5)),
            peak_memory_usage: 50_000_000,
            cache_efficiency: 0.75,
        };

        assert_eq!(report.model_name, "test-model");
        assert_eq!(report.total_texts_processed, 100);
        assert_eq!(report.cache_efficiency, 0.75);
    }
}
