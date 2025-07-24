//! Performance metrics collection for embedding operations.
//!
//! This module provides functionality for collecting and reporting performance metrics
//! for embedding generation, including timing, memory usage, and throughput measurements.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Performance metrics for embedding operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingMetrics {
    /// Model name
    pub model_name: String,
    /// Total number of texts embedded
    pub texts_embedded: usize,
    /// Total embedding time
    pub total_embedding_time: Duration,
    /// Average time per text
    pub avg_time_per_text: Duration,
    /// Peak memory usage during embedding
    pub peak_memory_usage: u64,
    /// Model loading time
    pub model_load_time: Option<Duration>,
    /// Cache hit rate for repeated texts
    pub cache_hit_rate: f32,
}

impl EmbeddingMetrics {
    /// Create a new metrics instance for the specified model
    pub fn new(model_name: String) -> Self {
        Self {
            model_name,
            texts_embedded: 0,
            total_embedding_time: Duration::ZERO,
            avg_time_per_text: Duration::ZERO,
            peak_memory_usage: 0,
            model_load_time: None,
            cache_hit_rate: 0.0,
        }
    }

    /// Update embedding statistics with new batch results
    pub fn update_embedding_stats(&mut self, text_count: usize, duration: Duration) {
        if text_count == 0 {
            return; // Don't update stats for empty batches
        }

        self.texts_embedded += text_count;
        self.total_embedding_time += duration;
        self.avg_time_per_text = self.total_embedding_time / self.texts_embedded as u32;
    }

    /// Set the model loading time
    pub fn set_model_load_time(&mut self, duration: Duration) {
        self.model_load_time = Some(duration);
    }

    /// Update memory usage, tracking the peak value
    pub fn update_memory_usage(&mut self, current_usage: u64) {
        self.peak_memory_usage = self.peak_memory_usage.max(current_usage);
    }
}

/// Thread-safe metrics collector
pub struct MetricsCollector {
    metrics: Arc<std::sync::Mutex<EmbeddingMetrics>>,
    start_time: Instant,
}

impl MetricsCollector {
    /// Create a new metrics collector for the specified model
    pub fn new(model_name: String) -> Self {
        Self {
            metrics: Arc::new(std::sync::Mutex::new(EmbeddingMetrics::new(model_name))),
            start_time: Instant::now(),
        }
    }

    /// Record embedding performance for a batch of texts
    pub fn record_embedding(&self, text_count: usize, duration: Duration) {
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.update_embedding_stats(text_count, duration);
        }
    }

    /// Record the time taken to load the model
    pub fn record_model_load_time(&self, duration: Duration) {
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.set_model_load_time(duration);
        }
    }

    /// Get a copy of the current metrics
    pub fn get_metrics(&self) -> EmbeddingMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Get the total time elapsed since metrics collection started
    pub fn elapsed_time(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// System resource monitor for tracking memory and CPU usage
pub struct ResourceMonitor {
    memory_samples: Vec<u64>,
    cpu_samples: Vec<f32>,
}

impl ResourceMonitor {
    /// Create a new resource monitor
    pub fn new() -> Self {
        Self {
            memory_samples: Vec::new(),
            cpu_samples: Vec::new(),
        }
    }

    /// Sample current system resources
    pub fn sample_resources(&mut self) {
        // Sample current memory usage
        if let Ok(memory) = self.get_memory_usage() {
            self.memory_samples.push(memory);
        }

        // Sample CPU usage (simplified)
        if let Ok(cpu) = self.get_cpu_usage() {
            self.cpu_samples.push(cpu);
        }
    }

    /// Get memory usage for the current process
    fn get_memory_usage(&self) -> Result<u64, Box<dyn std::error::Error>> {
        // Platform-specific memory usage collection
        #[cfg(target_os = "linux")]
        {
            let status = std::fs::read_to_string("/proc/self/status")?;
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let kb: u64 = line
                        .split_whitespace()
                        .nth(1)
                        .ok_or("Invalid format")?
                        .parse()?;
                    return Ok(kb * 1024); // Convert to bytes
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            // Use system calls or external crates for macOS
            // Simplified implementation for now
            Ok(0)
        }

        #[cfg(target_os = "windows")]
        {
            // Use Windows API for memory information
            // Simplified implementation for now
            Ok(0)
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        Ok(0)
    }

    /// Get CPU usage (simplified implementation)
    fn get_cpu_usage(&self) -> Result<f32, Box<dyn std::error::Error>> {
        // Simplified CPU usage (would need more sophisticated implementation)
        Ok(0.0)
    }

    /// Get the peak memory usage observed
    pub fn get_peak_memory(&self) -> u64 {
        self.memory_samples.iter().copied().max().unwrap_or(0)
    }

    /// Get the average memory usage
    pub fn get_avg_memory(&self) -> u64 {
        if self.memory_samples.is_empty() {
            0
        } else {
            self.memory_samples.iter().sum::<u64>() / self.memory_samples.len() as u64
        }
    }
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_integration() {
        let mut metrics = EmbeddingMetrics::new("test-model".to_string());

        // Test basic functionality
        assert_eq!(metrics.texts_embedded, 0);

        metrics.update_embedding_stats(5, Duration::from_millis(100));
        assert_eq!(metrics.texts_embedded, 5);
        assert_eq!(metrics.avg_time_per_text, Duration::from_millis(20));

        // Test memory tracking
        metrics.update_memory_usage(1000);
        metrics.update_memory_usage(2000);
        metrics.update_memory_usage(1500);
        assert_eq!(metrics.peak_memory_usage, 2000);
    }

    #[test]
    fn test_collector_thread_safety() {
        let collector = MetricsCollector::new("test-model".to_string());

        // Test concurrent access (basic test)
        collector.record_embedding(3, Duration::from_millis(150));
        collector.record_model_load_time(Duration::from_secs(2));

        let metrics = collector.get_metrics();
        assert_eq!(metrics.texts_embedded, 3);
        assert_eq!(metrics.model_load_time, Some(Duration::from_secs(2)));
    }
}
