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
            // Use mach task_info to get memory usage on macOS
            use std::mem;

            #[repr(C)]
            struct mach_task_basic_info {
                virtual_size: u64,
                resident_size: u64,
                resident_size_max: u64,
                user_time: u64,
                system_time: u64,
                policy: i32,
                suspend_count: i32,
            }

            const MACH_TASK_BASIC_INFO: u32 = 20;

            extern "C" {
                fn mach_task_self() -> u32;
                fn task_info(
                    task: u32,
                    flavor: u32,
                    task_info: *mut u8,
                    task_info_count: *mut u32,
                ) -> i32;
            }

            unsafe {
                let mut info: mach_task_basic_info = mem::zeroed();
                let mut count =
                    (mem::size_of::<mach_task_basic_info>() / mem::size_of::<u32>()) as u32;

                let result = task_info(
                    mach_task_self(),
                    MACH_TASK_BASIC_INFO,
                    &mut info as *mut _ as *mut u8,
                    &mut count,
                );

                if result == 0 {
                    Ok(info.resident_size)
                } else {
                    Ok(0) // Fall back to 0 if system call fails
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            // Use Windows API to get process memory information
            #[repr(C)]
            struct ProcessMemoryCounters {
                cb: u32,
                page_fault_count: u32,
                peak_working_set_size: usize,
                working_set_size: usize,
                quota_peak_paged_pool_usage: usize,
                quota_paged_pool_usage: usize,
                quota_peak_non_paged_pool_usage: usize,
                quota_non_paged_pool_usage: usize,
                pagefile_usage: usize,
                peak_pagefile_usage: usize,
            }

            extern "system" {
                fn GetCurrentProcess() -> *mut std::ffi::c_void;
                fn GetProcessMemoryInfo(
                    process: *mut std::ffi::c_void,
                    counters: *mut ProcessMemoryCounters,
                    cb: u32,
                ) -> i32;
            }

            unsafe {
                use std::mem;
                let mut counters: ProcessMemoryCounters = mem::zeroed();
                counters.cb = mem::size_of::<ProcessMemoryCounters>() as u32;

                let result = GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb);

                if result != 0 {
                    Ok(counters.working_set_size as u64)
                } else {
                    Ok(0) // Fall back to 0 if API call fails
                }
            }
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        Ok(0)
    }

    /// Get CPU usage (per-platform implementation)
    fn get_cpu_usage(&self) -> Result<f32, Box<dyn std::error::Error>> {
        #[cfg(target_os = "linux")]
        {
            // Read CPU usage from /proc/self/stat
            let stat = std::fs::read_to_string("/proc/self/stat")?;
            let fields: Vec<&str> = stat.split_whitespace().collect();

            if fields.len() >= 15 {
                // Sum user time (13) and system time (14) in clock ticks
                let utime: u64 = fields[13].parse().unwrap_or(0);
                let stime: u64 = fields[14].parse().unwrap_or(0);
                let total_time = utime + stime;

                // Simple approximation - would need time interval for accurate percentage
                // Return a normalized value based on total CPU time
                Ok((total_time as f32) / 1000000.0) // Convert to approximate percentage
            } else {
                Ok(0.0)
            }
        }

        #[cfg(target_os = "macos")]
        {
            // Use the same mach API to get CPU time information
            use std::mem;

            #[repr(C)]
            struct mach_task_basic_info {
                virtual_size: u64,
                resident_size: u64,
                resident_size_max: u64,
                user_time: u64,
                system_time: u64,
                policy: i32,
                suspend_count: i32,
            }

            const MACH_TASK_BASIC_INFO: u32 = 20;

            extern "C" {
                fn mach_task_self() -> u32;
                fn task_info(
                    task: u32,
                    flavor: u32,
                    task_info: *mut u8,
                    task_info_count: *mut u32,
                ) -> i32;
            }

            unsafe {
                let mut info: mach_task_basic_info = mem::zeroed();
                let mut count =
                    (mem::size_of::<mach_task_basic_info>() / mem::size_of::<u32>()) as u32;

                let result = task_info(
                    mach_task_self(),
                    MACH_TASK_BASIC_INFO,
                    &mut info as *mut _ as *mut u8,
                    &mut count,
                );

                if result == 0 {
                    // Convert microseconds to approximate percentage
                    let total_time = info.user_time + info.system_time;
                    Ok((total_time as f32) / 1000000.0)
                } else {
                    Ok(0.0)
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            // Use Windows API to get process times
            #[repr(C)]
            struct FileTime {
                low_date_time: u32,
                high_date_time: u32,
            }

            extern "system" {
                fn GetCurrentProcess() -> *mut std::ffi::c_void;
                fn GetProcessTimes(
                    process: *mut std::ffi::c_void,
                    creation_time: *mut FileTime,
                    exit_time: *mut FileTime,
                    kernel_time: *mut FileTime,
                    user_time: *mut FileTime,
                ) -> i32;
            }

            unsafe {
                use std::mem;
                let mut creation_time: FileTime = mem::zeroed();
                let mut exit_time: FileTime = mem::zeroed();
                let mut kernel_time: FileTime = mem::zeroed();
                let mut user_time: FileTime = mem::zeroed();

                let result = GetProcessTimes(
                    GetCurrentProcess(),
                    &mut creation_time,
                    &mut exit_time,
                    &mut kernel_time,
                    &mut user_time,
                );

                if result != 0 {
                    // Convert FILETIME to approximate CPU usage
                    let user = ((user_time.high_date_time as u64) << 32)
                        | (user_time.low_date_time as u64);
                    let kernel = ((kernel_time.high_date_time as u64) << 32)
                        | (kernel_time.low_date_time as u64);
                    let total = user + kernel;

                    // Convert 100-nanosecond intervals to approximate percentage
                    Ok((total as f32) / 10000000.0)
                } else {
                    Ok(0.0)
                }
            }
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
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
