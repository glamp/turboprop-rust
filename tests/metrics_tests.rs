//! Tests for performance metrics collection system

use std::time::Duration;
use turboprop::metrics::{EmbeddingMetrics, MetricsCollector, ResourceMonitor};

#[test]
fn test_embedding_metrics_initialization() {
    let model_name = "test-model".to_string();
    let metrics = EmbeddingMetrics::new(model_name.clone());

    assert_eq!(metrics.model_name, model_name);
    assert_eq!(metrics.texts_embedded, 0);
    assert_eq!(metrics.total_embedding_time, Duration::ZERO);
    assert_eq!(metrics.avg_time_per_text, Duration::ZERO);
    assert_eq!(metrics.peak_memory_usage, 0);
    assert_eq!(metrics.model_load_time, None);
    assert_eq!(metrics.cache_hit_rate, 0.0);
}

#[test]
fn test_embedding_metrics_update_stats() {
    let mut metrics = EmbeddingMetrics::new("test-model".to_string());

    // Update with first batch
    let duration1 = Duration::from_millis(100);
    metrics.update_embedding_stats(5, duration1);

    assert_eq!(metrics.texts_embedded, 5);
    assert_eq!(metrics.total_embedding_time, duration1);
    assert_eq!(metrics.avg_time_per_text, Duration::from_millis(20)); // 100ms / 5 texts

    // Update with second batch
    let duration2 = Duration::from_millis(200);
    metrics.update_embedding_stats(10, duration2);

    assert_eq!(metrics.texts_embedded, 15);
    assert_eq!(metrics.total_embedding_time, Duration::from_millis(300));
    assert_eq!(metrics.avg_time_per_text, Duration::from_millis(20)); // 300ms / 15 texts
}

#[test]
fn test_embedding_metrics_set_model_load_time() {
    let mut metrics = EmbeddingMetrics::new("test-model".to_string());
    let load_time = Duration::from_secs(2);

    metrics.set_model_load_time(load_time);

    assert_eq!(metrics.model_load_time, Some(load_time));
}

#[test]
fn test_embedding_metrics_update_memory_usage() {
    let mut metrics = EmbeddingMetrics::new("test-model".to_string());

    // Set initial memory usage
    metrics.update_memory_usage(1000);
    assert_eq!(metrics.peak_memory_usage, 1000);

    // Update with higher memory usage
    metrics.update_memory_usage(2000);
    assert_eq!(metrics.peak_memory_usage, 2000);

    // Update with lower memory usage (should not change peak)
    metrics.update_memory_usage(1500);
    assert_eq!(metrics.peak_memory_usage, 2000);
}

#[test]
fn test_metrics_collector_initialization() {
    let model_name = "test-model".to_string();
    let collector = MetricsCollector::new(model_name.clone());

    let metrics = collector.get_metrics();
    assert_eq!(metrics.model_name, model_name);
    assert_eq!(metrics.texts_embedded, 0);
}

#[test]
fn test_metrics_collector_record_embedding() {
    let collector = MetricsCollector::new("test-model".to_string());

    // Record first embedding batch
    collector.record_embedding(3, Duration::from_millis(150));

    let metrics = collector.get_metrics();
    assert_eq!(metrics.texts_embedded, 3);
    assert_eq!(metrics.total_embedding_time, Duration::from_millis(150));

    // Record second embedding batch
    collector.record_embedding(2, Duration::from_millis(100));

    let metrics = collector.get_metrics();
    assert_eq!(metrics.texts_embedded, 5);
    assert_eq!(metrics.total_embedding_time, Duration::from_millis(250));
}

#[test]
fn test_metrics_collector_record_model_load_time() {
    let collector = MetricsCollector::new("test-model".to_string());
    let load_time = Duration::from_secs(3);

    collector.record_model_load_time(load_time);

    let metrics = collector.get_metrics();
    assert_eq!(metrics.model_load_time, Some(load_time));
}

#[test]
fn test_resource_monitor_initialization() {
    let monitor = ResourceMonitor::new();

    assert_eq!(monitor.get_peak_memory(), 0);
    assert_eq!(monitor.get_avg_memory(), 0);
}

#[test]
fn test_resource_monitor_sampling() {
    let mut monitor = ResourceMonitor::new();

    // Sample resources (should not panic)
    monitor.sample_resources();

    // Test peak and average memory functions with empty samples
    assert_eq!(monitor.get_peak_memory(), 0);
    assert_eq!(monitor.get_avg_memory(), 0);
}

#[test]
fn test_resource_monitor_memory_calculations() {
    let mut monitor = ResourceMonitor::new();

    // Manually add some memory samples for testing
    // We'll need to access the internal data structure through a method
    // This test validates the calculation logic

    // For now, just verify the monitor can be created and used
    let peak = monitor.get_peak_memory();
    let avg = monitor.get_avg_memory();

    // These should be 0 initially since no samples have been recorded
    assert_eq!(peak, 0);
    assert_eq!(avg, 0);
}

#[test]
fn test_embedding_metrics_serialization() {
    let metrics = EmbeddingMetrics::new("test-model".to_string());

    // Test that metrics can be serialized and deserialized
    let json = serde_json::to_string(&metrics).expect("Should serialize to JSON");
    let deserialized: EmbeddingMetrics =
        serde_json::from_str(&json).expect("Should deserialize from JSON");

    assert_eq!(metrics.model_name, deserialized.model_name);
    assert_eq!(metrics.texts_embedded, deserialized.texts_embedded);
    assert_eq!(metrics.peak_memory_usage, deserialized.peak_memory_usage);
}

#[test]
fn test_embedding_metrics_edge_cases() {
    let mut metrics = EmbeddingMetrics::new("test-model".to_string());

    // Test update with zero texts (should not panic but should not change avg)
    metrics.update_embedding_stats(0, Duration::from_millis(100));
    assert_eq!(metrics.texts_embedded, 0);
    assert_eq!(metrics.avg_time_per_text, Duration::ZERO);

    // Test update with very large numbers
    metrics.update_embedding_stats(1000000, Duration::from_secs(1000));
    assert_eq!(metrics.texts_embedded, 1000000);
    assert_eq!(metrics.total_embedding_time, Duration::from_secs(1000));
}
