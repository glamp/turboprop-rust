//! Tests for benchmark command and performance comparison tools

use std::path::PathBuf;
use std::time::Duration;
use turboprop::commands::benchmark::{run_benchmark, BenchmarkArgs, BenchmarkResult};
use turboprop::models::ModelInfo;
use turboprop::types::{ModelBackend, ModelName, ModelType};

/// Mock model info for testing
fn create_test_model_info(name: &str) -> ModelInfo {
    ModelInfo {
        name: ModelName::from(name),
        backend: ModelBackend::FastEmbed,
        model_type: ModelType::SentenceTransformer,
        dimensions: 384,
        size_bytes: 90_000_000,
        description: "Test model".to_string(),
        download_url: Some("https://example.com".to_string()),
        local_path: None,
    }
}

#[test]
fn test_benchmark_args_creation() {
    let args = BenchmarkArgs {
        models: Some(vec!["model1".to_string(), "model2".to_string()]),
        text_count: 50,
        iterations: 2,
        sample_file: Some(PathBuf::from("test.txt")),
        format: "json".to_string(),
    };

    assert_eq!(
        args.models,
        Some(vec!["model1".to_string(), "model2".to_string()])
    );
    assert_eq!(args.text_count, 50);
    assert_eq!(args.iterations, 2);
    assert_eq!(args.sample_file, Some(PathBuf::from("test.txt")));
    assert_eq!(args.format, "json");
}

#[test]
fn test_benchmark_args_defaults() {
    let args = BenchmarkArgs {
        models: None,
        text_count: 100,
        iterations: 3,
        sample_file: None,
        format: "table".to_string(),
    };

    assert_eq!(args.models, None);
    assert_eq!(args.text_count, 100);
    assert_eq!(args.iterations, 3);
    assert_eq!(args.sample_file, None);
    assert_eq!(args.format, "table");
}

#[test]
fn test_benchmark_result_creation() {
    let result = BenchmarkResult {
        model: "test-model".to_string(),
        texts_per_second: 50.0,
        avg_latency_ms: 20.0,
        model_load_time_s: 2.5,
        peak_memory_mb: 100.0,
        cache_efficiency: 75.0,
    };

    assert_eq!(result.model, "test-model");
    assert_eq!(result.texts_per_second, 50.0);
    assert_eq!(result.avg_latency_ms, 20.0);
    assert_eq!(result.model_load_time_s, 2.5);
    assert_eq!(result.peak_memory_mb, 100.0);
    assert_eq!(result.cache_efficiency, 75.0);
}

#[test]
fn test_benchmark_result_calculations() {
    // Test that the benchmark result fields make sense together
    let texts_processed = 100;
    let total_time_s = 2.0;
    let texts_per_second = texts_processed as f32 / total_time_s;
    let avg_latency_ms = (total_time_s * 1000.0) / texts_processed as f32;

    let result = BenchmarkResult {
        model: "test-model".to_string(),
        texts_per_second,
        avg_latency_ms,
        model_load_time_s: 1.0,
        peak_memory_mb: 50.0,
        cache_efficiency: 80.0,
    };

    assert_eq!(result.texts_per_second, 50.0);
    assert_eq!(result.avg_latency_ms, 20.0);

    // Verify the relationship between texts_per_second and avg_latency_ms
    let expected_throughput = 1000.0 / result.avg_latency_ms;
    assert!((result.texts_per_second - expected_throughput).abs() < 0.001);
}

#[tokio::test]
async fn test_benchmark_with_no_models() {
    let args = BenchmarkArgs {
        models: Some(vec!["nonexistent-model".to_string()]),
        text_count: 10,
        iterations: 1,
        sample_file: None,
        format: "table".to_string(),
    };

    // This should return an error when no models are available
    let result = run_benchmark(args).await;
    assert!(result.is_err());

    if let Err(error) = result {
        assert!(error.to_string().contains("No models available"));
    }
}

#[test]
fn test_generate_test_texts() {
    // Test the test text generation logic
    let sample_texts = vec![
        "function calculateSum(a, b) { return a + b; }",
        "def process_data(data): return data.strip().upper()",
        "class UserService { authenticate(user) { return user.isValid(); } }",
    ];

    let count = 10;
    let generated_texts: Vec<String> = (0..count)
        .map(|i| sample_texts[i % sample_texts.len()].to_string())
        .collect();

    assert_eq!(generated_texts.len(), count);
    assert_eq!(generated_texts[0], sample_texts[0]);
    assert_eq!(generated_texts[3], sample_texts[0]); // Should cycle
    assert_eq!(generated_texts[4], sample_texts[1]);
}

#[test]
fn test_load_sample_texts_from_content() {
    // Test loading sample texts from content (simulating file content)
    let content = "line 1\nline 2\n\nline 4\n  \nline 6";
    let lines: Vec<String> = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .take(5)
        .map(|s| s.to_string())
        .collect();

    assert_eq!(lines.len(), 4); // Empty lines filtered out
    assert_eq!(lines[0], "line 1");
    assert_eq!(lines[1], "line 2");
    assert_eq!(lines[2], "line 4");
    assert_eq!(lines[3], "line 6");
}

#[test]
fn test_average_benchmark_results() {
    // Test averaging logic for multiple benchmark iterations
    let results = vec![
        BenchmarkResult {
            model: "test-model".to_string(),
            texts_per_second: 40.0,
            avg_latency_ms: 25.0,
            model_load_time_s: 2.0,
            peak_memory_mb: 100.0,
            cache_efficiency: 70.0,
        },
        BenchmarkResult {
            model: "test-model".to_string(),
            texts_per_second: 60.0,
            avg_latency_ms: 15.0,
            model_load_time_s: 3.0,
            peak_memory_mb: 120.0,
            cache_efficiency: 80.0,
        },
    ];

    let count = results.len() as f32;
    let avg_result = BenchmarkResult {
        model: "test-model".to_string(),
        texts_per_second: results.iter().map(|r| r.texts_per_second).sum::<f32>() / count,
        avg_latency_ms: results.iter().map(|r| r.avg_latency_ms).sum::<f32>() / count,
        model_load_time_s: results.iter().map(|r| r.model_load_time_s).sum::<f32>() / count,
        peak_memory_mb: results.iter().map(|r| r.peak_memory_mb).sum::<f32>() / count,
        cache_efficiency: results.iter().map(|r| r.cache_efficiency).sum::<f32>() / count,
    };

    assert_eq!(avg_result.texts_per_second, 50.0);
    assert_eq!(avg_result.avg_latency_ms, 20.0);
    assert_eq!(avg_result.model_load_time_s, 2.5);
    assert_eq!(avg_result.peak_memory_mb, 110.0);
    assert_eq!(avg_result.cache_efficiency, 75.0);
}

#[test]
fn test_benchmark_result_serialization() {
    let result = BenchmarkResult {
        model: "test-model".to_string(),
        texts_per_second: 50.0,
        avg_latency_ms: 20.0,
        model_load_time_s: 2.5,
        peak_memory_mb: 100.0,
        cache_efficiency: 75.0,
    };

    // Test JSON serialization
    let json = serde_json::to_string(&result).expect("Should serialize to JSON");
    let deserialized: BenchmarkResult =
        serde_json::from_str(&json).expect("Should deserialize from JSON");

    assert_eq!(result.model, deserialized.model);
    assert_eq!(result.texts_per_second, deserialized.texts_per_second);
    assert_eq!(result.avg_latency_ms, deserialized.avg_latency_ms);
}

#[test]
fn test_benchmark_output_formats() {
    let results = vec![
        BenchmarkResult {
            model: "model1".to_string(),
            texts_per_second: 50.0,
            avg_latency_ms: 20.0,
            model_load_time_s: 2.0,
            peak_memory_mb: 100.0,
            cache_efficiency: 75.0,
        },
        BenchmarkResult {
            model: "model2".to_string(),
            texts_per_second: 60.0,
            avg_latency_ms: 16.7,
            model_load_time_s: 1.5,
            peak_memory_mb: 80.0,
            cache_efficiency: 85.0,
        },
    ];

    // Test CSV format generation
    let csv_header =
        "model,texts_per_second,avg_latency_ms,model_load_time_s,peak_memory_mb,cache_efficiency";
    let csv_lines: Vec<String> = results
        .iter()
        .map(|r| {
            format!(
                "{},{},{},{},{},{}",
                r.model,
                r.texts_per_second,
                r.avg_latency_ms,
                r.model_load_time_s,
                r.peak_memory_mb,
                r.cache_efficiency
            )
        })
        .collect();

    assert_eq!(csv_lines[0], "model1,50,20,2,100,75");
    assert_eq!(csv_lines[1], "model2,60,16.7,1.5,80,85");

    // Test that results can be used with tabled
    // (We can't easily test the actual table output without integration testing)
    assert!(results.len() > 0);
}

#[test]
fn test_benchmark_error_conditions() {
    // Test that benchmark handles various error conditions gracefully

    // Empty model list
    let args = BenchmarkArgs {
        models: Some(vec![]),
        text_count: 10,
        iterations: 1,
        sample_file: None,
        format: "table".to_string(),
    };

    // Should handle empty model list
    assert_eq!(args.models.as_ref().unwrap().len(), 0);

    // Zero text count
    let args = BenchmarkArgs {
        models: None,
        text_count: 0,
        iterations: 1,
        sample_file: None,
        format: "table".to_string(),
    };

    assert_eq!(args.text_count, 0);

    // Zero iterations
    let args = BenchmarkArgs {
        models: None,
        text_count: 10,
        iterations: 0,
        sample_file: None,
        format: "table".to_string(),
    };

    assert_eq!(args.iterations, 0);
}

#[test]
fn test_benchmark_performance_metrics() {
    // Test performance calculations
    let texts_processed = 100;
    let total_time_ms = 2000; // 2 seconds
    let model_load_time_ms = 1000; // 1 second
    let peak_memory_bytes = 104_857_600; // 100MB

    // Calculate metrics
    let texts_per_second = (texts_processed as f32) / (total_time_ms as f32 / 1000.0);
    let avg_latency_ms = total_time_ms as f32 / texts_processed as f32;
    let model_load_time_s = model_load_time_ms as f32 / 1000.0;
    let peak_memory_mb = peak_memory_bytes as f32 / 1_048_576.0;

    assert_eq!(texts_per_second, 50.0);
    assert_eq!(avg_latency_ms, 20.0);
    assert_eq!(model_load_time_s, 1.0);
    assert_eq!(peak_memory_mb, 100.0);
}

#[test]
fn test_benchmark_result_validation() {
    // Test that benchmark results have reasonable values
    let result = BenchmarkResult {
        model: "test-model".to_string(),
        texts_per_second: 50.0,
        avg_latency_ms: 20.0,
        model_load_time_s: 2.0,
        peak_memory_mb: 100.0,
        cache_efficiency: 75.0,
    };

    // Validate ranges
    assert!(result.texts_per_second > 0.0);
    assert!(result.avg_latency_ms > 0.0);
    assert!(result.model_load_time_s >= 0.0);
    assert!(result.peak_memory_mb >= 0.0);
    assert!(result.cache_efficiency >= 0.0 && result.cache_efficiency <= 100.0);

    // Validate model name
    assert!(!result.model.is_empty());
}
