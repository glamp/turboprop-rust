//! Benchmark command for comparing model performance.
//!
//! This module provides benchmarking tools to compare the performance of different
//! embedding models, helping users make informed choices about which models to use.

use anyhow::Result;
use clap::Args;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Instant;
use tabled::{Table, Tabled};
use tracing::{debug, info};

use crate::embeddings::{EmbeddingConfig, OptimizedEmbeddingGenerator};
use crate::models::{ModelInfo, ModelManager};

/// Command-line arguments for the benchmark command
#[derive(Args)]
pub struct BenchmarkArgs {
    /// Models to benchmark (default: all available)
    #[arg(long, value_delimiter = ',')]
    pub models: Option<Vec<String>>,

    /// Number of texts to process for benchmark
    #[arg(long, default_value = "100")]
    pub text_count: usize,

    /// Number of benchmark iterations
    #[arg(long, default_value = "3")]
    pub iterations: usize,

    /// Sample text file to use for benchmarking
    #[arg(long)]
    pub sample_file: Option<PathBuf>,

    /// Output format (table, json, csv)
    #[arg(long, default_value = "table")]
    pub format: String,
}

/// Results from benchmarking a single model
#[derive(Tabled, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub model: String,
    pub texts_per_second: f32,
    pub avg_latency_ms: f32,
    pub model_load_time_s: f32,
    pub peak_memory_mb: f32,
    pub cache_efficiency: f32,
}

/// Run a comprehensive benchmark comparing multiple models
pub async fn run_benchmark(args: BenchmarkArgs) -> Result<()> {
    let model_manager = ModelManager::default();
    let available_models = model_manager.get_available_models();
    let models_to_test = if let Some(model_names) = args.models {
        available_models
            .into_iter()
            .filter(|m| model_names.contains(&m.name.as_str().to_string()))
            .collect()
    } else {
        available_models
    };

    if models_to_test.is_empty() {
        return Err(anyhow::anyhow!("No models available for benchmarking"));
    }

    // Generate or load test texts
    let test_texts = if let Some(sample_file) = args.sample_file {
        load_sample_texts(&sample_file, args.text_count)?
    } else {
        generate_test_texts(args.text_count)
    };

    info!(
        "Running benchmark with {} texts across {} iterations...",
        test_texts.len(),
        args.iterations
    );
    println!(
        "Running benchmark with {} texts across {} iterations...\n",
        test_texts.len(),
        args.iterations
    );

    let mut results = Vec::new();

    for model_info in models_to_test {
        println!("Benchmarking model: {}", model_info.name.as_str());

        let mut iteration_results = Vec::new();

        for iteration in 1..=args.iterations {
            print!("  Iteration {}/{}... ", iteration, args.iterations);

            let result = benchmark_single_model(&model_info, &test_texts).await?;
            iteration_results.push(result);

            println!("✓");
        }

        // Calculate average results
        let avg_result = average_benchmark_results(model_info.name.as_str(), iteration_results);
        results.push(avg_result);

        println!();
    }

    // Display results
    match args.format.as_str() {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
        "csv" => {
            // CSV output implementation
            println!("model,texts_per_second,avg_latency_ms,model_load_time_s,peak_memory_mb,cache_efficiency");
            for result in results {
                println!(
                    "{},{},{},{},{},{}",
                    result.model,
                    result.texts_per_second,
                    result.avg_latency_ms,
                    result.model_load_time_s,
                    result.peak_memory_mb,
                    result.cache_efficiency
                );
            }
        }
        _ => {
            let table = Table::new(results);
            println!("{}", table);
        }
    }

    Ok(())
}

/// Benchmark a single model with the provided test texts
async fn benchmark_single_model(
    model_info: &ModelInfo,
    test_texts: &[String],
) -> Result<BenchmarkResult> {
    debug!("Starting benchmark for model: {}", model_info.name.as_str());

    let config = EmbeddingConfig::default();
    let mut generator = OptimizedEmbeddingGenerator::new_with_model(model_info, config).await?;

    let start_time = Instant::now();
    let _embeddings = generator.embed_with_monitoring(test_texts)?;
    let total_time = start_time.elapsed();

    let report = generator.get_performance_report();

    let result = BenchmarkResult {
        model: model_info.name.as_str().to_string(),
        texts_per_second: test_texts.len() as f32 / total_time.as_secs_f32(),
        avg_latency_ms: report.average_time_per_text.as_millis() as f32,
        model_load_time_s: report
            .model_load_time
            .map(|d| d.as_secs_f32())
            .unwrap_or(0.0),
        peak_memory_mb: report.peak_memory_usage as f32 / 1_048_576.0,
        cache_efficiency: report.cache_efficiency * 100.0,
    };

    debug!(
        "Benchmark result for {}: {:.1} texts/sec, {:.1}ms avg latency",
        model_info.name.as_str(),
        result.texts_per_second,
        result.avg_latency_ms
    );

    Ok(result)
}

/// Generate test texts for benchmarking
fn generate_test_texts(count: usize) -> Vec<String> {
    let sample_texts = [
        "function calculateSum(a, b) { return a + b; }",
        "def process_data(data): return data.strip().upper()",
        "class UserService { authenticate(user) { return user.isValid(); } }",
        "async function fetchData(url) { const response = await fetch(url); return response.json(); }",
        "fn main() { println!(\"Hello, world!\"); }",
        "import React from 'react'; const App = () => <div>Hello</div>;",
        "SELECT * FROM users WHERE active = true AND created_date > '2023-01-01'",
        "public class Calculator { public int add(int a, int b) { return a + b; } }",
    ];

    (0..count)
        .map(|i| sample_texts[i % sample_texts.len()].to_string())
        .collect()
}

/// Load sample texts from a file
fn load_sample_texts(file_path: &PathBuf, max_count: usize) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(file_path)?;
    let lines: Vec<String> = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .take(max_count)
        .map(|s| s.to_string())
        .collect();

    if lines.is_empty() {
        return Err(anyhow::anyhow!("No valid text found in sample file"));
    }

    Ok(lines)
}

/// Calculate average results from multiple benchmark iterations
fn average_benchmark_results(model_name: &str, results: Vec<BenchmarkResult>) -> BenchmarkResult {
    let count = results.len() as f32;

    BenchmarkResult {
        model: model_name.to_string(),
        texts_per_second: results.iter().map(|r| r.texts_per_second).sum::<f32>() / count,
        avg_latency_ms: results.iter().map(|r| r.avg_latency_ms).sum::<f32>() / count,
        model_load_time_s: results.iter().map(|r| r.model_load_time_s).sum::<f32>() / count,
        peak_memory_mb: results.iter().map(|r| r.peak_memory_mb).sum::<f32>() / count,
        cache_efficiency: results.iter().map(|r| r.cache_efficiency).sum::<f32>() / count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModelBackend, ModelName, ModelType};

    fn _create_test_model_info() -> ModelInfo {
        ModelInfo {
            name: ModelName::from("test-model"),
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
    fn test_generate_test_texts() {
        let texts = generate_test_texts(10);
        assert_eq!(texts.len(), 10);
        assert!(!texts[0].is_empty());

        // Should cycle through sample texts
        let sample_count = 8; // Number of sample texts in generate_test_texts
        assert_eq!(texts[0], texts[sample_count]);
    }

    #[test]
    fn test_average_benchmark_results() {
        let results = vec![
            BenchmarkResult {
                model: "test".to_string(),
                texts_per_second: 40.0,
                avg_latency_ms: 25.0,
                model_load_time_s: 2.0,
                peak_memory_mb: 100.0,
                cache_efficiency: 70.0,
            },
            BenchmarkResult {
                model: "test".to_string(),
                texts_per_second: 60.0,
                avg_latency_ms: 15.0,
                model_load_time_s: 3.0,
                peak_memory_mb: 120.0,
                cache_efficiency: 80.0,
            },
        ];

        let avg = average_benchmark_results("test", results);
        assert_eq!(avg.texts_per_second, 50.0);
        assert_eq!(avg.avg_latency_ms, 20.0);
        assert_eq!(avg.model_load_time_s, 2.5);
        assert_eq!(avg.peak_memory_mb, 110.0);
        assert_eq!(avg.cache_efficiency, 75.0);
    }

    #[test]
    fn test_load_sample_texts_from_content() {
        // Test the logic that would be used in load_sample_texts
        let content = "line 1\nline 2\n\nline 4\n  \nline 6";
        let lines: Vec<String> = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .take(5)
            .map(|s| s.to_string())
            .collect();

        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0], "line 1");
        assert_eq!(lines[1], "line 2");
        assert_eq!(lines[2], "line 4");
        assert_eq!(lines[3], "line 6");
    }
}
