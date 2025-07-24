//! Performance benchmarks for model management and cache operations.
//!
//! This module provides benchmarks to measure performance characteristics
//! of model management operations, cache operations, and text processing
//! without requiring actual model loading.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tempfile::TempDir;
use turboprop::models::ModelManager;
use turboprop::types::ModelName;

/// Benchmark configuration constants
mod bench_config {
    pub const SAMPLE_SIZE: usize = 100;
    #[allow(dead_code)]
    pub const WARMUP_TIME: std::time::Duration = std::time::Duration::from_secs(3);
    #[allow(dead_code)]
    pub const MEASUREMENT_TIME: std::time::Duration = std::time::Duration::from_secs(10);
}

/// Generate test model names for benchmarking
fn generate_test_model_names(count: usize) -> Vec<ModelName> {
    let prefixes = [
        "microsoft",
        "sentence-transformers",
        "Qwen",
        "nomic-ai",
        "google",
    ];
    let suffixes = ["model", "embedding", "transformer", "bert", "code"];

    (0..count)
        .map(|i| {
            let prefix = &prefixes[i % prefixes.len()];
            let suffix = &suffixes[i % suffixes.len()];
            ModelName::from(format!("{}/{}-{}", prefix, suffix, i))
        })
        .collect()
}

/// Benchmark model loading and validation
fn bench_model_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_loading");
    group.sample_size(bench_config::SAMPLE_SIZE);

    let manager = ModelManager::default();
    let models = manager.get_available_models();

    // Benchmark model info validation
    group.bench_function("model_validation", |b| {
        b.iter(|| {
            for model in models.iter().take(10) {
                let _result = model.validate();
                black_box(&model.name);
            }
        })
    });

    // Benchmark model lookup by name
    group.bench_function("model_lookup", |b| {
        let test_names: Vec<_> = models.iter().take(10).map(|m| &m.name).collect();

        b.iter(|| {
            for name in &test_names {
                let found = models.iter().find(|m| &m.name == *name);
                black_box(found);
            }
        })
    });

    group.finish();
}

/// Benchmark cache operations and model management
fn bench_cache_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_operations");
    group.sample_size(bench_config::SAMPLE_SIZE);

    // Benchmark model path generation
    group.bench_function("model_path_generation", |b| {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new_with_defaults(temp_dir.path());
        let model_names = generate_test_model_names(100);

        b.iter(|| {
            for model_name in &model_names {
                let _path = manager.get_model_path(black_box(model_name));
            }
        })
    });

    // Benchmark cache status checking
    group.bench_function("cache_status_check", |b| {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new_with_defaults(temp_dir.path());
        manager.init_cache().unwrap();

        let model_names = generate_test_model_names(100);

        b.iter(|| {
            for model_name in &model_names {
                let _is_cached = manager.is_model_cached(black_box(model_name));
            }
        })
    });

    // Benchmark cache statistics calculation
    group.bench_function("cache_stats", |b| {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new_with_defaults(temp_dir.path());
        manager.init_cache().unwrap();

        // Create some fake cache entries
        for i in 0..20 {
            let model_path = manager.get_model_path(&ModelName::from(format!("test/model-{}", i)));
            std::fs::create_dir_all(&model_path).unwrap();
            std::fs::write(model_path.join("config.json"), "{}").unwrap();
            std::fs::write(model_path.join("model.bin"), vec![0u8; 1024 * (i + 1)]).unwrap();
        }

        b.iter(|| {
            let _stats = manager.get_cache_stats().unwrap();
        })
    });

    // Benchmark cache initialization
    group.bench_function("cache_init", |b| {
        b.iter_with_setup(
            || TempDir::new().unwrap(),
            |temp_dir| {
                let manager = ModelManager::new_with_defaults(temp_dir.path());
                let _result = manager.init_cache();
                black_box(manager);
            },
        )
    });

    group.finish();
}

/// Benchmark text processing and validation operations
fn bench_text_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_processing");
    group.sample_size(bench_config::SAMPLE_SIZE);

    // Generate test texts of various sizes
    let short_texts: Vec<String> = (0..100)
        .map(|i| format!("Short test text number {}", i))
        .collect();

    let medium_texts: Vec<String> = (0..100)
        .map(|i| format!("This is a medium length test text that contains more content and represents typical document content for embedding generation. Text number: {}", i))
        .collect();

    let long_texts: Vec<String> = (0..100)
        .map(|i| {
            let content = "This is a very long text that simulates processing large documents or code files. ".repeat(10);
            format!("{} Document number: {}", content, i)
        })
        .collect();

    // Benchmark text length calculation
    group.bench_with_input(
        BenchmarkId::new("text_length", "short"),
        &short_texts,
        |b, texts| {
            b.iter(|| {
                let total_length: usize = texts.iter().map(|t| t.len()).sum();
                black_box(total_length)
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("text_length", "medium"),
        &medium_texts,
        |b, texts| {
            b.iter(|| {
                let total_length: usize = texts.iter().map(|t| t.len()).sum();
                black_box(total_length)
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("text_length", "long"),
        &long_texts,
        |b, texts| {
            b.iter(|| {
                let total_length: usize = texts.iter().map(|t| t.len()).sum();
                black_box(total_length)
            })
        },
    );

    group.finish();
}

/// Benchmark model name processing and validation
fn bench_model_name_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_name_operations");
    group.sample_size(bench_config::SAMPLE_SIZE);

    let model_names = generate_test_model_names(1000);

    // Benchmark model name creation
    group.bench_function("model_name_creation", |b| {
        let name_strings: Vec<String> = (0..1000).map(|i| format!("org/model-{}", i)).collect();

        b.iter(|| {
            for name_str in &name_strings {
                let _model_name = ModelName::from(black_box(name_str.as_str()));
            }
        })
    });

    // Benchmark model name string operations
    group.bench_function("model_name_string_ops", |b| {
        b.iter(|| {
            for model_name in &model_names {
                let _as_str = model_name.as_str();
                let _to_string = model_name.to_string();
                black_box((_as_str, _to_string));
            }
        })
    });

    // Benchmark model name comparison
    group.bench_function("model_name_comparison", |b| {
        let first_half = &model_names[..500];
        let second_half = &model_names[500..];

        b.iter(|| {
            let mut matches = 0;
            for (name1, name2) in first_half.iter().zip(second_half.iter()) {
                if name1 == name2 {
                    matches += 1;
                }
            }
            black_box(matches)
        })
    });

    group.finish();
}

/// Benchmark file system operations related to model management
fn bench_filesystem_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("filesystem_operations");
    group.sample_size(50); // Fewer samples for I/O operations

    // Benchmark directory creation and cleanup
    group.bench_function("directory_operations", |b| {
        b.iter_with_setup(
            || TempDir::new().unwrap(),
            |temp_dir| {
                let manager = ModelManager::new_with_defaults(temp_dir.path());
                let model_names = generate_test_model_names(10);

                // Create model directories
                for model_name in &model_names {
                    let model_path = manager.get_model_path(model_name);
                    std::fs::create_dir_all(&model_path).unwrap();
                    std::fs::write(model_path.join("config.json"), "{}").unwrap();
                }

                // Check if models are cached
                let mut cached_count = 0;
                for model_name in &model_names {
                    if manager.is_model_cached(model_name) {
                        cached_count += 1;
                    }
                }

                black_box(cached_count);
            },
        )
    });

    group.finish();
}

// Define benchmark groups
criterion_group!(
    benches,
    bench_model_loading,
    bench_cache_operations,
    bench_text_processing,
    bench_model_name_operations,
    bench_filesystem_operations,
);

criterion_main!(benches);
