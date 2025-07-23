//! Performance benchmarks for glob pattern filtering functionality.
//!
//! This module provides comprehensive benchmarks to measure glob pattern
//! compilation, matching performance, and memory usage during filtering operations.
//! These benchmarks validate that glob filtering adds minimal overhead to search operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;
use turboprop::filters::{FilterConfig, GlobPattern, GlobPatternCache, SearchFilter};
use turboprop::types::{
    ChunkId, ChunkIndexNum, ContentChunk, IndexedChunk, SearchResult, SourceLocation, TokenCount,
};

/// Benchmark configuration constants
mod bench_config {
    use std::time::Duration;

    // Pattern complexity levels
    pub const SIMPLE_PATTERNS: [&str; 4] = ["*.rs", "*.js", "*.py", "*.md"];
    pub const DIRECTORY_PATTERNS: [&str; 4] = ["src/*.rs", "tests/*.js", "docs/*.md", "lib/*.py"];
    pub const RECURSIVE_PATTERNS: [&str; 4] = [
        "**/*.rs",
        "**/test_*.js",
        "src/**/*.py",
        "**/docs/**/*.md",
    ];
    pub const COMPLEX_PATTERNS: [&str; 4] = [
        "src/**/test_*.{rs,js}",
        "**/modules/**/handlers/*.rs",
        "tests/**/*_{test,spec}.{js,ts}",
        "**/src/**/*.{py,pyx,pyi}",
    ];

    // File set sizes for testing
    pub const SMALL_FILE_SET: usize = 100;
    pub const MEDIUM_FILE_SET: usize = 1000;
    pub const LARGE_FILE_SET: usize = 10000;
    pub const FILE_SET_SIZES: [usize; 3] = [SMALL_FILE_SET, MEDIUM_FILE_SET, LARGE_FILE_SET];

    // Pattern compilation test sizes
    pub const PATTERN_COMPILATION_COUNTS: [usize; 4] = [10, 100, 500, 1000];

    // Search result set sizes
    pub const RESULT_SET_SIZES: [usize; 4] = [100, 500, 1000, 5000];

    // Cache performance test sizes
    pub const CACHE_SIZES: [usize; 3] = [100, 500, 1000];

    // Benchmark execution configurations
    pub const MEMORY_INTENSIVE_SAMPLE_SIZE: usize = 10;
    pub const MEASUREMENT_TIME_SECONDS: Duration = Duration::from_secs(15);

    // Test data configurations
    pub const LINES_PER_CHUNK: usize = 20;
    pub const TOKEN_COUNT_DEFAULT: usize = 15;
    pub const EMBEDDING_DIMENSION: usize = 384;
    pub const DEFAULT_SIMILARITY: f32 = 0.8;

    // Performance thresholds
    pub const MAX_OVERHEAD_PERCENT: f64 = 10.0; // Max 10% overhead target
}

/// Create a realistic test codebase with various file types and directory structures
fn create_realistic_codebase(file_count: usize) -> TempDir {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let base_path = temp_dir.path();

    // Create directory structure
    let dirs = [
        "src",
        "src/auth",
        "src/api",
        "src/models",
        "src/utils",
        "tests",
        "tests/unit",
        "tests/integration",
        "docs",
        "scripts",
        "config",
        "lib",
        "lib/external",
        "examples",
    ];

    for dir in &dirs {
        fs::create_dir_all(base_path.join(dir)).expect("Failed to create directory");
    }

    // File templates for different types
    let templates = vec![
        // Rust files
        (
            "rs",
            r#"//! Module {} documentation
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct {}Entity {{
    pub id: u64,
    pub name: String,
    pub metadata: HashMap<String, String>,
}}

impl {}Entity {{
    pub fn new(id: u64, name: impl Into<String>) -> Self {{
        Self {{
            id,
            name: name.into(),
            metadata: HashMap::new(),
        }}
    }}

    pub fn authenticate(&self, credentials: &str) -> Result<bool> {{
        // Mock authentication logic
        Ok(!credentials.is_empty())
    }}

    pub fn process_data(&self) -> Result<String> {{
        Ok(format!("Processing data for entity: {{}}", self.name))
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_{}_creation() {{
        let entity = {}Entity::new(1, "test");
        assert_eq!(entity.id, 1);
    }}
}}
"#,
        ),
        // JavaScript files
        (
            "js",
            r#"/**
 * {} module for handling authentication
 */

const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');

class {}Service {{
    constructor(config) {{
        this.config = config;
        this.cache = new Map();
    }}

    async authenticate(username, password) {{
        try {{
            const user = await this.findUser(username);
            if (!user) {{
                return {{ success: false, error: 'User not found' }};
            }}

            const isValid = await bcrypt.compare(password, user.passwordHash);
            if (!isValid) {{
                return {{ success: false, error: 'Invalid credentials' }};
            }}

            const token = this.generateToken(user);
            return {{ success: true, token, user }};
        }} catch (error) {{
            return {{ success: false, error: error.message }};
        }}
    }}

    generateToken(user) {{
        return jwt.sign(
            {{ userId: user.id, username: user.username }},
            this.config.jwtSecret,
            {{ expiresIn: '24h' }}
        );
    }}

    async findUser(username) {{
        // Mock user lookup
        return {{
            id: 1,
            username,
            passwordHash: 'hashed_password',
            email: `${{username}}@example.com`
        }};
    }}
}}

module.exports = {}Service;
"#,
        ),
        // Python files
        (
            "py",
            r#"""
{} module for data processing and authentication.
"""

import hashlib
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class {}Entity:
    id: int
    name: str
    email: str
    metadata: Dict[str, str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {{}}

    def authenticate(self, password: str) -> bool:
        """Authenticate user with password"""
        if len(password) < 8:
            raise ValueError("Password too short")
        return True

    def to_dict(self) -> Dict:
        """Convert entity to dictionary"""
        return {{
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'metadata': self.metadata
        }}

    def process_data(self) -> str:
        """Process entity data"""
        return f"Processing data for {{self.name}}"

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_{}_database() -> Dict[int, {}Entity]:
    """Create sample database"""
    return {{
        1: {}Entity(1, "Alice", "alice@example.com"),
        2: {}Entity(2, "Bob", "bob@example.com")
    }}

if __name__ == "__main__":
    entity = {}Entity(1, "Test", "test@example.com")
    print(entity.to_dict())
"#,
        ),
        // TypeScript files
        (
            "ts",
            r#"/**
 * {} interface and implementation
 */

interface {}Entity {{
    id: number;
    name: string;
    email: string;
    metadata: Record<string, string>;
}}

interface AuthenticationResult {{
    success: boolean;
    user?: {}Entity;
    token?: string;
    error?: string;
}}

class {}Manager {{
    private users: Map<number, {}Entity> = new Map();
    private config: {{ jwtSecret: string }};

    constructor(config: {{ jwtSecret: string }}) {{
        this.config = config;
    }}

    async authenticate(id: number, password: string): Promise<AuthenticationResult> {{
        const user = this.users.get(id);
        if (!user) {{
            return {{ success: false, error: 'User not found' }};
        }}

        // Mock password validation
        if (password.length < 8) {{
            return {{ success: false, error: 'Invalid password' }};
        }}

        const token = this.generateJWT(user);
        return {{ success: true, user, token }};
    }}

    private generateJWT(user: {}Entity): string {{
        // Mock JWT generation
        return `jwt_${{user.id}}_${{Date.now()}}`;
    }}

    addUser(user: {}Entity): void {{
        this.users.set(user.id, user);
    }}

    getUser(id: number): {}Entity | undefined {{
        return this.users.get(id);
    }}
}}

export {{ {}Entity, AuthenticationResult, {}Manager }};
"#,
        ),
    ];

    // Generate files with realistic distribution
    let mut file_counter = 0;
    for i in 0..file_count {
        if file_counter >= file_count {
            break;
        }

        let (ext, template) = &templates[i % templates.len()];
        let module_name = format!("Module{}", i);
        let entity_name = format!("Test{}", i);

        // Choose directory based on file type and counter
        let dir = match *ext {
            "rs" => match i % 6 {
                0..=2 => "src",
                3 => "src/auth",
                4 => "src/api", 
                _ => "tests",
            },
            "js" => match i % 4 {
                0..=1 => "src",
                2 => "scripts",
                _ => "tests/integration",
            },
            "py" => match i % 3 {
                0 => "lib",
                1 => "scripts",
                _ => "tests/unit",
            },
            "ts" => match i % 3 {
                0 => "src",
                1 => "src/models",
                _ => "tests",
            },
            _ => "src",
        };

        let filename = match *ext {
            "rs" => format!("{}_test.rs", module_name.to_lowercase()),
            "js" => format!("{}.js", module_name.to_lowercase()),
            "py" => format!("{}.py", module_name.to_lowercase()),
            "ts" => format!("{}.ts", module_name.to_lowercase()),
            _ => format!("{}.{}", module_name.to_lowercase(), ext),
        };

        let content = template
            .replace("{}", &module_name)
            .replace("{}", &entity_name);

        let file_path = base_path.join(dir).join(filename);
        fs::write(&file_path, content).expect("Failed to write test file");
        file_counter += 1;
    }

    // Add some markdown and config files
    let additional_files = [
        ("README.md", "# Test Project\n\nThis is a test project for benchmarking glob patterns.\n\n## Authentication\n\nThe project includes authentication modules."),
        ("CHANGELOG.md", "# Changelog\n\n## Version 1.0.0\n- Initial release\n- Added authentication\n- Added user management"),
        ("config/app.json", r#"{"name": "test-app", "version": "1.0.0", "auth_enabled": true}"#),
        ("config/database.yaml", "host: localhost\nport: 5432\ndatabase: testdb\nssl: true"),
    ];

    for (file_path, content) in &additional_files {
        let full_path = base_path.join(file_path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent).ok();
        }
        fs::write(&full_path, content).expect("Failed to write additional file");
    }

    temp_dir
}

/// Create test search results for benchmarking filtering
fn create_test_search_results(count: usize, embedding_dim: usize) -> Vec<SearchResult> {
    use rand::prelude::*;
    let mut rng = thread_rng();

    let file_extensions = ["rs", "js", "py", "ts", "md", "json", "yaml"];
    let directories = ["src", "tests", "docs", "config", "lib", "scripts"];

    (0..count)
        .map(|i| {
            let ext = file_extensions[i % file_extensions.len()];
            let dir = directories[i % directories.len()];
            let filename = format!("file_{}.{}", i, ext);
            let filepath = PathBuf::from(dir).join(filename);

            let embedding: Vec<f32> = (0..embedding_dim)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();

            let chunk = ContentChunk {
                id: ChunkId::new(&format!("chunk-{}", i)),
                content: format!("test content for file {}", i),
                token_count: TokenCount::new(bench_config::TOKEN_COUNT_DEFAULT),
                source_location: SourceLocation {
                    file_path: filepath,
                    start_line: 1,
                    end_line: bench_config::LINES_PER_CHUNK,
                    start_char: 0,
                    end_char: 100,
                },
                chunk_index: ChunkIndexNum::new(0),
                total_chunks: 1,
            };

            let indexed_chunk = IndexedChunk { chunk, embedding };

            SearchResult::new(bench_config::DEFAULT_SIMILARITY, indexed_chunk, i)
        })
        .collect()
}

/// Benchmark glob pattern compilation performance
fn bench_glob_pattern_compilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("glob_pattern_compilation");

    // Test different pattern types
    let pattern_sets = [
        ("simple", bench_config::SIMPLE_PATTERNS.as_slice()),
        ("directory", bench_config::DIRECTORY_PATTERNS.as_slice()),
        ("recursive", bench_config::RECURSIVE_PATTERNS.as_slice()),
        ("complex", bench_config::COMPLEX_PATTERNS.as_slice()),
    ];

    for (pattern_type, patterns) in &pattern_sets {
        group.bench_function(
            BenchmarkId::new("pattern_compilation", pattern_type),
            |b| {
                b.iter(|| {
                    for pattern in *patterns {
                        let compiled = GlobPattern::new(pattern);
                        black_box(compiled.unwrap());
                    }
                });
            },
        );
    }

    // Test pattern compilation with different counts
    for &count in &bench_config::PATTERN_COMPILATION_COUNTS {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("bulk_compilation", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    for i in 0..count {
                        let pattern = format!("**/test_{}.rs", i);
                        let compiled = GlobPattern::new(&pattern);
                        black_box(compiled.unwrap());
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark pattern matching against file sets
fn bench_glob_pattern_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("glob_pattern_matching");

    // Create test file sets
    for &file_count in &bench_config::FILE_SET_SIZES {
        let temp_dir = create_realistic_codebase(file_count);
        let file_paths: Vec<PathBuf> = walkdir::WalkDir::new(temp_dir.path())
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.path().to_path_buf())
            .collect();

        group.throughput(Throughput::Elements(file_paths.len() as u64));

        // Test different pattern types against the file set
        let test_patterns = [
            ("simple_wildcard", "*.rs"),
            ("directory_specific", "src/*.js"),
            ("recursive", "**/*.py"),
            ("complex_nested", "**/test_*.{rs,js}"),
        ];

        for (pattern_name, pattern_str) in &test_patterns {
            let pattern = GlobPattern::new(pattern_str).unwrap();
            
            group.bench_with_input(
                BenchmarkId::new(format!("{}_{}", pattern_name, file_count), file_count),
                &file_paths,
                |b, paths| {
                    b.iter(|| {
                        let matches: Vec<&PathBuf> = paths
                            .iter()
                            .filter(|path| pattern.matches(path))
                            .collect();
                        black_box(matches.len())
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark search result filtering performance
fn bench_search_filtering_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_filtering");

    for &result_count in &bench_config::RESULT_SET_SIZES {
        let search_results = create_test_search_results(result_count, bench_config::EMBEDDING_DIMENSION);

        group.throughput(Throughput::Elements(result_count as u64));

        // Test filtering with different patterns
        let filter_tests = [
            ("no_filter", None),
            ("simple_filter", Some("*.rs")),
            ("directory_filter", Some("src/*.js")),
            ("recursive_filter", Some("**/*.py")),
            ("complex_filter", Some("**/test_*.{rs,js}")),
        ];

        for (filter_name, filter_pattern) in &filter_tests {
            group.bench_with_input(
                BenchmarkId::new(format!("{}_{}", filter_name, result_count), result_count),
                &search_results,
                |b, results| {
                    b.iter(|| {
                        let filter = SearchFilter::from_cli_args(
                            None,
                            filter_pattern.map(|s| s.to_string()),
                        );
                        let filtered = filter.apply_filters(results.clone()).unwrap();
                        black_box(filtered.len())
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark glob pattern caching performance
fn bench_glob_pattern_caching(c: &mut Criterion) {
    let mut group = c.benchmark_group("glob_pattern_caching");

    for &cache_size in &bench_config::CACHE_SIZES {
        let cache = GlobPatternCache::with_max_size(cache_size);
        let config = FilterConfig::default();

        group.throughput(Throughput::Elements(cache_size as u64));
        group.bench_with_input(
            BenchmarkId::new("cache_access", cache_size),
            &cache_size,
            |b, &cache_size| {
                b.iter(|| {
                    // Fill cache with patterns
                    for i in 0..cache_size {
                        let pattern = format!("**/file_{}.rs", i);
                        let _ = cache.get_or_create(&pattern, &config).unwrap();
                    }

                    // Access cached patterns
                    for i in 0..cache_size {
                        let pattern = format!("**/file_{}.rs", i);
                        let cached = cache.get_or_create(&pattern, &config).unwrap();
                        black_box(cached.pattern());
                    }
                });
            },
        );

        // Test cache hit ratio
        group.bench_with_input(
            BenchmarkId::new("cache_hit_ratio", cache_size),
            &cache_size,
            |b, &cache_size| {
                b.iter_with_setup(
                    || {
                        let cache = GlobPatternCache::with_max_size(cache_size);
                        // Pre-populate half the cache
                        for i in 0..cache_size / 2 {
                            let pattern = format!("**/cached_{}.rs", i);
                            let _ = cache.get_or_create(&pattern, &config).unwrap();
                        }
                        cache
                    },
                    |cache| {
                        // Access mix of cached and new patterns
                        for i in 0..cache_size {
                            let pattern = if i % 2 == 0 {
                                format!("**/cached_{}.rs", i / 2) // Cache hit
                            } else {
                                format!("**/new_{}.rs", i) // Cache miss
                            };
                            let result = cache.get_or_create(&pattern, &config).unwrap();
                            black_box(result.pattern());
                        }
                    },
                );
            },
        );
    }

    group.finish();
}

/// Benchmark filtering overhead vs baseline search
fn bench_filtering_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("filtering_overhead");
    group.measurement_time(bench_config::MEASUREMENT_TIME_SECONDS);

    let result_count = bench_config::MEDIUM_FILE_SET;
    let search_results = create_test_search_results(result_count, bench_config::EMBEDDING_DIMENSION);

    group.throughput(Throughput::Elements(result_count as u64));

    // Baseline: no filtering
    group.bench_function("baseline_no_filter", |b| {
        b.iter(|| {
            let filter = SearchFilter::from_cli_args(None, None);
            let filtered = filter.apply_filters(search_results.clone()).unwrap();
            black_box(filtered.len())
        });
    });

    // With simple glob filter
    group.bench_function("with_simple_glob", |b| {
        b.iter(|| {
            let filter = SearchFilter::from_cli_args(None, Some("*.rs".to_string()));
            let filtered = filter.apply_filters(search_results.clone()).unwrap();
            black_box(filtered.len())
        });
    });

    // With complex glob filter
    group.bench_function("with_complex_glob", |b| {
        b.iter(|| {
            let filter = SearchFilter::from_cli_args(None, Some("**/test_*.{rs,js}".to_string()));
            let filtered = filter.apply_filters(search_results.clone()).unwrap();
            black_box(filtered.len())
        });
    });

    // With very complex nested pattern
    group.bench_function("with_nested_glob", |b| {
        b.iter(|| {
            let filter = SearchFilter::from_cli_args(
                None,
                Some("src/**/modules/**/handlers/*.{rs,ts}".to_string()),
            );
            let filtered = filter.apply_filters(search_results.clone()).unwrap();
            black_box(filtered.len())
        });
    });

    group.finish();
}

/// Benchmark memory usage during glob filtering
fn bench_glob_filtering_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("glob_filtering_memory");
    group.sample_size(bench_config::MEMORY_INTENSIVE_SAMPLE_SIZE);

    let large_result_set = create_test_search_results(
        bench_config::LARGE_FILE_SET,
        bench_config::EMBEDDING_DIMENSION,
    );

    group.throughput(Throughput::Elements(bench_config::LARGE_FILE_SET as u64));

    // Test memory usage with large result sets
    group.bench_function("large_result_set_filtering", |b| {
        b.iter(|| {
            let filter = SearchFilter::from_cli_args(None, Some("**/*.rs".to_string()));
            let filtered = filter.apply_filters(large_result_set.clone()).unwrap();
            black_box(filtered.len())
        });
    });

    // Test memory usage with pattern cache under load
    group.bench_function("pattern_cache_memory", |b| {
        b.iter_with_setup(
            || {
                // Setup many different patterns to stress cache
                let patterns: Vec<String> = (0..1000)
                    .map(|i| format!("**/pattern_{}/*.rs", i))
                    .collect();
                patterns
            },
            |patterns| {
                let cache = GlobPatternCache::with_max_size(500); // Smaller than pattern count
                let config = FilterConfig::default();

                for pattern in &patterns {
                    let _ = cache.get_or_create(pattern, &config).unwrap();
                }

                black_box(cache.len())
            },
        );
    });

    group.finish();
}

// Define benchmark groups
criterion_group!(
    benches,
    bench_glob_pattern_compilation,
    bench_glob_pattern_matching,
    bench_search_filtering_performance,
    bench_glob_pattern_caching,
    bench_filtering_overhead,
    bench_glob_filtering_memory,
);

criterion_main!(benches);