# TurboProp API Reference

## Model Management API

### List Available Models
```rust
use turboprop::models::ModelManager;

let models = ModelManager::get_available_models();
for model in models {
    println!("Model: {}", model.name);
    println!("Type: {:?}", model.model_type);
    println!("Backend: {:?}", model.backend);
}
```

### Create Embedding Generator
```rust
use turboprop::embeddings::EmbeddingGenerator;
use turboprop::models::ModelManager;

// Get model info
let models = ModelManager::get_available_models();
let model_info = models.iter()
    .find(|m| m.name == "Qwen/Qwen3-Embedding-0.6B")
    .unwrap();

// Create generator
let generator = EmbeddingGenerator::new_with_model(model_info).await?;

// Generate embeddings
let texts = vec!["Hello world".to_string()];
let embeddings = generator.embed(&texts)?;
```

### Instruction-Based Embeddings
```rust
use turboprop::embeddings::{EmbeddingGenerator, EmbeddingOptions};

let options = EmbeddingOptions {
    instruction: Some("Represent this code for search".to_string()),
    normalize: true,
    max_length: None,
};

let embeddings = generator.embed_with_options(&texts, &options)?;
```

## Core Types

### ModelInfo
```rust
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub model_type: ModelType,
    pub backend: Backend,
    pub dimensions: usize,
    pub size_bytes: Option<u64>,
    pub languages: Vec<String>,
    pub features: Vec<ModelFeature>,
}
```

### EmbeddingGenerator
```rust
impl EmbeddingGenerator {
    /// Create new generator with default model
    pub async fn new() -> Result<Self, EmbeddingError> { ... }
    
    /// Create generator with specific model
    pub async fn new_with_model(model: &ModelInfo) -> Result<Self, EmbeddingError> { ... }
    
    /// Generate embeddings for text chunks
    pub async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> { ... }
    
    /// Generate embeddings with custom options
    pub async fn embed_with_options(
        &self, 
        texts: &[String], 
        options: &EmbeddingOptions
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> { ... }
}
```

### ChunkIndex
```rust
impl ChunkIndex {
    /// Create new empty index
    pub fn new() -> Self { ... }
    
    /// Add chunks to index
    pub fn add_chunks(&mut self, chunks: Vec<Chunk>) -> Result<(), IndexError> { ... }
    
    /// Search index with query
    pub fn search(
        &self, 
        query_embedding: &[f32], 
        limit: usize, 
        threshold: f32
    ) -> Result<Vec<SearchResult>, SearchError> { ... }
    
    /// Get index statistics
    pub fn stats(&self) -> IndexStats { ... }
}
```

## Configuration API

### EmbeddingConfig
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub model: Option<String>,
    pub instruction: Option<String>,
    pub batch_size: Option<usize>,
    pub normalize: Option<bool>,
    pub cache_dir: Option<PathBuf>,
    pub max_length: Option<usize>,
}
```

### ModelConfig
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub default_model: Option<String>,
    pub models: HashMap<String, ModelSpecificConfig>,
    pub embedding: Option<EmbeddingConfig>,
    pub max_memory_usage: Option<String>,
    pub warn_large_models: Option<bool>,
}
```

## Error Handling

### Error Types
```rust
#[derive(Debug, thiserror::Error)]
pub enum TurboPropError {
    #[error("Model error: {0}")]
    Model(#[from] ModelError),
    
    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),
    
    #[error("Index error: {0}")]
    Index(#[from] IndexError),
    
    #[error("Search error: {0}")]
    Search(#[from] SearchError),
    
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),
}
```

### ModelError
```rust
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Model not found: {model}")]
    NotFound { model: String },
    
    #[error("Model download failed: {reason}")]
    DownloadFailed { reason: String },
    
    #[error("Invalid model format: {details}")]
    InvalidFormat { details: String },
    
    #[error("Insufficient memory for model: requires {required_mb}MB, available {available_mb}MB")]
    InsufficientMemory { required_mb: u64, available_mb: u64 },
}
```

## Backend Integration

### FastEmbed Backend
```rust
use turboprop::backends::FastEmbedBackend;

let backend = FastEmbedBackend::new()?;
let models = backend.list_available_models().await?;
let generator = backend.create_generator("sentence-transformers/all-MiniLM-L6-v2").await?;
```

### GGUF Backend
```rust
use turboprop::backends::GGUFBackend;

let backend = GGUFBackend::new()?;
let generator = backend.create_generator("nomic-embed-code.Q5_K_S.gguf").await?;
```

### Hugging Face Backend
```rust
use turboprop::backends::HuggingFaceBackend;

let backend = HuggingFaceBackend::new()?;
let generator = backend.create_generator("Qwen/Qwen3-Embedding-0.6B").await?;

// With instruction
let options = EmbeddingOptions {
    instruction: Some("Represent this code for search".to_string()),
    ..Default::default()
};
let embeddings = generator.embed_with_options(&texts, &options).await?;
```

## Indexing Pipeline

### File Processing
```rust
use turboprop::pipeline::{IndexingPipeline, PipelineConfig};

let config = PipelineConfig {
    max_file_size: Some(2 * 1024 * 1024), // 2MB
    worker_threads: Some(4),
    batch_size: Some(32),
    ..Default::default()
};

let pipeline = IndexingPipeline::new(config)?;
let index = pipeline.index_repository("/path/to/repo").await?;
```

### Incremental Updates
```rust
use turboprop::incremental::IncrementalIndexer;

let indexer = IncrementalIndexer::new(index)?;
let updated_files = vec!["src/main.rs", "src/lib.rs"];
indexer.update_files(&updated_files).await?;
```

## Search Interface

### Basic Search
```rust
use turboprop::search::{SearchEngine, SearchOptions};

let engine = SearchEngine::new(index)?;
let results = engine.search("jwt authentication", SearchOptions::default()).await?;
```

### Advanced Search
```rust
let options = SearchOptions {
    limit: Some(20),
    threshold: Some(0.7),
    file_filter: Some("*.rs".to_string()),
    model: Some("nomic-embed-code.Q5_K_S.gguf".to_string()),
    instruction: Some("Find authentication-related code".to_string()),
};

let results = engine.search("user authentication", options).await?;
```

## Performance Monitoring

### Metrics Collection
```rust
use turboprop::metrics::{Metrics, MetricsCollector};

let collector = MetricsCollector::new();
collector.start_operation("embedding_generation");
// ... perform operation ...
collector.end_operation("embedding_generation");

let metrics = collector.get_metrics();
println!("Average embedding time: {}ms", metrics.avg_embedding_time_ms);
```

### Benchmarking
```rust
use turboprop::benchmark::{Benchmark, BenchmarkConfig};

let config = BenchmarkConfig {
    models: vec![
        "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        "nomic-embed-code.Q5_K_S.gguf".to_string(),
    ],
    text_count: 100,
    iterations: 3,
};

let benchmark = Benchmark::new(config)?;
let results = benchmark.run().await?;
```

## Utilities

### Model Validation
```rust
use turboprop::validation::ModelValidator;

let validator = ModelValidator::new();
let validation_result = validator.validate_model("Qwen/Qwen3-Embedding-0.6B").await?;

if validation_result.is_valid {
    println!("Model is valid and ready to use");
} else {
    println!("Model validation failed: {:?}", validation_result.errors);
}
```

### Cache Management
```rust
use turboprop::cache::CacheManager;

let cache = CacheManager::new("/path/to/cache")?;

// Clear all cached models
cache.clear_all().await?;

// Clear specific model
cache.clear_model("nomic-embed-code.Q5_K_S.gguf").await?;

// Get cache statistics
let stats = cache.get_stats().await?;
println!("Cache size: {} MB", stats.total_size_mb);
```

## Example: Complete Integration

```rust
use turboprop::{
    models::ModelManager,
    embeddings::EmbeddingGenerator,
    index::ChunkIndex,
    search::SearchEngine,
    config::Config,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = Config::load_from_file(".turboprop.yml")?;
    
    // Get available models
    let models = ModelManager::get_available_models();
    let model = models.iter()
        .find(|m| m.name == config.default_model.unwrap_or_default())
        .ok_or("Model not found")?;
    
    // Create embedding generator
    let generator = EmbeddingGenerator::new_with_model(model).await?;
    
    // Create index
    let mut index = ChunkIndex::new();
    
    // Add some example text
    let texts = vec![
        "function authenticate(user, password) { return validateCredentials(user, password); }".to_string(),
        "def process_payment(amount, currency): return payment_gateway.charge(amount, currency)".to_string(),
    ];
    
    // Generate embeddings
    let embeddings = generator.embed(&texts).await?;
    
    // Add to index (simplified - normally you'd create proper Chunk objects)
    // index.add_chunks(chunks)?;
    
    // Create search engine
    let engine = SearchEngine::new(index)?;
    
    // Perform search
    let results = engine.search("user authentication", Default::default()).await?;
    
    for result in results {
        println!("Score: {:.3} | File: {}", result.score, result.file_path);
        println!("{}\n", result.content.trim());
    }
    
    Ok(())
}
```