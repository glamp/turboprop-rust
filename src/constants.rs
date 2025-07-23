//! Constants used throughout the TurboProp codebase.
//!
//! This module centralizes all magic numbers and constants to improve maintainability
//! and make the codebase more readable.

/// GGUF file format validation constants
pub mod gguf {
    /// Minimum file size for a valid GGUF file (magic + version + metadata header)
    pub const MINIMUM_FILE_SIZE_BYTES: u64 = 12;

    /// Size of the GGUF magic header in bytes
    pub const MAGIC_HEADER_SIZE_BYTES: usize = 4;

    /// Size of the GGUF version field in bytes
    pub const VERSION_FIELD_SIZE_BYTES: usize = 4;
}

/// Text processing and tokenization constants
pub mod text {
    /// Rough estimate of characters per token for length validation
    /// This is used for approximating token counts from character counts
    pub const CHARS_PER_TOKEN_ESTIMATE: usize = 4;

    /// Maximum length for text snippets in error messages to prevent log spam
    pub const ERROR_MESSAGE_TEXT_PREVIEW_LENGTH: usize = 50;

    /// Length threshold below which texts are considered "short" for batch optimization
    pub const SHORT_TEXT_LENGTH_THRESHOLD: usize = 100;

    /// Length threshold above which texts are considered "long" for batch optimization
    pub const LONG_TEXT_LENGTH_THRESHOLD: usize = 1000;

    /// Default threshold for batch size warnings
    pub const BATCH_SIZE_WARNING_THRESHOLD: usize = 1000;
}

/// Memory management and batch processing constants
pub mod memory {
    /// Available memory threshold (in GB) considered "high" for batch scaling
    pub const HIGH_MEMORY_THRESHOLD_GB: u64 = 8;

    /// Available memory threshold (in GB) considered "medium" for batch scaling  
    pub const MEDIUM_MEMORY_THRESHOLD_GB: u64 = 4;

    /// Memory scaling factor for systems with high available memory
    pub const HIGH_MEMORY_SCALING_FACTOR: f64 = 2.0;

    /// Memory scaling factor for systems with medium available memory
    pub const MEDIUM_MEMORY_SCALING_FACTOR: f64 = 1.5;

    /// Memory scaling factor for systems with limited available memory
    pub const LOW_MEMORY_SCALING_FACTOR: f64 = 0.75;

    /// Conversion factor from KB to GB (1024 * 1024)
    pub const KB_TO_GB_DIVISOR: u64 = 1024 * 1024;

    /// Conversion factor from seconds to milliseconds
    pub const SECONDS_TO_MILLISECONDS: f64 = 1000.0;
}

/// Batch processing size limits
pub mod batch {
    /// Maximum batch size for memory-constrained environments
    pub const CONSERVATIVE_MAX_SIZE: usize = 128;

    /// Maximum batch size for high-memory environments
    pub const MAX_SIZE: usize = 256;

    /// Minimum batch size (always at least 1)
    pub const MIN_SIZE: usize = 1;
}

/// Test and placeholder constants
pub mod test {
    /// Default model size in bytes for test scenarios
    pub const DEFAULT_MODEL_SIZE_BYTES: u64 = 1000;

    /// Text hash modulo for creating variation in test embeddings
    pub const TEXT_HASH_MODULO: usize = 100;

    /// Base value for test embedding generation
    pub const TEST_EMBEDDING_BASE_VALUE: f32 = 0.1;

    /// Scaling factor for test embedding variation
    pub const TEST_EMBEDDING_VARIATION_FACTOR: f32 = 0.001;
}

/// Configuration defaults for threading and GPU
pub mod config {
    /// Default number of GPU layers for GGUF models
    pub const DEFAULT_GPU_LAYERS: u32 = 32;

    /// Default number of CPU threads for GGUF models
    pub const DEFAULT_CPU_THREADS: u32 = 8;

    /// Default context length for GGUF models
    pub const DEFAULT_CONTEXT_LENGTH: usize = 256;

    /// Default maximum sequence length for testing
    pub const TEST_MAX_SEQUENCE_LENGTH: usize = 50;
}

/// Model configuration defaults for HuggingFace models
pub mod model_config {
    /// Default maximum position embeddings for transformer models
    pub const DEFAULT_MAX_POSITION_EMBEDDINGS: u64 = 2048;

    /// Default RoPE theta value for positional encoding
    pub const DEFAULT_ROPE_THETA: f64 = 10000.0;

    /// Default RMS normalization epsilon value
    pub const DEFAULT_RMS_NORM_EPS: f64 = 1e-6;
}

pub mod logging {
    //! # TurboProp Logging Guidelines
    //!
    //! This module defines when to use each logging level for consistent logging throughout the codebase.
    //!
    //! ## Logging Level Usage:
    //!
    //! ### ERROR - `error!`
    //! - Fatal errors that prevent operation completion
    //! - Data corruption or consistency violations
    //! - External service failures (network, filesystem)
    //! - Configuration errors that prevent startup
    //! - Examples: Model loading failures, index corruption, network timeouts
    //!
    //! ### WARN - `warn!`
    //! - Recoverable errors or fallback scenarios
    //! - Deprecated features or configurations
    //! - Performance degradation conditions
    //! - Missing optional features or configurations
    //! - Examples: Fallback to alternate model format, missing cache directory (created automatically)
    //!
    //! ### INFO - `info!`
    //! - High-level application flow and state changes
    //! - Major operation start/completion (model loading, indexing, searching)
    //! - Configuration decisions and backend selection
    //! - User-visible progress and results
    //! - Examples: "Loading embedding model", "Generated 150 embeddings", "Using FastEmbed backend"
    //!
    //! ### DEBUG - `debug!`
    //! - Detailed execution flow and internal state
    //! - Performance metrics and timing information
    //! - Batch processing details and intermediate results
    //! - Internal algorithm decisions and calculations
    //! - Examples: "Processing batch 3/10", "Calculated optimal batch size: 64", "Memory scaling factor: 1.5"
    //!
    //! ### TRACE - `trace!` (rarely used)
    //! - Extremely detailed debugging information
    //! - Function entry/exit with parameters
    //! - Individual data transformations
    //! - Only for complex debugging scenarios
}
