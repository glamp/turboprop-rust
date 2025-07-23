use clap::{Parser, Subcommand};
use std::path::PathBuf;

// Default path for indexing when no path is specified
const DEFAULT_INDEX_PATH: &str = ".";

#[derive(Debug, Parser)]
#[command(name = "tp")]
#[command(about = "TurboProp - Semantic code search and indexing")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Embedding model to use
    #[arg(long, value_name = "MODEL")]
    pub model: Option<String>,

    /// Model-specific instruction (for Qwen3 models)
    #[arg(long, value_name = "INSTRUCTION")]
    pub instruction: Option<String>,

    /// Force model download even if cached
    #[arg(long)]
    pub force_download: bool,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Build or update the search index
    Index {
        /// Repository path to index
        #[arg(long, default_value = DEFAULT_INDEX_PATH)]
        repo: PathBuf,
        /// Maximum file size to index (e.g., "2mb", "100kb", "1gb")
        #[arg(short, long)]
        max_filesize: Option<String>,
        /// Embedding model to use (overrides global setting)
        #[arg(
            long,
            value_name = "MODEL",
            help = r#"Embedding model to use. Available options:
    - sentence-transformers/all-MiniLM-L6-v2 (default, fast)
    - sentence-transformers/all-MiniLM-L12-v2 (better accuracy)
    - nomic-embed-code.Q5_K_S.gguf (specialized for code)
    - Qwen/Qwen3-Embedding-0.6B (multilingual, instruction-capable)
Use 'tp model list' to see all available models"#
        )]
        model: Option<String>,
        /// Model instruction for context-aware embeddings
        #[arg(
            long,
            value_name = "INSTRUCTION",
            help = r#"Instruction for context-aware embeddings (Qwen3 only).
Examples:
    --instruction "Represent this code for search"
    --instruction "Encode this text for similarity search""#
        )]
        instruction: Option<String>,
        /// Cache directory for models and data
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,
        /// Number of worker threads for processing
        #[arg(long)]
        worker_threads: Option<usize>,
        /// Batch size for embedding generation
        #[arg(long, default_value = "32")]
        batch_size: usize,
        /// Watch for file changes and update index automatically
        #[arg(short, long)]
        watch: bool,
    },
    /// Search the indexed repository
    Search {
        /// Search query
        query: String,
        /// Repository path to search in
        #[arg(long, default_value = DEFAULT_INDEX_PATH)]
        repo: PathBuf,
        /// Embedding model to use (overrides global setting)
        #[arg(long, value_name = "MODEL")]
        model: Option<String>,
        /// Model instruction for query embeddings
        #[arg(long, value_name = "INSTRUCTION")]
        instruction: Option<String>,
        /// Maximum number of results to return (default: 10)
        #[arg(short, long, default_value = "10")]
        limit: usize,
        /// Minimum similarity threshold (0.0 to 1.0)
        #[arg(short, long)]
        threshold: Option<f32>,
        /// Output format: 'json' (default) or 'text'
        #[arg(long, default_value = "json")]
        output: String,
        /// Filter results by file extension (e.g., '.rs', '.js', '.py')
        #[arg(long)]
        filetype: Option<String>,
        /// Filter results by file glob pattern.
        ///
        /// Glob patterns use Unix shell-style wildcards to match file paths.
        ///
        /// Basic wildcards: * (any chars in dir), ? (single char), ** (recursive),
        /// [abc] (char set), [!abc] (not in set).
        ///
        /// Examples: "*.rs" (all Rust files), "src/*.js" (JS in src only),
        /// "**/*.py" (Python anywhere), "tests/**/*_test.rs" (test files),
        /// "*.{js,ts,jsx,tsx}" (JS/TS files), "src/**/handlers/*.rs" (handlers).
        ///
        /// Notes: Case-sensitive, matches full path, use forward slashes,
        /// can combine with --filetype. See 'tp search --help' for more details.
        #[arg(long)]
        filter: Option<String>,
    },

    /// Model management commands
    Model {
        #[command(subcommand)]
        action: ModelCommands,
    },
}

#[derive(Debug, Subcommand)]
pub enum ModelCommands {
    /// List available models
    List,
    /// Download a specific model
    Download {
        /// Model name to download
        model: String,
    },
    /// Show model information
    Info {
        /// Model name
        model: String,
    },
    /// Clear model cache
    Clear {
        /// Specific model to clear (optional)
        model: Option<String>,
    },
}
