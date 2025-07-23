use clap::{Parser, Subcommand};
use std::path::PathBuf;

// Default path for indexing when no path is specified
const DEFAULT_INDEX_PATH: &str = ".";

#[derive(Parser)]
#[command(name = "tp")]
#[command(about = "TurboProp - Fast code search and indexing tool")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Index files for searching
    Index {
        /// Repository path to index
        #[arg(long, default_value = DEFAULT_INDEX_PATH)]
        repo: PathBuf,
        /// Maximum file size to index (e.g., "2mb", "100kb", "1gb")
        #[arg(short, long)]
        max_filesize: Option<String>,
        /// Embedding model to use for vector generation
        #[arg(long, default_value = "sentence-transformers/all-MiniLM-L6-v2")]
        model: String,
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
    /// Search indexed files
    Search {
        /// Search query
        query: String,
        /// Repository path to search in
        #[arg(long, default_value = DEFAULT_INDEX_PATH)]
        repo: PathBuf,
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
    },
}
