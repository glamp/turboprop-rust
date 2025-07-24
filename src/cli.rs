use clap::{Args, Parser, Subcommand};
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

    /// model management commands
    Model {
        #[command(subcommand)]
        action: ModelCommands,
    },

    /// Benchmark embedding models for performance comparison
    Benchmark {
        /// Models to benchmark (default: all available)
        #[arg(long, value_delimiter = ',')]
        models: Option<Vec<String>>,
        /// Number of texts to process for benchmark
        #[arg(long, default_value = "100")]
        text_count: usize,
        /// Number of benchmark iterations
        #[arg(long, default_value = "3")]
        iterations: usize,
        /// Sample text file to use for benchmarking
        #[arg(long)]
        sample_file: Option<std::path::PathBuf>,
        /// Output format (table, json, csv)
        #[arg(long, default_value = "table")]
        format: String,
    },

    /// Start MCP server for real-time semantic search
    /// 
    /// The MCP (Model Context Protocol) server enables integration with coding
    /// agents like Claude Code, GitHub Copilot, Cursor, and Windsurf. It provides
    /// real-time semantic search capabilities over your codebase.
    Mcp(McpArgs),
}

#[derive(Debug, Subcommand)]
pub enum ModelCommands {
    /// List all available embedding models
    List,
    /// Download a specific model
    Download {
        /// Model name to download
        model: String,
    },
    /// Show detailed information about a specific model
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

/// Arguments for the MCP server command
#[derive(Debug, Args)]
pub struct McpArgs {
    /// Repository path to index and watch
    /// 
    /// The MCP server will index all files in this repository and watch for
    /// changes to keep the index up-to-date.
    #[arg(long, default_value = ".", value_name = "PATH")]
    pub repo: PathBuf,
    
    /// Embedding model to use for semantic search
    /// 
    /// Overrides the model specified in the configuration file. Use 'tp model list'
    /// to see available models.
    #[arg(long, value_name = "MODEL")]
    pub model: Option<String>,
    
    /// Maximum file size to index
    /// 
    /// Files larger than this size will be skipped during indexing.
    /// Examples: "1mb", "2.5MB", "500kb"
    #[arg(long, value_name = "SIZE")]
    pub max_filesize: Option<String>,
    
    /// Only index files matching this glob pattern
    /// 
    /// Examples: "*.rs", "src/**/*.js", "**/*.py"
    #[arg(long, value_name = "PATTERN")]
    pub filter: Option<String>,
    
    /// Only index files of this type
    /// 
    /// Examples: "rust", "javascript", "python"
    #[arg(long, value_name = "TYPE")]  
    pub filetype: Option<String>,
    
    /// Force rebuild of the index even if it exists
    /// 
    /// Useful when changing models or after major configuration changes.
    #[arg(long)]
    pub force_rebuild: bool,
    
    /// Enable verbose logging
    /// 
    /// Logs are written to stderr to avoid interfering with MCP protocol
    /// messages on stdout.
    #[arg(short, long)]
    pub verbose: bool,
    
    /// Show additional debug information
    /// 
    /// Enables debug-level logging for troubleshooting MCP server issues.
    #[arg(long)]
    pub debug: bool,
}

impl McpArgs {
    /// Validate MCP command arguments
    pub fn validate(&self) -> Result<(), String> {
        // Validate repository path
        if !self.repo.exists() {
            return Err(format!("Repository path does not exist: {}", self.repo.display()));
        }
        
        if !self.repo.is_dir() {
            return Err(format!("Repository path is not a directory: {}", self.repo.display()));
        }
        
        // Note: More detailed validation is done in the command handler
        // to avoid dependency issues in CLI module
        
        Ok(())
    }
}
