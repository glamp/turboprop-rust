use clap::{Parser, Subcommand};
use std::path::PathBuf;

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
        /// Path to index
        #[arg(short, long, default_value = crate::DEFAULT_INDEX_PATH)]
        path: PathBuf,
        /// Maximum file size to index (e.g., "2mb", "100kb", "1gb")
        #[arg(short, long)]
        max_filesize: Option<String>,
    },
    /// Search indexed files
    Search {
        /// Search query
        query: String,
    },
}
