use clap::{Parser, Subcommand};

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
        #[arg(short, long, default_value = ".")]
        path: String,
    },
    /// Search indexed files
    Search {
        /// Search query
        query: String,
    },
}