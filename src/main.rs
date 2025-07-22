use clap::Parser;
use tp::cli::{Cli, Commands};
use tp::{index_files, search_files};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Index { path, max_filesize } => {
            index_files(&path, max_filesize.as_deref())?;
        }
        Commands::Search { query } => {
            search_files(&query)?;
        }
    }

    Ok(())
}
