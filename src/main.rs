use clap::Parser;
use tp::cli::{Cli, Commands};
use tp::commands::{execute_index_command_cli, execute_search_command_cli};
use tp::config::{CliConfigOverrides, TurboPropConfig};
use tp::types::parse_filesize;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Index {
            path,
            max_filesize,
            model,
            cache_dir,
            verbose,
            worker_threads,
            batch_size,
            watch,
        } => {
            // Load base configuration
            let mut config = TurboPropConfig::load()?;

            // Build CLI overrides
            let mut cli_overrides = CliConfigOverrides::new()
                .with_model(model)
                .with_verbose(verbose);

            if let Some(cache_dir) = cache_dir {
                cli_overrides = cli_overrides.with_cache_dir(cache_dir);
            }

            if let Some(max_filesize_str) = max_filesize {
                let max_bytes = parse_filesize(&max_filesize_str)
                    .map_err(|e| anyhow::anyhow!("Failed to parse filesize: {}", e))?;
                cli_overrides = cli_overrides.with_max_filesize(max_bytes);
            }

            if let Some(threads) = worker_threads {
                cli_overrides = cli_overrides.with_worker_threads(threads);
            }

            cli_overrides = cli_overrides.with_batch_size(batch_size);

            // Merge CLI overrides with config
            config = config.merge_cli_args(&cli_overrides);

            // Validate configuration
            config.validate()?;

            // Call the enhanced index command with progress tracking
            // Show progress bars by default (true), they work fine with verbose logging
            if watch {
                // Execute the watch mode
                use tp::commands::index::execute_watch_command_cli;
                execute_watch_command_cli(&path, &config, true).await?;
            } else {
                execute_index_command_cli(&path, &config, true).await?;
            }
        }
        Commands::Search {
            query,
            repo,
            limit,
            threshold,
            output,
            filetype,
        } => {
            // Execute the search command using the new implementation
            execute_search_command_cli(query, repo, limit, threshold, output, filetype).await?;
        }
    }

    Ok(())
}
