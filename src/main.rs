use clap::Parser;
use turboprop::cli::{Cli, Commands};
use turboprop::commands::{
    execute_index_command_cli, execute_search_command_cli, handle_model_command, SearchCliArgs,
};
use turboprop::config::{CliConfigOverrides, TurboPropConfig};
use turboprop::types::parse_filesize;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Index {
            repo,
            max_filesize,
            model,
            instruction,
            cache_dir,
            verbose,
            worker_threads,
            batch_size,
            watch,
        } => {
            // Load base configuration
            let mut config = TurboPropConfig::load()?;

            // Build CLI overrides
            let mut cli_overrides = CliConfigOverrides::new().with_verbose(verbose);

            // Apply global model options (command-level overrides global)
            let effective_model = model.or(cli.model);
            let effective_instruction = instruction.or(cli.instruction);

            if let Some(model_name) = effective_model {
                cli_overrides = cli_overrides.with_model(model_name);
            }

            if let Some(instruction_text) = effective_instruction {
                cli_overrides = cli_overrides.with_instruction(instruction_text);
            }

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
                use turboprop::commands::index::execute_watch_command_cli;
                execute_watch_command_cli(&repo, &config, true).await?;
            } else {
                execute_index_command_cli(&repo, &config, true).await?;
            }
        }
        Commands::Search {
            query,
            repo,
            model,
            instruction,
            limit,
            threshold,
            output,
            filetype,
            filter,
        } => {
            // Load base configuration for search command
            let config = TurboPropConfig::load()?;

            // Apply global model options (command-level overrides global)
            let effective_model = model.or(cli.model);
            let effective_instruction = instruction.or(cli.instruction);

            // Apply model and instruction overrides to configuration
            let mut config_overrides = CliConfigOverrides::new();
            if let Some(model_name) = effective_model {
                config_overrides = config_overrides.with_model(model_name);
            }
            if let Some(instruction_text) = effective_instruction {
                config_overrides = config_overrides.with_instruction(instruction_text);
            }

            // TODO: Apply config overrides to search command when supported
            let config =
                if config_overrides.model.is_some() || config_overrides.instruction.is_some() {
                    config.merge_cli_args(&config_overrides)
                } else {
                    config
                };

            // Create CLI args struct
            let args = SearchCliArgs::new(query, repo, limit, threshold, output, filetype, filter);

            // Execute the search command using the new implementation
            execute_search_command_cli(args, &config).await?;
        }
        Commands::Model { action } => {
            handle_model_command(action).await?;
        }
    }

    Ok(())
}
