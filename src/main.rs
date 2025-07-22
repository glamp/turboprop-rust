use clap::Parser;
use tp::cli::{Cli, Commands};
use tp::config::{CliConfigOverrides, TurboPropConfig};
use tp::types::parse_filesize;
use tp::{index_files_with_config, search_with_config};

/// Default content preview length for search results in CLI
const DEFAULT_CONTENT_PREVIEW_LENGTH: usize = 80;

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

            // Call the new config-aware indexing function
            index_files_with_config(&path, &config).await?;
        }
        Commands::Search {
            query,
            repo,
            limit,
            threshold,
        } => {
            // Validate threshold if provided
            if let Some(t) = threshold {
                if !(0.0..=1.0).contains(&t) {
                    anyhow::bail!("Threshold must be between 0.0 and 1.0, got: {}", t);
                }
            }

            // Perform the search with the new parameters
            let results = search_with_config(&query, &repo, Some(limit), threshold).await?;

            // Display the results
            if results.is_empty() {
                println!("No results found for query: '{}'", query);
                if let Some(threshold) = threshold {
                    println!(
                        "Try lowering the similarity threshold (currently {:.3})",
                        threshold
                    );
                }
            } else {
                println!("Found {} results for query: '{}'", results.len(), query);
                if let Some(threshold) = threshold {
                    println!("(minimum similarity: {:.3})", threshold);
                }
                println!();

                for (i, result) in results.iter().enumerate() {
                    println!(
                        "{:2}. {} (similarity: {:.3})",
                        i + 1,
                        result.location_display(),
                        result.similarity
                    );

                    // Show content preview
                    let preview = result.content_preview(DEFAULT_CONTENT_PREVIEW_LENGTH);
                    let preview_lines: Vec<&str> = preview.lines().take(2).collect();
                    for line in preview_lines {
                        println!("    {}", line.trim());
                    }
                    if preview.lines().count() > 2 {
                        println!("    ...");
                    }
                    println!();
                }

                println!("Search completed successfully");
            }
        }
    }

    Ok(())
}
