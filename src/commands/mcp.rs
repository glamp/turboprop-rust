//! MCP server command implementation
//!
//! Provides the `tp mcp` command that starts an MCP server exposing
//! semantic search functionality for coding agents.

use anyhow::{Context, Result};
use std::path::Path;
use tracing::{debug, info};

use crate::cli::McpArgs;
use crate::config::TurboPropConfig;
use crate::mcp::McpServer;
use crate::types::parse_filesize;

/// Logging configuration for MCP server
#[derive(Debug, Clone)]
pub struct McpLoggingConfig {
    /// Enable verbose logging
    pub verbose: bool,
    /// Enable debug logging
    pub debug: bool,
}

impl McpLoggingConfig {
    /// Create new logging config
    pub fn new(verbose: bool, debug: bool) -> Self {
        Self { verbose, debug }
    }
}

impl From<&McpArgs> for McpLoggingConfig {
    fn from(args: &McpArgs) -> Self {
        Self::new(args.verbose, args.debug)
    }
}

/// Comprehensive validation of MCP command arguments
/// This is the single source of validation for all MCP arguments
fn validate_args(args: &McpArgs) -> Result<()> {
    // Validate repository path
    if !args.repo.exists() {
        return Err(anyhow::anyhow!(
            "Repository path does not exist: {}",
            args.repo.display()
        ));
    }

    if !args.repo.is_dir() {
        return Err(anyhow::anyhow!(
            "Repository path is not a directory: {}",
            args.repo.display()
        ));
    }

    // Validate file size format if provided
    if let Some(max_filesize) = &args.max_filesize {
        if let Err(e) = parse_filesize(max_filesize) {
            return Err(anyhow::anyhow!(
                "Invalid file size format '{}': {}",
                max_filesize,
                e
            ));
        }
    }

    // Validate glob pattern if provided
    if let Some(filter) = &args.filter {
        if let Err(e) = glob::Pattern::new(filter) {
            return Err(anyhow::anyhow!("Invalid glob pattern '{}': {}", filter, e));
        }
    }

    Ok(())
}

/// Execute the MCP server command
pub async fn execute_mcp_command(args: McpArgs) -> Result<()> {
    // Set up logging (to stderr to avoid interfering with MCP protocol)
    setup_logging(McpLoggingConfig::from(&args))
        .context("Failed to initialize logging for MCP server")?;

    info!("Starting TurboProp MCP server");
    debug!("MCP arguments: {:?}", args);

    // Comprehensive argument validation
    validate_args(&args).context("MCP command argument validation failed")?;

    // Print setup information after validation
    print_setup_info(&args.repo);

    // Load configuration from current directory or look for config in repo
    let mut config =
        if let Ok(config_path) = std::fs::canonicalize(args.repo.join(".turboprop.yml")) {
            TurboPropConfig::load_from_file(&config_path).with_context(|| {
                format!(
                    "Failed to load configuration from {}",
                    config_path.display()
                )
            })?
        } else {
            // Fall back to default configuration
            TurboPropConfig::load().with_context(|| "Failed to load default configuration")?
        };

    // Apply CLI overrides
    apply_config_overrides(&mut config, &args)
        .context("Failed to apply CLI argument overrides to configuration")?;

    // Log configuration summary
    log_config_summary(&config, &args);

    // Create and run MCP server
    let server = McpServer::new(&args.repo, &config)
        .await
        .context("Failed to create MCP server")?;

    info!(
        "MCP server starting for repository: {}",
        args.repo.display()
    );
    info!("Ready to accept connections from coding agents");
    info!("Use Ctrl+C or close stdin to shutdown");

    // Run the server (this blocks until shutdown)
    server.run().await.context("MCP server execution failed")?;

    info!("MCP server shutdown complete");
    Ok(())
}

/// Apply CLI argument overrides to configuration
fn apply_config_overrides(config: &mut TurboPropConfig, args: &McpArgs) -> Result<()> {
    // Override model if specified
    if let Some(model) = &args.model {
        config.embedding.model_name = model.clone();
    }

    // Override max file size if specified
    if let Some(max_filesize_str) = &args.max_filesize {
        let max_filesize_bytes = parse_filesize(max_filesize_str)
            .map_err(|e| anyhow::anyhow!("Invalid file size '{}': {}", max_filesize_str, e))?;
        config.file_discovery.max_filesize_bytes = Some(max_filesize_bytes);
    }

    // Check for unsupported CLI overrides and provide helpful feedback
    if let Some(filter) = &args.filter {
        return Err(anyhow::anyhow!(
            "Filter patterns (--filter '{}') are not yet supported in MCP mode; \
             use .turboprop.yml configuration file to specify filtering options",
            filter
        ));
    }

    if let Some(filetype) = &args.filetype {
        return Err(anyhow::anyhow!(
            "File type filtering (--filetype '{}') is not yet supported in MCP mode; \
             use .turboprop.yml configuration file to specify filtering options",
            filetype
        ));
    }

    if args.force_rebuild {
        return Err(anyhow::anyhow!(
            "Force rebuild (--force-rebuild) is not yet supported in MCP mode; \
             remove or recreate the index directory to force a rebuild"
        ));
    }

    Ok(())
}

/// Initialize logging for the MCP server
///
/// Currently uses the global logging configuration set up by main.rs.
/// The verbose and debug parameters are accepted for API compatibility
/// but do not currently affect the logging level.
///
/// # Future Enhancement
/// In the future, this function could be enhanced to configure MCP-specific
/// logging levels or output destinations while avoiding conflicts with the
/// global subscriber.
fn setup_logging(config: McpLoggingConfig) -> Result<()> {
    debug!(
        "MCP server using global logging configuration (verbose: {}, debug: {})",
        config.verbose, config.debug
    );
    Ok(())
}

/// Log configuration summary for debugging
fn log_config_summary(config: &TurboPropConfig, args: &McpArgs) {
    info!("Configuration summary:");
    info!("  Repository: {}", args.repo.display());
    info!("  Model: {}", config.embedding.model_name);
    info!(
        "  Max file size: {:?}",
        config.file_discovery.max_filesize_bytes
    );
    info!("  Batch size: {}", config.embedding.batch_size);

    if let Some(filter) = &args.filter {
        info!("  Filter pattern: {}", filter);
    }

    if let Some(filetype) = &args.filetype {
        info!("  File type: {}", filetype);
    }

    if args.force_rebuild {
        info!("  Force rebuild: enabled");
    }

    debug!("Full configuration: {:#?}", config);
}

/// Display helpful information for setting up MCP with coding agents
pub fn print_setup_info(repo_path: &Path) {
    eprintln!();
    eprintln!("🚀 TurboProp Semantic Search MCP Server Started");
    eprintln!("──────────────────────────────────────────────");
    eprintln!("Repository: {}", repo_path.display());
    eprintln!();
    eprintln!("To integrate with coding agents, add this to your MCP configuration:");
    eprintln!();
    eprintln!("Claude Code (.mcp.json):");
    eprintln!(
        r#"{{
  "mcpServers": {{
    "turboprop": {{
      "command": "tp",
      "args": ["mcp", "--repo", "{}"]
    }}
  }}
}}"#,
        repo_path.display()
    );
    eprintln!();
    eprintln!("Cursor (.cursor/mcp.json):");
    eprintln!(
        r#"{{
  "mcpServers": {{
    "turboprop": {{
      "command": "tp",
      "args": ["mcp", "--repo", "{}"]
    }}
  }}
}}"#,
        repo_path.display()
    );
    eprintln!();
    eprintln!("Semantic search tool parameters:");
    eprintln!("  • query (required): Natural language search query");
    eprintln!("  • limit: Maximum results (default: 10)");
    eprintln!("  • threshold: Similarity threshold (0.0-1.0)");
    eprintln!("  • filetype: File extension filter (e.g., '.rs', '.js')");
    eprintln!("  • filter: Glob pattern filter (e.g., 'src/**/*.rs')");
    eprintln!();
    eprintln!("Press Ctrl+C to stop the server");
    eprintln!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_mcp_args_validation() {
        let temp_dir = TempDir::new().unwrap();

        // Valid arguments
        let valid_args = McpArgs {
            repo: temp_dir.path().to_path_buf(),
            model: None,
            max_filesize: Some("1mb".to_string()),
            filter: Some("*.rs".to_string()),
            filetype: None,
            force_rebuild: false,
            verbose: false,
            debug: false,
        };
        assert!(validate_args(&valid_args).is_ok());

        // Invalid repository path
        let invalid_repo = McpArgs {
            repo: "/nonexistent/path".into(),
            model: None,
            max_filesize: None,
            filter: None,
            filetype: None,
            force_rebuild: false,
            verbose: false,
            debug: false,
        };
        assert!(validate_args(&invalid_repo).is_err());

        // Invalid file size
        let invalid_size = McpArgs {
            repo: temp_dir.path().to_path_buf(),
            model: None,
            max_filesize: Some("invalid_size".to_string()),
            filter: None,
            filetype: None,
            force_rebuild: false,
            verbose: false,
            debug: false,
        };
        assert!(validate_args(&invalid_size).is_err());

        // Invalid glob pattern
        let invalid_glob = McpArgs {
            repo: temp_dir.path().to_path_buf(),
            model: None,
            max_filesize: None,
            filter: Some("[invalid".to_string()),
            filetype: None,
            force_rebuild: false,
            verbose: false,
            debug: false,
        };
        assert!(validate_args(&invalid_glob).is_err());
    }

    #[test]
    fn test_config_overrides() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = TurboPropConfig::default();

        // Test successful overrides (model and max_filesize)
        let supported_args = McpArgs {
            repo: temp_dir.path().to_path_buf(),
            model: Some("custom-model".to_string()),
            max_filesize: Some("5mb".to_string()),
            filter: None,
            filetype: None,
            force_rebuild: false,
            verbose: false,
            debug: false,
        };

        apply_config_overrides(&mut config, &supported_args).unwrap();

        assert_eq!(config.embedding.model_name, "custom-model");
        assert_eq!(
            config.file_discovery.max_filesize_bytes,
            Some(5 * 1024 * 1024)
        );

        // Test unsupported filter option returns error
        let filter_args = McpArgs {
            repo: temp_dir.path().to_path_buf(),
            model: None,
            max_filesize: None,
            filter: Some("src/**/*.rs".to_string()),
            filetype: None,
            force_rebuild: false,
            verbose: false,
            debug: false,
        };

        assert!(apply_config_overrides(&mut config, &filter_args).is_err());

        // Test unsupported filetype option returns error
        let filetype_args = McpArgs {
            repo: temp_dir.path().to_path_buf(),
            model: None,
            max_filesize: None,
            filter: None,
            filetype: Some("rust".to_string()),
            force_rebuild: false,
            verbose: false,
            debug: false,
        };

        assert!(apply_config_overrides(&mut config, &filetype_args).is_err());

        // Test unsupported force_rebuild option returns error
        let force_rebuild_args = McpArgs {
            repo: temp_dir.path().to_path_buf(),
            model: None,
            max_filesize: None,
            filter: None,
            filetype: None,
            force_rebuild: true,
            verbose: false,
            debug: false,
        };

        assert!(apply_config_overrides(&mut config, &force_rebuild_args).is_err());
    }
}
