//! MCP server command implementation
//!
//! Provides the `tp mcp` command that starts an MCP server for real-time
//! semantic search integration with coding agents.

use anyhow::{Context, Result};
use std::path::Path;
use tracing::{info, warn, debug, Level};

use crate::cli::McpArgs;
use crate::config::TurboPropConfig;
use crate::mcp::McpServer;
use crate::types::parse_filesize;

/// Validate MCP command arguments with proper imports
fn validate_mcp_args(args: &McpArgs) -> Result<()> {
    // Validate repository path
    if !args.repo.exists() {
        return Err(anyhow::anyhow!("Repository path does not exist: {}", args.repo.display()));
    }
    
    if !args.repo.is_dir() {
        return Err(anyhow::anyhow!("Repository path is not a directory: {}", args.repo.display()));
    }
    
    // Validate file size format if provided
    if let Some(max_filesize) = &args.max_filesize {
        if let Err(e) = parse_filesize(max_filesize) {
            return Err(anyhow::anyhow!("Invalid file size format '{}': {}", max_filesize, e));
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
    setup_logging(args.verbose, args.debug)?;
    
    info!("Starting TurboProp MCP server");
    debug!("MCP arguments: {:?}", args);
    
    // Validate arguments - do detailed validation here where we have imports
    validate_mcp_args(&args)?;
    
    // Print setup information after validation
    print_setup_info(&args.repo);
    
    // Load configuration from current directory or look for config in repo
    let mut config = if let Ok(config_path) = std::fs::canonicalize(args.repo.join(".turboprop.yml")) {
        TurboPropConfig::load_from_file(&config_path)
            .with_context(|| format!("Failed to load configuration from {}", config_path.display()))?
    } else {
        // Fall back to default configuration
        TurboPropConfig::load()
            .with_context(|| "Failed to load default configuration")?
    };
    
    // Apply CLI overrides
    apply_config_overrides(&mut config, &args)?;
    
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
    server.run()
        .await
        .context("MCP server execution failed")?;
    
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
    
    // Override filters if specified - TODO: implement proper filtering
    if let Some(_filter) = &args.filter {
        warn!("Filter patterns are not yet implemented in config override");
    }
    
    if let Some(_filetype) = &args.filetype {
        warn!("File type filtering is not yet implemented in config override");
    }
    
    // Set force rebuild flag - TODO: implement proper force rebuild
    if args.force_rebuild {
        warn!("Force rebuild is not yet implemented in config override");
    }
    
    Ok(())
}

/// Set up logging for the MCP server
fn setup_logging(verbose: bool, debug: bool) -> Result<()> {
    let _log_level = if debug {
        Level::DEBUG
    } else if verbose {
        Level::INFO
    } else {
        Level::WARN
    };
    
    // Skip logging setup since main.rs already sets up a global subscriber
    // The MCP server will use the existing logging configuration
    debug!("Using existing logging configuration");
    Ok(())
}

/// Log configuration summary for debugging
fn log_config_summary(config: &TurboPropConfig, args: &McpArgs) {
    info!("Configuration summary:");
    info!("  Repository: {}", args.repo.display());
    info!("  Model: {}", config.embedding.model_name);
    info!("  Max file size: {:?}", config.file_discovery.max_filesize_bytes);
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
    eprintln!("🚀 TurboProp MCP Server Started");
    eprintln!("────────────────────────────────");
    eprintln!("Repository: {}", repo_path.display());
    eprintln!();
    eprintln!("To integrate with coding agents, add this to your MCP configuration:");
    eprintln!();
    eprintln!("Claude Code (.mcp.json):");
    eprintln!(r#"{{
  "mcpServers": {{
    "turboprop": {{
      "command": "tp",
      "args": ["mcp", "--repo", "{}"]
    }}
  }}
}}"#, repo_path.display());
    eprintln!();
    eprintln!("Cursor (.cursor/mcp.json):");
    eprintln!(r#"{{
  "mcpServers": {{
    "turboprop": {{
      "command": "tp",
      "args": ["mcp", "--repo", "{}"]
    }}
  }}
}}"#, repo_path.display());
    eprintln!();
    eprintln!("Available search tool parameters:");
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
        assert!(validate_mcp_args(&valid_args).is_ok());
        
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
        assert!(validate_mcp_args(&invalid_repo).is_err());
        
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
        assert!(validate_mcp_args(&invalid_size).is_err());
        
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
        assert!(validate_mcp_args(&invalid_glob).is_err());
    }
    
    #[test]
    fn test_config_overrides() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = TurboPropConfig::default();
        
        let args = McpArgs {
            repo: temp_dir.path().to_path_buf(),
            model: Some("custom-model".to_string()),
            max_filesize: Some("5mb".to_string()),
            filter: Some("src/**/*.rs".to_string()),
            filetype: Some("rust".to_string()),
            force_rebuild: true,
            verbose: false,
            debug: false,
        };
        
        apply_config_overrides(&mut config, &args).unwrap();
        
        assert_eq!(config.embedding.model_name, "custom-model");
        assert_eq!(config.file_discovery.max_filesize_bytes, Some(5 * 1024 * 1024));
        // Note: Other fields are not yet implemented in config override
    }
}