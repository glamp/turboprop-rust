//! Search command implementation for the TurboProp CLI.
//!
//! This module provides the complete search command functionality including
//! query processing, result filtering, output formatting, and error handling.

use anyhow::{Context, Result};
use std::path::Path;
use tracing::{debug, info, warn};

use crate::filters::SearchFilter;
use crate::output::{OutputFormat, ResultFormatter};
use crate::search_with_config;

/// Configuration for the search command
#[derive(Debug, Clone)]
pub struct SearchCommandConfig {
    /// The search query string
    pub query: String,
    /// Repository path to search in
    pub repo_path: String,
    /// Maximum number of results to return
    pub limit: usize,
    /// Minimum similarity threshold (0.0 to 1.0)
    pub threshold: Option<f32>,
    /// Output format (json or text)
    pub output_format: OutputFormat,
    /// Optional file extension filter
    pub filetype: Option<String>,
}

impl SearchCommandConfig {
    /// Create a new search command configuration
    pub fn new(
        query: String,
        repo_path: String,
        limit: usize,
        threshold: Option<f32>,
        output_format: OutputFormat,
        filetype: Option<String>,
    ) -> Self {
        Self {
            query,
            repo_path,
            limit,
            threshold,
            output_format,
            filetype,
        }
    }

    /// Validate the search command configuration
    pub fn validate(&self) -> Result<()> {
        // Validate query
        crate::query::validate_query(&self.query).context("Invalid search query")?;

        // Validate repository path
        let repo_path = Path::new(&self.repo_path);
        if !repo_path.exists() {
            anyhow::bail!("Repository path does not exist: {}", self.repo_path);
        }

        // Validate threshold range
        if let Some(threshold) = self.threshold {
            if !(0.0..=1.0).contains(&threshold) {
                anyhow::bail!("Threshold must be between 0.0 and 1.0, got: {}", threshold);
            }
        }

        // Validate limit
        if self.limit == 0 {
            anyhow::bail!("Limit must be greater than 0, got: {}", self.limit);
        }

        if self.limit > 1000 {
            warn!(
                "Large result limit specified ({}), this may impact performance",
                self.limit
            );
        }

        // Validate file extension if provided
        if let Some(ref filetype) = self.filetype {
            crate::filters::normalize_file_extension(filetype)
                .context("Invalid file extension format")?;
        }

        Ok(())
    }
}

/// Execute the search command with comprehensive error handling and logging
pub async fn execute_search_command(config: SearchCommandConfig) -> Result<()> {
    info!("Starting search command execution");
    debug!("Search config: {:?}", config);

    // Validate configuration
    config
        .validate()
        .context("Search configuration validation failed")?;

    // Log search parameters
    info!("Searching for: '{}'", config.query);
    info!("Repository: {}", config.repo_path);
    info!("Output format: {}", config.output_format);
    info!("Result limit: {}", config.limit);

    if let Some(threshold) = config.threshold {
        info!("Similarity threshold: {:.1}%", threshold * 100.0);
    }

    // Create search filter
    let search_filter = SearchFilter::from_cli_args(config.filetype.clone());

    if search_filter.has_active_filters() {
        let filter_descriptions = search_filter.describe_filters();
        info!("Active filters: {}", filter_descriptions.join(", "));
    }

    // Perform the search
    info!("Executing search query...");
    let repo_path = Path::new(&config.repo_path);

    let results = search_with_config(
        &config.query,
        repo_path,
        Some(config.limit),
        config.threshold,
    )
    .await
    .context("Search execution failed")?;

    debug!("Raw search returned {} results", results.len());

    // Apply filters to results
    let filtered_results = search_filter
        .apply_filters(results)
        .context("Failed to apply result filters")?;

    info!("Found {} results after filtering", filtered_results.len());

    // Format and output results
    let formatter = ResultFormatter::new(config.output_format);

    if filtered_results.is_empty() {
        formatter.print_no_results(&config.query, config.threshold)?;
    } else {
        formatter.print_results(&filtered_results, &config.query)?;
    }

    info!("Search command completed successfully");
    Ok(())
}

/// Execute search command from CLI arguments
pub async fn execute_search_command_cli(
    query: String,
    repo: std::path::PathBuf,
    limit: usize,
    threshold: Option<f32>,
    output: String,
    filetype: Option<String>,
) -> Result<()> {
    // Parse output format
    let output_format: OutputFormat = output
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid output format: {}", e))?;

    // Create configuration
    let config = SearchCommandConfig::new(
        query,
        repo.to_string_lossy().to_string(),
        limit,
        threshold,
        output_format,
        filetype,
    );

    // Execute the command
    execute_search_command(config).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_search_command_config_validation() {
        // Valid configuration
        let config = SearchCommandConfig::new(
            "test query".to_string(),
            ".".to_string(),
            10,
            Some(0.5),
            OutputFormat::Json,
            Some("rs".to_string()),
        );
        assert!(config.validate().is_ok());

        // Empty query should fail
        let config = SearchCommandConfig::new(
            "".to_string(),
            ".".to_string(),
            10,
            None,
            OutputFormat::Json,
            None,
        );
        assert!(config.validate().is_err());

        // Invalid threshold should fail
        let config = SearchCommandConfig::new(
            "test".to_string(),
            ".".to_string(),
            10,
            Some(1.5),
            OutputFormat::Json,
            None,
        );
        assert!(config.validate().is_err());

        // Zero limit should fail
        let config = SearchCommandConfig::new(
            "test".to_string(),
            ".".to_string(),
            0,
            None,
            OutputFormat::Json,
            None,
        );
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_search_command_config_nonexistent_path() {
        let config = SearchCommandConfig::new(
            "test query".to_string(),
            "/nonexistent/path".to_string(),
            10,
            None,
            OutputFormat::Json,
            None,
        );
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_search_command_config_invalid_filetype() {
        let config = SearchCommandConfig::new(
            "test query".to_string(),
            ".".to_string(),
            10,
            None,
            OutputFormat::Json,
            Some("".to_string()), // Empty filetype should be invalid
        );
        assert!(config.validate().is_err());
    }

    #[tokio::test]
    async fn test_search_command_with_temp_directory() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path().to_string_lossy().to_string();

        let config = SearchCommandConfig::new(
            "test query".to_string(),
            temp_path,
            10,
            None,
            OutputFormat::Json,
            None,
        );

        // This will fail because there's no index, but configuration should be valid
        assert!(config.validate().is_ok());

        // The actual search execution will fail due to missing index, which is expected
        let result = execute_search_command(config).await;
        assert!(result.is_err());
    }
}
