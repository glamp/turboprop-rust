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

/// Threshold for warning about large result limits that may impact performance
const LARGE_RESULT_LIMIT_WARNING_THRESHOLD: usize = 1000;

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
    /// Optional glob pattern filter
    pub glob_pattern: Option<String>,
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
        glob_pattern: Option<String>,
    ) -> Self {
        Self {
            query,
            repo_path,
            limit,
            threshold,
            output_format,
            filetype,
            glob_pattern,
        }
    }

    /// Validate the search command configuration
    pub fn validate(&self) -> Result<()> {
        // Validate query
        crate::query::validate_query(&self.query)
            .with_context(|| format!("Search query validation failed: '{}'", self.query))?;

        // Validate repository path
        let repo_path = Path::new(&self.repo_path);
        if !repo_path.exists() {
            return Err(anyhow::anyhow!("Repository path does not exist"))
                .with_context(|| format!("Invalid repository path: {}", self.repo_path));
        }

        // Validate threshold range
        if let Some(threshold) = self.threshold {
            if !(0.0..=1.0).contains(&threshold) {
                return Err(anyhow::anyhow!("Threshold out of valid range")).with_context(|| {
                    format!("Threshold must be between 0.0 and 1.0, got: {}", threshold)
                });
            }
        }

        // Validate limit
        if self.limit == 0 {
            return Err(anyhow::anyhow!("Invalid limit value"))
                .with_context(|| format!("Limit must be greater than 0, got: {}", self.limit));
        }

        if self.limit > LARGE_RESULT_LIMIT_WARNING_THRESHOLD {
            warn!(
                "Large result limit specified ({}), this may impact performance",
                self.limit
            );
        }

        // Validate file extension if provided
        if let Some(ref filetype) = self.filetype {
            crate::filters::normalize_file_extension(filetype)
                .with_context(|| format!("File extension validation failed for '{}'", filetype))?;
        }

        // Validate glob pattern if provided
        if let Some(ref glob_pattern) = self.glob_pattern {
            crate::filters::validate_glob_pattern(glob_pattern).with_context(|| {
                format!("Glob pattern validation failed for '{}'", glob_pattern)
            })?;
        }

        Ok(())
    }
}

/// Execute the search command with comprehensive error handling and logging
pub async fn execute_search_command(config: SearchCommandConfig, turboprop_config: &crate::config::TurboPropConfig) -> Result<()> {
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

    if let Some(ref glob_pattern) = config.glob_pattern {
        info!("Glob pattern filter: {}", glob_pattern);
    }

    // Create search filter with configuration
    let search_filter =
        SearchFilter::from_cli_args_with_config(config.filetype.clone(), config.glob_pattern.clone(), turboprop_config);

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
    filter: Option<String>,
    turboprop_config: &crate::config::TurboPropConfig,
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
        filter,
    );

    // Execute the command
    execute_search_command(config, turboprop_config).await
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
            Some("*.rs".to_string()),
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
            None,
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
            None,
        );

        // This will fail because there's no index, but configuration should be valid
        assert!(config.validate().is_ok());

        // The actual search execution will fail due to missing index, which is expected
        let turboprop_config = crate::config::TurboPropConfig::default();
        let result = execute_search_command(config, &turboprop_config).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_search_command_config_with_valid_glob_pattern() {
        // Valid glob patterns should pass validation
        let valid_patterns = vec!["*.rs", "src/*.js", "**/*.py", "test_*.txt"];

        for pattern in valid_patterns {
            let config = SearchCommandConfig::new(
                "test query".to_string(),
                ".".to_string(),
                10,
                None,
                OutputFormat::Json,
                None,
                Some(pattern.to_string()),
            );
            assert!(
                config.validate().is_ok(),
                "Pattern '{}' should be valid",
                pattern
            );
        }
    }

    #[test]
    fn test_search_command_config_with_invalid_glob_pattern() {
        // Invalid glob patterns should fail validation
        let invalid_patterns = vec!["", "   ", "[invalid"];

        for pattern in invalid_patterns {
            let config = SearchCommandConfig::new(
                "test query".to_string(),
                ".".to_string(),
                10,
                None,
                OutputFormat::Json,
                None,
                Some(pattern.to_string()),
            );
            assert!(
                config.validate().is_err(),
                "Pattern '{}' should be invalid",
                pattern
            );
        }
    }

    #[test]
    fn test_search_command_config_with_no_glob_pattern() {
        // No glob pattern should be valid
        let config = SearchCommandConfig::new(
            "test query".to_string(),
            ".".to_string(),
            10,
            None,
            OutputFormat::Json,
            None,
            None,
        );
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_search_command_config_with_both_filetype_and_glob() {
        // Should be able to have both filetype and glob pattern
        let config = SearchCommandConfig::new(
            "test query".to_string(),
            ".".to_string(),
            10,
            None,
            OutputFormat::Json,
            Some("rs".to_string()),
            Some("src/*.rs".to_string()),
        );
        assert!(config.validate().is_ok());
    }
}
