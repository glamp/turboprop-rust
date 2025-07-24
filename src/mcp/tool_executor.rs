//! Tool executor for MCP server - handles tool execution logic
//!
//! Separates tool execution concerns from protocol and server management

use anyhow::Result;
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, warn};

use crate::config::TurboPropConfig;
use crate::index::PersistentChunkIndex;
use crate::mcp::error::McpError;

use super::tools::{SearchTool, ToolCallRequest, ToolResponse, Tools};

// Operation timeouts
const SEARCH_OPERATION_TIMEOUT_SECS: u64 = 20; // 20 seconds for search operations
const TOOL_EXECUTION_TIMEOUT_SECS: u64 = 25; // 25 seconds for tool execution

/// Handles tool execution for MCP server
pub struct ToolExecutor {
    /// Search tool instance
    search_tool: SearchTool,
    /// Tools registry
    tools: Option<Tools>,
}

impl ToolExecutor {
    /// Create a new tool executor
    pub fn new() -> Self {
        Self {
            search_tool: SearchTool::new(),
            tools: None,
        }
    }

    /// Create a new tool executor with tools registry
    pub fn with_tools(tools: Tools) -> Self {
        Self {
            search_tool: SearchTool::new(),
            tools: Some(tools),
        }
    }

    /// Set the tools registry
    pub fn set_tools(&mut self, tools: Tools) {
        self.tools = Some(tools);
    }

    /// Execute a tool call with the given parameters
    pub async fn execute_tool(
        &self,
        tool_name: &str,
        arguments: Value,
        index: &Arc<RwLock<Option<PersistentChunkIndex>>>,
        config: &TurboPropConfig,
        repo_path: &Path,
    ) -> Result<Value, ToolExecutionError> {
        // Security validation for tool parameters
        Self::validate_tool_parameters(&arguments)?;

        // Execute tool using tools registry if available
        if let Some(ref tools_registry) = self.tools {
            self.execute_with_registry(tool_name, arguments, tools_registry)
                .await
        } else {
            // Fallback to direct tool execution
            self.execute_direct_tool(tool_name, arguments, index, config, repo_path)
                .await
        }
    }

    /// Execute tool using tools registry with timeout
    async fn execute_with_registry(
        &self,
        tool_name: &str,
        arguments: Value,
        tools_registry: &Tools,
    ) -> Result<Value, ToolExecutionError> {
        let tool_call_request = ToolCallRequest {
            name: tool_name.to_string(),
            arguments: serde_json::from_value(arguments).unwrap_or_default(),
        };

        match tokio::time::timeout(
            Duration::from_secs(TOOL_EXECUTION_TIMEOUT_SECS),
            tools_registry.execute_tool(tool_call_request),
        )
        .await
        {
            Ok(Ok(tool_response)) => {
                if tool_response.success {
                    debug!("Tool executed successfully: {}", tool_name);
                    Ok(tool_response.content.unwrap_or(json!({})))
                } else {
                    let error_msg = tool_response
                        .error
                        .unwrap_or_else(|| "Unknown error".to_string());
                    error!("Tool execution failed: {}", error_msg);
                    Err(ToolExecutionError::ExecutionFailed(error_msg))
                }
            }
            Ok(Err(e)) => {
                error!("Tool execution failed: {}", e);
                Err(ToolExecutionError::ExecutionFailed(e.to_string()))
            }
            Err(_) => {
                warn!("Tool execution timed out: {}", tool_name);
                Err(ToolExecutionError::Timeout(TOOL_EXECUTION_TIMEOUT_SECS))
            }
        }
    }

    /// Execute tool directly (fallback path)
    async fn execute_direct_tool(
        &self,
        tool_name: &str,
        arguments: Value,
        index: &Arc<RwLock<Option<PersistentChunkIndex>>>,
        config: &TurboPropConfig,
        repo_path: &Path,
    ) -> Result<Value, ToolExecutionError> {
        match tool_name {
            "search" => {
                // Get index
                let index_guard = index.read().await;
                let index = match index_guard.as_ref() {
                    Some(index) => index,
                    None => {
                        return Err(ToolExecutionError::IndexNotReady);
                    }
                };

                // Execute search tool with timeout
                match tokio::time::timeout(
                    Duration::from_secs(SEARCH_OPERATION_TIMEOUT_SECS),
                    self.search_tool.execute(arguments, index, config, repo_path),
                )
                .await
                {
                    Ok(Ok(result)) => {
                        debug!("Search tool executed successfully");
                        Ok(result)
                    }
                    Ok(Err(e)) => {
                        error!("Search tool execution failed: {}", e);
                        Err(ToolExecutionError::ExecutionFailed(e.to_string()))
                    }
                    Err(_) => {
                        warn!("Search tool execution timed out");
                        Err(ToolExecutionError::Timeout(SEARCH_OPERATION_TIMEOUT_SECS))
                    }
                }
            }
            _ => Err(ToolExecutionError::ToolNotFound(tool_name.to_string())),
        }
    }

    /// Get list of available tools
    pub fn list_tools(&self) -> Vec<super::tools::ToolDefinition> {
        if let Some(ref tools_registry) = self.tools {
            tools_registry.list_tools()
        } else {
            // Create a ToolDefinition for the SearchTool
            vec![super::tools::ToolDefinition {
                name: "search".to_string(),
                description: "Semantic search across the codebase using natural language queries"
                    .to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query to find relevant code"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Minimum similarity threshold (0.0 to 1.0)",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "filetype": {
                            "type": "string",
                            "description": "Filter by file extension (e.g., '.rs', '.js')"
                        },
                        "filter": {
                            "type": "string",
                            "description": "Glob pattern filter (e.g., '*.rs', 'src/**/*.js')"
                        },
                        "include_content": {
                            "type": "boolean",
                            "description": "Include file content in results",
                            "default": true
                        }
                    },
                    "required": ["query"]
                }),
            }]
        }
    }

    /// Validate tool parameters for security
    fn validate_tool_parameters(params: &Value) -> Result<(), ToolExecutionError> {
        // Validate query
        if let Some(query) = params.get("query").and_then(|q| q.as_str()) {
            Self::validate_query_input(query)?;
        }

        // Validate limit parameter
        if let Some(limit) = params.get("limit").and_then(|l| l.as_u64()) {
            if limit > 100 {
                return Err(ToolExecutionError::InvalidParameters(
                    "Limit too high (max 100)".to_string(),
                ));
            }
        }

        // Validate threshold parameter
        if let Some(threshold) = params.get("threshold").and_then(|t| t.as_f64()) {
            if threshold < 0.0 || threshold > 1.0 {
                return Err(ToolExecutionError::InvalidParameters(
                    "Threshold must be between 0.0 and 1.0".to_string(),
                ));
            }
        }

        // Validate filter patterns to prevent path traversal
        if let Some(filter) = params.get("filter").and_then(|f| f.as_str()) {
            if filter.contains("../") || filter.contains("..\\") {
                return Err(ToolExecutionError::SecurityViolation(
                    "Suspicious filter pattern detected".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Validate query input to prevent security issues
    fn validate_query_input(query: &str) -> Result<(), ToolExecutionError> {
        // Enforce length limits
        if query.len() > 1000 {
            return Err(ToolExecutionError::InvalidParameters(
                "Query too long (max 1000 characters)".to_string(),
            ));
        }

        // Check for suspicious patterns that might indicate path traversal attempts
        if query.contains("../") || query.contains("..\\") {
            return Err(ToolExecutionError::SecurityViolation(
                "Suspicious query pattern detected".to_string(),
            ));
        }

        // Check for null bytes that might be used in path manipulation
        if query.contains('\0') {
            return Err(ToolExecutionError::SecurityViolation(
                "Null bytes not allowed in query".to_string(),
            ));
        }

        // Check for extremely long individual words that might indicate buffer overflow attempts
        if query.split_whitespace().any(|word| word.len() > 200) {
            return Err(ToolExecutionError::SecurityViolation(
                "Extremely long words detected".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for ToolExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Tool execution errors
#[derive(Debug, thiserror::Error)]
pub enum ToolExecutionError {
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Tool execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Tool execution timed out after {0} seconds")]
    Timeout(u64),

    #[error("Index not ready")]
    IndexNotReady,

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("Security violation: {0}")]
    SecurityViolation(String),
}

impl From<McpError> for ToolExecutionError {
    fn from(error: McpError) -> Self {
        match error {
            McpError::InvalidPath => ToolExecutionError::InvalidParameters("Invalid path".to_string()),
            McpError::PathTraversal => ToolExecutionError::SecurityViolation("Path traversal detected".to_string()),
            McpError::SymlinkAttack => ToolExecutionError::SecurityViolation("Symbolic link attack detected".to_string()),
            McpError::QueryTooLong { max } => ToolExecutionError::InvalidParameters(format!("Query too long (max {} characters)", max)),
            McpError::SuspiciousQuery => ToolExecutionError::SecurityViolation("Suspicious query pattern detected".to_string()),
            _ => ToolExecutionError::ExecutionFailed(error.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tool_executor_creation() {
        let executor = ToolExecutor::new();
        // Basic creation test - executor should be created successfully
        drop(executor);
    }

    #[test]
    fn test_validate_query_input_valid() {
        let result = ToolExecutor::validate_query_input("normal search query");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_query_input_path_traversal() {
        let result = ToolExecutor::validate_query_input("../etc/passwd");
        assert!(result.is_err());
        matches!(result.unwrap_err(), ToolExecutionError::SecurityViolation(_));
    }

    #[test]
    fn test_validate_query_input_too_long() {
        let long_query = "a".repeat(1001);
        let result = ToolExecutor::validate_query_input(&long_query);
        assert!(result.is_err());
        matches!(result.unwrap_err(), ToolExecutionError::InvalidParameters(_));
    }

    #[test]
    fn test_validate_tool_parameters_valid() {
        let params = json!({
            "query": "test",
            "limit": 10,
            "threshold": 0.5
        });
        let result = ToolExecutor::validate_tool_parameters(&params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_tool_parameters_invalid_limit() {
        let params = json!({
            "query": "test",
            "limit": 200
        });
        let result = ToolExecutor::validate_tool_parameters(&params);
        assert!(result.is_err());
        matches!(result.unwrap_err(), ToolExecutionError::InvalidParameters(_));
    }

    #[test]
    fn test_list_tools_default() {
        let executor = ToolExecutor::new();
        let tools = executor.list_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "search");
    }
}