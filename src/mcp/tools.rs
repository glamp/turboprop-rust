//! MCP tools implementation
//!
//! Provides tools that expose TurboProp's semantic search capabilities
//! to MCP clients following the tool calling specification.

use crate::mcp::error::{McpError, McpResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use tracing::{debug, info};

/// Tool definition for MCP protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Input schema (JSON Schema)
    pub input_schema: Value,
}

/// Tool call request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequest {
    /// Tool name to call
    pub name: String,
    /// Arguments for the tool
    pub arguments: HashMap<String, Value>,
}

/// Tool call response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResponse {
    /// Whether the tool call was successful
    pub success: bool,
    /// Result data (if successful)
    pub content: Option<Value>,
    /// Error message (if failed)
    pub error: Option<String>,
}

/// Trait defining tool execution interface
#[async_trait]
pub trait ToolExecutor {
    /// Execute a tool with given arguments
    async fn execute(&self, request: ToolCallRequest) -> McpResult<ToolCallResponse>;

    /// Get tool definition
    fn definition(&self) -> ToolDefinition;

    /// Check if tool supports given arguments
    fn validate_arguments(&self, arguments: &HashMap<String, Value>) -> McpResult<()>;
}

/// Semantic search tool for TurboProp integration
pub struct SemanticSearchTool {
    /// Tool configuration
    config: SearchToolConfig,
}

/// Configuration for semantic search tool
#[derive(Debug, Clone)]
pub struct SearchToolConfig {
    /// Default search limit
    pub default_limit: usize,
    /// Maximum search limit
    pub max_limit: usize,
    /// Default similarity threshold
    pub default_threshold: f32,
}

impl Default for SearchToolConfig {
    fn default() -> Self {
        Self {
            default_limit: 10,
            max_limit: 100,
            default_threshold: 0.3,
        }
    }
}

impl SemanticSearchTool {
    /// Create a new semantic search tool
    pub fn new() -> Self {
        Self::with_config(SearchToolConfig::default())
    }

    /// Create a new semantic search tool with custom configuration
    pub fn with_config(config: SearchToolConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl ToolExecutor for SemanticSearchTool {
    async fn execute(&self, request: ToolCallRequest) -> McpResult<ToolCallResponse> {
        debug!(
            "Executing semantic search tool with arguments: {:?}",
            request.arguments
        );

        // Validate arguments first
        self.validate_arguments(&request.arguments)?;

        // Extract search parameters
        let query = request
            .arguments
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::tool_execution(&request.name, "Missing 'query' parameter"))?;

        let limit = request
            .arguments
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.config.default_limit as u64) as usize;

        let threshold = request
            .arguments
            .get("threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(self.config.default_threshold as f64) as f32;

        info!(
            "Performing semantic search: query='{}', limit={}, threshold={}",
            query, limit, threshold
        );

        // TODO: Integrate with actual TurboProp search functionality
        // For now, return a placeholder response
        let mock_results = serde_json::json!({
            "results": [],
            "query": query,
            "limit": limit,
            "threshold": threshold,
            "message": "Semantic search functionality not yet implemented"
        });

        Ok(ToolCallResponse {
            success: true,
            content: Some(mock_results),
            error: None,
        })
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "semantic_search".to_string(),
            description: "Perform semantic search on the indexed codebase using TurboProp's embedding-based search".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find semantically similar code"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "minimum": 1,
                        "maximum": self.config.max_limit,
                        "default": self.config.default_limit
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum similarity threshold (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": self.config.default_threshold
                    }
                },
                "required": ["query"]
            }),
        }
    }

    fn validate_arguments(&self, arguments: &HashMap<String, Value>) -> McpResult<()> {
        // Check required query parameter
        if !arguments.contains_key("query") {
            return Err(McpError::tool_execution(
                "semantic_search",
                "Missing required 'query' parameter",
            ));
        }

        // Validate query is a string
        if !arguments.get("query").unwrap().is_string() {
            return Err(McpError::tool_execution(
                "semantic_search",
                "'query' parameter must be a string",
            ));
        }

        // Validate limit if provided
        if let Some(limit_value) = arguments.get("limit") {
            if let Some(limit) = limit_value.as_u64() {
                if limit == 0 || limit > self.config.max_limit as u64 {
                    return Err(McpError::tool_execution(
                        "semantic_search",
                        &format!("'limit' must be between 1 and {}", self.config.max_limit),
                    ));
                }
            } else {
                return Err(McpError::tool_execution(
                    "semantic_search",
                    "'limit' parameter must be an integer",
                ));
            }
        }

        // Validate threshold if provided
        if let Some(threshold_value) = arguments.get("threshold") {
            if let Some(threshold) = threshold_value.as_f64() {
                if threshold < 0.0 || threshold > 1.0 {
                    return Err(McpError::tool_execution(
                        "semantic_search",
                        "'threshold' must be between 0.0 and 1.0",
                    ));
                }
            } else {
                return Err(McpError::tool_execution(
                    "semantic_search",
                    "'threshold' parameter must be a number",
                ));
            }
        }

        Ok(())
    }
}

impl Default for SemanticSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

/// MCP tools registry that manages available tools
pub struct Tools {
    /// Available tools
    tools: HashMap<String, Box<dyn ToolExecutor + Send + Sync>>,
}

impl Tools {
    /// Create a new tools registry with default tools
    pub fn new() -> Self {
        let mut tools: HashMap<String, Box<dyn ToolExecutor + Send + Sync>> = HashMap::new();

        // Register semantic search tool
        let search_tool = SemanticSearchTool::new();
        tools.insert("semantic_search".to_string(), Box::new(search_tool));

        Self { tools }
    }

    /// Register a new tool
    pub fn register_tool(&mut self, tool: Box<dyn ToolExecutor + Send + Sync>) {
        let definition = tool.definition();
        self.tools.insert(definition.name.clone(), tool);
    }

    /// Get all available tool definitions
    pub fn list_tools(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|tool| tool.definition()).collect()
    }

    /// Execute a tool by name
    pub async fn execute_tool(&self, request: ToolCallRequest) -> McpResult<ToolCallResponse> {
        let tool = self
            .tools
            .get(&request.name)
            .ok_or_else(|| McpError::tool_execution(&request.name, "Tool not found"))?;

        tool.execute(request).await
    }

    /// Check if a tool exists
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }
}

impl Default for Tools {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_search_tool_definition() {
        let tool = SemanticSearchTool::new();
        let definition = tool.definition();

        assert_eq!(definition.name, "semantic_search");
        assert!(!definition.description.is_empty());
        assert!(definition.input_schema.is_object());
    }

    #[test]
    fn test_semantic_search_tool_validation() {
        let tool = SemanticSearchTool::new();

        // Valid arguments
        let mut args = HashMap::new();
        args.insert("query".to_string(), Value::String("test query".to_string()));
        assert!(tool.validate_arguments(&args).is_ok());

        // Missing query
        let empty_args = HashMap::new();
        assert!(tool.validate_arguments(&empty_args).is_err());

        // Invalid limit
        let mut invalid_args = HashMap::new();
        invalid_args.insert("query".to_string(), Value::String("test".to_string()));
        invalid_args.insert(
            "limit".to_string(),
            Value::Number(serde_json::Number::from(0)),
        );
        assert!(tool.validate_arguments(&invalid_args).is_err());
    }

    #[tokio::test]
    async fn test_tools_registry() {
        let tools = Tools::new();

        // Check default tools are registered
        assert!(tools.has_tool("semantic_search"));

        // List tools
        let tool_list = tools.list_tools();
        assert!(!tool_list.is_empty());
        assert_eq!(tool_list[0].name, "semantic_search");
    }

    #[tokio::test]
    async fn test_tool_execution() {
        let tools = Tools::new();

        let mut args = HashMap::new();
        args.insert("query".to_string(), Value::String("test query".to_string()));

        let request = ToolCallRequest {
            name: "semantic_search".to_string(),
            arguments: args,
        };

        let response = tools.execute_tool(request).await;
        assert!(response.is_ok());

        let result = response.unwrap();
        assert!(result.success);
        assert!(result.content.is_some());
    }
}
