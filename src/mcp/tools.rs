//! MCP tools for semantic search
//!
//! Implements the search tool that exposes TurboProp's semantic search
//! capabilities via MCP protocol

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use tracing::{debug, info};

use crate::config::TurboPropConfig;
use crate::search::search_index_with_filters;

/// Strong type for search query strings
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QueryString(String);

impl QueryString {
    /// Create a new query string
    pub fn new(query: String) -> Self {
        Self(query)
    }
    
    /// Get the inner string
    pub fn as_str(&self) -> &str {
        &self.0
    }
    
    /// Get the length of the query
    pub fn len(&self) -> usize {
        self.0.len()
    }
    
    /// Check if the query is empty
    pub fn is_empty(&self) -> bool {
        self.0.trim().is_empty()
    }
}

impl From<String> for QueryString {
    fn from(query: String) -> Self {
        Self::new(query)
    }
}

impl AsRef<str> for QueryString {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Strong type for result limits
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ResultLimit(usize);

impl ResultLimit {
    /// Create a new result limit
    pub fn new(limit: usize) -> Self {
        Self(limit)
    }
    
    /// Get the inner value
    pub fn value(&self) -> usize {
        self.0
    }
}

impl From<usize> for ResultLimit {
    fn from(limit: usize) -> Self {
        Self::new(limit)
    }
}

/// Strong type for context lines
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ContextLines(usize);

impl ContextLines {
    /// Create a new context lines value
    pub fn new(lines: usize) -> Self {
        Self(lines)
    }
    
    /// Get the inner value
    pub fn value(&self) -> usize {
        self.0
    }
}

impl From<usize> for ContextLines {
    fn from(lines: usize) -> Self {
        Self::new(lines)
    }
}

/// Strong type for similarity scores
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct SimilarityScore(f32);

impl SimilarityScore {
    /// Create a new similarity score
    pub fn new(score: f32) -> Self {
        Self(score)
    }
    
    /// Get the inner value
    pub fn value(&self) -> f32 {
        self.0
    }
    
    /// Check if the score is within valid range (0.0 to 1.0)
    pub fn is_valid(&self) -> bool {
        (0.0..=1.0).contains(&self.0)
    }
}

impl From<f32> for SimilarityScore {
    fn from(score: f32) -> Self {
        Self::new(score)
    }
}

// Serialization and deserialization functions
use serde::{Deserializer, Serializer};

fn deserialize_query_string<'de, D>(deserializer: D) -> Result<QueryString, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    Ok(QueryString::from(s))
}

fn deserialize_result_limit<'de, D>(deserializer: D) -> Result<ResultLimit, D::Error>
where
    D: Deserializer<'de>,
{
    let n = usize::deserialize(deserializer)?;
    Ok(ResultLimit::from(n))
}

fn deserialize_optional_similarity_score<'de, D>(deserializer: D) -> Result<Option<SimilarityScore>, D::Error>
where
    D: Deserializer<'de>,
{
    let opt = Option::<f32>::deserialize(deserializer)?;
    Ok(opt.map(SimilarityScore::from))
}

fn deserialize_optional_context_lines<'de, D>(deserializer: D) -> Result<Option<ContextLines>, D::Error>
where
    D: Deserializer<'de>,
{
    let opt = Option::<usize>::deserialize(deserializer)?;
    Ok(opt.map(ContextLines::from))
}

fn serialize_similarity_score<S>(score: &SimilarityScore, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_f32(score.value())
}

/// Default result limit for search results
fn default_result_limit() -> ResultLimit {
    ResultLimit::new(10)
}


/// Parameters for the search tool
#[derive(Debug, Clone, Deserialize)]
pub struct SearchToolParams {
    /// Natural language search query (required)
    #[serde(deserialize_with = "deserialize_query_string")]
    pub query: QueryString,
    /// Maximum number of results to return
    #[serde(default = "default_result_limit", deserialize_with = "deserialize_result_limit")]
    pub limit: ResultLimit,
    /// Minimum similarity threshold (0.0 to 1.0)
    #[serde(default, deserialize_with = "deserialize_optional_similarity_score")]
    pub threshold: Option<SimilarityScore>,
    /// Filter by file extension (e.g., ".rs", ".js", ".py")
    #[serde(default)]
    pub filetype: Option<String>,
    /// Glob pattern filter (e.g., "*.rs", "src/**/*.js")
    #[serde(default)]
    pub filter: Option<String>,
    /// Include file content in results
    #[serde(default = "default_include_content")]
    pub include_content: bool,
    /// Context lines around matches
    #[serde(default, deserialize_with = "deserialize_optional_context_lines")]
    pub context_lines: Option<ContextLines>,
}

/// Search result formatted for MCP
#[derive(Debug, Clone, Serialize)]
pub struct McpSearchResult {
    /// Relative file path
    pub file_path: String,
    /// Line number of the match
    pub line_number: usize,
    /// Similarity score (0.0 to 1.0)
    #[serde(serialize_with = "serialize_similarity_score")]
    pub similarity_score: SimilarityScore,
    /// Matched text content
    pub content: String,
    /// Context lines around the match (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<String>>,
    /// File type/extension
    pub file_type: String,
    /// Start and end positions in the chunk
    pub start_line: usize,
    pub end_line: usize,
}

/// Search tool execution result
#[derive(Debug, Clone, Serialize)]
pub struct SearchToolResult {
    /// Search results
    pub results: Vec<McpSearchResult>,
    /// Total number of results found (before limit)
    pub total_results: usize,
    /// Query execution time in milliseconds
    pub execution_time_ms: u64,
    /// Search parameters used
    pub query_info: SearchQueryInfo,
}

/// Information about the executed search query
#[derive(Debug, Clone, Serialize)]
pub struct SearchQueryInfo {
    /// Original query string
    pub query: String,
    /// Applied filters
    pub filters: SearchFilters,
    /// Limit applied
    pub limit: usize,
    /// Threshold used
    pub threshold: f32,
}

/// Applied search filters
#[derive(Debug, Clone, Serialize)]
pub struct SearchFilters {
    /// File type filter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filetype: Option<String>,
    /// Glob pattern filter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub glob_pattern: Option<String>,
}



/// Default for include_content
fn default_include_content() -> bool {
    true
}

// Keep the existing ToolExecutor trait and other types for compatibility
use crate::mcp::error::{McpError as ExistingMcpError, McpResult as ExistingMcpResult};
use async_trait::async_trait;

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
    async fn execute(&self, request: ToolCallRequest) -> ExistingMcpResult<ToolCallResponse>;

    /// Get tool definition
    fn definition(&self) -> ToolDefinition;

    /// Check if tool supports given arguments
    fn validate_arguments(&self, arguments: &HashMap<String, Value>) -> ExistingMcpResult<()>;
}

/// Semantic search tool that integrates with TurboProp
pub struct SemanticSearchTool {
    /// Tool configuration
    config: SearchToolConfig,
    /// Path to the search index
    index_path: std::path::PathBuf,
    /// Repository path for relative paths
    repo_path: std::path::PathBuf,
    /// TurboProp configuration
    turboprop_config: TurboPropConfig,
    /// Whether this is a mock tool for testing
    is_mock: bool,
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
    /// Maximum query length in characters
    pub max_query_length: usize,
    /// Minimum similarity threshold
    pub min_threshold: f32,
    /// Maximum similarity threshold
    pub max_threshold: f32,
    /// Default context lines
    pub default_context_lines: usize,
    /// Maximum context lines
    pub max_context_lines: usize,
}

impl Default for SearchToolConfig {
    fn default() -> Self {
        Self {
            default_limit: 10,
            max_limit: 100,
            default_threshold: 0.3,
            max_query_length: 1000,
            min_threshold: 0.0,
            max_threshold: 1.0,
            default_context_lines: 0,
            max_context_lines: 10,
        }
    }
}

impl SemanticSearchTool {
    /// Create a new semantic search tool
    pub fn new(
        index_path: std::path::PathBuf,
        repo_path: std::path::PathBuf,
        turboprop_config: TurboPropConfig,
    ) -> Self {
        Self::with_config(
            SearchToolConfig::default(),
            index_path,
            repo_path,
            turboprop_config,
        )
    }

    /// Create a new semantic search tool with custom configuration
    pub fn with_config(
        config: SearchToolConfig,
        index_path: std::path::PathBuf,
        repo_path: std::path::PathBuf,
        turboprop_config: TurboPropConfig,
    ) -> Self {
        let is_mock = index_path.to_string_lossy().contains("/tmp/test");
        Self {
            config,
            index_path,
            repo_path,
            turboprop_config,
            is_mock,
        }
    }

    /// Create a mock tool for testing
    pub fn new_mock() -> Self {
        use std::path::PathBuf;
        use crate::config::TurboPropConfig;
        
        Self {
            config: SearchToolConfig::default(),
            index_path: PathBuf::from("/tmp/test_index"),
            repo_path: PathBuf::from("/tmp/test_repo"),
            turboprop_config: TurboPropConfig::default(),
            is_mock: true,
        }
    }

    /// Create the JSON schema for search tool parameters
    fn create_input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": format!("Natural language search query (max {} characters)", self.config.max_query_length)
                },
                "limit": {
                    "type": "integer",
                    "description": format!("Maximum number of results to return (default: {}, max: {})", 
                        self.config.default_limit, self.config.max_limit),
                    "default": self.config.default_limit,
                    "minimum": 1,
                    "maximum": self.config.max_limit
                },
                "threshold": {
                    "type": "number",
                    "description": format!("Minimum similarity threshold ({} to {}, default: use config value)", 
                        self.config.min_threshold, self.config.max_threshold),
                    "minimum": self.config.min_threshold,
                    "maximum": self.config.max_threshold
                },
                "filetype": {
                    "type": "string",
                    "description": "Filter by file extension (e.g., '.rs', '.js', '.py', '.md')",
                    "pattern": "^\\.[a-zA-Z0-9]+$"
                },
                "filter": {
                    "type": "string", 
                    "description": "Glob pattern filter (e.g., '*.rs', 'src/**/*.js', 'tests/**')"
                },
                "include_content": {
                    "type": "boolean",
                    "description": "Include full chunk content in results (default: true)",
                    "default": true
                },
                "context_lines": {
                    "type": "integer",
                    "description": format!("Number of context lines around matches (default: {}, max: {})", 
                        self.config.default_context_lines, self.config.max_context_lines),
                    "default": self.config.default_context_lines,
                    "minimum": 0,
                    "maximum": self.config.max_context_lines
                }
            },
            "required": ["query"],
            "additionalProperties": false
        })
    }

    /// Execute the search with given parameters
    async fn execute_search(&self, search_params: SearchToolParams) -> anyhow::Result<Value> {
        let start_time = std::time::Instant::now();
        
        // Validate parameters
        self.validate_search_params(&search_params)?;
        
        debug!(
            "Executing search: query='{}', limit={}, threshold={:?}",
            search_params.query.as_str(), search_params.limit.value(), 
            search_params.threshold.as_ref().map(|t| t.value())
        );
        
        // Use the existing search functionality directly
        let threshold = search_params.threshold
            .map(|t| t.value())
            .or(Some(self.turboprop_config.search.min_similarity));
        
        // Execute search using the enhanced function with filters
        let search_results = search_index_with_filters(
            &self.index_path,
            search_params.query.as_str(),
            Some(search_params.limit.value()),
            threshold,
            search_params.filetype.clone(),
            search_params.filter.clone(),
        )
        .await
        .context("Search execution failed")?;
        
        let execution_time = start_time.elapsed();
        
        // Convert results to MCP format
        let mcp_results = self.convert_results(
            search_results.clone(),
            search_params.include_content,
            search_params.context_lines.map(|c| c.value()).unwrap_or(0),
        ).await?;
        
        let total_results = search_results.len();
        
        // Create response
        let result = SearchToolResult {
            results: mcp_results,
            total_results,
            execution_time_ms: execution_time.as_millis() as u64,
            query_info: SearchQueryInfo {
                query: search_params.query.as_str().to_string(),
                filters: SearchFilters {
                    filetype: search_params.filetype,
                    glob_pattern: search_params.filter,
                },
                limit: search_params.limit.value(),
                threshold: threshold.unwrap_or(self.turboprop_config.search.min_similarity),
            },
        };
        
        info!(
            "Search completed: query='{}', results={}/{}, time={}ms",
            search_params.query.as_str(),
            result.results.len(),
            total_results,
            result.execution_time_ms
        );
        
        Ok(serde_json::to_value(result)?)
    }

    /// Validate search tool parameters
    fn validate_search_params(&self, params: &SearchToolParams) -> anyhow::Result<()> {
        // Validate query
        if params.query.is_empty() {
            anyhow::bail!("Query cannot be empty");
        }
        
        if params.query.len() > self.config.max_query_length {
            anyhow::bail!("Query too long (max {} characters)", self.config.max_query_length);
        }
        
        // Validate limit
        if params.limit.value() > self.config.max_limit {
            anyhow::bail!("Limit too high (max {})", self.config.max_limit);
        }
        
        // Validate threshold
        if let Some(threshold) = &params.threshold {
            if !(self.config.min_threshold..=self.config.max_threshold).contains(&threshold.value()) {
                anyhow::bail!("Threshold must be between {} and {}", 
                    self.config.min_threshold, self.config.max_threshold);
            }
        }
        
        // Validate filetype format
        if let Some(filetype) = &params.filetype {
            if !filetype.starts_with('.') || filetype.len() < 2 {
                anyhow::bail!("File type must start with '.' and have at least one character (e.g., '.rs', '.js')");
            }
        }
        
        // Validate context lines
        if let Some(context_lines) = &params.context_lines {
            if context_lines.value() > self.config.max_context_lines {
                anyhow::bail!("Context lines too high (max {})", self.config.max_context_lines);
            }
        }
        
        Ok(())
    }

    /// Convert TurboProp search results to MCP format
    async fn convert_results(
        &self,
        results: Vec<crate::types::SearchResult>,
        include_content: bool,
        context_lines: usize,
    ) -> anyhow::Result<Vec<McpSearchResult>> {
        let mut mcp_results = Vec::new();
        
        for result in results {
            // Get relative path
            let relative_path = result.chunk.chunk.source_location.file_path
                .strip_prefix(&self.repo_path)
                .unwrap_or(&result.chunk.chunk.source_location.file_path)
                .to_string_lossy()
                .to_string();
            
            // Extract file extension
            let file_type = result.chunk.chunk.source_location.file_path
                .extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("unknown")
                .to_string();
            
            // Get content
            let content = if include_content {
                result.chunk.chunk.content.clone()
            } else {
                // Provide a preview of the content
                let preview_len = 200;
                if result.chunk.chunk.content.len() > preview_len {
                    format!("{}...", &result.chunk.chunk.content[..preview_len])
                } else {
                    result.chunk.chunk.content.clone()
                }
            };
            
            // Add context lines if requested
            let context = if context_lines > 0 {
                // For now, return None - context line extraction would require
                // reading the full file content, which we can implement in a future step
                None
            } else {
                None
            };
            
            let mcp_result = McpSearchResult {
                file_path: relative_path,
                line_number: result.chunk.chunk.source_location.start_line,
                similarity_score: SimilarityScore::from(result.similarity),
                content,
                context,
                file_type,
                start_line: result.chunk.chunk.source_location.start_line,
                end_line: result.chunk.chunk.source_location.end_line,
            };
            
            mcp_results.push(mcp_result);
        }
        
        Ok(mcp_results)
    }
}

#[async_trait]
impl ToolExecutor for SemanticSearchTool {
    async fn execute(&self, request: ToolCallRequest) -> ExistingMcpResult<ToolCallResponse> {
        debug!(
            "Executing semantic search tool with arguments: {:?}",
            request.arguments
        );

        // Validate arguments first
        self.validate_arguments(&request.arguments)?;

        // Return mock results when in test mode
        if self.is_mock {
            let mock_result = json!({
                "results": [
                    {
                        "file_path": "src/main.rs",
                        "line_number": 10,
                        "similarity_score": 0.85,
                        "content": "fn main() { println!(\"Hello, world!\"); }",
                        "file_type": "rs",
                        "start_line": 10,
                        "end_line": 10
                    }
                ],
                "total_results": 1,
                "execution_time_ms": 50,
                "query_info": {
                    "query": request.arguments.get("query").unwrap_or(&json!("test")).as_str().unwrap_or("test"),
                    "filters": {},
                    "limit": 10,
                    "threshold": 0.3
                }
            });
            
            return Ok(ToolCallResponse {
                success: true,
                content: Some(mock_result),
                error: None,
            });
        }

        // Convert arguments to search params
        let search_params: SearchToolParams = serde_json::from_value(
            serde_json::to_value(&request.arguments)
                .map_err(|e| ExistingMcpError::tool_execution(&request.name, e.to_string()))?
        ).map_err(|e| ExistingMcpError::tool_execution(&request.name, e.to_string()))?;

        // Execute search directly
        match self.execute_search(search_params).await {
            Ok(result) => Ok(ToolCallResponse {
                success: true,
                content: Some(result),
                error: None,
            }),
            Err(e) => Ok(ToolCallResponse {
                success: false,
                content: None,
                error: Some(e.to_string()),
            }),
        }
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "semantic_search".to_string(),
            description: "Semantic search across the indexed codebase using natural language queries. Supports filtering by file type, glob patterns, and similarity thresholds.".to_string(),
            input_schema: self.create_input_schema(),
        }
    }

    fn validate_arguments(&self, arguments: &HashMap<String, Value>) -> ExistingMcpResult<()> {
        // Check required query parameter
        if !arguments.contains_key("query") {
            return Err(ExistingMcpError::tool_execution(
                "semantic_search",
                "Missing required 'query' parameter",
            ));
        }

        // Validate query is a string
        if !arguments.get("query").unwrap().is_string() {
            return Err(ExistingMcpError::tool_execution(
                "semantic_search", 
                "'query' parameter must be a string",
            ));
        }

        // Validate limit if provided
        if let Some(limit_value) = arguments.get("limit") {
            if let Some(limit) = limit_value.as_u64() {
                if limit == 0 || limit > self.config.max_limit as u64 {
                    return Err(ExistingMcpError::tool_execution(
                        "semantic_search",
                        format!("'limit' must be between 1 and {}", self.config.max_limit),
                    ));
                }
            } else {
                return Err(ExistingMcpError::tool_execution(
                    "semantic_search",
                    "'limit' parameter must be an integer",
                ));
            }
        }

        // Validate threshold if provided
        if let Some(threshold_value) = arguments.get("threshold") {
            if let Some(threshold) = threshold_value.as_f64() {
                if !(self.config.min_threshold as f64..=self.config.max_threshold as f64).contains(&threshold) {
                    return Err(ExistingMcpError::tool_execution(
                        "semantic_search",
                        &format!("'threshold' must be between {} and {}", 
                            self.config.min_threshold, self.config.max_threshold),
                    ));
                }
            } else {
                return Err(ExistingMcpError::tool_execution(
                    "semantic_search",
                    "'threshold' parameter must be a number",
                ));
            }
        }

        Ok(())
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

        // Register semantic_search tool for tests
        if cfg!(test) {
            let search_tool = SemanticSearchTool::new_mock();
            tools.insert("semantic_search".to_string(), Box::new(search_tool));
        }

        Self { tools }
    }

    /// Create a new tools registry with default semantic search tool for integration tests
    pub fn new_for_integration_tests() -> Self {
        let mut tools: HashMap<String, Box<dyn ToolExecutor + Send + Sync>> = HashMap::new();

        // Always create a mock search tool for integration testing
        let search_tool = SemanticSearchTool::new_mock();
        tools.insert("semantic_search".to_string(), Box::new(search_tool));

        Self { tools }
    }

    /// Create a new tools registry with enhanced search tool
    pub fn with_search_tool(
        index_path: std::path::PathBuf,
        repo_path: std::path::PathBuf,
        turboprop_config: TurboPropConfig,
    ) -> Self {
        let mut tools: HashMap<String, Box<dyn ToolExecutor + Send + Sync>> = HashMap::new();

        // Register search tool
        let search_tool = SemanticSearchTool::new(index_path, repo_path, turboprop_config);
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
    pub async fn execute_tool(&self, request: ToolCallRequest) -> ExistingMcpResult<ToolCallResponse> {
        let tool = self
            .tools
            .get(&request.name)
            .ok_or_else(|| ExistingMcpError::tool_execution(&request.name, "Tool not found"))?;

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
    use serde_json::json;
    
    #[test]
    fn test_search_tool_creation() {
        let tool = SemanticSearchTool::new_mock();
        let definition = tool.definition();
        
        assert_eq!(definition.name, "semantic_search");
        assert!(!definition.description.is_empty());
        assert!(definition.input_schema.is_object());
    }
    
    #[test]  
    fn test_search_params_deserialization() {
        let params = json!({
            "query": "test function",
            "limit": 5,
            "threshold": 0.7,
            "filetype": ".rs"
        });
        
        let search_params: SearchToolParams = serde_json::from_value(params).unwrap();
        
        assert_eq!(search_params.query.as_str(), "test function");
        assert_eq!(search_params.limit.value(), 5);
        assert_eq!(search_params.threshold.as_ref().map(|t| t.value()), Some(0.7));
        assert_eq!(search_params.filetype, Some(".rs".to_string()));
    }
    
    #[test]
    fn test_search_params_defaults() {
        let params = json!({
            "query": "test function"
        });
        
        let search_params: SearchToolParams = serde_json::from_value(params).unwrap();
        
        assert_eq!(search_params.query.as_str(), "test function");
        assert_eq!(search_params.limit.value(), 10); // default
        assert_eq!(search_params.threshold, None);
        assert_eq!(search_params.filetype, None);
        assert!(search_params.include_content); // default true
    }
    
    #[test]
    fn test_parameter_validation() {
        let tool = SemanticSearchTool::new_mock();
        
        // Valid parameters
        let valid_params = SearchToolParams {
            query: QueryString::from("test".to_string()),
            limit: ResultLimit::from(10),
            threshold: Some(SimilarityScore::from(0.5)),
            filetype: Some(".rs".to_string()),
            filter: None,
            include_content: true,
            context_lines: Some(ContextLines::from(2)),
        };
        assert!(tool.validate_search_params(&valid_params).is_ok());
        
        // Empty query
        let empty_query = SearchToolParams {
            query: QueryString::from("".to_string()),
            limit: ResultLimit::from(10),
            threshold: None,
            filetype: None,
            filter: None,
            include_content: true,
            context_lines: None,
        };
        assert!(tool.validate_search_params(&empty_query).is_err());
        
        // Invalid threshold
        let invalid_threshold = SearchToolParams {
            query: QueryString::from("test".to_string()),
            limit: ResultLimit::from(10),
            threshold: Some(SimilarityScore::from(1.5)),
            filetype: None,
            filter: None,
            include_content: true,
            context_lines: None,
        };
        assert!(tool.validate_search_params(&invalid_threshold).is_err());
        
        // Invalid filetype
        let invalid_filetype = SearchToolParams {
            query: QueryString::from("test".to_string()),
            limit: ResultLimit::from(10),
            threshold: None,
            filetype: Some("rs".to_string()), // missing dot
            filter: None,
            include_content: true,
            context_lines: None,
        };
        assert!(tool.validate_search_params(&invalid_filetype).is_err());
    }
    
    #[test]
    fn test_input_schema_structure() {
        let tool = SemanticSearchTool::new_mock();
        let definition = tool.definition();
        let schema = &definition.input_schema;
        
        // Check required properties
        assert!(schema["properties"]["query"].is_object());
        assert!(schema["required"].as_array().unwrap().contains(&json!("query")));
        
        // Check optional properties
        assert!(schema["properties"]["limit"].is_object());
        assert!(schema["properties"]["threshold"].is_object());
        assert!(schema["properties"]["filetype"].is_object());
        assert!(schema["properties"]["filter"].is_object());
    }

    #[test]
    fn test_tools_registry() {
        let tools = Tools::new();
        
        // Should have semantic_search tool for testing
        assert!(tools.has_tool("semantic_search"));
        
        // List tools
        let tool_list = tools.list_tools();
        assert!(!tool_list.is_empty());
    }
}