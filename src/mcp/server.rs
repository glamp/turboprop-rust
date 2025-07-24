//! Main MCP server implementation
//!
//! Coordinates file watching, incremental indexing, and search tool handling

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::config::TurboPropConfig;
use crate::index::PersistentChunkIndex;
use crate::types::{ConnectionLimit, Port, TimeoutSeconds};

use super::protocol::{
    constants, InitializeParams, InitializeResult, JsonRpcError, JsonRpcRequest, JsonRpcResponse,
    ServerCapabilities, ServerInfo, ToolsCapability,
};
use super::tools::Tools;
use super::transport::StdioTransport;

/// Configuration for MCP server
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    /// Server address (for future TCP transport)
    pub address: String,
    /// Server port (for future TCP transport)
    pub port: Port,
    /// Maximum number of connections
    pub max_connections: ConnectionLimit,
    /// Request timeout in seconds
    pub request_timeout: TimeoutSeconds,
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            address: "127.0.0.1".to_string(),
            port: Port::dynamic(), // STDIO by default (port 0)
            max_connections: ConnectionLimit::default(),
            request_timeout: TimeoutSeconds::default(),
        }
    }
}

/// Trait defining the MCP server interface
#[async_trait]
pub trait McpServerTrait {
    /// Initialize the server with given parameters
    async fn initialize(&mut self, params: InitializeParams) -> Result<InitializeResult>;

    /// Handle a request
    async fn handle_request(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse>;

    /// Check if server is running
    fn is_running(&self) -> bool;
}

/// Main MCP server
pub struct McpServer {
    /// Repository path being indexed
    repo_path: PathBuf,
    /// TurboProp configuration
    config: TurboPropConfig,
    /// Server configuration
    #[allow(dead_code)]
    server_config: McpServerConfig,
    /// Tools registry
    tools: Tools,
    /// Persistent index (wrapped for thread safety)
    index: Arc<RwLock<Option<PersistentChunkIndex>>>,
    /// Server initialization state
    initialized: Arc<RwLock<bool>>,
    /// Server running state
    #[allow(dead_code)]
    running: Arc<RwLock<bool>>,
}

impl McpServer {
    /// Create a new MCP server
    pub async fn new(repo_path: &Path, config: &TurboPropConfig) -> Result<Self> {
        info!("Initializing MCP server for {}", repo_path.display());

        let tools = Tools::with_search_tool(
            repo_path.to_path_buf(),
            repo_path.to_path_buf(),
            config.clone(),
        );

        let server = Self {
            repo_path: repo_path.to_path_buf(),
            config: config.clone(),
            server_config: McpServerConfig::default(),
            tools,
            index: Arc::new(RwLock::new(None)),
            initialized: Arc::new(RwLock::new(false)),
            running: Arc::new(RwLock::new(false)),
        };

        Ok(server)
    }

    /// Create a new MCP server with custom configuration and tools
    pub fn with_config_and_tools(server_config: McpServerConfig, tools: Tools) -> Self {
        Self {
            repo_path: PathBuf::new(), // Will be set later
            config: TurboPropConfig::default(),
            server_config,
            tools,
            index: Arc::new(RwLock::new(None)),
            initialized: Arc::new(RwLock::new(false)),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Run the MCP server
    pub async fn run(self) -> Result<()> {
        let server = Arc::new(self);

        // Initialize transport
        let mut transport = StdioTransport::new();

        info!("MCP server ready and listening on stdio...");

        // Main message processing loop
        loop {
            match transport.receive_request().await {
                Some(Ok(request)) => {
                    let response = server.handle_request_internal(request).await;
                    if let Err(e) = transport.send_response(response).await {
                        error!("Failed to send response: {}", e);
                        break;
                    }
                }
                Some(Err(e)) => {
                    error!("Error receiving request: {}", e);
                    // Send error response if possible
                    let error_response = StdioTransport::create_error_response(
                        None,
                        JsonRpcError::parse_error(e.to_string()),
                    );
                    let _ = transport.send_response(error_response).await;
                }
                None => {
                    // STDIN closed
                    info!("STDIN closed, shutting down MCP server");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle an incoming JSON-RPC request (internal implementation)
    async fn handle_request_internal(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        debug!(
            "Handling request: method={}, id={:?}",
            request.method, request.id
        );

        // Validate request format
        if let Err(error) = request.validate() {
            return request.create_error_response(error);
        }

        // Dispatch to appropriate handler
        match request.method.as_str() {
            constants::methods::INITIALIZE => self.handle_initialize(request).await,
            constants::methods::TOOLS_LIST => self.handle_tools_list(request).await,
            constants::methods::TOOLS_CALL => self.handle_tools_call(request).await,
            _ => {
                let error = JsonRpcError::method_not_found(request.method.clone());
                request.create_error_response(error)
            }
        }
    }

    /// Handle MCP initialization
    async fn handle_initialize(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        debug!("Handling initialize request");

        // Parse initialization parameters
        let params = match &request.params {
            Some(params) => match serde_json::from_value::<InitializeParams>(params.clone()) {
                Ok(params) => params,
                Err(e) => {
                    let error =
                        JsonRpcError::invalid_params(format!("Invalid initialize params: {}", e));
                    return request.create_error_response(error);
                }
            },
            None => {
                let error =
                    JsonRpcError::invalid_params("Missing initialize parameters".to_string());
                return request.create_error_response(error);
            }
        };

        info!(
            "Initializing MCP server for client: {} v{}",
            params.client_info.name, params.client_info.version
        );

        // Validate protocol version
        if params.protocol_version != constants::PROTOCOL_VERSION {
            warn!(
                "Client protocol version {} differs from server version {}",
                params.protocol_version,
                constants::PROTOCOL_VERSION
            );
        }

        // Start index initialization in background
        let index_clone = Arc::clone(&self.index);
        let repo_path = self.repo_path.clone();
        let config = self.config.clone();
        let initialized_flag = Arc::clone(&self.initialized);

        tokio::spawn(async move {
            match Self::initialize_index(&repo_path, &config).await {
                Ok(index) => {
                    {
                        let mut index_guard = index_clone.write().await;
                        *index_guard = Some(index);
                    }
                    {
                        let mut initialized_guard = initialized_flag.write().await;
                        *initialized_guard = true;
                    }
                    info!("Index initialization completed");
                }
                Err(e) => {
                    error!("Failed to initialize index: {}", e);
                    // Note: Server continues to run but search will return errors
                }
            }
        });

        // Create initialization result
        let result = InitializeResult {
            protocol_version: constants::PROTOCOL_VERSION.to_string(),
            server_info: ServerInfo {
                name: constants::SERVER_NAME.to_string(),
                version: constants::SERVER_VERSION.to_string(),
            },
            capabilities: ServerCapabilities {
                tools: Some(ToolsCapability {
                    list_changed: false, // Static tool list
                }),
                experimental: std::collections::HashMap::new(),
            },
        };

        match serde_json::to_value(result) {
            Ok(result_value) => {
                info!("MCP server initialized successfully");
                request.create_success_response(result_value)
            }
            Err(e) => {
                let error =
                    JsonRpcError::internal_error(format!("Failed to serialize result: {}", e));
                request.create_error_response(error)
            }
        }
    }

    /// Handle tools/list request
    async fn handle_tools_list(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        debug!("Handling tools/list request");

        // Check if server is initialized first
        let initialized = {
            let initialized_guard = self.initialized.read().await;
            *initialized_guard
        };

        if !initialized {
            let error = JsonRpcError::internal_error("Server not initialized");
            return request.create_error_response(error);
        }

        // Use the tools registry
        let tools = self.tools.list_tools();

        let result = json!({
            "tools": tools
        });

        request.create_success_response(result)
    }

    /// Handle tools/call request
    async fn handle_tools_call(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        debug!("Handling tools/call request");

        // Check if server is initialized
        let initialized = {
            let initialized_guard = self.initialized.read().await;
            *initialized_guard
        };

        if !initialized {
            let error = JsonRpcError::index_not_ready();
            return request.create_error_response(error);
        }

        // Parse tool call parameters
        let params = match &request.params {
            Some(params) => params.clone(),
            None => {
                let error =
                    JsonRpcError::invalid_params("Missing tool call parameters".to_string());
                return request.create_error_response(error);
            }
        };

        // Extract tool name and arguments
        let tool_name = match params.get("name").and_then(|v| v.as_str()) {
            Some(name) => name,
            None => {
                let error = JsonRpcError::invalid_params("Missing tool name".to_string());
                return request.create_error_response(error);
            }
        };

        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

        // Use the tools registry
        let tool_call_request = crate::mcp::tools::ToolCallRequest {
            name: tool_name.to_string(),
            arguments: serde_json::from_value(arguments).unwrap_or_default(),
        };

        match self.tools.execute_tool(tool_call_request).await {
            Ok(tool_response) => {
                if tool_response.success {
                    debug!("Tool executed successfully: {}", tool_name);
                    let result = tool_response.content.unwrap_or(json!({}));
                    request.create_success_response(result)
                } else {
                    error!(
                        "Tool execution failed: {}",
                        tool_response
                            .error
                            .as_ref()
                            .unwrap_or(&"Unknown error".to_string())
                    );
                    let error = JsonRpcError::tool_execution_error(
                        tool_response
                            .error
                            .unwrap_or_else(|| "Unknown error".to_string()),
                    );
                    request.create_error_response(error)
                }
            }
            Err(e) => {
                error!("Tool execution failed: {}", e);
                let error = JsonRpcError::tool_execution_error(e.to_string());
                request.create_error_response(error)
            }
        }
    }

    /// Initialize the search index
    async fn initialize_index(
        repo_path: &Path,
        config: &TurboPropConfig,
    ) -> Result<PersistentChunkIndex> {
        info!("Initializing search index for {}", repo_path.display());

        // Check if index already exists by trying to load it
        let index = if let Ok(existing_index) = PersistentChunkIndex::load(repo_path) {
            info!("Loading existing index from {}", repo_path.display());
            existing_index
        } else {
            info!("Creating new index");

            // Use existing TurboProp indexing logic
            let index = crate::commands::index::build_index(repo_path, config)
                .await
                .context("Failed to build initial index")?;

            info!("Index created successfully with {} chunks", index.len());
            index
        };

        Ok(index)
    }

    /// Initialize the server (public method for tests)
    pub async fn initialize(&mut self, params: InitializeParams) -> Result<InitializeResult> {
        info!(
            "Initializing MCP server for client: {} v{}",
            params.client_info.name, params.client_info.version
        );

        // Validate protocol version
        if params.protocol_version != constants::PROTOCOL_VERSION {
            anyhow::bail!(
                "Unsupported protocol version: {} (expected: {})",
                params.protocol_version,
                constants::PROTOCOL_VERSION
            );
        }

        // Validate parameters
        params
            .validate()
            .map_err(|e| anyhow::anyhow!("Invalid initialization parameters: {:?}", e))?;

        // Mark as initialized
        {
            let mut initialized_guard = self.initialized.write().await;
            *initialized_guard = true;
        }

        // Create initialization result
        let result = InitializeResult {
            protocol_version: constants::PROTOCOL_VERSION.to_string(),
            server_info: ServerInfo {
                name: constants::SERVER_NAME.to_string(),
                version: constants::SERVER_VERSION.to_string(),
            },
            capabilities: ServerCapabilities {
                tools: Some(ToolsCapability {
                    list_changed: false, // Static tool list
                }),
                experimental: std::collections::HashMap::new(),
            },
        };

        info!("MCP server initialized successfully");
        Ok(result)
    }

    /// Check if the server is running
    pub fn is_running(&self) -> bool {
        // For now, return false by default since we don't track running state in the current implementation
        // In a real implementation, this would check if the server loop is active
        false
    }
}

/// MCP server builder for configuration
pub struct McpServerBuilder {
    repo_path: Option<PathBuf>,
    config: Option<TurboPropConfig>,
}

impl McpServerBuilder {
    /// Create a new server builder
    pub fn new() -> Self {
        Self {
            repo_path: None,
            config: None,
        }
    }

    /// Set the repository path
    pub fn repo_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.repo_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set the configuration
    pub fn config(mut self, config: TurboPropConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Build the MCP server
    pub async fn build(self) -> Result<McpServer> {
        let repo_path = self
            .repo_path
            .ok_or_else(|| anyhow::anyhow!("Repository path is required"))?;
        let config = self
            .config
            .ok_or_else(|| anyhow::anyhow!("Configuration is required"))?;

        McpServer::new(&repo_path, &config).await
    }
}

impl Default for McpServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Implementation of McpServerTrait for McpServer
#[async_trait]
impl McpServerTrait for McpServer {
    async fn initialize(&mut self, params: InitializeParams) -> Result<InitializeResult> {
        self.initialize(params).await
    }

    async fn handle_request(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
        // The internal method returns JsonRpcResponse, but the trait expects Result<JsonRpcResponse>
        // We need to wrap the response in Ok() for success cases
        let response = self.handle_request_internal(request).await;

        // Check if the response contains an error and convert to Result accordingly
        if response.error.is_some() {
            // Extract error message for the Err case
            let error_msg = response
                .error
                .as_ref()
                .map(|e| e.message.clone())
                .unwrap_or_else(|| "Unknown error".to_string());
            Err(anyhow::anyhow!("Request failed: {}", error_msg))
        } else {
            Ok(response)
        }
    }

    fn is_running(&self) -> bool {
        self.is_running()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_mcp_server_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();

        let server = McpServer::new(temp_dir.path(), &config).await;
        assert!(server.is_ok());
    }

    #[tokio::test]
    async fn test_initialize_request() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();
        let server = McpServer::new(temp_dir.path(), &config).await.unwrap();

        let request = JsonRpcRequest::new(
            constants::methods::INITIALIZE.to_string(),
            Some(json!({
                "protocol_version": constants::PROTOCOL_VERSION,
                "client_info": {
                    "name": "test-client",
                    "version": "1.0.0"
                },
                "capabilities": {}
            })),
        );

        let response = server.handle_initialize(request).await;

        if let Some(error) = &response.error {
            println!("Error: {:?}", error);
        }

        assert!(response.error.is_none());
        assert!(response.result.is_some());

        let result = response.result.unwrap();
        println!("Result: {:?}", result);
        assert_eq!(result["protocol_version"], constants::PROTOCOL_VERSION);
        assert_eq!(result["server_info"]["name"], constants::SERVER_NAME);
    }

    #[tokio::test]
    async fn test_tools_list_request() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();
        let server = McpServer::new(temp_dir.path(), &config).await.unwrap();

        // Mark server as initialized for this test
        {
            let mut initialized_guard = server.initialized.write().await;
            *initialized_guard = true;
        }

        let request = JsonRpcRequest::new(constants::methods::TOOLS_LIST.to_string(), None);

        let response = server.handle_tools_list(request).await;

        assert!(response.error.is_none());
        assert!(response.result.is_some());

        let result = response.result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "semantic_search");
    }

    #[test]
    fn test_server_builder() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();

        let builder = McpServerBuilder::new()
            .repo_path(temp_dir.path())
            .config(config);

        // Builder should be created successfully
        // Actual build() test would require async context
        drop(builder);
    }

    #[tokio::test]
    async fn test_invalid_method_request() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();
        let server = McpServer::new(temp_dir.path(), &config).await.unwrap();

        let request = JsonRpcRequest::new("invalid_method".to_string(), None);

        let response = server.handle_request_internal(request).await;

        assert!(response.result.is_none());
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32601); // Method not found
    }

    #[tokio::test]
    async fn test_tools_call_before_initialization() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();
        let server = McpServer::new(temp_dir.path(), &config).await.unwrap();

        let request = JsonRpcRequest::new(
            constants::methods::TOOLS_CALL.to_string(),
            Some(json!({
                "name": "semantic_search",
                "arguments": {
                    "query": "test"
                }
            })),
        );

        let response = server.handle_tools_call(request).await;

        assert!(response.result.is_none());
        assert!(response.error.is_some());
        // Should return index not ready error
    }
}
