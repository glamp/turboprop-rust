//! MCP server implementation
//!
//! Provides the core MCP server that handles JSON-RPC protocol communication
//! and integrates with TurboProp's semantic search capabilities.

use crate::mcp::error::{McpError, McpResult};
use crate::mcp::protocol::{
    ClientCapabilities, InitializeParams, InitializeResult, JsonRpcRequest, JsonRpcResponse,
    ServerCapabilities, ServerInfo, ToolsCapability,
};
use crate::mcp::transport::StdioTransport;
use std::collections::HashMap;
use tracing::{debug, error, info};

/// Configuration for the MCP server
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    /// Server listening address
    pub address: String,
    /// Server listening port
    pub port: u16,
    /// Maximum number of concurrent connections
    pub max_connections: usize,
    /// Request timeout in seconds
    pub request_timeout_seconds: u64,
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            address: "127.0.0.1".to_string(),
            port: 8080,
            max_connections: 100,
            request_timeout_seconds: 30,
        }
    }
}

/// Trait defining the core MCP server interface
pub trait McpServerTrait {
    /// Initialize the server with client capabilities
    async fn initialize(&mut self, params: InitializeParams) -> McpResult<InitializeResult>;

    /// Handle incoming JSON-RPC request
    async fn handle_request(&self, request: JsonRpcRequest) -> McpResult<JsonRpcResponse>;

    /// Start the server
    async fn start(&mut self) -> McpResult<()>;

    /// Stop the server
    async fn stop(&mut self) -> McpResult<()>;

    /// Check if server is running
    fn is_running(&self) -> bool;
}

/// MCP server implementation
pub struct McpServer {
    config: McpServerConfig,
    transport: StdioTransport,
    initialized: bool,
    running: bool,
    client_capabilities: Option<ClientCapabilities>,
}

impl McpServer {
    /// Create a new MCP server with default configuration
    pub fn new() -> Self {
        Self::with_config(McpServerConfig::default())
    }

    /// Create a new MCP server with custom configuration
    pub fn with_config(config: McpServerConfig) -> Self {
        Self {
            config,
            transport: StdioTransport::new(),
            initialized: false,
            running: false,
            client_capabilities: None,
        }
    }

    /// Get server configuration
    pub fn config(&self) -> &McpServerConfig {
        &self.config
    }

    /// Create server capabilities based on TurboProp features
    fn create_server_capabilities() -> ServerCapabilities {
        let mut experimental = HashMap::new();
        experimental.insert("semantic_search".to_string(), serde_json::json!(true));

        ServerCapabilities {
            tools: Some(ToolsCapability {
                list_changed: false,
            }),
            experimental,
        }
    }

    /// Create server information
    fn create_server_info() -> ServerInfo {
        ServerInfo {
            name: crate::mcp::protocol::constants::SERVER_NAME.to_string(),
            version: crate::mcp::protocol::constants::SERVER_VERSION.to_string(),
        }
    }
}

impl McpServerTrait for McpServer {
    async fn initialize(&mut self, params: InitializeParams) -> McpResult<InitializeResult> {
        info!(
            "Initializing MCP server with client: {}",
            params.client_info.name
        );

        // Validate protocol version
        if params.protocol_version != crate::mcp::protocol::constants::PROTOCOL_VERSION {
            return Err(McpError::protocol(format!(
                "Unsupported protocol version: {}. Expected: {}",
                params.protocol_version,
                crate::mcp::protocol::constants::PROTOCOL_VERSION
            )));
        }

        // Store client capabilities
        self.client_capabilities = Some(params.capabilities);
        self.initialized = true;

        debug!("MCP server initialized successfully");

        Ok(InitializeResult {
            protocol_version: crate::mcp::protocol::constants::PROTOCOL_VERSION.to_string(),
            server_info: Self::create_server_info(),
            capabilities: Self::create_server_capabilities(),
        })
    }

    async fn handle_request(&self, request: JsonRpcRequest) -> McpResult<JsonRpcResponse> {
        if !self.initialized
            && request.method != crate::mcp::protocol::constants::methods::INITIALIZE
        {
            return Err(McpError::protocol(
                "Server not initialized. Call 'initialize' first.".to_string(),
            ));
        }

        debug!("Handling MCP request: {}", request.method);

        match request.method.as_str() {
            crate::mcp::protocol::constants::methods::INITIALIZE => {
                Err(McpError::protocol("Server already initialized".to_string()))
            }
            crate::mcp::protocol::constants::methods::TOOLS_LIST => {
                // TODO: Implement tools listing
                Ok(JsonRpcResponse::from_request_success(
                    &request,
                    serde_json::json!({"tools": []}),
                ))
            }
            crate::mcp::protocol::constants::methods::TOOLS_CALL => {
                // TODO: Implement tool calling
                Err(McpError::tool_execution(
                    "unknown",
                    "Tool execution not yet implemented",
                ))
            }
            _ => {
                error!("Unknown method: {}", request.method);
                Err(McpError::protocol(format!(
                    "Method not found: {}",
                    request.method
                )))
            }
        }
    }

    async fn start(&mut self) -> McpResult<()> {
        info!(
            "Starting MCP server on {}:{}",
            self.config.address, self.config.port
        );

        if self.running {
            return Err(McpError::server_initialization(
                "Server already running".to_string(),
            ));
        }

        // TODO: Implement actual server startup logic
        self.running = true;

        info!("MCP server started successfully");

        Ok(())
    }

    async fn stop(&mut self) -> McpResult<()> {
        info!("Stopping MCP server");

        if !self.running {
            return Err(McpError::server_initialization(
                "Server not running".to_string(),
            ));
        }

        // TODO: Implement actual server shutdown logic
        self.running = false;

        info!("MCP server stopped successfully");

        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running
    }
}

impl Default for McpServer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::protocol::{ClientCapabilities, ClientInfo};
    use std::collections::HashMap;

    fn create_test_initialize_params() -> InitializeParams {
        InitializeParams {
            protocol_version: crate::mcp::protocol::constants::PROTOCOL_VERSION.to_string(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities {
                experimental: HashMap::new(),
            },
        }
    }

    #[tokio::test]
    async fn test_server_initialization() {
        let mut server = McpServer::new();
        let params = create_test_initialize_params();

        let result = server.initialize(params).await;
        assert!(result.is_ok());

        let init_result = result.unwrap();
        assert_eq!(
            init_result.protocol_version,
            crate::mcp::protocol::constants::PROTOCOL_VERSION
        );
        assert_eq!(
            init_result.server_info.name,
            crate::mcp::protocol::constants::SERVER_NAME
        );
    }

    #[tokio::test]
    async fn test_server_lifecycle() {
        let mut server = McpServer::new();

        assert!(!server.is_running());

        let start_result = server.start().await;
        assert!(start_result.is_ok());
        assert!(server.is_running());

        let stop_result = server.stop().await;
        assert!(stop_result.is_ok());
        assert!(!server.is_running());
    }
}
