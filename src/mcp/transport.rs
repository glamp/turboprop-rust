//! MCP transport layer
//!
//! Provides transport layer abstractions for MCP communication
//! supporting different transport mechanisms like stdio and HTTP.

use crate::mcp::error::{McpError, McpResult};
use crate::mcp::protocol::{JsonRpcRequest, JsonRpcResponse};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use tracing::{debug, error, info};

/// Transport configuration
#[derive(Debug, Clone)]
pub struct TransportConfig {
    /// Transport type
    pub transport_type: TransportType,
    /// Transport-specific settings
    pub settings: HashMap<String, Value>,
}

/// Supported transport types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportType {
    /// Standard I/O transport (stdin/stdout)
    Stdio,
    /// HTTP transport
    Http,
    /// WebSocket transport
    WebSocket,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            transport_type: TransportType::Stdio,
            settings: HashMap::new(),
        }
    }
}

/// Trait defining transport layer interface
#[async_trait]
pub trait TransportLayer {
    /// Initialize the transport
    async fn initialize(&mut self, config: TransportConfig) -> McpResult<()>;

    /// Send a JSON-RPC response
    async fn send_response(&self, response: JsonRpcResponse) -> McpResult<()>;

    /// Receive a JSON-RPC request (blocking)
    async fn receive_request(&self) -> McpResult<JsonRpcRequest>;

    /// Close the transport connection
    async fn close(&mut self) -> McpResult<()>;

    /// Check if transport is connected
    fn is_connected(&self) -> bool;
}

/// Standard I/O transport implementation
pub struct StdioTransport {
    initialized: bool,
    connected: bool,
}

impl StdioTransport {
    /// Create a new stdio transport
    pub fn new() -> Self {
        Self {
            initialized: false,
            connected: false,
        }
    }
}

#[async_trait]
impl TransportLayer for StdioTransport {
    async fn initialize(&mut self, config: TransportConfig) -> McpResult<()> {
        if config.transport_type != TransportType::Stdio {
            return Err(McpError::transport("Invalid transport type for StdioTransport"));
        }

        info!("Initializing stdio transport");
        self.initialized = true;
        self.connected = true;

        debug!("Stdio transport initialized successfully");
        Ok(())
    }

    async fn send_response(&self, response: JsonRpcResponse) -> McpResult<()> {
        if !self.connected {
            return Err(McpError::transport("Transport not connected"));
        }

        debug!("Sending JSON-RPC response via stdio: {:?}", response.id);

        // TODO: Implement actual stdio output
        let json_str = serde_json::to_string(&response)
            .map_err(|e| McpError::transport(format!("Failed to serialize response: {}", e)))?;

        println!("{}", json_str);

        Ok(())
    }

    async fn receive_request(&self) -> McpResult<JsonRpcRequest> {
        if !self.connected {
            return Err(McpError::transport("Transport not connected"));
        }

        debug!("Waiting for JSON-RPC request via stdio");

        // TODO: Implement actual stdio input reading
        // For now, return a mock request to demonstrate the interface
        Err(McpError::transport("Stdio input reading not yet implemented"))
    }

    async fn close(&mut self) -> McpResult<()> {
        info!("Closing stdio transport");
        self.connected = false;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected
    }
}

impl Default for StdioTransport {
    fn default() -> Self {
        Self::new()
    }
}

/// HTTP transport implementation
pub struct HttpTransport {
    initialized: bool,
    connected: bool,
    config: Option<TransportConfig>,
}

impl HttpTransport {
    /// Create a new HTTP transport
    pub fn new() -> Self {
        Self {
            initialized: false,
            connected: false,
            config: None,
        }
    }
}

#[async_trait]
impl TransportLayer for HttpTransport {
    async fn initialize(&mut self, config: TransportConfig) -> McpResult<()> {
        if config.transport_type != TransportType::Http {
            return Err(McpError::transport("Invalid transport type for HttpTransport"));
        }

        info!("Initializing HTTP transport");
        self.config = Some(config);
        self.initialized = true;
        self.connected = true;

        debug!("HTTP transport initialized successfully");
        Ok(())
    }

    async fn send_response(&self, response: JsonRpcResponse) -> McpResult<()> {
        if !self.connected {
            return Err(McpError::transport("Transport not connected"));
        }

        debug!("Sending JSON-RPC response via HTTP: {:?}", response.id);

        // TODO: Implement actual HTTP response sending
        error!("HTTP response sending not yet implemented");
        Err(McpError::transport("HTTP response sending not yet implemented"))
    }

    async fn receive_request(&self) -> McpResult<JsonRpcRequest> {
        if !self.connected {
            return Err(McpError::transport("Transport not connected"));
        }

        debug!("Waiting for JSON-RPC request via HTTP");

        // TODO: Implement actual HTTP request handling
        Err(McpError::transport("HTTP request handling not yet implemented"))
    }

    async fn close(&mut self) -> McpResult<()> {
        info!("Closing HTTP transport");
        self.connected = false;
        self.config = None;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected
    }
}

impl Default for HttpTransport {
    fn default() -> Self {
        Self::new()
    }
}

/// Main transport abstraction that can use different underlying transports
pub struct Transport {
    inner: Box<dyn TransportLayer + Send + Sync>,
    transport_type: TransportType,
}

impl Transport {
    /// Create a new transport with stdio (default)
    pub fn new() -> Self {
        Self::with_stdio()
    }

    /// Create a new transport with stdio
    pub fn with_stdio() -> Self {
        Self {
            inner: Box::new(StdioTransport::new()),
            transport_type: TransportType::Stdio,
        }
    }

    /// Create a new transport with HTTP
    pub fn with_http() -> Self {
        Self {
            inner: Box::new(HttpTransport::new()),
            transport_type: TransportType::Http,
        }
    }

    /// Initialize the transport
    pub async fn initialize(&mut self, config: TransportConfig) -> McpResult<()> {
        self.transport_type = config.transport_type.clone();
        self.inner.initialize(config).await
    }

    /// Send a JSON-RPC response
    pub async fn send_response(&self, response: JsonRpcResponse) -> McpResult<()> {
        self.inner.send_response(response).await
    }

    /// Receive a JSON-RPC request
    pub async fn receive_request(&self) -> McpResult<JsonRpcRequest> {
        self.inner.receive_request().await
    }

    /// Close the transport
    pub async fn close(&mut self) -> McpResult<()> {
        self.inner.close().await
    }

    /// Check if transport is connected
    pub fn is_connected(&self) -> bool {
        self.inner.is_connected()
    }

    /// Get transport type
    pub fn transport_type(&self) -> &TransportType {
        &self.transport_type
    }
}

impl Default for Transport {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::protocol::JsonRpcError;

    #[tokio::test]
    async fn test_stdio_transport_initialization() {
        let mut transport = StdioTransport::new();
        let config = TransportConfig {
            transport_type: TransportType::Stdio,
            settings: HashMap::new(),
        };

        let result = transport.initialize(config).await;
        assert!(result.is_ok());
        assert!(transport.is_connected());
    }

    #[tokio::test]
    async fn test_http_transport_initialization() {
        let mut transport = HttpTransport::new();
        let config = TransportConfig {
            transport_type: TransportType::Http,
            settings: HashMap::new(),
        };

        let result = transport.initialize(config).await;
        assert!(result.is_ok());
        assert!(transport.is_connected());
    }

    #[tokio::test]
    async fn test_transport_factory() {
        let stdio_transport = Transport::with_stdio();
        assert_eq!(*stdio_transport.transport_type(), TransportType::Stdio);

        let http_transport = Transport::with_http();
        assert_eq!(*http_transport.transport_type(), TransportType::Http);
    }

    #[tokio::test]
    async fn test_stdio_response_sending() {
        let mut transport = StdioTransport::new();
        let config = TransportConfig::default();
        transport.initialize(config).await.unwrap();

        let response = JsonRpcResponse::success(
            crate::mcp::protocol::RequestId::from_number(1),
            serde_json::json!({"result": "test"}),
        );

        let result = transport.send_response(response).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_transport_error_when_not_connected() {
        let transport = StdioTransport::new();
        let response = JsonRpcResponse::error(
            crate::mcp::protocol::RequestId::from_number(1),
            JsonRpcError::internal_error("test error".to_string()),
        );

        let result = transport.send_response(response).await;
        assert!(result.is_err());
    }
}
