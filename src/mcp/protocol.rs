//! MCP JSON-RPC 2.0 protocol message types
//!
//! Implements the core message structures for the Model Context Protocol
//! following the JSON-RPC 2.0 specification.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// JSON-RPC 2.0 request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// JSON-RPC version (must be "2.0")
    pub jsonrpc: String,
    /// Request identifier (required for MCP)
    pub id: Value,
    /// Method name
    pub method: String,
    /// Optional parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

/// JSON-RPC 2.0 response message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    /// JSON-RPC version (must be "2.0")
    pub jsonrpc: String,
    /// Request identifier (echoed from request)
    pub id: Value,
    /// Success result (mutually exclusive with error)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error details (mutually exclusive with result)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 error object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    /// Error code
    pub code: i32,
    /// Error message
    pub message: String,
    /// Optional additional error data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// MCP initialization request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeParams {
    /// Protocol version
    pub protocol_version: String,
    /// Client information
    pub client_info: ClientInfo,
    /// Client capabilities
    pub capabilities: ClientCapabilities,
}

/// Client information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    /// Client name
    pub name: String,
    /// Client version
    pub version: String,
}

/// Client capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]  
pub struct ClientCapabilities {
    /// Supports experimental features
    #[serde(default)]
    pub experimental: HashMap<String, Value>,
}

/// MCP initialization response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeResult {
    /// Protocol version
    pub protocol_version: String,
    /// Server information
    pub server_info: ServerInfo,
    /// Server capabilities
    pub capabilities: ServerCapabilities,
}

/// Server information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    /// Server name
    pub name: String,
    /// Server version
    pub version: String,
}

/// Server capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilities {
    /// Tools capability
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
    /// Experimental capabilities
    #[serde(default)]
    pub experimental: HashMap<String, Value>,
}

/// Tools capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsCapability {
    /// List of available tools is static
    #[serde(default)]
    pub list_changed: bool,
}

impl JsonRpcRequest {
    /// Create a new JSON-RPC request
    pub fn new(id: Value, method: String, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            method,
            params,
        }
    }
}

impl JsonRpcResponse {
    /// Create a successful response
    pub fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }
    
    /// Create an error response
    pub fn error(id: Value, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(error),
        }
    }
}

impl JsonRpcError {
    /// Create a new error
    pub fn new(code: i32, message: String, data: Option<Value>) -> Self {
        Self { code, message, data }
    }
    
    /// Create a parse error (-32700)
    pub fn parse_error(message: String) -> Self {
        Self::new(-32700, message, None)
    }
    
    /// Create an invalid request error (-32600)
    pub fn invalid_request(message: String) -> Self {
        Self::new(-32600, message, None)
    }
    
    /// Create a method not found error (-32601)
    pub fn method_not_found(method: String) -> Self {
        Self::new(-32601, format!("Method not found: {}", method), None)
    }
    
    /// Create an invalid params error (-32602)
    pub fn invalid_params(message: String) -> Self {
        Self::new(-32602, message, None)
    }
    
    /// Create an internal error (-32603)
    pub fn internal_error(message: String) -> Self {
        Self::new(-32603, message, None)
    }
}

/// MCP protocol constants
pub mod constants {
    /// Supported MCP protocol version
    pub const PROTOCOL_VERSION: &str = "2024-11-05";
    
    /// Server name
    pub const SERVER_NAME: &str = "turboprop";
    
    /// Server version (should match Cargo.toml version)
    pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");
    
    /// MCP method names
    pub mod methods {
        pub const INITIALIZE: &str = "initialize";
        pub const TOOLS_LIST: &str = "tools/list";
        pub const TOOLS_CALL: &str = "tools/call";
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    
    #[test]
    fn test_json_rpc_request_serialization() {
        let request = JsonRpcRequest::new(
            json!(1),
            "test_method".to_string(),
            Some(json!({"param": "value"}))
        );
        
        let serialized = serde_json::to_string(&request).unwrap();
        let deserialized: JsonRpcRequest = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(request.jsonrpc, "2.0");
        assert_eq!(deserialized.method, "test_method");
    }
    
    #[test]
    fn test_json_rpc_response_success() {
        let response = JsonRpcResponse::success(json!(1), json!({"success": true}));
        
        assert_eq!(response.jsonrpc, "2.0");
        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }
    
    #[test]
    fn test_json_rpc_response_error() {
        let error = JsonRpcError::method_not_found("unknown_method".to_string());
        let response = JsonRpcResponse::error(json!(1), error);
        
        assert_eq!(response.jsonrpc, "2.0");
        assert!(response.result.is_none());
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32601);
    }
}