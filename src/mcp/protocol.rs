//! MCP JSON-RPC 2.0 protocol message types
//!
//! Implements the core message structures for the Model Context Protocol
//! following the JSON-RPC 2.0 specification.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

/// Type-safe wrapper for JSON-RPC request IDs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(untagged)]
pub enum RequestId {
    /// UUID-based request ID (recommended for new requests)
    Uuid(Uuid),
    /// String-based request ID (for compatibility)
    String(String),
    /// Numeric request ID (for compatibility)
    Number(u64),
}

impl RequestId {
    /// Generate a new UUID-based request ID
    pub fn new() -> Self {
        Self::Uuid(Uuid::new_v4())
    }

    /// Create from UUID
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self::Uuid(uuid)
    }

    /// Create from string
    pub fn from_string(s: impl Into<String>) -> Self {
        Self::String(s.into())
    }

    /// Create from number
    pub fn from_number(n: u64) -> Self {
        Self::Number(n)
    }

    /// Convert to serde_json::Value for backward compatibility
    pub fn to_value(&self) -> Value {
        match self {
            RequestId::Uuid(uuid) => Value::String(uuid.to_string()),
            RequestId::String(s) => Value::String(s.clone()),
            RequestId::Number(n) => Value::Number(serde_json::Number::from(*n)),
        }
    }

    /// Try to create from serde_json::Value
    pub fn from_value(value: Value) -> Option<Self> {
        match value {
            Value::String(s) => {
                // Try to parse as UUID first, fall back to string
                if let Ok(uuid) = Uuid::parse_str(&s) {
                    Some(RequestId::Uuid(uuid))
                } else {
                    Some(RequestId::String(s))
                }
            }
            Value::Number(n) => n.as_u64().map(RequestId::Number),
            _ => None,
        }
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RequestId::Uuid(uuid) => write!(f, "{}", uuid),
            RequestId::String(s) => write!(f, "{}", s),
            RequestId::Number(n) => write!(f, "{}", n),
        }
    }
}

/// JSON-RPC 2.0 request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// JSON-RPC version (must be "2.0")
    pub jsonrpc: String,
    /// Request identifier (required for MCP)
    pub id: RequestId,
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
    pub id: RequestId,
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

impl InitializeParams {
    /// Create new initialization parameters
    pub fn new(
        protocol_version: impl Into<String>,
        client_name: impl Into<String>,
        client_version: impl Into<String>,
    ) -> Self {
        Self {
            protocol_version: protocol_version.into(),
            client_info: ClientInfo {
                name: client_name.into(),
                version: client_version.into(),
            },
            capabilities: ClientCapabilities::default(),
        }
    }

    /// Create with capabilities
    pub fn with_capabilities(
        protocol_version: impl Into<String>,
        client_name: impl Into<String>,
        client_version: impl Into<String>,
        capabilities: ClientCapabilities,
    ) -> Self {
        Self {
            protocol_version: protocol_version.into(),
            client_info: ClientInfo {
                name: client_name.into(),
                version: client_version.into(),
            },
            capabilities,
        }
    }

    /// Validate initialization parameters
    pub fn validate(&self) -> Result<(), JsonRpcError> {
        // Check protocol version format (basic validation)
        if self.protocol_version.is_empty() {
            return Err(JsonRpcError::invalid_params("protocol_version cannot be empty"));
        }

        // Check client info
        if self.client_info.name.is_empty() {
            return Err(JsonRpcError::invalid_params("client_info.name cannot be empty"));
        }

        if self.client_info.version.is_empty() {
            return Err(JsonRpcError::invalid_params("client_info.version cannot be empty"));
        }

        Ok(())
    }
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

impl Default for ClientCapabilities {
    fn default() -> Self {
        Self {
            experimental: HashMap::new(),
        }
    }
}

impl ClientCapabilities {
    /// Create new client capabilities
    pub fn new() -> Self {
        Self::default()
    }

    /// Add experimental capability
    pub fn with_experimental(mut self, key: impl Into<String>, value: Value) -> Self {
        self.experimental.insert(key.into(), value);
        self
    }

    /// Check if experimental capability is supported
    pub fn has_experimental(&self, key: &str) -> bool {
        self.experimental.contains_key(key)
    }
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

impl Default for ServerCapabilities {
    fn default() -> Self {
        Self {
            tools: None,
            experimental: HashMap::new(),
        }
    }
}

impl ServerCapabilities {
    /// Create new server capabilities
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable tools capability
    pub fn with_tools(mut self, list_changed: bool) -> Self {
        self.tools = Some(ToolsCapability { list_changed });
        self
    }

    /// Add experimental capability
    pub fn with_experimental(mut self, key: impl Into<String>, value: Value) -> Self {
        self.experimental.insert(key.into(), value);
        self
    }

    /// Check if tools are supported
    pub fn supports_tools(&self) -> bool {
        self.tools.is_some()
    }

    /// Check if experimental capability is supported
    pub fn has_experimental(&self, key: &str) -> bool {
        self.experimental.contains_key(key)
    }
}

/// Tools capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsCapability {
    /// List of available tools is static
    #[serde(default)]
    pub list_changed: bool,
}

impl JsonRpcRequest {
    /// Create a new JSON-RPC request with generated UUID
    pub fn new(method: String, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: RequestId::new(),
            method,
            params,
        }
    }

    /// Create a new JSON-RPC request with specific ID
    pub fn with_id(id: RequestId, method: String, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            method,
            params,
        }
    }

    /// Create a new request with string ID (for compatibility)
    pub fn with_string_id(id: impl Into<String>, method: String, params: Option<Value>) -> Self {
        Self::with_id(RequestId::from_string(id), method, params)
    }

    /// Create a new request with numeric ID (for compatibility)
    pub fn with_number_id(id: u64, method: String, params: Option<Value>) -> Self {
        Self::with_id(RequestId::from_number(id), method, params)
    }
}

impl JsonRpcResponse {
    /// Create a successful response
    pub fn success(id: RequestId, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Create an error response
    pub fn error(id: RequestId, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(error),
        }
    }

    /// Try to create a response from a request (inherits request ID)
    pub fn from_request_success(request: &JsonRpcRequest, result: Value) -> Self {
        Self::success(request.id.clone(), result)
    }

    /// Try to create an error response from a request (inherits request ID)
    pub fn from_request_error(request: &JsonRpcRequest, error: JsonRpcError) -> Self {
        Self::error(request.id.clone(), error)
    }
}

impl JsonRpcError {
    /// Create a new error
    pub fn new(code: i32, message: String, data: Option<Value>) -> Self {
        Self {
            code,
            message,
            data,
        }
    }

    /// Create a parse error (-32700)
    pub fn parse_error(message: impl Into<String>) -> Self {
        Self::new(-32700, message.into(), None)
    }

    /// Create a parse error with additional data (-32700)
    pub fn parse_error_with_data(message: impl Into<String>, data: Value) -> Self {
        Self::new(-32700, message.into(), Some(data))
    }

    /// Create an invalid request error (-32600)
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(-32600, message.into(), None)
    }

    /// Create an invalid request error with additional data (-32600)
    pub fn invalid_request_with_data(message: impl Into<String>, data: Value) -> Self {
        Self::new(-32600, message.into(), Some(data))
    }

    /// Create a method not found error (-32601)
    pub fn method_not_found(method: impl Into<String>) -> Self {
        let method = method.into();
        Self::new(-32601, format!("Method not found: {}", method), None)
    }

    /// Create an invalid params error (-32602)
    pub fn invalid_params(message: impl Into<String>) -> Self {
        Self::new(-32602, message.into(), None)
    }

    /// Create an invalid params error with details (-32602)
    pub fn invalid_params_with_details(message: impl Into<String>, details: Value) -> Self {
        Self::new(-32602, message.into(), Some(details))
    }

    /// Create an internal error (-32603)
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::new(-32603, message.into(), None)
    }

    /// Create an internal error with additional data (-32603)
    pub fn internal_error_with_data(message: impl Into<String>, data: Value) -> Self {
        Self::new(-32603, message.into(), Some(data))
    }

    /// Create a custom application error (should be >= -32000 and <= -32099)
    pub fn application_error(code: i32, message: impl Into<String>) -> Self {
        assert!(code >= -32099 && code <= -32000, "Application error codes must be between -32099 and -32000");
        Self::new(code, message.into(), None)
    }

    /// Create a custom application error with data (should be >= -32000 and <= -32099)
    pub fn application_error_with_data(code: i32, message: impl Into<String>, data: Value) -> Self {
        assert!(code >= -32099 && code <= -32000, "Application error codes must be between -32099 and -32000");
        Self::new(code, message.into(), Some(data))
    }

    /// Check if this is a parse error
    pub fn is_parse_error(&self) -> bool {
        self.code == -32700
    }

    /// Check if this is an invalid request error
    pub fn is_invalid_request(&self) -> bool {
        self.code == -32600
    }

    /// Check if this is a method not found error
    pub fn is_method_not_found(&self) -> bool {
        self.code == -32601
    }

    /// Check if this is an invalid params error
    pub fn is_invalid_params(&self) -> bool {
        self.code == -32602
    }

    /// Check if this is an internal error
    pub fn is_internal_error(&self) -> bool {
        self.code == -32603
    }

    /// Check if this is an application-defined error
    pub fn is_application_error(&self) -> bool {
        self.code >= -32099 && self.code <= -32000
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
    fn test_request_id_functionality() {
        // Test UUID generation
        let uuid_id = RequestId::new();
        matches!(uuid_id, RequestId::Uuid(_));

        // Test string creation
        let string_id = RequestId::from_string("test-id");
        matches!(string_id, RequestId::String(_));

        // Test number creation
        let number_id = RequestId::from_number(42);
        matches!(number_id, RequestId::Number(42));

        // Test conversion to Value
        let value = string_id.to_value();
        assert_eq!(value, json!("test-id"));

        // Test conversion from Value
        let recovered = RequestId::from_value(json!("test-id")).unwrap();
        matches!(recovered, RequestId::String(_));
    }

    #[test]
    fn test_json_rpc_request_serialization() {
        let request = JsonRpcRequest::with_number_id(
            1,
            "test_method".to_string(),
            Some(json!({"param": "value"})),
        );

        let serialized = serde_json::to_string(&request).unwrap();
        let deserialized: JsonRpcRequest = serde_json::from_str(&serialized).unwrap();

        assert_eq!(request.jsonrpc, "2.0");
        assert_eq!(deserialized.method, "test_method");
        assert_eq!(deserialized.id, RequestId::from_number(1));
    }

    #[test]
    fn test_json_rpc_response_success() {
        let id = RequestId::from_number(1);
        let response = JsonRpcResponse::success(id.clone(), json!({"success": true}));

        assert_eq!(response.jsonrpc, "2.0");
        assert_eq!(response.id, id);
        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[test]
    fn test_json_rpc_response_error() {
        let id = RequestId::from_number(1);
        let error = JsonRpcError::method_not_found("unknown_method".to_string());
        let response = JsonRpcResponse::error(id.clone(), error);

        assert_eq!(response.jsonrpc, "2.0");
        assert_eq!(response.id, id);
        assert!(response.result.is_none());
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32601);
    }

    #[test]
    fn test_json_rpc_response_from_request() {
        let request = JsonRpcRequest::new("test_method".to_string(), None);
        let response = JsonRpcResponse::from_request_success(&request, json!({"result": "ok"}));

        assert_eq!(response.id, request.id);
        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[test]
    fn test_json_rpc_error_improvements() {
        // Test improved error constructors
        let parse_error = JsonRpcError::parse_error("Invalid JSON");
        assert!(parse_error.is_parse_error());
        assert_eq!(parse_error.code, -32700);

        let method_error = JsonRpcError::method_not_found("unknown_method");
        assert!(method_error.is_method_not_found());
        assert!(method_error.message.contains("unknown_method"));

        // Test application error
        let app_error = JsonRpcError::application_error(-32001, "Custom error");
        assert!(app_error.is_application_error());
        assert_eq!(app_error.code, -32001);

        // Test error with data
        let error_with_data = JsonRpcError::invalid_params_with_details(
            "Invalid parameters",
            json!({"expected": "string", "received": "number"})
        );
        assert!(error_with_data.is_invalid_params());
        assert!(error_with_data.data.is_some());
    }

    #[test]
    fn test_initialize_params_helpers() {
        let params = InitializeParams::new("2024-11-05", "test-client", "1.0.0");
        assert_eq!(params.protocol_version, "2024-11-05");
        assert_eq!(params.client_info.name, "test-client");
        assert_eq!(params.client_info.version, "1.0.0");

        // Test validation
        assert!(params.validate().is_ok());

        // Test invalid params
        let invalid_params = InitializeParams::new("", "test-client", "1.0.0");
        assert!(invalid_params.validate().is_err());
    }

    #[test]
    fn test_capabilities_helpers() {
        let client_caps = ClientCapabilities::new()
            .with_experimental("feature1", json!(true))
            .with_experimental("feature2", json!({"enabled": true}));
        
        assert!(client_caps.has_experimental("feature1"));
        assert!(client_caps.has_experimental("feature2"));
        assert!(!client_caps.has_experimental("feature3"));

        let server_caps = ServerCapabilities::new()
            .with_tools(false)
            .with_experimental("semantic_search", json!(true));
        
        assert!(server_caps.supports_tools());
        assert!(server_caps.has_experimental("semantic_search"));
        assert!(!server_caps.has_experimental("other_feature"));
    }
}
