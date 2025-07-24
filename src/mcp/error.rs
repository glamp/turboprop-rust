//! MCP-specific error types that integrate with TurboProp's error system

use crate::error::TurboPropError;
use crate::mcp::protocol::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};
use anyhow::Result;
use serde_json::Value;
use thiserror::Error;

/// Result type alias for MCP operations
pub type McpResult<T> = Result<T, McpError>;

/// MCP-specific error types
#[derive(Error, Debug)]
pub enum McpError {
    /// JSON-RPC protocol errors
    #[error("JSON-RPC protocol error: {message}")]
    ProtocolError { message: String },

    /// Server initialization errors
    #[error("MCP server initialization failed: {reason}")]
    ServerInitializationError { reason: String },

    /// Transport layer errors
    #[error("MCP transport error: {message}")]
    TransportError { message: String },

    /// Tool execution errors
    #[error("MCP tool execution failed for '{tool_name}': {reason}")]
    ToolExecutionError { tool_name: String, reason: String },

    /// Configuration errors
    #[error("MCP configuration error: {message}")]
    ConfigurationError { message: String },

    /// Client capability errors
    #[error("MCP client capability error: {capability} not supported")]
    UnsupportedCapability { capability: String },

    /// Security validation errors
    #[error("Security validation failed: invalid path")]
    InvalidPath,

    /// Path traversal attack detected
    #[error("Security validation failed: path traversal detected")]
    PathTraversal,

    /// Symbolic link attack detected  
    #[error("Security validation failed: symbolic link attack detected")]
    SymlinkAttack,

    /// Query too long
    #[error("Security validation failed: query too long (max {max} characters)")]
    QueryTooLong { max: usize },

    /// Suspicious query pattern detected
    #[error("Security validation failed: suspicious query pattern detected")]
    SuspiciousQuery,
}

impl McpError {
    /// Create a protocol error
    pub fn protocol(message: impl Into<String>) -> Self {
        Self::ProtocolError {
            message: message.into(),
        }
    }

    /// Create a server initialization error
    pub fn server_initialization(reason: impl Into<String>) -> Self {
        Self::ServerInitializationError {
            reason: reason.into(),
        }
    }

    /// Create a transport error
    pub fn transport(message: impl Into<String>) -> Self {
        Self::TransportError {
            message: message.into(),
        }
    }

    /// Create a tool execution error
    pub fn tool_execution(tool_name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ToolExecutionError {
            tool_name: tool_name.into(),
            reason: reason.into(),
        }
    }

    /// Create a configuration error
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::ConfigurationError {
            message: message.into(),
        }
    }

    /// Create an unsupported capability error
    pub fn unsupported_capability(capability: impl Into<String>) -> Self {
        Self::UnsupportedCapability {
            capability: capability.into(),
        }
    }

    /// Create an invalid path security error
    pub fn invalid_path() -> Self {
        Self::InvalidPath
    }

    /// Create a path traversal security error
    pub fn path_traversal() -> Self {
        Self::PathTraversal
    }

    /// Create a symbolic link attack security error
    pub fn symlink_attack() -> Self {
        Self::SymlinkAttack
    }

    /// Create a query too long security error
    pub fn query_too_long(max: usize) -> Self {
        Self::QueryTooLong { max }
    }

    /// Create a suspicious query security error
    pub fn suspicious_query() -> Self {
        Self::SuspiciousQuery
    }
}

/// Convert anyhow errors to MCP errors
impl From<anyhow::Error> for McpError {
    fn from(error: anyhow::Error) -> Self {
        Self::ToolExecutionError {
            tool_name: "unknown".to_string(),
            reason: error.to_string(),
        }
    }
}

/// Convert MCP errors to TurboProp errors
impl From<McpError> for TurboPropError {
    fn from(error: McpError) -> Self {
        TurboPropError::other(error.to_string())
    }
}

/// Convert MCP errors to JSON-RPC errors
impl From<McpError> for JsonRpcError {
    fn from(error: McpError) -> Self {
        match error {
            McpError::ProtocolError { message } => JsonRpcError::invalid_request(message),
            McpError::ServerInitializationError { reason } => JsonRpcError::internal_error(reason),
            McpError::TransportError { message } => JsonRpcError::internal_error(message),
            McpError::ToolExecutionError {
                tool_name: _,
                reason,
            } => JsonRpcError::application_error(-32001, reason),
            McpError::ConfigurationError { message } => JsonRpcError::internal_error(message),
            McpError::UnsupportedCapability { capability } => {
                JsonRpcError::method_not_found(capability)
            }
            McpError::InvalidPath => JsonRpcError::invalid_params("Invalid path".to_string()),
            McpError::PathTraversal => {
                JsonRpcError::invalid_params("Path traversal detected".to_string())
            }
            McpError::SymlinkAttack => {
                JsonRpcError::invalid_params("Symbolic link attack detected".to_string())
            }
            McpError::QueryTooLong { max } => {
                JsonRpcError::invalid_params(format!("Query too long (max {} characters)", max))
            }
            McpError::SuspiciousQuery => {
                JsonRpcError::invalid_params("Suspicious query pattern detected".to_string())
            }
        }
    }
}

/// Error handling utilities for consistent error conversion
pub struct ErrorHandler;

impl ErrorHandler {
    /// Convert an anyhow::Result to a JSON-RPC response
    /// Use this for internal operations that return anyhow::Result
    pub fn handle_internal_result(
        request: &JsonRpcRequest,
        result: Result<Value>,
    ) -> JsonRpcResponse {
        match result {
            Ok(value) => request.create_success_response(value),
            Err(e) => {
                let error = JsonRpcError::internal_error(format!("Internal error: {}", e));
                request.create_error_response(error)
            }
        }
    }

    /// Convert an McpResult to a JSON-RPC response
    /// Use this for MCP-specific operations that return McpResult
    pub fn handle_mcp_result(
        request: &JsonRpcRequest,
        result: McpResult<Value>,
    ) -> JsonRpcResponse {
        match result {
            Ok(value) => request.create_success_response(value),
            Err(mcp_error) => {
                let json_rpc_error = JsonRpcError::from(mcp_error);
                request.create_error_response(json_rpc_error)
            }
        }
    }

    /// Create a standardized error response for tool execution errors
    pub fn handle_tool_execution_error(
        request: &JsonRpcRequest,
        tool_name: &str,
        error: anyhow::Error,
    ) -> JsonRpcResponse {
        let mcp_error = McpError::tool_execution(tool_name, error.to_string());
        let json_rpc_error = JsonRpcError::from(mcp_error);
        request.create_error_response(json_rpc_error)
    }

    /// Create a standardized error response for configuration errors
    pub fn handle_configuration_error(
        request: &JsonRpcRequest,
        message: String,
    ) -> JsonRpcResponse {
        let mcp_error = McpError::configuration(message);
        let json_rpc_error = JsonRpcError::from(mcp_error);
        request.create_error_response(json_rpc_error)
    }

    /// Create a standardized error response for security violations
    pub fn handle_security_error(
        request: &JsonRpcRequest,
        security_error: McpError,
    ) -> JsonRpcResponse {
        let json_rpc_error = JsonRpcError::from(security_error);
        request.create_error_response(json_rpc_error)
    }

    /// Add context to an anyhow error and convert to JSON-RPC response
    pub fn handle_internal_result_with_context(
        request: &JsonRpcRequest,
        result: Result<Value>,
        context: &str,
    ) -> JsonRpcResponse {
        match result {
            Ok(value) => request.create_success_response(value),
            Err(e) => {
                let error = JsonRpcError::internal_error(format!("{}: {}", context, e));
                request.create_error_response(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_error_constructors() {
        let protocol_error = McpError::protocol("Invalid JSON-RPC format");
        matches!(protocol_error, McpError::ProtocolError { .. });

        let tool_error = McpError::tool_execution("search", "Parameter validation failed");
        matches!(tool_error, McpError::ToolExecutionError { .. });
    }

    #[test]
    fn test_mcp_to_turboprop_error_conversion() {
        let mcp_error = McpError::protocol("Test error");
        let turboprop_error: TurboPropError = mcp_error.into();
        matches!(turboprop_error, TurboPropError::Other { .. });
    }
}
