//! MCP-specific error types that integrate with TurboProp's error system

use crate::error::TurboPropError;
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
}

/// Convert MCP errors to TurboProp errors
impl From<McpError> for TurboPropError {
    fn from(error: McpError) -> Self {
        TurboPropError::other(error.to_string())
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