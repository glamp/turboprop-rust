//! Protocol handler for MCP server - handles JSON-RPC protocol specifics
//!
//! Separates protocol handling concerns from business logic

use anyhow::Result;
use serde_json::json;
use std::collections::HashMap;
use tracing::{debug, info};

use super::protocol::{
    constants, InitializeParams, InitializeResult, JsonRpcError, JsonRpcRequest, JsonRpcResponse,
    ServerCapabilities, ServerInfo, ToolsCapability,
};

/// Handles JSON-RPC protocol specifics for MCP server
pub struct ProtocolHandler;

impl ProtocolHandler {
    /// Create a new protocol handler
    pub fn new() -> Self {
        Self
    }

    /// Validate an incoming JSON-RPC request
    pub fn validate_request(&self, request: &JsonRpcRequest) -> Result<(), JsonRpcError> {
        request.validate()
    }

    /// Create a standard MCP initialization response
    pub fn create_initialize_response(
        &self,
        request: &JsonRpcRequest,
        params: &InitializeParams,
    ) -> JsonRpcResponse {
        info!(
            "Creating initialize response for client: {} v{}",
            params.client_info.name, params.client_info.version
        );

        // Validate protocol version
        if params.protocol_version != constants::PROTOCOL_VERSION {
            let error = JsonRpcError::invalid_params(format!(
                "Unsupported protocol version: {} (expected: {})",
                params.protocol_version,
                constants::PROTOCOL_VERSION
            ));
            return request.create_error_response(error);
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
                experimental: HashMap::new(),
            },
        };

        match serde_json::to_value(result) {
            Ok(result_value) => {
                info!("MCP server initialization response created successfully");
                request.create_success_response(result_value)
            }
            Err(e) => {
                let error =
                    JsonRpcError::internal_error(format!("Failed to serialize result: {}", e));
                request.create_error_response(error)
            }
        }
    }

    /// Create tools list response
    pub fn create_tools_list_response(
        &self,
        request: &JsonRpcRequest,
        tools: Vec<super::tools::ToolDefinition>,
    ) -> JsonRpcResponse {
        debug!("Creating tools list response with {} tools", tools.len());

        let result = json!({
            "tools": tools
        });

        request.create_success_response(result)
    }

    /// Create a tool execution response
    pub fn create_tool_execution_response(
        &self,
        request: &JsonRpcRequest,
        result: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Creating tool execution response");
        request.create_success_response(result)
    }

    /// Create an error response for tool execution
    pub fn create_tool_execution_error(
        &self,
        request: &JsonRpcRequest,
        error_message: String,
    ) -> JsonRpcResponse {
        let error = JsonRpcError::tool_execution_error(error_message);
        request.create_error_response(error)
    }

    /// Create an error response for method not found
    pub fn create_method_not_found_response(
        &self,
        request: &JsonRpcRequest,
        method: String,
    ) -> JsonRpcResponse {
        let error = JsonRpcError::method_not_found(method);
        request.create_error_response(error)
    }

    /// Create an error response for server overload
    pub fn create_server_overload_response(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let error = JsonRpcError::internal_error("Server overloaded".to_string());
        request.create_error_response(error)
    }

    /// Create an error response for timeout
    pub fn create_timeout_response(
        &self,
        request: &JsonRpcRequest,
        timeout_seconds: u64,
    ) -> JsonRpcResponse {
        let error = JsonRpcError::internal_error(format!(
            "Request processing timed out after {} seconds",
            timeout_seconds
        ));
        request.create_error_response(error)
    }
}

impl Default for ProtocolHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_protocol_handler_creation() {
        let handler = ProtocolHandler::new();
        // Basic creation test - handler should be created successfully
        drop(handler);
    }

    #[test]
    fn test_create_tools_list_response() {
        let handler = ProtocolHandler::new();
        let request = JsonRpcRequest::new("tools/list".to_string(), None);
        let tools = vec![];

        let response = handler.create_tools_list_response(&request, tools);

        assert!(response.error.is_none());
        assert!(response.result.is_some());
    }

    #[test]
    fn test_create_method_not_found_response() {
        let handler = ProtocolHandler::new();
        let request = JsonRpcRequest::new("invalid_method".to_string(), None);

        let response = handler.create_method_not_found_response(&request, "invalid_method".to_string());

        assert!(response.result.is_none());
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32601); // Method not found
    }
}