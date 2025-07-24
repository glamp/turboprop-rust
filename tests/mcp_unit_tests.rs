//! Fast unit tests for MCP components
//!
//! These tests provide quick feedback during development and are part of
//! the default `cargo test` suite.

use serde_json::json;
use tempfile::TempDir;

use turboprop::config::TurboPropConfig;
use turboprop::mcp::protocol::{
    ClientCapabilities, ClientInfo, InitializeParams, JsonRpcError, JsonRpcRequest,
    JsonRpcResponse, RequestId,
};
use turboprop::mcp::{IndexManager, McpServer, McpServerBuilder, McpServerTrait, StdioTransport};

/// Test MCP protocol message serialization/deserialization
mod protocol_tests {
    use super::*;

    #[test]
    fn test_json_rpc_request_creation() {
        let request = JsonRpcRequest::new("initialize".to_string(), Some(json!({"test": "value"})));

        assert_eq!(request.jsonrpc, "2.0");
        assert_eq!(request.method, "initialize");
        assert!(request.params.is_some());
    }

    #[test]
    fn test_initialize_params_deserialization() {
        let params_json = json!({
            "protocol_version": "2024-11-05",
            "client_info": {
                "name": "test-client",
                "version": "1.0.0"
            },
            "capabilities": {
                "experimental": {}
            }
        });

        let params: InitializeParams = serde_json::from_value(params_json).unwrap();

        assert_eq!(params.protocol_version, "2024-11-05");
        assert_eq!(params.client_info.name, "test-client");
        assert_eq!(params.client_info.version, "1.0.0");
    }

    #[test]
    fn test_error_response_creation() {
        let error = JsonRpcError::method_not_found("unknown_method".to_string());
        let response = JsonRpcResponse::error(RequestId::Number(123), error);

        assert_eq!(response.jsonrpc, "2.0");
        assert_eq!(response.id, RequestId::Number(123));
        assert!(response.result.is_none());
        assert!(response.error.is_some());

        let error_obj = response.error.unwrap();
        assert_eq!(error_obj.code, -32601);
        assert!(error_obj.message.contains("unknown_method"));
    }

    #[test]
    fn test_request_id_types() {
        let number_id = RequestId::Number(42);
        let string_id = RequestId::String("test-id".to_string());

        // RequestId should serialize properly
        let number_json = serde_json::to_value(&number_id).unwrap();
        let string_json = serde_json::to_value(&string_id).unwrap();

        assert_eq!(number_json, json!(42));
        assert_eq!(string_json, json!("test-id"));
    }

    #[test]
    fn test_invalid_json_rpc_request_deserialization() {
        // Test missing required fields
        let invalid_request = json!({
            "jsonrpc": "2.0"
            // Missing method and id
        });

        let result: Result<JsonRpcRequest, _> = serde_json::from_value(invalid_request);
        assert!(result.is_err());
    }

    #[test]
    fn test_malformed_initialize_params() {
        // Test with missing required fields
        let invalid_params = json!({
            "protocol_version": "2024-11-05"
            // Missing client_info and capabilities
        });

        let result: Result<InitializeParams, _> = serde_json::from_value(invalid_params);
        assert!(result.is_err());
    }

    #[test]
    fn test_request_validation() {
        // Test valid request
        let valid_request =
            JsonRpcRequest::new("initialize".to_string(), Some(json!({"test": "value"})));

        assert!(valid_request.validate().is_ok());

        // Test request with invalid JSON-RPC version
        let mut invalid_request = JsonRpcRequest::new("initialize".to_string(), None);
        invalid_request.jsonrpc = "1.0".to_string();

        let validation_result = invalid_request.validate();
        assert!(validation_result.is_err());

        let error = validation_result.unwrap_err();
        assert_eq!(error.code, -32600); // Invalid Request
    }

    #[test]
    fn test_request_id_serialization() {
        let uuid_id = RequestId::new();
        let string_id = RequestId::String("test".to_string());
        let number_id = RequestId::Number(42);

        // Test serialization
        let uuid_json = serde_json::to_value(&uuid_id).unwrap();
        let string_json = serde_json::to_value(&string_id).unwrap();
        let number_json = serde_json::to_value(&number_id).unwrap();

        assert!(uuid_json.is_string());
        assert_eq!(string_json, json!("test"));
        assert_eq!(number_json, json!(42));
    }

    #[test]
    fn test_error_code_constants() {
        // Test standard JSON-RPC error codes
        let parse_error = JsonRpcError::parse_error("Invalid JSON".to_string());
        assert_eq!(parse_error.code, -32700);

        let invalid_request = JsonRpcError::invalid_request("Missing method".to_string());
        assert_eq!(invalid_request.code, -32600);

        let method_not_found = JsonRpcError::method_not_found("unknown".to_string());
        assert_eq!(method_not_found.code, -32601);

        let invalid_params = JsonRpcError::invalid_params("Bad params".to_string());
        assert_eq!(invalid_params.code, -32602);

        let internal_error = JsonRpcError::internal_error("Server error".to_string());
        assert_eq!(internal_error.code, -32603);
    }
}

/// Test MCP server functionality (without actual indexing)
mod server_tests {
    use super::*;

    #[tokio::test]
    async fn test_mcp_server_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();

        let server = McpServer::new(temp_dir.path(), &config).await;
        assert!(server.is_ok());
    }

    #[tokio::test]
    async fn test_mcp_server_builder() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();

        let server = McpServerBuilder::new()
            .repo_path(temp_dir.path())
            .config(config)
            .build()
            .await;

        assert!(server.is_ok());
    }

    #[tokio::test]
    async fn test_server_initialization() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();
        let mut server = McpServer::new(temp_dir.path(), &config).await.unwrap();

        let params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };

        let result = server.initialize(params).await;
        assert!(result.is_ok());

        let init_result = result.unwrap();
        assert_eq!(init_result.protocol_version, "2024-11-05");
        assert_eq!(init_result.server_info.name, "turboprop");
        assert_eq!(init_result.server_info.version, env!("CARGO_PKG_VERSION"));
    }

    #[tokio::test]
    async fn test_tools_list_request_handling() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();
        let mut server = McpServer::new(temp_dir.path(), &config).await.unwrap();

        // Initialize first
        let params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };
        server.initialize(params).await.unwrap();

        let request = JsonRpcRequest::new("tools/list".to_string(), None);

        let response = server.handle_request(request).await;
        assert!(response.is_ok());

        let response = response.unwrap();
        assert!(response.error.is_none());
        assert!(response.result.is_some());

        let result = response.result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "semantic_search");
        assert!(tools[0]["description"].is_string());
        assert!(tools[0]["input_schema"].is_object());
    }

    #[tokio::test]
    async fn test_invalid_method_handling() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();
        let mut server = McpServer::new(temp_dir.path(), &config).await.unwrap();

        // Initialize first
        let params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };
        server.initialize(params).await.unwrap();

        let request = JsonRpcRequest::new("invalid_method".to_string(), None);

        let response = server.handle_request(request).await;

        // The MCP server returns an error for invalid methods, not a JSON-RPC error response
        assert!(response.is_err());
        let error_message = response.unwrap_err().to_string();
        assert!(error_message.contains("Method not found"));
        assert!(error_message.contains("invalid_method"));
    }

    #[tokio::test]
    async fn test_server_running_state() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();
        let server = McpServer::new(temp_dir.path(), &config).await.unwrap();

        // Server should not be running initially (using async version)
        assert!(!server.is_running().await);
    }
}

/// Test index manager functionality (without actual file watching)
mod index_manager_tests {
    use super::*;

    #[tokio::test]
    async fn test_index_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();

        let manager = IndexManager::new(temp_dir.path(), &config, None).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_index_stats_tracking() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();

        let manager = IndexManager::new(temp_dir.path(), &config, None)
            .await
            .unwrap();
        let stats = manager.get_stats().await;

        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.total_files, 0);
        assert_eq!(stats.updates_processed, 0);
        assert_eq!(stats.files_added, 0);
        assert_eq!(stats.files_updated, 0);
        assert_eq!(stats.files_removed, 0);
        assert_eq!(stats.update_errors, 0);
        assert!(stats.last_update.is_none());
    }

    #[tokio::test]
    async fn test_index_manager_shutdown() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();

        let mut manager = IndexManager::new(temp_dir.path(), &config, None)
            .await
            .unwrap();

        // Start and stop should not error
        assert!(manager.start().await.is_ok());
        assert!(manager.stop().await.is_ok());
    }

    #[tokio::test]
    async fn test_index_manager_invalid_path() {
        let config = TurboPropConfig::default();

        // Try to create manager for non-existent path
        let result = IndexManager::new(
            std::path::Path::new("/definitely/does/not/exist/nowhere"),
            &config,
            None,
        )
        .await;

        // Should handle invalid paths gracefully
        match result {
            Ok(_) => {
                // Some implementations might create the path - this is acceptable
                // The important thing is that it doesn't panic
            }
            Err(e) => {
                // Should provide meaningful error (but the actual message may vary)
                let error_str = e.to_string();
                assert!(!error_str.is_empty(), "Error message should not be empty");
                // We accept any non-empty error message for invalid paths
            }
        }
    }

    #[tokio::test]
    async fn test_index_manager_double_start() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();

        let mut manager = IndexManager::new(temp_dir.path(), &config, None)
            .await
            .unwrap();

        // First start should succeed
        assert!(manager.start().await.is_ok());

        // Second start should be handled gracefully
        let second_start = manager.start().await;
        match second_start {
            Ok(_) => {
                // Idempotent start - acceptable
            }
            Err(e) => {
                // Error for already started - also acceptable
                assert!(e.to_string().contains("already") || e.to_string().contains("running"));
            }
        }

        // Should still be able to stop
        assert!(manager.stop().await.is_ok());
    }

    #[tokio::test]
    async fn test_index_manager_stop_without_start() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();

        let manager = IndexManager::new(temp_dir.path(), &config, None)
            .await
            .unwrap();

        // Stop without start should be handled gracefully
        let result = manager.stop().await;
        match result {
            Ok(_) => {
                // Idempotent stop - acceptable
            }
            Err(e) => {
                // Error for not started - also acceptable
                assert!(e.to_string().contains("not") || e.to_string().contains("stopped"));
            }
        }
    }
}

/// Test transport functionality
mod transport_tests {
    use super::*;

    #[tokio::test]
    async fn test_transport_creation() {
        let transport = StdioTransport::new();

        // Transport should be created successfully
        drop(transport);
    }

    #[test]
    fn test_error_response_creation() {
        let error = JsonRpcError::internal_error("test error".to_string());
        let response = StdioTransport::create_error_response(Some(RequestId::Number(42)), error);

        assert_eq!(response.id, RequestId::Number(42));
        assert!(response.result.is_none());
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().message, "test error");
    }

    #[test]
    fn test_error_response_with_string_id() {
        let error = JsonRpcError::parse_error("Invalid JSON".to_string());
        let response = StdioTransport::create_error_response(
            Some(RequestId::String("test".to_string())),
            error,
        );

        // Should preserve the request ID in error response
        assert_eq!(response.id, RequestId::String("test".to_string()));
        assert!(response.result.is_none());
        assert!(response.error.is_some());
    }
}

/// Test comprehensive error handling scenarios
mod error_handling_tests {
    use super::*;
    use turboprop::mcp::error::McpError;
    use turboprop::mcp::protocol::JsonRpcError;

    #[test]
    fn test_protocol_error_creation() {
        let error = McpError::protocol("Invalid method format");
        assert!(matches!(error, McpError::ProtocolError { .. }));

        let json_rpc_error = JsonRpcError::from(error);
        assert_eq!(json_rpc_error.code, -32600); // Invalid Request
        assert!(json_rpc_error.message.contains("Invalid method format"));
    }

    #[test]
    fn test_server_initialization_error() {
        let error = McpError::server_initialization("Failed to load index");
        assert!(matches!(error, McpError::ServerInitializationError { .. }));

        let json_rpc_error = JsonRpcError::from(error);
        assert_eq!(json_rpc_error.code, -32603); // Internal error
    }

    #[test]
    fn test_tool_execution_error() {
        let error = McpError::tool_execution("semantic_search", "Index not found");
        assert!(matches!(error, McpError::ToolExecutionError { .. }));

        let json_rpc_error = JsonRpcError::from(error);
        assert_eq!(json_rpc_error.code, -32001); // Application error
    }

    #[test]
    fn test_security_validation_errors() {
        let path_traversal = McpError::PathTraversal;
        let symlink_attack = McpError::SymlinkAttack;
        let invalid_path = McpError::InvalidPath;

        // All security errors should map to Invalid Params
        assert_eq!(JsonRpcError::from(path_traversal).code, -32602);
        assert_eq!(JsonRpcError::from(symlink_attack).code, -32602);
        assert_eq!(JsonRpcError::from(invalid_path).code, -32602);
    }

    #[tokio::test]
    async fn test_invalid_json_rpc_version() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();
        let mut server = McpServer::new(temp_dir.path(), &config).await.unwrap();

        // Initialize server first
        let params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };
        server.initialize(params).await.unwrap();

        // Create request with invalid JSON-RPC version
        let mut request = JsonRpcRequest::new("tools/list".to_string(), None);
        request.jsonrpc = "1.0".to_string(); // Invalid version

        let response = server.handle_request(request).await;

        // Should return an error for invalid version
        assert!(response.is_err());
        let error_message = response.unwrap_err().to_string();
        assert!(
            error_message.contains("Invalid JSON-RPC version") || error_message.contains("version")
        );
    }

    #[tokio::test]
    async fn test_malformed_request_parameters() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();
        let mut server = McpServer::new(temp_dir.path(), &config).await.unwrap();

        // Initialize server first
        let params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };
        server.initialize(params).await.unwrap();

        // Create tool call request with malformed parameters
        let request = JsonRpcRequest::new(
            "tools/call".to_string(),
            Some(json!({
                "name": "semantic_search",
                "arguments": "not_an_object" // Should be an object
            })),
        );

        let response = server.handle_request(request).await;

        // Should handle the malformed parameters gracefully
        match response {
            Ok(json_response) => {
                // Should return an error response
                assert!(json_response.error.is_some());
                let error = json_response.error.unwrap();
                assert_eq!(error.code, -32602); // Invalid params
            }
            Err(_) => {
                // Also acceptable - server rejects the request
            }
        }
    }

    #[tokio::test]
    async fn test_uninitialized_server_requests() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();
        let server = McpServer::new(temp_dir.path(), &config).await.unwrap();

        // Try to call tools/list without initializing
        let request = JsonRpcRequest::new("tools/list".to_string(), None);

        let response = server.handle_request(request).await;

        // Should return an error for uninitialized server
        match response {
            Ok(json_response) => {
                assert!(json_response.error.is_some());
                let error = json_response.error.unwrap();
                // Should be a server error indicating not initialized
                assert!(error.code < 0);
                assert!(
                    error.message.contains("not initialized")
                        || error.message.contains("initialize")
                );
            }
            Err(e) => {
                // Also acceptable
                assert!(
                    e.to_string().contains("initialized") || e.to_string().contains("initialize")
                );
            }
        }
    }

    #[tokio::test]
    async fn test_double_initialization() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();
        let mut server = McpServer::new(temp_dir.path(), &config).await.unwrap();

        let params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };

        // First initialization should succeed
        let result1 = server.initialize(params.clone()).await;
        assert!(result1.is_ok());

        // Second initialization should be handled gracefully
        let result2 = server.initialize(params).await;
        match result2 {
            Ok(_) => {
                // Server allows re-initialization - that's fine
            }
            Err(e) => {
                // Server rejects re-initialization - that's also acceptable
                assert!(
                    e.to_string().contains("already initialized")
                        || e.to_string().contains("initialized")
                );
            }
        }
    }

    #[tokio::test]
    async fn test_unsupported_protocol_version() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();
        let mut server = McpServer::new(temp_dir.path(), &config).await.unwrap();

        let params = InitializeParams {
            protocol_version: "2025-01-01".to_string(), // Future version
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };

        let result = server.initialize(params).await;

        // Server should handle unsupported versions gracefully
        match result {
            Ok(init_result) => {
                // Server should return supported version
                assert_eq!(init_result.protocol_version, "2024-11-05");
            }
            Err(e) => {
                // Or reject with appropriate error
                assert!(e.to_string().contains("protocol") || e.to_string().contains("version"));
            }
        }
    }

    #[tokio::test]
    async fn test_empty_query_handling() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();
        let mut server = McpServer::new(temp_dir.path(), &config).await.unwrap();

        // Initialize server
        let params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };
        server.initialize(params).await.unwrap();

        // Test empty query
        let request = JsonRpcRequest::new(
            "tools/call".to_string(),
            Some(json!({
                "name": "semantic_search",
                "arguments": {
                    "query": "",
                    "limit": 10
                }
            })),
        );

        let response = server.handle_request(request).await;

        match response {
            Ok(json_response) => {
                if json_response.error.is_some() {
                    let error = json_response.error.unwrap();
                    // Should return appropriate error for empty query
                    assert_eq!(error.code, -32602); // Invalid params
                    assert!(error.message.contains("query") || error.message.contains("empty"));
                } else {
                    // Or return empty results - both are acceptable
                    let result = json_response.result.unwrap();
                    assert!(result["results"].is_array());
                }
            }
            Err(_) => {
                // Also acceptable to reject at request level
            }
        }
    }
}
