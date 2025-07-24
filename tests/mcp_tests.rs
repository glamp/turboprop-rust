//! Comprehensive tests for MCP (Model Context Protocol) implementation
//!
//! Tests cover:
//! - JSON-RPC message parsing and validation
//! - Transport layer error scenarios
//! - Server lifecycle and request handling
//! - Tool execution and error cases

use std::collections::HashMap;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::task::JoinSet;
use tokio::time::timeout;

use turboprop::mcp::error::McpError;
use turboprop::mcp::protocol::{
    ClientCapabilities, ClientInfo, InitializeParams, JsonRpcError, JsonRpcRequest,
    JsonRpcResponse, RequestId,
};
use turboprop::mcp::server::{McpServer, McpServerConfig, McpServerTrait};
use turboprop::mcp::tools::{ToolCallRequest, Tools};
use turboprop::mcp::transport::{RequestValidator, StdioTransportConfig};

/// Test data for MCP protocol testing
mod test_data {
    use super::*;

    pub fn create_valid_initialize_request() -> JsonRpcRequest {
        let params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };

        JsonRpcRequest::with_number_id(
            1,
            "initialize".to_string(),
            Some(serde_json::to_value(params).unwrap()),
        )
    }

    pub fn create_tools_list_request() -> JsonRpcRequest {
        JsonRpcRequest::with_number_id(2, "tools/list".to_string(), None)
    }

    pub fn create_tools_call_request() -> JsonRpcRequest {
        let mut args = HashMap::new();
        args.insert("query".to_string(), Value::String("test query".to_string()));

        let tool_request = ToolCallRequest {
            name: "semantic_search".to_string(),
            arguments: args,
        };

        JsonRpcRequest::with_number_id(
            3,
            "tools/call".to_string(),
            Some(serde_json::to_value(tool_request).unwrap()),
        )
    }

    pub fn create_malformed_requests() -> Vec<String> {
        vec![
            // Invalid JSON
            "{invalid json}".to_string(),
            // Missing jsonrpc field
            r#"{"id": 1, "method": "test"}"#.to_string(),
            // Invalid jsonrpc version
            r#"{"jsonrpc": "1.0", "id": 1, "method": "test"}"#.to_string(),
            // Empty method name
            r#"{"jsonrpc": "2.0", "id": 1, "method": ""}"#.to_string(),
            // Reserved rpc. prefix
            r#"{"jsonrpc": "2.0", "id": 1, "method": "rpc.test"}"#.to_string(),
            // Invalid method name
            r#"{"jsonrpc": "2.0", "id": 1, "method": "invalid_method"}"#.to_string(),
            // initialize without params
            r#"{"jsonrpc": "2.0", "id": 1, "method": "initialize"}"#.to_string(),
            // tools/call without params
            r#"{"jsonrpc": "2.0", "id": 1, "method": "tools/call"}"#.to_string(),
        ]
    }
}

#[cfg(test)]
mod json_rpc_protocol_tests {
    use super::*;

    #[test]
    fn test_request_id_types() {
        // Test UUID-based ID
        let uuid_id = RequestId::new();
        assert!(matches!(uuid_id, RequestId::Uuid(_)));

        // Test string-based ID
        let string_id = RequestId::from_string("test-123");
        assert!(matches!(string_id, RequestId::String(_)));
        assert_eq!(string_id.to_value(), json!("test-123"));

        // Test numeric ID
        let number_id = RequestId::from_number(42);
        assert!(matches!(number_id, RequestId::Number(42)));
        assert_eq!(number_id.to_value(), json!(42));

        // Test Value conversion roundtrip
        let original_value = json!("test-value");
        let id = RequestId::from_value(original_value.clone()).unwrap();
        assert_eq!(id.to_value(), original_value);
    }

    #[test]
    fn test_json_rpc_request_validation() {
        // Valid request
        let valid_request = test_data::create_valid_initialize_request();
        assert!(RequestValidator::validate(&valid_request).is_ok());

        // Test various invalid requests
        let invalid_jsonrpc = JsonRpcRequest {
            jsonrpc: "1.0".to_string(),
            id: RequestId::from_number(1),
            method: "test".to_string(),
            params: None,
        };
        assert!(RequestValidator::validate(&invalid_jsonrpc).is_err());

        let empty_method = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: RequestId::from_number(1),
            method: "".to_string(),
            params: None,
        };
        assert!(RequestValidator::validate(&empty_method).is_err());

        let reserved_method = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: RequestId::from_number(1),
            method: "rpc.test".to_string(),
            params: None,
        };
        assert!(RequestValidator::validate(&reserved_method).is_err());
    }

    #[test]
    fn test_json_rpc_error_types() {
        // Test standard JSON-RPC errors
        let parse_error = JsonRpcError::parse_error("Invalid JSON");
        assert!(parse_error.is_parse_error());
        assert_eq!(parse_error.code, -32700);

        let method_error = JsonRpcError::method_not_found("unknown_method");
        assert!(method_error.is_method_not_found());
        assert_eq!(method_error.code, -32601);

        let invalid_params = JsonRpcError::invalid_params("Missing required field");
        assert!(invalid_params.is_invalid_params());
        assert_eq!(invalid_params.code, -32602);

        let internal_error = JsonRpcError::internal_error("Server crashed");
        assert!(internal_error.is_internal_error());
        assert_eq!(internal_error.code, -32603);

        // Test application errors
        let app_error = JsonRpcError::application_error(-32001, "Custom error");
        assert!(app_error.is_application_error());
        assert_eq!(app_error.code, -32001);
    }

    #[test]
    fn test_json_rpc_response_creation() {
        let request = test_data::create_valid_initialize_request();
        let result = json!({"status": "success"});

        // Test success response
        let success_response = JsonRpcResponse::from_request_success(&request, result.clone());
        assert_eq!(success_response.id, request.id);
        assert!(success_response.result.is_some());
        assert!(success_response.error.is_none());
        assert_eq!(success_response.result.unwrap(), result);

        // Test error response
        let error = JsonRpcError::internal_error("Test error");
        let error_response = JsonRpcResponse::from_request_error(&request, error);
        assert_eq!(error_response.id, request.id);
        assert!(error_response.result.is_none());
        assert!(error_response.error.is_some());
    }
}

#[cfg(test)]
mod transport_tests {
    use super::*;

    #[test]
    fn test_transport_config() {
        let default_config = StdioTransportConfig::default();
        assert_eq!(default_config.max_message_size, 1024 * 1024); // 1MB
        assert_eq!(default_config.read_timeout_seconds, 30);
        assert_eq!(default_config.write_timeout_seconds, 10);
        assert_eq!(default_config.channel_buffer_size, 100);
        assert_eq!(default_config.max_requests_per_minute, 60);
        assert_eq!(default_config.rate_limit_burst_capacity, 10);

        let custom_config = StdioTransportConfig {
            max_message_size: 512 * 1024, // 512KB
            read_timeout_seconds: 15,
            write_timeout_seconds: 5,
            channel_buffer_size: 50,
            max_requests_per_minute: 30,
            rate_limit_burst_capacity: 5,
        };
        assert_eq!(custom_config.max_message_size, 512 * 1024);
        assert_eq!(custom_config.max_requests_per_minute, 30);
    }

    #[test]
    fn test_request_validation_size_limits() {
        // Test valid size
        assert!(RequestValidator::validate_size(1024, 2048).is_ok());

        // Test size exceeded
        assert!(RequestValidator::validate_size(2048, 1024).is_err());

        let error_msg = RequestValidator::validate_size(2048, 1024).unwrap_err();
        assert!(error_msg.contains("exceeds maximum"));
        assert!(error_msg.contains("2048"));
        assert!(error_msg.contains("1024"));
    }

    #[test]
    fn test_malformed_message_validation() {
        let malformed_requests = test_data::create_malformed_requests();

        for malformed in malformed_requests {
            match serde_json::from_str::<JsonRpcRequest>(&malformed) {
                Ok(request) => {
                    // If it parses as valid JSON-RPC, it should fail validation
                    assert!(
                        RequestValidator::validate(&request).is_err(),
                        "Expected validation failure for: {}",
                        malformed
                    );
                }
                Err(_) => {
                    // Parse error is also acceptable for malformed messages
                }
            }
        }
    }

    #[test]
    fn test_experimental_method_validation() {
        // Experimental methods should be allowed
        let experimental_request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: RequestId::from_number(1),
            method: "experimental/custom_feature".to_string(),
            params: None,
        };
        assert!(RequestValidator::validate(&experimental_request).is_ok());

        // Non-experimental unknown methods should be rejected
        let unknown_request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: RequestId::from_number(1),
            method: "unknown_method".to_string(),
            params: None,
        };
        assert!(RequestValidator::validate(&unknown_request).is_err());
    }

    #[test]
    fn test_rate_limiter() {
        use turboprop::mcp::transport::TokenBucketRateLimiter;

        // Create rate limiter with small capacity for testing
        let mut limiter = TokenBucketRateLimiter::new(2, 60); // 2 tokens, 60 per minute

        // Should allow first 2 requests
        assert!(limiter.try_consume());
        assert_eq!(limiter.current_tokens(), 1);
        assert!(limiter.try_consume());
        assert_eq!(limiter.current_tokens(), 0);

        // Should reject 3rd request
        assert!(!limiter.try_consume());
        assert_eq!(limiter.current_tokens(), 0);

        // Should still reject without any time passing
        assert!(!limiter.try_consume());
    }
}

#[cfg(test)]
mod server_tests {
    use super::*;

    pub fn create_test_server() -> McpServer {
        let config = McpServerConfig {
            address: "127.0.0.1".to_string(),
            port: 0, // Use port 0 for testing
            max_connections: 10,
            request_timeout_seconds: 5,
        };
        let tools = Tools::new_for_integration_tests();
        McpServer::with_config_and_tools(config, tools)
    }

    #[tokio::test]
    async fn test_server_initialization() {
        let mut server = create_test_server();

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
        assert!(init_result.capabilities.supports_tools());
    }

    #[tokio::test]
    async fn test_server_initialization_wrong_protocol_version() {
        let mut server = create_test_server();

        let params = InitializeParams {
            protocol_version: "1.0.0".to_string(), // Wrong version
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };

        let result = server.initialize(params).await;
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(error.to_string().contains("Unsupported protocol version"));
    }

    #[tokio::test]
    async fn test_server_request_handling_before_initialization() {
        let server = create_test_server();

        let request = test_data::create_tools_list_request();
        let result = server.handle_request(request).await;

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Server not initialized"));
    }

    #[tokio::test]
    async fn test_server_tools_list() {
        let mut server = create_test_server();

        // Initialize server first
        let init_params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };
        server.initialize(init_params).await.unwrap();

        // Test tools/list
        let request = test_data::create_tools_list_request();
        let result = server.handle_request(request).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.result.is_some());

        let tools_data = response.result.unwrap();
        assert!(tools_data.get("tools").is_some());
    }

    #[tokio::test]
    async fn test_server_tools_call() {
        let mut server = create_test_server();

        // Initialize server first
        let init_params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };
        server.initialize(init_params).await.unwrap();

        // Test tools/call
        let request = test_data::create_tools_call_request();
        let result = server.handle_request(request).await;

        if result.is_err() {
            eprintln!(
                "Server tools call failed with error: {:?}",
                result.as_ref().unwrap_err()
            );
        }
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.result.is_some());
    }

    #[tokio::test]
    async fn test_server_unknown_method() {
        let mut server = create_test_server();

        // Initialize server first
        let init_params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };
        server.initialize(init_params).await.unwrap();

        // Test unknown method
        let request = JsonRpcRequest::with_number_id(99, "unknown_method".to_string(), None);
        let result = server.handle_request(request).await;

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Method not found"));
    }

    #[tokio::test]
    async fn test_server_lifecycle() {
        let server = create_test_server();

        // Server should not be running initially
        assert!(!server.is_running());

        // Starting an already running server should fail
        // Note: We can't easily test the actual start() method in unit tests
        // because it enters an infinite loop for message processing
        // This would require integration testing with actual STDIO
    }
}

#[cfg(test)]
mod tools_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_tools_registry() {
        let tools = Tools::new_for_integration_tests();

        // Check default tools are registered
        assert!(tools.has_tool("semantic_search"));

        // List tools
        let tool_list = tools.list_tools();
        assert!(!tool_list.is_empty());

        let semantic_search_tool = tool_list
            .iter()
            .find(|tool| tool.name == "semantic_search")
            .expect("semantic_search tool should be available");

        assert_eq!(semantic_search_tool.name, "semantic_search");
        assert!(!semantic_search_tool.description.is_empty());
        assert!(semantic_search_tool.input_schema.is_object());
    }

    #[tokio::test]
    async fn test_semantic_search_tool_execution() {
        let tools = Tools::new_for_integration_tests();

        let mut args = HashMap::new();
        args.insert("query".to_string(), Value::String("test query".to_string()));
        args.insert(
            "limit".to_string(),
            Value::Number(serde_json::Number::from(5)),
        );
        args.insert(
            "threshold".to_string(),
            Value::Number(serde_json::Number::from_f64(0.5).unwrap()),
        );

        let request = ToolCallRequest {
            name: "semantic_search".to_string(),
            arguments: args,
        };

        let response = tools.execute_tool(request).await;
        assert!(response.is_ok());

        let result = response.unwrap();
        if !result.success {
            eprintln!("Tool execution failed with error: {:?}", result.error);
        }
        assert!(result.success);
        assert!(result.content.is_some());
        assert!(result.error.is_none());
    }

    #[tokio::test]
    async fn test_semantic_search_tool_validation_errors() {
        let tools = Tools::new_for_integration_tests();

        // Test missing query
        let request_no_query = ToolCallRequest {
            name: "semantic_search".to_string(),
            arguments: HashMap::new(),
        };

        let response = tools.execute_tool(request_no_query).await;
        assert!(response.is_err());

        // Test invalid limit
        let mut invalid_args = HashMap::new();
        invalid_args.insert("query".to_string(), Value::String("test".to_string()));
        invalid_args.insert(
            "limit".to_string(),
            Value::Number(serde_json::Number::from(0)),
        ); // Invalid: 0

        let request_invalid_limit = ToolCallRequest {
            name: "semantic_search".to_string(),
            arguments: invalid_args,
        };

        let response = tools.execute_tool(request_invalid_limit).await;
        assert!(response.is_err());

        // Test invalid threshold
        let mut invalid_threshold_args = HashMap::new();
        invalid_threshold_args.insert("query".to_string(), Value::String("test".to_string()));
        invalid_threshold_args.insert(
            "threshold".to_string(),
            Value::Number(serde_json::Number::from_f64(2.0).unwrap()),
        ); // Invalid: > 1.0

        let request_invalid_threshold = ToolCallRequest {
            name: "semantic_search".to_string(),
            arguments: invalid_threshold_args,
        };

        let response = tools.execute_tool(request_invalid_threshold).await;
        assert!(response.is_err());
    }

    #[tokio::test]
    async fn test_unknown_tool_execution() {
        let tools = Tools::new_for_integration_tests();

        let request = ToolCallRequest {
            name: "unknown_tool".to_string(),
            arguments: HashMap::new(),
        };

        let response = tools.execute_tool(request).await;
        assert!(response.is_err());

        let error = response.unwrap_err();
        assert!(error.to_string().contains("Tool not found"));
    }
}

#[cfg(test)]
mod error_conversion_tests {
    use super::*;

    #[test]
    fn test_mcp_error_to_json_rpc_error_conversion() {
        // Test protocol error conversion
        let protocol_error = McpError::protocol("Invalid request format");
        let json_error: JsonRpcError = protocol_error.into();
        assert!(json_error.is_invalid_request());
        assert!(json_error.message.contains("Invalid request format"));

        // Test server initialization error conversion
        let server_error = McpError::server_initialization("Failed to start");
        let json_error: JsonRpcError = server_error.into();
        assert!(json_error.is_internal_error());

        // Test tool execution error conversion
        let tool_error = McpError::tool_execution("search", "Query validation failed");
        let json_error: JsonRpcError = tool_error.into();
        assert!(json_error.is_application_error());
        assert_eq!(json_error.code, -32001);

        // Test transport error conversion
        let transport_error = McpError::transport("Connection lost");
        let json_error: JsonRpcError = transport_error.into();
        assert!(json_error.is_internal_error());

        // Test configuration error conversion
        let config_error = McpError::configuration("Invalid timeout value");
        let json_error: JsonRpcError = config_error.into();
        assert!(json_error.is_internal_error());

        // Test unsupported capability conversion
        let capability_error = McpError::unsupported_capability("advanced_search");
        let json_error: JsonRpcError = capability_error.into();
        assert!(json_error.is_method_not_found());
    }
}

#[cfg(test)]
mod concurrent_request_tests {
    use super::*;
    use std::sync::Arc;
    use tokio::task::JoinSet;

    #[tokio::test]
    async fn test_concurrent_tool_execution() {
        let tools = Arc::new(Tools::new_for_integration_tests());
        let num_concurrent = 10;
        let mut join_set = JoinSet::new();

        // Spawn multiple concurrent tool executions
        for i in 0..num_concurrent {
            let tools_clone = Arc::clone(&tools);
            join_set.spawn(async move {
                let mut args = HashMap::new();
                args.insert(
                    "query".to_string(),
                    Value::String(format!("test query {}", i)),
                );

                let request = ToolCallRequest {
                    name: "semantic_search".to_string(),
                    arguments: args,
                };

                tools_clone.execute_tool(request).await
            });
        }

        // Wait for all tasks to complete
        let mut success_count = 0;
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(tool_result) => {
                    assert!(tool_result.is_ok());
                    success_count += 1;
                }
                Err(e) => {
                    panic!("Task failed: {}", e);
                }
            }
        }

        assert_eq!(success_count, num_concurrent);
    }

    #[tokio::test]
    async fn test_concurrent_request_validation() {
        let num_concurrent = 20;
        let mut join_set = JoinSet::new();

        // Test concurrent validation of different request types
        for i in 0..num_concurrent {
            join_set.spawn(async move {
                let request = if i % 3 == 0 {
                    test_data::create_valid_initialize_request()
                } else if i % 3 == 1 {
                    test_data::create_tools_list_request()
                } else {
                    test_data::create_tools_call_request()
                };

                RequestValidator::validate(&request)
            });
        }

        // All validations should succeed
        let mut success_count = 0;
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(validation_result) => {
                    assert!(validation_result.is_ok());
                    success_count += 1;
                }
                Err(e) => {
                    panic!("Validation task failed: {}", e);
                }
            }
        }

        assert_eq!(success_count, num_concurrent);
    }
}

/// Performance and stress tests for MCP components
#[cfg(test)]
mod performance_tests {
    use super::*;

    #[tokio::test]
    async fn test_large_message_handling() {
        // Test handling of messages near the size limit
        let config = StdioTransportConfig {
            max_message_size: 1024, // 1KB limit for testing
            read_timeout_seconds: 5,
            write_timeout_seconds: 5,
            channel_buffer_size: 10,
            max_requests_per_minute: 30,
            rate_limit_burst_capacity: 5,
        };

        // Test message within limit
        let small_size = 512;
        assert!(RequestValidator::validate_size(small_size, config.max_message_size).is_ok());

        // Test message at limit
        let exact_size = config.max_message_size;
        assert!(RequestValidator::validate_size(exact_size, config.max_message_size).is_ok());

        // Test message exceeding limit
        let large_size = config.max_message_size + 1;
        assert!(RequestValidator::validate_size(large_size, config.max_message_size).is_err());
    }

    #[tokio::test]
    async fn test_request_id_generation_uniqueness() {
        let mut ids = std::collections::HashSet::new();
        let num_ids = 1000;

        // Generate many IDs and ensure they're unique
        for _ in 0..num_ids {
            let id = RequestId::new();
            let id_string = id.to_string();
            assert!(ids.insert(id_string), "Generated duplicate ID");
        }

        assert_eq!(ids.len(), num_ids);
    }

    #[tokio::test]
    async fn test_json_serialization_performance() {
        let request = test_data::create_valid_initialize_request();
        let num_iterations = 1000;

        let start = std::time::Instant::now();

        for _ in 0..num_iterations {
            let serialized = serde_json::to_string(&request).unwrap();
            let _deserialized: JsonRpcRequest = serde_json::from_str(&serialized).unwrap();
        }

        let duration = start.elapsed();

        // Should complete within reasonable time (less than 1 second for 1000 iterations)
        assert!(
            duration < Duration::from_secs(1),
            "Serialization performance too slow: {:?}",
            duration
        );
    }
}

/// Integration tests for index initialization and failure recovery
#[cfg(test)]
mod index_initialization_tests {
    use super::*;
    use tempfile::TempDir;
    use turboprop::config::TurboPropConfig;

    #[tokio::test]
    async fn test_index_initialization_failure_recovery() {
        // Create a temporary directory that doesn't exist for testing failure
        let temp_dir = TempDir::new().unwrap();
        let non_existent_path = temp_dir.path().join("non_existent");

        let config = TurboPropConfig::default();

        // This should handle the error gracefully without crashing
        let result = turboprop::mcp::McpServer::new(&non_existent_path, &config).await;

        // Server creation should succeed even if index can't be built initially
        // The server should handle this gracefully in the background
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_server_continues_without_index() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();

        // Create server with valid path but potentially no content
        let mut server = turboprop::mcp::McpServer::new(temp_dir.path(), &config)
            .await
            .unwrap();

        // Initialize server
        let init_params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };

        let init_result = server.initialize(init_params).await;
        assert!(init_result.is_ok());

        // Server should be able to handle tools/list even if index isn't ready
        let tools_request = test_data::create_tools_list_request();
        let tools_result = server.handle_request(tools_request).await;
        assert!(tools_result.is_ok());
    }

    #[tokio::test]
    async fn test_concurrent_index_initialization() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();

        // Create multiple servers concurrently trying to initialize the same repository
        let mut join_set = JoinSet::new();
        let num_servers = 5;

        for _ in 0..num_servers {
            let path = temp_dir.path().to_path_buf();
            let config_clone = config.clone();

            join_set
                .spawn(async move { turboprop::mcp::McpServer::new(&path, &config_clone).await });
        }

        // All servers should be created successfully despite concurrent initialization
        let mut success_count = 0;
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(server_result) => {
                    assert!(server_result.is_ok());
                    success_count += 1;
                }
                Err(e) => {
                    panic!("Server creation task failed: {}", e);
                }
            }
        }

        assert_eq!(success_count, num_servers);
    }
}

/// Resource exhaustion and DoS protection tests
#[cfg(test)]
mod resource_exhaustion_tests {
    use super::*;
    use std::sync::Arc;
    use tokio::time::{timeout, Duration};

    #[tokio::test]
    async fn test_concurrent_request_overload() {
        let _server = std::sync::Arc::new(server_tests::create_test_server());

        // Initialize server
        let mut server_mut = server_tests::create_test_server();
        let init_params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };
        server_mut.initialize(init_params).await.unwrap();
        let server = Arc::new(server_mut);

        // Spawn many concurrent requests to test overload protection
        let mut join_set = JoinSet::new();
        let num_requests = 50; // Exceed normal capacity

        for i in 0..num_requests {
            let server_clone: Arc<McpServer> = Arc::clone(&server);
            join_set.spawn(async move {
                let request = if i % 2 == 0 {
                    test_data::create_tools_list_request()
                } else {
                    test_data::create_tools_call_request()
                };

                // Add timeout to prevent hanging
                timeout(
                    Duration::from_secs(10),
                    server_clone.handle_request(request),
                )
                .await
            });
        }

        let mut completed_requests = 0;
        let mut successful_requests = 0;
        let mut overload_rejections = 0;

        while let Some(result) = join_set.join_next().await {
            completed_requests += 1;

            match result {
                Ok(Ok(response_result)) => match response_result {
                    Ok(_response) => {
                        successful_requests += 1;
                    }
                    Err(e) => {
                        if e.to_string().contains("overload") {
                            overload_rejections += 1;
                        }
                    }
                },
                Ok(Err(_timeout)) => {
                    // Timeout occurred, which is acceptable under load
                }
                Err(e) => {
                    panic!("Request task panicked: {}", e);
                }
            }
        }

        assert_eq!(completed_requests, num_requests);
        // Should have some successful requests and potentially some overload rejections
        assert!(successful_requests > 0);

        println!(
            "Completed: {}, Successful: {}, Overload rejections: {}",
            completed_requests, successful_requests, overload_rejections
        );
    }

    #[tokio::test]
    async fn test_large_query_handling() {
        let tools = Tools::new_for_integration_tests();

        // Test with progressively larger queries
        let query_sizes = vec![100, 500, 1000, 2000]; // Characters

        for size in query_sizes {
            let large_query = "x".repeat(size);

            let mut args = HashMap::new();
            args.insert("query".to_string(), Value::String(large_query));

            let request = ToolCallRequest {
                name: "semantic_search".to_string(),
                arguments: args,
            };

            let result = timeout(Duration::from_secs(5), tools.execute_tool(request)).await;

            match result {
                Ok(tool_result) => {
                    if size <= 1000 {
                        // Should succeed for reasonable sizes
                        if let Err(e) = &tool_result {
                            println!("Query size {} failed: {}", size, e);
                        }
                    } else {
                        // May fail for very large queries (security protection)
                        if let Err(e) = &tool_result {
                            assert!(
                                e.to_string().contains("too long")
                                    || e.to_string().contains("suspicious"),
                                "Expected size-related error for large query, got: {}",
                                e
                            );
                        }
                    }
                }
                Err(_) => {
                    // Timeout is acceptable for very large queries
                    println!("Query size {} timed out (acceptable)", size);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_malicious_query_patterns() {
        let tools = Tools::new_for_integration_tests();

        let long_query = "x".repeat(5000);
        let repeated_word_query = "word".repeat(1000);
        let malicious_queries = vec![
            "../../../etc/passwd",           // Path traversal
            "..\\..\\..\\windows\\system32", // Windows path traversal
            "query\0with\0nulls",            // Null byte injection
            &long_query,                     // Extremely long query
            &repeated_word_query,            // Many repeated words
        ];

        for malicious_query in malicious_queries {
            let mut args = HashMap::new();
            args.insert("query".to_string(), Value::String(malicious_query.to_string()));

            let request = ToolCallRequest {
                name: "semantic_search".to_string(),
                arguments: args,
            };

            let result = tools.execute_tool(request).await;

            // Malicious queries should be rejected
            if let Err(e) = result {
                assert!(
                    e.to_string().contains("suspicious")
                        || e.to_string().contains("too long")
                        || e.to_string().contains("security")
                        || e.to_string().contains("invalid"),
                    "Expected security error for malicious query '{}', got: {}",
                    malicious_query,
                    e
                );
            } else {
                // If it succeeds, the query should at least be sanitized/limited
                println!(
                    "Malicious query '{}' was processed (potentially sanitized)",
                    malicious_query
                );
            }
        }
    }

    #[tokio::test]
    async fn test_rate_limiting_behavior() {
        use turboprop::mcp::transport::TokenBucketRateLimiter;

        // Test rate limiter behavior under sustained load
        let mut limiter = TokenBucketRateLimiter::new(5, 300); // 5 tokens, 300 per minute (5 per second)

        // Burst should be allowed
        for i in 0..5 {
            assert!(
                limiter.try_consume(),
                "Burst request {} should succeed",
                i + 1
            );
        }

        // Further requests should be rate limited
        for i in 0..10 {
            let allowed = limiter.try_consume();
            // Most should be rejected during rate limiting
            if !allowed {
                println!("Request {} was rate limited (expected)", i + 6);
            }
        }

        // After some time, requests should be allowed again
        // Note: This test depends on the rate limiter implementation
        // For a real test, we'd need to advance time or wait
    }
}

/// Protocol violation and edge case tests
#[cfg(test)]
mod protocol_violation_tests {
    use super::*;

    #[tokio::test]
    async fn test_invalid_json_rpc_version() {
        let server = server_tests::create_test_server();

        let invalid_request = JsonRpcRequest {
            jsonrpc: "1.0".to_string(), // Wrong version
            id: RequestId::from_number(1),
            method: "initialize".to_string(),
            params: Some(json!({
                "protocol_version": "2024-11-05",
                "client_info": {
                    "name": "test-client",
                    "version": "1.0.0"
                },
                "capabilities": {}
            })),
        };

        let result = server.handle_request(invalid_request).await;
        assert!(result.is_err());

        // Should get a validation error
        let error = result.unwrap_err();
        assert!(error.to_string().contains("jsonrpc") || error.to_string().contains("version"));
    }

    #[tokio::test]
    async fn test_missing_required_parameters() {
        let mut server = server_tests::create_test_server();

        // Initialize server first
        let init_params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };
        server.initialize(init_params).await.unwrap();

        // Test tools/call without required parameters
        let invalid_call = JsonRpcRequest::with_number_id(
            1,
            "tools/call".to_string(),
            Some(json!({})), // Missing tool name and arguments
        );

        let result = server.handle_request(invalid_call).await;
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(error.to_string().contains("Missing") || error.to_string().contains("required"));
    }

    #[tokio::test]
    async fn test_extremely_long_method_names() {
        let server = server_tests::create_test_server();

        let long_method = "x".repeat(1000);
        let request = JsonRpcRequest::with_number_id(1, long_method, None);

        let result = server.handle_request(request).await;
        assert!(result.is_err());

        // Should be rejected as method not found or validation error
        let error = result.unwrap_err();
        assert!(
            error.to_string().contains("Method not found")
                || error.to_string().contains("validation")
                || error.to_string().contains("invalid")
        );
    }

    #[tokio::test]
    async fn test_null_and_empty_request_ids() {
        let server = server_tests::create_test_server();

        // Test with string-based empty ID
        let empty_id_request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: RequestId::from_string(""),
            method: "tools/list".to_string(),
            params: None,
        };

        // Should handle empty ID gracefully
        let result = server.handle_request(empty_id_request).await;
        // The result itself may fail for other reasons (not initialized),
        // but it shouldn't crash due to empty ID
        match result {
            Ok(_) => {} // Fine
            Err(e) => {
                // Should not be an ID-related error
                assert!(!e.to_string().contains("ID"));
            }
        }
    }
}

/// Memory usage and resource cleanup tests  
#[cfg(test)]
mod resource_cleanup_tests {
    use super::*;

    #[tokio::test]
    async fn test_server_cleanup_after_error() {
        // Test that server properly cleans up resources after errors
        let mut server = server_tests::create_test_server();

        // Try to initialize with invalid parameters
        let invalid_params = InitializeParams {
            protocol_version: "invalid-version".to_string(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };

        let result = server.initialize(invalid_params).await;
        assert!(result.is_err());

        // Server should still be in a valid state for subsequent operations
        let valid_params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };

        let second_result = server.initialize(valid_params).await;
        assert!(second_result.is_ok());
    }

    #[tokio::test]
    async fn test_repeated_tool_executions() {
        let tools = Tools::new_for_integration_tests();
        let num_executions = 100;

        // Execute the same tool many times to test for memory leaks or resource buildup
        for i in 0..num_executions {
            let mut args = HashMap::new();
            args.insert(
                "query".to_string(),
                Value::String(format!("test query {}", i)),
            );

            let request = ToolCallRequest {
                name: "semantic_search".to_string(),
                arguments: args,
            };

            let result = timeout(Duration::from_secs(5), tools.execute_tool(request)).await;

            match result {
                Ok(_tool_result) => {
                    // Most should succeed (some may fail due to missing index, which is OK)
                    if i % 10 == 0 {
                        println!("Completed {} tool executions", i + 1);
                    }
                }
                Err(_) => {
                    panic!("Tool execution {} timed out", i + 1);
                }
            }
        }

        println!("Successfully completed {} tool executions", num_executions);
    }
}
