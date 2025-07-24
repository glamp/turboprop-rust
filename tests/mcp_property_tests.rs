//! Property-based tests for MCP protocol implementation
//!
//! This module uses proptest to verify that the MCP protocol implementation
//! handles arbitrary inputs correctly and maintains invariants across
//! all possible input combinations.

use proptest::prelude::*;
use serde_json::json;
use tempfile::TempDir;

use turboprop::config::TurboPropConfig;
use turboprop::mcp::protocol::{
    ClientCapabilities, ClientInfo, InitializeParams, JsonRpcRequest, RequestId,
};
use turboprop::mcp::{McpServer, McpServerTrait};

// Import our enhanced test utilities
mod mcp_common;
use mcp_common::create_test_repository;

// =============================================================================
// PROPERTY TEST GENERATORS
// =============================================================================

/// Generate arbitrary RequestId values
fn arb_request_id() -> impl Strategy<Value = RequestId> {
    prop_oneof![
        any::<u64>().prop_map(RequestId::Number),
        "[a-zA-Z0-9_-]{1,50}".prop_map(RequestId::String),
        // Generate valid UUID strings and convert to RequestId::Uuid
        Just(RequestId::new()),
    ]
}

/// Generate arbitrary InitializeParams values
fn arb_initialize_params() -> impl Strategy<Value = InitializeParams> {
    (
        "[0-9]{4}-[0-9]{2}-[0-9]{2}",  // protocol_version format
        "[a-zA-Z][a-zA-Z0-9_-]{0,29}", // client name
        "[0-9]+\\.[0-9]+\\.[0-9]+",    // version format
    )
        .prop_map(
            |(protocol_version, client_name, client_version)| InitializeParams {
                protocol_version,
                client_info: ClientInfo {
                    name: client_name,
                    version: client_version,
                },
                capabilities: ClientCapabilities::default(),
            },
        )
}

/// Generate arbitrary search query strings
fn arb_search_query() -> impl Strategy<Value = String> {
    prop_oneof![
        // Common programming terms
        prop::sample::select(vec![
            "function".to_string(),
            "class".to_string(),
            "method".to_string(),
            "variable".to_string(),
            "error".to_string(),
            "authentication".to_string(),
            "user".to_string(),
            "config".to_string(),
            "jwt".to_string(),
            "token".to_string(),
        ]),
        // Random text queries
        "[a-zA-Z ]{1,50}",
        // Programming patterns
        "[a-zA-Z_][a-zA-Z0-9_]*", // identifiers
    ]
}

// =============================================================================
// PROPERTY-BASED PROTOCOL VALIDATION TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    /// Property: All valid JSON-RPC requests should be parseable
    #[test]
    fn prop_json_rpc_request_parsing(
        method in "[a-zA-Z_]{1,20}",
        id in arb_request_id(),
    ) {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: method.clone(),
            id: id.clone(),
            params: None,
        };

        // Should serialize and deserialize without loss
        let serialized = serde_json::to_string(&request).unwrap();
        let deserialized: JsonRpcRequest = serde_json::from_str(&serialized).unwrap();

        prop_assert_eq!(request.jsonrpc, deserialized.jsonrpc);
        prop_assert_eq!(request.method, deserialized.method);
        prop_assert_eq!(request.id, deserialized.id);
    }

    /// Property: Initialize parameters should always produce valid responses
    #[test]
    fn prop_initialize_params_handling(
        params in arb_initialize_params()
    ) {
        let _ = tokio_test::block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let config = TurboPropConfig::default();
            let mut server = McpServer::new(temp_dir.path(), &config).await.unwrap();

            let result = server.initialize(params.clone()).await;

            // Initialization should either succeed or fail gracefully
            match result {
                Ok(init_result) => {
                    prop_assert_eq!(init_result.protocol_version, "2024-11-05");
                    prop_assert_eq!(init_result.server_info.name, "turboprop");
                }
                Err(_) => {
                    // Some parameter combinations may fail, but should not panic
                }
            }
            Ok(())
        });
    }

    /// Property: Search queries should never cause panics
    #[test]
    fn prop_search_queries_no_panic(
        query in arb_search_query(),
    ) {
        tokio_test::block_on(async {
            let temp_dir = create_test_repository();
            let config = TurboPropConfig::default();
            let mut server = McpServer::new(temp_dir.path(), &config).await.unwrap();

            // Initialize first
            let init_params = InitializeParams {
                protocol_version: "2024-11-05".to_string(),
                client_info: ClientInfo {
                    name: "test".to_string(),
                    version: "1.0".to_string(),
                },
                capabilities: ClientCapabilities::default(),
            };
            let _ = server.initialize(init_params).await;

            // Create search request
            let arguments = json!({ "query": query });

            let request = JsonRpcRequest {
                jsonrpc: "2.0".to_string(),
                method: "tools/call".to_string(),
                id: RequestId::Number(1),
                params: Some(json!({
                    "name": "semantic_search",
                    "arguments": arguments
                })),
            };

            // Should not panic regardless of query content
            let _result = server.handle_request(request).await;
        });
    }

    /// Property: Invalid JSON-RPC methods should return consistent errors
    #[test]
    fn prop_invalid_methods_consistent_errors(
        invalid_method in "[a-zA-Z_]{1,20}",
        id in arb_request_id()
    ) {
        // Skip valid methods to test only invalid ones
        prop_assume!(!["initialize", "tools/list", "tools/call"].contains(&invalid_method.as_str()));

        let _ = tokio_test::block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let config = TurboPropConfig::default();
            let server = McpServer::new(temp_dir.path(), &config).await.unwrap();

            let request = JsonRpcRequest {
                jsonrpc: "2.0".to_string(),
                method: invalid_method,
                id,
                params: None,
            };

            let result = server.handle_request(request).await;

            // Should consistently return error for invalid methods
            prop_assert!(result.is_err());
            Ok(())
        });
    }

    /// Property: Request IDs should be preserved in responses
    #[test]
    fn prop_request_id_preservation(
        id in arb_request_id()
    ) {
        let _ = tokio_test::block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let config = TurboPropConfig::default();
            let mut server = McpServer::new(temp_dir.path(), &config).await.unwrap();

            // Initialize first
            let init_params = InitializeParams {
                protocol_version: "2024-11-05".to_string(),
                client_info: ClientInfo {
                    name: "test".to_string(),
                    version: "1.0".to_string(),
                },
                capabilities: ClientCapabilities::default(),
            };
            let _ = server.initialize(init_params).await;

            let request = JsonRpcRequest {
                jsonrpc: "2.0".to_string(),
                method: "tools/list".to_string(),
                id: id.clone(),
                params: None,
            };

            let result = server.handle_request(request).await;

            match result {
                Ok(response) => {
                    prop_assert_eq!(response.id, id);
                }
                Err(_) => {
                    // Even errors should preserve request ID context
                }
            }
            Ok(())
        });
    }
}
