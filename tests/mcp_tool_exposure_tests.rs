//! Tests to verify that the MCP server only exposes the expected tools
//!
//! This test suite ensures that the MCP server has a controlled interface
//! and doesn't accidentally expose internal tools or unintended functionality.

use serde_json::json;
use tempfile::TempDir;

use turboprop::config::TurboPropConfig;
use turboprop::mcp::protocol::{ClientCapabilities, ClientInfo, InitializeParams, JsonRpcRequest};
use turboprop::mcp::server::{McpServer, McpServerTrait};
use turboprop::mcp::tools::Tools;

/// Test that verifies exactly one tool is exposed by the MCP server
#[cfg(test)]
mod tool_exposure_tests {
    use super::*;

    #[tokio::test]
    async fn test_mcp_server_exposes_only_semantic_search_tool() {
        // Create a test server
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();
        let mut server = McpServer::new(temp_dir.path(), &config).await.unwrap();

        // Initialize the server
        let init_params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };
        server.initialize(init_params).await.unwrap();

        // Request the tools list
        let tools_request = JsonRpcRequest::new("tools/list".to_string(), None);
        let response = server.handle_request(tools_request).await.unwrap();

        // Verify the response structure
        assert!(response.error.is_none(), "Tools list request should succeed");
        assert!(response.result.is_some(), "Response should contain result");

        let result = response.result.unwrap();
        let tools_array = result["tools"]
            .as_array()
            .expect("Result should contain a 'tools' array");

        // Verify exactly one tool is exposed
        assert_eq!(
            tools_array.len(),
            1,
            "Expected exactly 1 tool to be exposed, found {}. Tools: {:?}",
            tools_array.len(),
            tools_array
        );

        // Verify the single tool is semantic_search
        let tool = &tools_array[0];
        assert_eq!(
            tool["name"].as_str().unwrap(),
            "semantic_search",
            "The single exposed tool should be 'semantic_search', found: {}",
            tool["name"]
        );

        // Verify the tool has required fields
        assert!(
            tool["description"].is_string(),
            "Tool should have a description"
        );
        assert!(
            tool["input_schema"].is_object(),
            "Tool should have an input schema"
        );

        // Verify the description is appropriate
        let description = tool["description"].as_str().unwrap();
        assert!(
            description.contains("semantic") || description.contains("search"),
            "Tool description should mention semantic search functionality"
        );

        // Verify the input schema has required structure
        let schema = &tool["input_schema"];
        assert_eq!(schema["type"], "object", "Schema should be an object type");
        assert!(
            schema["properties"].is_object(),
            "Schema should have properties"
        );
        assert!(
            schema["required"].is_array(),
            "Schema should specify required fields"
        );

        // Verify required 'query' parameter is present
        let required_fields = schema["required"].as_array().unwrap();
        assert!(
            required_fields.contains(&json!("query")),
            "Schema should require 'query' parameter"
        );
    }

    #[tokio::test]
    async fn test_tools_registry_direct_verification() {
        // Create a tools registry for integration tests
        let tools = Tools::new_for_integration_tests();

        // Get the list of all available tools
        let tool_list = tools.list_tools();

        // Verify exactly one tool is registered
        assert_eq!(
            tool_list.len(),
            1,
            "Expected exactly 1 tool in registry, found {}. Tools: {:?}",
            tool_list.len(),
            tool_list.iter().map(|t| &t.name).collect::<Vec<_>>()
        );

        // Verify the single tool is semantic_search
        let tool_definition = &tool_list[0];
        assert_eq!(
            tool_definition.name, "semantic_search",
            "The single tool should be 'semantic_search', found: {}",
            tool_definition.name
        );

        // Verify the tool is accessible by name
        assert!(
            tools.has_tool("semantic_search"),
            "Registry should contain 'semantic_search' tool"
        );

        // Verify no other common tool names are present
        let common_tool_names = [
            "search",
            "index",
            "list_files",
            "read_file",
            "write_file",
            "execute",
            "shell",
            "filesystem",
            "git",
            "database",
            "network",
            "system",
            "admin",
            "config",
            "debug",
            "test",
        ];

        for tool_name in &common_tool_names {
            assert!(
                !tools.has_tool(tool_name),
                "Registry should not contain unexpected tool: {}",
                tool_name
            );
        }
    }

    #[tokio::test]
    async fn test_semantic_search_tool_definition_completeness() {
        let tools = Tools::new_for_integration_tests();
        let tool_list = tools.list_tools();
        let semantic_search_tool = &tool_list[0];

        // Verify tool name is exactly what we expect
        assert_eq!(semantic_search_tool.name, "semantic_search");

        // Verify description is comprehensive
        let description = &semantic_search_tool.description;
        assert!(
            !description.is_empty(),
            "Tool description should not be empty"
        );
        assert!(
            description.len() > 20,
            "Tool description should be descriptive (>20 chars), got: {}",
            description
        );

        // Verify input schema is well-defined
        let schema = &semantic_search_tool.input_schema;
        assert!(schema.is_object(), "Input schema should be an object");

        // Check for essential schema properties
        let properties = schema["properties"]
            .as_object()
            .expect("Schema should have properties");

        // Verify required parameters are defined
        assert!(
            properties.contains_key("query"),
            "Schema should define 'query' parameter"
        );

        // Verify optional parameters are defined
        let expected_optional_params = ["limit", "threshold", "filetype", "filter"];
        for param in &expected_optional_params {
            assert!(
                properties.contains_key(*param),
                "Schema should define '{}' parameter",
                param
            );
        }

        // Verify the schema specifies the type correctly
        assert_eq!(
            schema["type"], "object",
            "Schema type should be 'object'"
        );

        // Verify additionalProperties is set to false for strict validation
        assert_eq!(
            schema["additionalProperties"], false,
            "Schema should not allow additional properties for security"
        );
    }

    #[tokio::test]
    async fn test_no_hidden_or_debug_tools_exposed() {
        let tools = Tools::new_for_integration_tests();

        // List of potentially dangerous or debug tools that should never be exposed
        let forbidden_tools = [
            "exec",
            "execute",
            "shell",
            "bash",
            "cmd",
            "system",
            "filesystem",
            "fs",
            "file_read",
            "file_write",
            "file_delete",
            "dir_list",
            "process",
            "network",
            "http",
            "request",
            "sql",
            "database",
            "db",
            "admin",
            "config_edit",
            "settings",
            "debug",
            "test_internal",
            "development",
            "dev",
            "internal",
            "private",
            "_hidden",
            "experimental",
            "unsafe",
            "raw",
            "direct",
        ];

        for forbidden_tool in &forbidden_tools {
            assert!(
                !tools.has_tool(forbidden_tool),
                "Server should never expose potentially dangerous tool: {}",
                forbidden_tool
            );
        }

        // Also verify variations with common prefixes/suffixes
        let prefixes = ["mcp_", "turboprop_", "internal_", "debug_", "test_"];
        let suffixes = ["_tool", "_exec", "_internal", "_debug", "_test"];

        for prefix in &prefixes {
            for suffix in &suffixes {
                let tool_name = format!("{}{}{}", prefix, "semantic_search", suffix);
                if tool_name != "semantic_search" {
                    assert!(
                        !tools.has_tool(&tool_name),
                        "Server should not expose tool with suspicious name: {}",
                        tool_name
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_tool_registry_isolation() {
        // Create multiple tool registries to ensure they don't cross-contaminate
        let tools1 = Tools::new_for_integration_tests();
        let tools2 = Tools::new_for_integration_tests();

        // Both should have exactly the same tools
        let tools1_list = tools1.list_tools();
        let tools2_list = tools2.list_tools();

        assert_eq!(
            tools1_list.len(),
            tools2_list.len(),
            "Tool registries should be consistent"
        );

        assert_eq!(
            tools1_list[0].name, tools2_list[0].name,
            "Tool registries should contain the same tools"
        );

        // Verify they both only have semantic_search
        assert_eq!(tools1_list.len(), 1);
        assert_eq!(tools2_list.len(), 1);
        assert_eq!(tools1_list[0].name, "semantic_search");
        assert_eq!(tools2_list[0].name, "semantic_search");
    }

    #[tokio::test]
    async fn test_tools_list_response_format() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();
        let mut server = McpServer::new(temp_dir.path(), &config).await.unwrap();

        // Initialize the server
        let init_params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };
        server.initialize(init_params).await.unwrap();

        // Request tools list
        let tools_request = JsonRpcRequest::new("tools/list".to_string(), None);
        let response = server.handle_request(tools_request).await.unwrap();

        // Verify JSON-RPC response format
        assert_eq!(response.jsonrpc, "2.0");
        assert!(response.error.is_none());
        assert!(response.result.is_some());

        let result = response.result.unwrap();

        // Verify the response structure follows MCP specification
        assert!(result.is_object(), "Result should be an object");
        assert!(
            result.get("tools").is_some(),
            "Result should contain 'tools' field"
        );

        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1, "Should contain exactly one tool");

        let tool = &tools[0];

        // Verify tool object structure per MCP spec
        let required_fields = ["name", "description", "input_schema"];
        for field in &required_fields {
            assert!(
                tool.get(*field).is_some(),
                "Tool should have required field: {}",
                field
            );
        }

        // Verify field types
        assert!(tool["name"].is_string(), "Tool name should be a string");
        assert!(
            tool["description"].is_string(),
            "Tool description should be a string"
        );
        assert!(
            tool["input_schema"].is_object(),
            "Tool input_schema should be an object"
        );
    }

    #[tokio::test]
    async fn test_server_state_consistency_across_requests() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();
        let mut server = McpServer::new(temp_dir.path(), &config).await.unwrap();

        // Initialize the server
        let init_params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };
        server.initialize(init_params).await.unwrap();

        // Make multiple tools/list requests
        for i in 0..5 {
            let tools_request = JsonRpcRequest::new("tools/list".to_string(), None);
            let response = server.handle_request(tools_request).await.unwrap();

            assert!(
                response.error.is_none(),
                "Request {} should succeed",
                i + 1
            );

            let result = response.result.unwrap();
            let tools = result["tools"].as_array().unwrap();

            assert_eq!(
                tools.len(),
                1,
                "Request {} should return exactly 1 tool",
                i + 1
            );

            assert_eq!(
                tools[0]["name"], "semantic_search",
                "Request {} should return semantic_search tool",
                i + 1
            );
        }
    }
}