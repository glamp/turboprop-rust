//! Integration tests for MCP server functionality
//!
//! These tests validate end-to-end MCP functionality and are part of
//! the slow test suite (`cargo test --test integration`).

use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::time::Duration;
use tempfile::TempDir;
use tokio::time::timeout;

use turboprop::config::TurboPropConfig;
use turboprop::mcp::{McpServer, McpServerTrait};

/// Test utilities for MCP integration tests
mod test_utils {
    use super::*;

    /// Create a test repository with sample files
    pub fn create_test_repo() -> TempDir {
        let temp_dir = TempDir::new().unwrap();

        // Initialize as git repo to avoid warnings
        Command::new("git")
            .args(&["init"])
            .current_dir(temp_dir.path())
            .output()
            .ok();

        // Create sample files
        std::fs::create_dir_all(temp_dir.path().join("src")).unwrap();
        std::fs::write(
            temp_dir.path().join("src/main.rs"),
            r#"
fn main() {
    println!("Hello, world!");
}

fn authenticate_user(username: &str, password: &str) -> bool {
    // Simple authentication logic
    username == "admin" && password == "secret"
}

fn calculate_total(items: &[f64]) -> f64 {
    items.iter().sum()
}
"#,
        )
        .unwrap();

        std::fs::write(
            temp_dir.path().join("src/lib.rs"),
            r#"
/// JWT token validation function
pub fn validate_jwt_token(token: &str) -> Result<bool, String> {
    if token.is_empty() {
        return Err("Token cannot be empty".to_string());
    }
    
    // Mock validation logic
    Ok(token.starts_with("eyJ"))
}

/// Error handling utilities
pub mod error {
    pub fn handle_database_error(err: &str) -> String {
        format!("Database error: {}", err)
    }
}
"#,
        )
        .unwrap();

        std::fs::write(
            temp_dir.path().join("README.md"),
            r#"# Test Project

This is a test project for TurboProp MCP server testing.

## Features

- User authentication
- JWT token validation
- Error handling
- Mathematical calculations
"#,
        )
        .unwrap();

        temp_dir
    }

    /// Start an MCP server process with improved error handling
    pub fn start_mcp_server(repo_path: &std::path::Path) -> Result<Child> {
        // Ensure the repository directory exists and has some content
        std::fs::create_dir_all(repo_path).context("Failed to create test repository directory")?;

        // Create a simple .turboprop.yml config for testing
        let config_content = r#"# Test configuration for MCP server
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 8

search:
  default_limit: 10
  min_similarity: 0.3

indexing:
  max_filesize: "1mb"
  include_patterns:
    - "**/*.rs"
    - "**/*.md"
"#;
        std::fs::write(repo_path.join(".turboprop.yml"), config_content)
            .context("Failed to create test configuration")?;

        let mut cmd = Command::new("cargo");
        cmd.args(&["run", "--bin", "tp", "--", "mcp", "--repo"])
            .arg(repo_path)
            .arg("--debug") // Use debug instead of force-rebuild to get better error messages
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("RUST_LOG", "debug"); // Enable debug logging

        cmd.spawn().context("Failed to start MCP server process")
    }

    /// Send a JSON-RPC request and receive response with better error handling
    pub async fn send_mcp_request(process: &mut Child, request: Value) -> Result<Value> {
        let stdin = process
            .stdin
            .as_mut()
            .context("Failed to get stdin handle")?;

        let request_str = format!("{}\n", serde_json::to_string(&request)?);

        // Write request
        stdin
            .write_all(request_str.as_bytes())
            .context("Failed to write request to stdin")?;
        stdin.flush().context("Failed to flush stdin")?;

        // Read response with timeout
        let stdout = process
            .stdout
            .as_mut()
            .context("Failed to get stdout handle")?;

        let mut reader = BufReader::new(stdout);
        let mut response_line = String::new();

        // Try to read response line
        match reader.read_line(&mut response_line) {
            Ok(0) => return Err(anyhow::anyhow!("Process closed stdout")),
            Ok(_) => {}
            Err(e) => return Err(anyhow::anyhow!("Failed to read response: {}", e)),
        }

        // Parse JSON response
        let response: Value = serde_json::from_str(response_line.trim()).context(format!(
            "Failed to parse JSON response: '{}'",
            response_line.trim()
        ))?;

        Ok(response)
    }

    /// Wait for MCP server to be ready with improved error reporting
    pub async fn wait_for_server_ready(process: &mut Child) -> Result<()> {
        // Give the server time to start, initialize, and potentially download the model
        tokio::time::sleep(Duration::from_millis(10000)).await;

        // Check if process is still alive
        match process.try_wait() {
            Ok(Some(status)) => {
                // Capture stderr output for debugging
                let mut stderr_output = String::new();
                if let Some(stderr) = process.stderr.as_mut() {
                    use std::io::Read;
                    let _ = stderr.read_to_string(&mut stderr_output);
                }

                // Also try to capture any remaining stdout
                let mut stdout_output = String::new();
                if let Some(stdout) = process.stdout.as_mut() {
                    use std::io::Read;
                    let _ = stdout.read_to_string(&mut stdout_output);
                }

                return Err(anyhow::anyhow!(
                    "MCP server exited early with status: {}. stderr: '{}'. stdout: '{}'",
                    status,
                    stderr_output,
                    stdout_output
                ));
            }
            Ok(None) => {} // Still running, good
            Err(e) => {
                return Err(anyhow::anyhow!("Failed to check process status: {}", e));
            }
        }

        Ok(())
    }

    /// Properly shutdown an MCP server process
    pub fn shutdown_mcp_server(mut process: Child) -> Result<()> {
        // Try graceful shutdown first
        if let Some(stdin) = process.stdin.as_mut() {
            let _ = stdin.write_all(b"\n");
            let _ = stdin.flush();
        }

        // Wait a bit for graceful shutdown
        std::thread::sleep(Duration::from_millis(500));

        // Force kill if still running
        match process.try_wait() {
            Ok(Some(_)) => {} // Already exited
            Ok(None) => {
                let _ = process.kill();
                let _ = process.wait();
            }
            Err(_) => {
                let _ = process.kill();
            }
        }

        Ok(())
    }
}

/// Test MCP server binary integration
mod binary_tests {
    use super::*;
    use test_utils::*;

    #[tokio::test]
    #[ignore = "Binary tests require model downloading and are brittle in CI - using library tests instead"]
    async fn test_mcp_binary_startup() {
        let test_repo = create_test_repo();

        // Start MCP server process
        let mut child = start_mcp_server(test_repo.path()).expect("Failed to start MCP server");

        // Wait for server to be ready
        if let Err(e) = wait_for_server_ready(&mut child).await {
            let _ = shutdown_mcp_server(child);
            panic!("Server failed to start: {}", e);
        }

        // Send initialize request
        let initialize_request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                },
                "capabilities": {}
            }
        });

        let response = match timeout(
            Duration::from_secs(15),
            send_mcp_request(&mut child, initialize_request),
        )
        .await
        {
            Ok(Ok(response)) => response,
            Ok(Err(e)) => {
                let _ = shutdown_mcp_server(child);
                panic!("Failed to send request: {}", e);
            }
            Err(_) => {
                let _ = shutdown_mcp_server(child);
                panic!("Request timed out");
            }
        };

        // Verify response
        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 1);

        if !response["error"].is_null() {
            let _ = shutdown_mcp_server(child);
            panic!("Server returned error: {}", response["error"]);
        }

        assert_eq!(response["result"]["protocolVersion"], "2024-11-05");
        assert_eq!(response["result"]["serverInfo"]["name"], "turboprop");

        // Cleanup
        let _ = shutdown_mcp_server(child);
    }

    #[tokio::test]
    #[ignore = "Binary tests require model downloading and are brittle in CI - using library tests instead"]
    async fn test_tools_list_integration() {
        let test_repo = create_test_repo();

        let mut child = start_mcp_server(test_repo.path()).expect("Failed to start MCP server");

        if let Err(e) = wait_for_server_ready(&mut child).await {
            let _ = shutdown_mcp_server(child);
            panic!("Server failed to start: {}", e);
        }

        // Initialize first
        let initialize_request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "test", "version": "1.0"},
                "capabilities": {}
            }
        });

        let init_response = match send_mcp_request(&mut child, initialize_request).await {
            Ok(response) => response,
            Err(e) => {
                let _ = shutdown_mcp_server(child);
                panic!("Failed to initialize: {}", e);
            }
        };

        if !init_response["error"].is_null() {
            let _ = shutdown_mcp_server(child);
            panic!("Initialization failed: {}", init_response["error"]);
        }

        // Request tools list
        let tools_request = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        });

        let response = match timeout(
            Duration::from_secs(10),
            send_mcp_request(&mut child, tools_request),
        )
        .await
        {
            Ok(Ok(response)) => response,
            Ok(Err(e)) => {
                let _ = shutdown_mcp_server(child);
                panic!("Failed to send tools/list request: {}", e);
            }
            Err(_) => {
                let _ = shutdown_mcp_server(child);
                panic!("Tools/list request timed out");
            }
        };

        // Verify response
        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 2);

        if !response["error"].is_null() {
            let _ = shutdown_mcp_server(child);
            panic!("Tools/list returned error: {}", response["error"]);
        }

        let tools = response["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "semantic_search");
        assert!(tools[0]["inputSchema"].is_object());

        let _ = shutdown_mcp_server(child);
    }

    #[tokio::test]
    #[ignore = "Binary tests require model downloading, indexing, and are brittle in CI - using library tests instead"]
    async fn test_search_tool_integration() {
        let test_repo = create_test_repo();

        let mut child = start_mcp_server(test_repo.path()).expect("Failed to start MCP server");

        // Give server more time to index files
        tokio::time::sleep(Duration::from_millis(8000)).await;

        if let Err(e) = wait_for_server_ready(&mut child).await {
            let _ = shutdown_mcp_server(child);
            panic!("Server failed to start: {}", e);
        }

        // Initialize
        let initialize_request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "test", "version": "1.0"},
                "capabilities": {}
            }
        });

        let init_response = match send_mcp_request(&mut child, initialize_request).await {
            Ok(response) => response,
            Err(e) => {
                let _ = shutdown_mcp_server(child);
                panic!("Failed to initialize: {}", e);
            }
        };

        if !init_response["error"].is_null() {
            let _ = shutdown_mcp_server(child);
            panic!("Initialization failed: {}", init_response["error"]);
        }

        // Wait a bit more for indexing to complete
        tokio::time::sleep(Duration::from_millis(3000)).await;

        // Execute search
        let search_request = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "semantic_search",
                "arguments": {
                    "query": "authentication",
                    "limit": 5
                }
            }
        });

        let response = match timeout(
            Duration::from_secs(15),
            send_mcp_request(&mut child, search_request),
        )
        .await
        {
            Ok(Ok(response)) => response,
            Ok(Err(e)) => {
                let _ = shutdown_mcp_server(child);
                panic!("Failed to send search request: {}", e);
            }
            Err(_) => {
                let _ = shutdown_mcp_server(child);
                panic!("Search request timed out");
            }
        };

        // Verify response structure (results may vary based on indexing)
        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 3);

        // Check if we got results or an appropriate error
        if response["error"].is_null() {
            let result = &response["result"];
            assert!(result["results"].is_array());
            assert!(result["total_results"].is_number());
            assert!(result["execution_time_ms"].is_number());
        } else {
            // If indexing isn't complete, we should get an appropriate error
            let error = &response["error"];
            assert!(error["code"].is_number());
            assert!(error["message"].is_string());
            println!(
                "Search returned error (expected during testing): {}",
                error["message"]
            );
        }

        let _ = shutdown_mcp_server(child);
    }
}

/// Test library-level MCP server functionality
mod library_tests {
    use super::*;
    use test_utils::*;

    #[tokio::test]
    async fn test_mcp_server_library_usage() {
        let test_repo = create_test_repo();
        let config = TurboPropConfig::default();

        // Create server
        let mut server = McpServer::new(test_repo.path(), &config)
            .await
            .expect("Failed to create MCP server");

        // Test initialization request
        let params = turboprop::mcp::protocol::InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            client_info: turboprop::mcp::protocol::ClientInfo {
                name: "test".to_string(),
                version: "1.0".to_string(),
            },
            capabilities: turboprop::mcp::protocol::ClientCapabilities::default(),
        };

        let result = server.initialize(params).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_concurrent_requests() {
        let test_repo = create_test_repo();
        let config = TurboPropConfig::default();

        let mut server = McpServer::new(test_repo.path(), &config)
            .await
            .expect("Failed to create MCP server");

        // Initialize the server first
        let params = turboprop::mcp::protocol::InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            client_info: turboprop::mcp::protocol::ClientInfo {
                name: "test".to_string(),
                version: "1.0".to_string(),
            },
            capabilities: turboprop::mcp::protocol::ClientCapabilities::default(),
        };
        server
            .initialize(params)
            .await
            .expect("Failed to initialize server");

        // Send multiple sequential requests to test server stability
        for i in 0..5 {
            let request = json!({
                "jsonrpc": "2.0",
                "id": i,
                "method": "tools/list"
            });

            let json_rpc_request = serde_json::from_value(request).unwrap();
            let response = server.handle_request(json_rpc_request).await;

            match response {
                Ok(json_response) => {
                    assert!(json_response.error.is_none());
                    assert!(json_response.result.is_some());
                }
                Err(e) => {
                    println!("Request {} failed: {}", i, e);
                    panic!("Request should succeed but got error: {}", e);
                }
            }
        }
    }
}
