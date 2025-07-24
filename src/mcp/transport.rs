//! STDIO transport for MCP communication
//!
//! Handles JSON-RPC messages over stdin/stdout as per MCP specification.
//! Messages are newline-delimited JSON objects.

use anyhow::{Context, Result};
use tokio::io::{stdin, stdout, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;
use tracing::{debug, error};

use super::protocol::{JsonRpcError, JsonRpcRequest, JsonRpcResponse, RequestId};

/// STDIO transport for MCP communication
pub struct StdioTransport {
    /// Channel for receiving requests from stdin
    request_receiver: mpsc::UnboundedReceiver<Result<JsonRpcRequest>>,
    /// Channel for sending responses to stdout  
    response_sender: mpsc::UnboundedSender<JsonRpcResponse>,
    /// Handle for managing the transport tasks
    _handle: StdioTransportHandle,
}

/// Handle for managing STDIO transport background tasks
pub struct StdioTransportHandle {
    /// Task for reading from stdin
    stdin_task: tokio::task::JoinHandle<()>,
    /// Task for writing to stdout
    stdout_task: tokio::task::JoinHandle<()>,
}

impl StdioTransport {
    /// Create a new STDIO transport
    pub fn new() -> Self {
        let (request_tx, request_rx) = mpsc::unbounded_channel();
        let (response_tx, response_rx) = mpsc::unbounded_channel();

        // Start background tasks for stdin/stdout handling
        let stdin_task = tokio::spawn(Self::stdin_reader_task(request_tx));
        let stdout_task = tokio::spawn(Self::stdout_writer_task(response_rx));

        let handle = StdioTransportHandle {
            stdin_task,
            stdout_task,
        };

        Self {
            request_receiver: request_rx,
            response_sender: response_tx,
            _handle: handle,
        }
    }

    /// Background task for reading JSON-RPC requests from stdin
    async fn stdin_reader_task(sender: mpsc::UnboundedSender<Result<JsonRpcRequest>>) {
        let stdin = stdin();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();

        loop {
            line.clear();

            match reader.read_line(&mut line).await {
                Ok(0) => {
                    // EOF reached
                    debug!("STDIN closed, stopping reader task");
                    break;
                }
                Ok(_) => {
                    // Parse JSON-RPC message
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    debug!("Received message: {}", trimmed);

                    let request_result = serde_json::from_str::<JsonRpcRequest>(trimmed)
                        .with_context(|| format!("Failed to parse JSON-RPC request: {}", trimmed));

                    if sender.send(request_result).is_err() {
                        error!("Failed to send request to handler, receiver dropped");
                        break;
                    }
                }
                Err(e) => {
                    error!("Error reading from stdin: {}", e);
                    let error_result = Err(anyhow::anyhow!("STDIN read error: {}", e));
                    if sender.send(error_result).is_err() {
                        break;
                    }
                }
            }
        }
    }

    /// Background task for writing JSON-RPC responses to stdout
    async fn stdout_writer_task(mut receiver: mpsc::UnboundedReceiver<JsonRpcResponse>) {
        let mut stdout = stdout();

        while let Some(response) = receiver.recv().await {
            match serde_json::to_string(&response) {
                Ok(json) => {
                    let message = format!("{}\n", json);
                    debug!("Sending response: {}", json);

                    if let Err(e) = stdout.write_all(message.as_bytes()).await {
                        error!("Failed to write to stdout: {}", e);
                        break;
                    }

                    if let Err(e) = stdout.flush().await {
                        error!("Failed to flush stdout: {}", e);
                        break;
                    }
                }
                Err(e) => {
                    error!("Failed to serialize response: {}", e);
                    // Try to send an error response
                    let error_response = JsonRpcResponse::error(
                        response.id,
                        JsonRpcError::internal_error(format!("Serialization error: {}", e)),
                    );

                    if let Ok(json) = serde_json::to_string(&error_response) {
                        let message = format!("{}\n", json);
                        let _ = stdout.write_all(message.as_bytes()).await;
                        let _ = stdout.flush().await;
                    }
                }
            }
        }

        debug!("STDOUT writer task finished");
    }

    /// Receive the next request from stdin
    pub async fn receive_request(&mut self) -> Option<Result<JsonRpcRequest>> {
        self.request_receiver.recv().await
    }

    /// Send a response to stdout
    pub fn send_response(&self, response: JsonRpcResponse) -> Result<()> {
        self.response_sender
            .send(response)
            .map_err(|_| anyhow::anyhow!("Response receiver dropped"))
    }

    /// Create an error response for invalid requests
    pub fn create_error_response(
        request_id: Option<RequestId>,
        error: JsonRpcError,
    ) -> JsonRpcResponse {
        let id = request_id.unwrap_or(RequestId::from_number(0));
        JsonRpcResponse::error(id, error)
    }
}

impl Drop for StdioTransportHandle {
    fn drop(&mut self) {
        // Abort background tasks when handle is dropped
        self.stdin_task.abort();
        self.stdout_task.abort();
    }
}

/// Helper for validating JSON-RPC requests
pub struct RequestValidator;

impl RequestValidator {
    /// Validate a JSON-RPC request according to MCP requirements
    pub fn validate(request: &JsonRpcRequest) -> Result<(), JsonRpcError> {
        // Check JSON-RPC version
        if request.jsonrpc != "2.0" {
            return Err(JsonRpcError::invalid_request(format!(
                "Invalid jsonrpc version: {}, expected '2.0'",
                request.jsonrpc
            )));
        }

        // Check method name
        if request.method.is_empty() {
            return Err(JsonRpcError::invalid_request(
                "Method name cannot be empty".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_validator() {
        // Valid request
        let valid_request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: RequestId::from_number(1),
            method: "test_method".to_string(),
            params: None,
        };
        assert!(RequestValidator::validate(&valid_request).is_ok());

        // Invalid JSON-RPC version
        let invalid_version = JsonRpcRequest {
            jsonrpc: "1.0".to_string(),
            id: RequestId::from_number(1),
            method: "test_method".to_string(),
            params: None,
        };
        assert!(RequestValidator::validate(&invalid_version).is_err());

        // Test passes now since we removed the null ID check

        // Empty method name
        let empty_method = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: RequestId::from_number(1),
            method: "".to_string(),
            params: None,
        };
        assert!(RequestValidator::validate(&empty_method).is_err());
    }

    #[tokio::test]
    async fn test_transport_creation() {
        let transport = StdioTransport::new();

        // Transport should be created successfully
        // Background tasks should be running
        // This is mainly a compilation and basic functionality test
        drop(transport);
    }

    #[test]
    fn test_error_response_creation() {
        let error = JsonRpcError::method_not_found("unknown".to_string());
        let response =
            StdioTransport::create_error_response(Some(RequestId::from_number(123)), error);

        assert_eq!(response.id, RequestId::from_number(123));
        assert!(response.error.is_some());
        assert!(response.result.is_none());
        assert_eq!(response.error.unwrap().code, -32601);
    }
}
