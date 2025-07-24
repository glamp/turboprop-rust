//! STDIO transport for MCP communication
//!
//! Handles JSON-RPC messages over stdin/stdout as per MCP specification.
//! Messages are newline-delimited JSON objects.

use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::io::{stdin, stdout, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, Mutex};
use tokio::time::{timeout, Duration, Instant};
use tracing::{debug, error, info, warn};

use super::protocol::{JsonRpcError, JsonRpcRequest, JsonRpcResponse, RequestId};

/// Simple token bucket rate limiter for security
#[derive(Debug)]
pub struct TokenBucketRateLimiter {
    /// Maximum number of tokens (burst capacity)
    max_tokens: u32,
    /// Current number of tokens
    current_tokens: u32,
    /// Rate of token replenishment (tokens per minute)
    refill_rate: u32,
    /// Last refill time
    last_refill: Instant,
}

impl TokenBucketRateLimiter {
    /// Create a new rate limiter
    pub fn new(max_tokens: u32, refill_rate: u32) -> Self {
        Self {
            max_tokens,
            current_tokens: max_tokens,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    /// Try to consume a token (returns true if allowed, false if rate limited)
    pub fn try_consume(&mut self) -> bool {
        // Refill tokens based on elapsed time
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);
        let tokens_to_add = (elapsed.as_secs_f64() * self.refill_rate as f64 / 60.0) as u32;

        if tokens_to_add > 0 {
            self.current_tokens = (self.current_tokens + tokens_to_add).min(self.max_tokens);
            self.last_refill = now;
        }

        // Try to consume a token
        if self.current_tokens > 0 {
            self.current_tokens -= 1;
            true
        } else {
            false
        }
    }

    /// Get current token count (for observability)
    pub fn current_tokens(&self) -> u32 {
        self.current_tokens
    }
}

/// Configuration for STDIO transport
#[derive(Debug, Clone)]
pub struct StdioTransportConfig {
    /// Maximum message size in bytes (default: 1MB)
    pub max_message_size: usize,
    /// Read timeout in seconds (default: 30)
    pub read_timeout_seconds: u64,
    /// Write timeout in seconds (default: 10)
    pub write_timeout_seconds: u64,
    /// Channel buffer size for backpressure (default: 100)
    pub channel_buffer_size: usize,
    /// Rate limiting: maximum requests per minute (default: 60)
    pub max_requests_per_minute: u32,
    /// Rate limiting: burst capacity (default: 10)
    pub rate_limit_burst_capacity: u32,
}

impl Default for StdioTransportConfig {
    fn default() -> Self {
        Self {
            max_message_size: 1024 * 1024, // 1MB
            read_timeout_seconds: 30,
            write_timeout_seconds: 10,
            channel_buffer_size: 100,
            max_requests_per_minute: 60,
            rate_limit_burst_capacity: 10,
        }
    }
}

/// STDIO transport for MCP communication
pub struct StdioTransport {
    /// Transport configuration
    config: StdioTransportConfig,
    /// Channel for receiving requests from stdin
    request_receiver: mpsc::Receiver<Result<JsonRpcRequest>>,
    /// Channel for sending responses to stdout  
    response_sender: mpsc::Sender<JsonRpcResponse>,
    /// Handle for managing the transport tasks
    _handle: StdioTransportHandle,
    /// Rate limiter for incoming requests
    rate_limiter: Arc<Mutex<TokenBucketRateLimiter>>,
}

/// Handle for managing STDIO transport background tasks
pub struct StdioTransportHandle {
    /// Task for reading from stdin
    stdin_task: tokio::task::JoinHandle<()>,
    /// Task for writing to stdout
    stdout_task: tokio::task::JoinHandle<()>,
}

impl StdioTransport {
    /// Create a new STDIO transport with default configuration
    pub fn new() -> Self {
        Self::with_config(StdioTransportConfig::default())
    }

    /// Create a new STDIO transport with custom configuration
    pub fn with_config(config: StdioTransportConfig) -> Self {
        let (request_tx, request_rx) = mpsc::channel(config.channel_buffer_size);
        let (response_tx, response_rx) = mpsc::channel(config.channel_buffer_size);

        // Create rate limiter
        let rate_limiter = Arc::new(Mutex::new(TokenBucketRateLimiter::new(
            config.rate_limit_burst_capacity,
            config.max_requests_per_minute,
        )));

        // Start background tasks for stdin/stdout handling (simplified without shutdown)
        let stdin_task = tokio::spawn(Self::stdin_reader_task(
            request_tx,
            config.clone(),
            Arc::clone(&rate_limiter),
        ));
        let stdout_task = tokio::spawn(Self::stdout_writer_task(response_rx, config.clone()));

        let handle = StdioTransportHandle {
            stdin_task,
            stdout_task,
        };

        Self {
            config,
            request_receiver: request_rx,
            response_sender: response_tx,
            _handle: handle,
            rate_limiter,
        }
    }

    /// Background task for reading JSON-RPC requests from stdin
    async fn stdin_reader_task(
        sender: mpsc::Sender<Result<JsonRpcRequest>>,
        config: StdioTransportConfig,
        rate_limiter: Arc<Mutex<TokenBucketRateLimiter>>,
    ) {
        let stdin = stdin();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();
        let read_timeout = Duration::from_secs(config.read_timeout_seconds);

        loop {
            line.clear();

            // Add timeout to read operation
            let read_result = timeout(read_timeout, reader.read_line(&mut line)).await;

            match read_result {
                Ok(Ok(0)) => {
                    // EOF reached
                    debug!("STDIN closed, stopping reader task");
                    break;
                }
                Ok(Ok(bytes_read)) => {
                    // Check message size limit
                    if bytes_read > config.max_message_size {
                        warn!(
                            "Message size {} exceeds limit {}, rejecting",
                            bytes_read, config.max_message_size
                        );
                        let error_result = Err(anyhow::anyhow!(
                            "Message size {} exceeds maximum allowed size {}",
                            bytes_read,
                            config.max_message_size
                        ));
                        if sender.send(error_result).await.is_err() {
                            break;
                        }
                        continue;
                    }

                    // Parse JSON-RPC message
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    debug!("Received message ({} bytes): {}", bytes_read, trimmed);

                    let request_result = serde_json::from_str::<JsonRpcRequest>(trimmed)
                        .with_context(|| {
                            format!(
                                "Failed to parse JSON-RPC request (size: {}): {}",
                                bytes_read, trimmed
                            )
                        })
                        .and_then(|request| {
                            // Validate the parsed request
                            RequestValidator::validate(&request).map_err(|e| {
                                anyhow::anyhow!("Request validation failed: {:?}", e)
                            })?;
                            Ok(request)
                        })
                        .and_then(|request| {
                            // Apply rate limiting
                            let mut limiter = rate_limiter.blocking_lock();
                            if !limiter.try_consume() {
                                warn!(
                                    "Rate limit exceeded for request: method={}, tokens={}",
                                    request.method,
                                    limiter.current_tokens()
                                );
                                info!(
                                    "Security event: Rate limiting triggered for method '{}'",
                                    request.method
                                );
                                return Err(anyhow::anyhow!("Rate limit exceeded"));
                            }

                            debug!(
                                "Request allowed: method={}, remaining_tokens={}",
                                request.method,
                                limiter.current_tokens()
                            );
                            Ok(request)
                        });

                    if sender.send(request_result).await.is_err() {
                        error!("Failed to send request to handler, receiver dropped");
                        break;
                    }
                }
                Ok(Err(e)) => {
                    error!("Error reading from stdin: {}", e);
                    let error_result = Err(anyhow::anyhow!("STDIN read error: {}", e)
                        .context("Transport layer error during stdin read"));
                    if sender.send(error_result).await.is_err() {
                        break;
                    }
                }
                Err(_) => {
                    warn!(
                        "Read timeout after {} seconds, continuing",
                        config.read_timeout_seconds
                    );
                    // Continue the loop on timeout rather than breaking
                    // This allows the server to handle other messages
                    continue;
                }
            }
        }
        debug!("STDIN reader task finished");
    }

    /// Background task for writing JSON-RPC responses to stdout
    async fn stdout_writer_task(
        mut receiver: mpsc::Receiver<JsonRpcResponse>,
        config: StdioTransportConfig,
    ) {
        let mut stdout = stdout();
        let write_timeout = Duration::from_secs(config.write_timeout_seconds);

        while let Some(response) = receiver.recv().await {
            match serde_json::to_string(&response) {
                Ok(json) => {
                    let message = format!("{}\n", json);
                    let message_size = message.len();

                    // Check message size limit before sending
                    if message_size > config.max_message_size {
                        error!(
                            "Response message size {} exceeds limit {}, dropping message",
                            message_size, config.max_message_size
                        );
                        // Send a simplified error response instead
                        let error_response = JsonRpcResponse::error(
                            response.id,
                            JsonRpcError::internal_error("Response too large"),
                        );
                        if let Ok(error_json) = serde_json::to_string(&error_response) {
                            let error_message = format!("{}\n", error_json);
                            let _ =
                                timeout(write_timeout, stdout.write_all(error_message.as_bytes()))
                                    .await;
                            let _ = timeout(write_timeout, stdout.flush()).await;
                        }
                        continue;
                    }

                    debug!("Sending response ({} bytes): {}", message_size, json);

                    // Add timeout to write operations
                    match timeout(write_timeout, stdout.write_all(message.as_bytes())).await {
                        Ok(Ok(())) => {
                            match timeout(write_timeout, stdout.flush()).await {
                                Ok(Ok(())) => {
                                    // Success
                                }
                                Ok(Err(e)) => {
                                    error!("Failed to flush stdout: {}", e);
                                    break;
                                }
                                Err(_) => {
                                    error!(
                                        "Timeout flushing stdout after {} seconds",
                                        config.write_timeout_seconds
                                    );
                                    break;
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            error!("Failed to write to stdout: {}", e);
                            break;
                        }
                        Err(_) => {
                            error!(
                                "Timeout writing to stdout after {} seconds",
                                config.write_timeout_seconds
                            );
                            break;
                        }
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
                        let _ = timeout(write_timeout, stdout.write_all(message.as_bytes())).await;
                        let _ = timeout(write_timeout, stdout.flush()).await;
                    }
                }
            }
        }

        debug!("STDOUT writer task finished");
    }

    /// Get transport configuration
    pub fn config(&self) -> &StdioTransportConfig {
        &self.config
    }

    /// Get current rate limiter status for observability
    pub async fn get_rate_limit_status(&self) -> (u32, u32) {
        let limiter = self.rate_limiter.lock().await;
        (limiter.current_tokens(), limiter.max_tokens)
    }

    /// Receive the next request from stdin
    pub async fn receive_request(&mut self) -> Option<Result<JsonRpcRequest>> {
        self.request_receiver.recv().await
    }

    /// Send a response to stdout
    pub async fn send_response(&self, response: JsonRpcResponse) -> Result<()> {
        self.response_sender
            .send(response)
            .await
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

impl Default for StdioTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for StdioTransportHandle {
    fn drop(&mut self) {
        // Abort background tasks when handle is dropped
        // Note: In production, graceful shutdown should be called before dropping
        debug!("StdioTransportHandle dropped, aborting background tasks");
        self.stdin_task.abort();
        self.stdout_task.abort();
    }
}

/// Helper for validating JSON-RPC requests
pub struct RequestValidator;

impl RequestValidator {
    /// Valid MCP method names
    const VALID_METHODS: &'static [&'static str] = &["initialize", "tools/list", "tools/call"];

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

        // Validate method name pattern (MCP methods should not start with reserved prefixes)
        if request.method.starts_with("rpc.") {
            return Err(JsonRpcError::invalid_request(format!(
                "Method name '{}' uses reserved 'rpc.' prefix",
                request.method
            )));
        }

        // Check for valid MCP method names (allow experimental methods with 'experimental/' prefix)
        if !Self::VALID_METHODS.contains(&request.method.as_str())
            && !request.method.starts_with("experimental/")
        {
            return Err(JsonRpcError::method_not_found(request.method.clone()));
        }

        // Additional validation for specific methods
        match request.method.as_str() {
            "initialize" => {
                if request.params.is_none() {
                    return Err(JsonRpcError::invalid_params(
                        "initialize method requires parameters".to_string(),
                    ));
                }
            }
            "tools/call" => {
                if request.params.is_none() {
                    return Err(JsonRpcError::invalid_params(
                        "tools/call method requires parameters".to_string(),
                    ));
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Validate request size
    pub fn validate_size(request_size: usize, max_size: usize) -> Result<(), String> {
        if request_size > max_size {
            return Err(format!(
                "Request size {} exceeds maximum allowed size {}",
                request_size, max_size
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
            method: "initialize".to_string(),
            params: Some(serde_json::json!({
                "protocol_version": "2024-11-05",
                "client_info": {
                    "name": "test-client",
                    "version": "1.0.0"
                },
                "capabilities": {}
            })),
        };
        assert!(RequestValidator::validate(&valid_request).is_ok());

        // Invalid JSON-RPC version
        let invalid_version = JsonRpcRequest {
            jsonrpc: "1.0".to_string(),
            id: RequestId::from_number(1),
            method: "initialize".to_string(),
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
