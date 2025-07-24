//! MCP (Model Context Protocol) server domain types.
//!
//! This module provides strongly-typed wrappers for MCP server configuration
//! and domain concepts to prevent parameter mix-ups and provide better type safety.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Network port number with validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Port(u16);

impl Port {
    /// Create a new Port, ensuring it's not zero (which is invalid for server binding)
    pub fn new(port: u16) -> Result<Self, InvalidPortError> {
        if port == 0 {
            return Err(InvalidPortError::Zero);
        }
        Ok(Port(port))
    }

    /// Get the underlying port value
    pub fn value(self) -> u16 {
        self.0
    }

    /// Create a Port for dynamic allocation (port 0 is allowed for this use case)
    pub fn dynamic() -> Self {
        Port(0)
    }
}

impl fmt::Display for Port {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Port> for u16 {
    fn from(port: Port) -> u16 {
        port.0
    }
}

/// Maximum number of concurrent connections
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConnectionLimit(usize);

impl ConnectionLimit {
    /// Create a new ConnectionLimit with validation
    pub fn new(limit: usize) -> Result<Self, InvalidConnectionLimitError> {
        if limit == 0 {
            return Err(InvalidConnectionLimitError::Zero);
        }
        Ok(ConnectionLimit(limit))
    }

    /// Get the underlying limit value
    pub fn value(self) -> usize {
        self.0
    }
}

impl fmt::Display for ConnectionLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<ConnectionLimit> for usize {
    fn from(limit: ConnectionLimit) -> usize {
        limit.0
    }
}

/// Request timeout in seconds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimeoutSeconds(u64);

impl TimeoutSeconds {
    /// Create a new TimeoutSeconds
    pub fn new(seconds: u64) -> Self {
        TimeoutSeconds(seconds)
    }

    /// Get the underlying seconds value
    pub fn value(self) -> u64 {
        self.0
    }

    /// Convert to std::time::Duration
    pub fn as_duration(self) -> std::time::Duration {
        std::time::Duration::from_secs(self.0)
    }
}

impl fmt::Display for TimeoutSeconds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}s", self.0)
    }
}

impl From<TimeoutSeconds> for u64 {
    fn from(timeout: TimeoutSeconds) -> u64 {
        timeout.0
    }
}

/// Maximum message size in bytes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageSize(usize);

impl MessageSize {
    /// Create a new MessageSize with validation
    pub fn new(size: usize) -> Result<Self, InvalidMessageSizeError> {
        if size == 0 {
            return Err(InvalidMessageSizeError::Zero);
        }
        Ok(MessageSize(size))
    }

    /// Get the underlying size value
    pub fn value(self) -> usize {
        self.0
    }
}

impl fmt::Display for MessageSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} bytes", self.0)
    }
}

impl From<MessageSize> for usize {
    fn from(size: MessageSize) -> usize {
        size.0
    }
}

/// Content preview length in characters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PreviewLength(usize);

impl PreviewLength {
    /// Create a new PreviewLength
    pub fn new(length: usize) -> Self {
        PreviewLength(length)
    }

    /// Get the underlying length value
    pub fn value(self) -> usize {
        self.0
    }
}

impl fmt::Display for PreviewLength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} chars", self.0)
    }
}

impl From<PreviewLength> for usize {
    fn from(length: PreviewLength) -> usize {
        length.0
    }
}

/// JSON-RPC error code
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ErrorCode(i32);

impl ErrorCode {
    /// Create a new ErrorCode
    pub fn new(code: i32) -> Self {
        ErrorCode(code)
    }

    /// Get the underlying code value
    pub fn value(self) -> i32 {
        self.0
    }

    // Standard JSON-RPC error codes
    pub const PARSE_ERROR: ErrorCode = ErrorCode(-32700);
    pub const INVALID_REQUEST: ErrorCode = ErrorCode(-32600);
    pub const METHOD_NOT_FOUND: ErrorCode = ErrorCode(-32601);
    pub const INVALID_PARAMS: ErrorCode = ErrorCode(-32602);
    pub const INTERNAL_ERROR: ErrorCode = ErrorCode(-32603);
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<ErrorCode> for i32 {
    fn from(code: ErrorCode) -> i32 {
        code.0
    }
}

// Error types for validation failures

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidPortError {
    Zero,
}

impl fmt::Display for InvalidPortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InvalidPortError::Zero => write!(f, "Port cannot be zero for server binding"),
        }
    }
}

impl std::error::Error for InvalidPortError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidConnectionLimitError {
    Zero,
}

impl fmt::Display for InvalidConnectionLimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InvalidConnectionLimitError::Zero => write!(f, "Connection limit cannot be zero"),
        }
    }
}

impl std::error::Error for InvalidConnectionLimitError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidMessageSizeError {
    Zero,
}

impl fmt::Display for InvalidMessageSizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InvalidMessageSizeError::Zero => write!(f, "Message size cannot be zero"),
        }
    }
}

impl std::error::Error for InvalidMessageSizeError {}

// Default implementations for configuration convenience

impl Default for Port {
    fn default() -> Self {
        Port::dynamic() // Port 0 for dynamic allocation
    }
}

impl Default for ConnectionLimit {
    fn default() -> Self {
        ConnectionLimit(10) // Reasonable default for most use cases
    }
}

impl Default for TimeoutSeconds {
    fn default() -> Self {
        TimeoutSeconds(30) // 30 second default timeout
    }
}

impl Default for MessageSize {
    fn default() -> Self {
        MessageSize(1024 * 1024) // 1MB default message size
    }
}

impl Default for PreviewLength {
    fn default() -> Self {
        PreviewLength(200) // 200 character default preview
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_port_validation() {
        assert!(Port::new(0).is_err());
        assert!(Port::new(80).is_ok());
        assert!(Port::new(65535).is_ok());
        
        let dynamic_port = Port::dynamic();
        assert_eq!(dynamic_port.value(), 0);
    }

    #[test]
    fn test_connection_limit_validation() {
        assert!(ConnectionLimit::new(0).is_err());
        assert!(ConnectionLimit::new(1).is_ok());
        assert!(ConnectionLimit::new(1000).is_ok());
    }

    #[test]
    fn test_message_size_validation() {
        assert!(MessageSize::new(0).is_err());
        assert!(MessageSize::new(1).is_ok());
        assert!(MessageSize::new(1024 * 1024).is_ok());
    }

    #[test]
    fn test_timeout_duration_conversion() {
        let timeout = TimeoutSeconds::new(30);
        let duration = timeout.as_duration();
        assert_eq!(duration.as_secs(), 30);
    }

    #[test]
    fn test_error_code_constants() {
        assert_eq!(ErrorCode::PARSE_ERROR.value(), -32700);
        assert_eq!(ErrorCode::INVALID_REQUEST.value(), -32600);
        assert_eq!(ErrorCode::METHOD_NOT_FOUND.value(), -32601);
        assert_eq!(ErrorCode::INVALID_PARAMS.value(), -32602);
        assert_eq!(ErrorCode::INTERNAL_ERROR.value(), -32603);
    }

    #[test]
    fn test_display_formatting() {
        assert_eq!(Port::new(8080).unwrap().to_string(), "8080");
        assert_eq!(ConnectionLimit::new(50).unwrap().to_string(), "50");
        assert_eq!(TimeoutSeconds::new(60).to_string(), "60s");
        assert_eq!(MessageSize::new(1024).unwrap().to_string(), "1024 bytes");
        assert_eq!(PreviewLength::new(100).to_string(), "100 chars");
        assert_eq!(ErrorCode::new(-32700).to_string(), "-32700");
    }

    #[test]
    fn test_defaults() {
        let port = Port::default();
        let conn_limit = ConnectionLimit::default();
        let timeout = TimeoutSeconds::default();
        let msg_size = MessageSize::default();
        let preview_len = PreviewLength::default();

        assert_eq!(port.value(), 0);
        assert_eq!(conn_limit.value(), 10);
        assert_eq!(timeout.value(), 30);
        assert_eq!(msg_size.value(), 1024 * 1024);
        assert_eq!(preview_len.value(), 200);
    }
}