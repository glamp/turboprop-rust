//! Model Context Protocol (MCP) implementation for TurboProp
//!
//! Provides JSON-RPC 2.0 server implementation for exposing semantic search
//! capabilities to coding agents via the MCP protocol.
//!
//! The MCP server follows the official specification and provides:
//! - Real-time semantic search via the 'search' tool
//! - Automatic repository indexing and file watching
//! - Integration with existing TurboProp infrastructure

pub mod error;
pub mod protocol;
pub mod server;
pub mod tools;
pub mod transport;

pub use error::ErrorHandler;
pub use server::{McpServer, McpServerBuilder, McpServerConfig, McpServerTrait};
pub use transport::{
    RequestValidator, StdioTransport, StdioTransportConfig, TokenBucketRateLimiter,
};
