//! Model Context Protocol (MCP) implementation for TurboProp
//!
//! Provides JSON-RPC 2.0 server implementation for exposing semantic search
//! capabilities to coding agents via the MCP protocol.
//!
//! The MCP server follows the official specification and provides:
//! - Real-time semantic search via the 'search' tool
//! - Automatic repository indexing and file watching
//! - Integration with existing TurboProp infrastructure
//!
//! ## Architecture
//!
//! The MCP implementation is structured as follows:
//!
//! - `protocol`: JSON-RPC 2.0 message types and protocol constants
//! - `server`: Core MCP server implementation with request handling
//! - `tools`: Tool registry and semantic search tool implementation
//! - `transport`: Transport layer abstractions (stdio, HTTP, WebSocket)
//! - `error`: MCP-specific error types that integrate with TurboProp errors
//!
//! ## Usage
//!
//! ```rust,no_run
//! use turboprop::mcp::{McpServer, McpServerConfig, McpServerTrait};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = McpServerConfig::default();
//!     let mut server = McpServer::with_config(config);
//!     
//!     server.start().await?;
//!     // Server is now running and ready to handle MCP requests
//!     
//!     Ok(())
//! }
//! ```

pub mod error;
pub mod protocol;
pub mod server;
pub mod tools;
pub mod transport;

// Re-export commonly used types
pub use error::{McpError, McpResult};
pub use protocol::RequestId;
pub use server::{McpServer, McpServerConfig, McpServerTrait};
pub use tools::{ToolCallRequest, ToolCallResponse, ToolDefinition, ToolExecutor, Tools};
pub use transport::StdioTransport;
