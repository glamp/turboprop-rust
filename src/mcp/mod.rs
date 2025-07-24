//! Model Context Protocol (MCP) implementation for TurboProp
//! 
//! Provides JSON-RPC 2.0 server implementation for exposing semantic search
//! capabilities to coding agents via the MCP protocol.
//!
//! The MCP server follows the official specification and provides:
//! - Real-time semantic search via the 'search' tool
//! - Automatic repository indexing and file watching
//! - Integration with existing TurboProp infrastructure

pub mod protocol;
pub mod server;
pub mod transport;
pub mod tools;

pub use server::McpServer;