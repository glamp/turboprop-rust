//! Command implementations for the TurboProp CLI.
//!
//! This module contains the enhanced command implementations that provide
//! rich user feedback, progress tracking, and comprehensive error handling.

pub mod benchmark;
pub mod index;
pub mod mcp;
pub mod model;
pub mod search;

pub use benchmark::{run_benchmark, BenchmarkArgs, BenchmarkResult};
pub use index::{execute_index_command, execute_index_command_cli};
pub use mcp::execute_mcp_command;
pub use model::handle_model_command;
pub use search::{execute_search_command, execute_search_command_cli, SearchCliArgs};
