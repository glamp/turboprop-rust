//! Command implementations for the TurboProp CLI.
//!
//! This module contains the enhanced command implementations that provide
//! rich user feedback, progress tracking, and comprehensive error handling.

pub mod index;
pub mod model;
pub mod search;

pub use index::{execute_index_command, execute_index_command_cli};
pub use model::handle_model_command;
pub use search::{execute_search_command, execute_search_command_cli, SearchCliArgs};
