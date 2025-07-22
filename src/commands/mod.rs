//! Command implementations for the TurboProp CLI.
//!
//! This module contains the enhanced command implementations that provide
//! rich user feedback, progress tracking, and comprehensive error handling.

pub mod index;

pub use index::{execute_index_command, execute_index_command_cli};
