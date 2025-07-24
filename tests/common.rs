//! Common test utilities and helpers.
//!
//! This module provides shared functionality for test setup, test data creation,
//! and common test operations to reduce duplication across test files.

use std::path::Path;

/// Get the path to the poker test fixture
#[allow(dead_code)]
pub fn get_poker_fixture_path() -> &'static Path {
    Path::new("tests/fixtures/poker")
}
