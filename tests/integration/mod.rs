//! Integration tests for TurboProp
//!
//! This module contains slow integration tests that either:
//! - Spawn external `tp` binary processes (binary_tests)
//! - Call library functions that perform expensive operations like ML embedding (library_tests, embedding_tests)
//!
//! Run these tests with: `cargo test --test integration`
//!
//! These tests are separated from the fast unit tests to provide quick feedback
//! during development while still maintaining thorough end-to-end testing.

mod binary_tests;
mod embedding_tests;
mod library_tests;
