//! Backend implementations for different embedding model types.
//!
//! This module provides different backend implementations for loading and running
//! embedding models. Each backend is optimized for specific model formats.

pub mod gguf;

pub use gguf::{GGUFBackend, GGUFEmbeddingModel};