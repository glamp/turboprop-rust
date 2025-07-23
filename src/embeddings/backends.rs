//! Backend enumeration and delegation for different embedding models.
//!
//! This module provides the EmbeddingBackendType enum that wraps different
//! embedding backend implementations and provides a unified interface.

use fastembed::TextEmbedding;

use crate::backends::{GGUFEmbeddingModel, Qwen3EmbeddingModel};

/// Enum wrapping different embedding backend implementations
pub enum EmbeddingBackendType {
    /// FastEmbed backend for sentence-transformer models
    FastEmbed(TextEmbedding),
    /// GGUF backend for quantized models using candle framework
    GGUF(GGUFEmbeddingModel),
    /// HuggingFace backend for models like Qwen3 not supported by FastEmbed
    HuggingFace(Qwen3EmbeddingModel),
}

impl std::fmt::Debug for EmbeddingBackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingBackendType::FastEmbed(_) => write!(f, "FastEmbed"),
            EmbeddingBackendType::GGUF(model) => write!(f, "GGUF({:?})", model),
            EmbeddingBackendType::HuggingFace(model) => write!(f, "HuggingFace({:?})", model),
        }
    }
}