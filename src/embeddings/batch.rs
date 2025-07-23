//! Batch processing utilities for embedding generation.
//!
//! This module provides the BatchProcessor trait and helper functions for efficiently
//! processing multiple texts in batches to avoid memory issues with large inputs.

use anyhow::{Context, Result};
use fastembed::TextEmbedding;
use tracing::debug;

use crate::backends::{GGUFEmbeddingModel, Qwen3EmbeddingModel};
use crate::models::EmbeddingModel;

/// Trait for processing embeddings in batches with common error handling and logging
pub trait BatchProcessor<T> {
    /// Process a batch of texts and return embeddings
    fn process_batch(&mut self, texts: &[T]) -> Result<Vec<Vec<f32>>>;

    /// Get the backend name for logging purposes
    fn backend_name(&self) -> &'static str;
}

/// Helper function to process texts in batches using any BatchProcessor
pub fn process_texts_in_batches<T, P: BatchProcessor<T>>(
    processor: &mut P,
    texts: &[T],
    batch_size: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut all_embeddings = Vec::with_capacity(texts.len());
    let backend_name = processor.backend_name();

    // Process in batches to avoid memory issues with large inputs
    for chunk in texts.chunks(batch_size) {
        debug!("Processing {} batch of {} texts", backend_name, chunk.len());

        let batch_embeddings = processor.process_batch(chunk).with_context(|| {
            format!(
                "Failed to generate {} embeddings for batch of {} texts",
                backend_name,
                chunk.len()
            )
        })?;

        all_embeddings.extend(batch_embeddings);
    }

    debug!(
        "Generated {} {} embeddings total",
        all_embeddings.len(),
        backend_name
    );

    Ok(all_embeddings)
}

/// BatchProcessor implementation for FastEmbed backend
impl BatchProcessor<String> for TextEmbedding {
    fn process_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.embed(text_refs, None)
    }

    fn backend_name(&self) -> &'static str {
        "FastEmbed"
    }
}

/// BatchProcessor implementation for GGUF backend
impl BatchProcessor<String> for GGUFEmbeddingModel {
    fn process_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed(texts)
    }

    fn backend_name(&self) -> &'static str {
        "GGUF"
    }
}

/// BatchProcessor implementation for HuggingFace backend
impl BatchProcessor<String> for Qwen3EmbeddingModel {
    fn process_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed(texts)
    }

    fn backend_name(&self) -> &'static str {
        "HuggingFace"
    }
}
