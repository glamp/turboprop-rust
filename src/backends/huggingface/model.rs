//! Qwen3 embedding model implementation.
//!
//! This module provides the Qwen3EmbeddingModel struct that handles inference
//! for Qwen3-based embedding models using the Candle framework.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM};
use tokenizers::Tokenizer;

use crate::models::EmbeddingModel as EmbeddingModelTrait;

/// Qwen3 embedding model for generating text embeddings
pub struct Qwen3EmbeddingModel {
    pub model: tokio::sync::RwLock<ModelForCausalLM>,
    pub tokenizer: Tokenizer,
    pub config: Qwen2Config,
    pub device: Device,
    pub model_name: String,
}

impl std::fmt::Debug for Qwen3EmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3EmbeddingModel")
            .field("model_name", &self.model_name)
            .field("hidden_size", &self.config.hidden_size)
            .field("device", &self.device)
            .finish()
    }
}

impl Qwen3EmbeddingModel {
    /// Generate embeddings for a batch of texts
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_with_instruction(texts, None)
    }

    /// Generate embeddings with optional instruction (Qwen3 feature)
    pub fn embed_with_instruction(
        &self,
        texts: &[String],
        instruction: Option<&str>,
    ) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            // Apply instruction if provided (Qwen3 supports instruction-based embeddings)
            let processed_text = if let Some(instr) = instruction {
                format!("Instruct: {}\nQuery: {}", instr, text)
            } else {
                text.clone()
            };

            // Tokenize the text
            let encoding = self
                .tokenizer
                .encode(processed_text.as_str(), true)
                .map_err(|e| {
                    anyhow::anyhow!(
                        "[HUGGINGFACE] [TOKENIZE] failed: Cannot tokenize input text. Error: {}",
                        e
                    )
                })?;

            let token_ids: Vec<u32> = encoding.get_ids().to_vec();
            let attention_mask: Vec<u32> = vec![1u32; token_ids.len()];

            // Convert to tensors
            let input_ids = Tensor::new(&token_ids[..], &self.device)
                .context("[HUGGINGFACE] [TENSOR] failed: Cannot create input_ids tensor from tokenized text")?
                .unsqueeze(0)?; // Add batch dimension

            let attention_mask_tensor = Tensor::new(&attention_mask[..], &self.device)
                .context("[HUGGINGFACE] [TENSOR] failed: Cannot create attention_mask tensor for model input")?
                .unsqueeze(0)?; // Add batch dimension

            // Run model inference - get hidden states from the language model
            // Use blocking_write() to avoid async in sync method while still using tokio::sync::RwLock
            let hidden_states = {
                let mut model_guard = self.model.blocking_write();
                model_guard
                    .forward(&input_ids, 0)
                    .context("[HUGGINGFACE] [INFERENCE] failed: Model forward pass execution failed. Check input tensor dimensions and model state.")?
            };

            // Extract embedding using mean pooling of last hidden states
            let embedding = self.mean_pooling(&hidden_states, &attention_mask_tensor)?;
            let normalized = self.normalize_embedding(embedding)?;

            embeddings.push(normalized);
        }

        Ok(embeddings)
    }

    /// Apply mean pooling to get sentence embedding from token embeddings
    fn mean_pooling(&self, last_hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // last_hidden_states shape: [batch_size, sequence_length, hidden_size]
        // attention_mask shape: [batch_size, sequence_length]

        // Expand attention mask to match hidden states dimensions
        let attention_mask = attention_mask.unsqueeze(2)?; // [batch_size, sequence_length, 1]
        let expanded_mask = attention_mask.broadcast_as(last_hidden_states.shape())?;

        // Apply mask to hidden states
        let masked_hidden_states = (last_hidden_states * &expanded_mask)?;

        // Sum along sequence dimension
        let sum_hidden_states = masked_hidden_states.sum(1)?; // [batch_size, hidden_size]

        // Sum attention mask to get sequence lengths
        let sum_mask = attention_mask.sum(1)?; // [batch_size, 1]

        // Divide by sequence length to get mean
        let mean_pooled = sum_hidden_states.broadcast_div(&sum_mask)?;

        Ok(mean_pooled)
    }

    /// L2 normalize the embedding vector
    fn normalize_embedding(&self, embedding: Tensor) -> Result<Vec<f32>> {
        // Calculate L2 norm
        let norm = embedding.sqr()?.sum_keepdim(1)?.sqrt()?;

        // Normalize
        let normalized = embedding.broadcast_div(&norm)?;

        // Convert to Vec<f32>
        let normalized_vec = normalized.squeeze(0)?.to_vec1::<f32>()?;

        Ok(normalized_vec)
    }
}

impl EmbeddingModelTrait for Qwen3EmbeddingModel {
    /// Generate embeddings for a batch of texts
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed(texts)
    }

    /// Get the embedding dimensions produced by this model
    fn dimensions(&self) -> usize {
        self.config.hidden_size
    }

    /// Get the maximum sequence length supported by this model
    fn max_sequence_length(&self) -> usize {
        self.config.max_position_embeddings
    }
}
