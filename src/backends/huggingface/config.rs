//! Configuration parsing utilities for HuggingFace models.
//!
//! This module provides functions to parse HuggingFace model configurations
//! into Candle-compatible configuration structures.

use anyhow::Result;
use candle_nn::Activation;
use candle_transformers::models::qwen2::Config as Qwen2Config;
use serde_json::Value;

use crate::constants;

/// Parse Qwen2 configuration from HuggingFace config.json
pub fn parse_qwen2_config(config_json: &Value) -> Result<Qwen2Config> {
    let vocab_size = config_json["vocab_size"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("[HUGGINGFACE] [CONFIG] failed: Missing required vocab_size field in model configuration"))?
        as usize;

    let hidden_size = config_json["hidden_size"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("[HUGGINGFACE] [CONFIG] failed: Missing required hidden_size field in model configuration"))?
        as usize;

    let intermediate_size = config_json["intermediate_size"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("[HUGGINGFACE] [CONFIG] failed: Missing required intermediate_size field in model configuration"))?
        as usize;

    let num_hidden_layers = config_json["num_hidden_layers"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("[HUGGINGFACE] [CONFIG] failed: Missing required num_hidden_layers field in model configuration"))?
        as usize;

    let num_attention_heads = config_json["num_attention_heads"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("[HUGGINGFACE] [CONFIG] failed: Missing required num_attention_heads field in model configuration"))?
        as usize;

    let num_key_value_heads = config_json["num_key_value_heads"]
        .as_u64()
        .unwrap_or(num_attention_heads as u64) as usize;

    let max_position_embeddings = config_json["max_position_embeddings"]
        .as_u64()
        .unwrap_or(constants::model_config::DEFAULT_MAX_POSITION_EMBEDDINGS)
        as usize;

    let sliding_window = config_json["sliding_window"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or(max_position_embeddings);

    let rope_theta = config_json["rope_theta"]
        .as_f64()
        .unwrap_or(constants::model_config::DEFAULT_ROPE_THETA);

    let hidden_act = match config_json["hidden_act"].as_str().unwrap_or("silu") {
        "silu" => Activation::Silu,
        "relu" => Activation::Relu,
        "gelu" => Activation::NewGelu,
        _ => Activation::Silu, // default to silu
    };

    Ok(Qwen2Config {
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        max_position_embeddings,
        sliding_window,
        rope_theta,
        rms_norm_eps: config_json["rms_norm_eps"]
            .as_f64()
            .unwrap_or(constants::model_config::DEFAULT_RMS_NORM_EPS),
        hidden_act,
        max_window_layers: config_json["max_window_layers"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or(num_hidden_layers),
        tie_word_embeddings: config_json["tie_word_embeddings"]
            .as_bool()
            .unwrap_or(false),
        use_sliding_window: config_json["use_sliding_window"].as_bool().unwrap_or(false),
    })
}