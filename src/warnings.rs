//! Resource usage warnings and recommendations system.
//!
//! This module provides warnings and guidance to users about model resource requirements,
//! helping them make informed decisions about model selection and system configuration.

use crate::models::ModelInfo;
use crate::types::ModelBackend;
use tracing::warn;

/// Resource warning system for model requirements and recommendations
pub struct ResourceWarnings;

impl ResourceWarnings {
    /// Check model resource requirements and provide warnings if needed
    pub fn check_model_requirements(model_info: &ModelInfo) {
        // Check available memory
        if let Ok(available_memory) = Self::get_available_memory() {
            let required_memory = Self::estimate_model_memory_usage(model_info);

            if required_memory > available_memory {
                warn!(
                    "Model '{}' may require {}MB of memory, but only {}MB available. \
                     Consider using a smaller model or increasing system memory.",
                    model_info.name.as_str(),
                    required_memory / 1_048_576,
                    available_memory / 1_048_576
                );
            }
        }

        // Check model download size
        if model_info.size_bytes > 1_000_000_000 {
            // > 1GB
            warn!(
                "Model '{}' is large ({:.1}GB). Initial download may take significant time.",
                model_info.name.as_str(),
                model_info.size_bytes as f32 / 1_073_741_824.0
            );
        }

        // Model-specific warnings
        match model_info.name.as_str() {
            name if name.contains("gguf") => {
                warn!("GGUF models may have slower inference on CPU. Consider GPU acceleration for better performance.");
            }
            name if name.contains("Qwen3") => {
                warn!("Qwen3 models support instruction-based embeddings. Use --instruction flag for optimal results.");
            }
            _ => {}
        }
    }

    /// Get available system memory
    pub fn get_available_memory() -> Result<u64, Box<dyn std::error::Error>> {
        #[cfg(target_os = "linux")]
        {
            let meminfo = std::fs::read_to_string("/proc/meminfo")?;
            for line in meminfo.lines() {
                if line.starts_with("MemAvailable:") {
                    let kb: u64 = line
                        .split_whitespace()
                        .nth(1)
                        .ok_or("Invalid format")?
                        .parse()?;
                    return Ok(kb * 1024);
                }
            }
        }

        #[cfg(any(target_os = "macos", target_os = "windows"))]
        {
            // Use system info crate for cross-platform memory info
            match sys_info::mem_info() {
                Ok(mem_info) => {
                    Ok(mem_info.avail * 1024) // sys_info returns KB, we want bytes
                }
                Err(e) => Err(e.into()),
            }
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            // Default/fallback - assume 8GB
            Ok(8 * 1_073_741_824)
        }
    }

    /// Estimate memory usage for a given model
    pub fn estimate_model_memory_usage(model_info: &ModelInfo) -> u64 {
        // Rough estimation based on model size and type
        // Use saturating multiplication to avoid overflow
        let multiplier = match model_info.backend {
            ModelBackend::FastEmbed => 2,
            ModelBackend::Candle => 3,
            ModelBackend::Custom => 4,
        };

        model_info.size_bytes.saturating_mul(multiplier)
    }

    /// Check if a model is considered large
    pub fn is_large_model(model_info: &ModelInfo) -> bool {
        model_info.size_bytes > 1_000_000_000 // > 1GB
    }

    /// Get a human-readable description of memory requirements
    pub fn describe_memory_requirements(model_info: &ModelInfo) -> String {
        let estimated_bytes = Self::estimate_model_memory_usage(model_info);
        let estimated_mb = estimated_bytes as f32 / 1_048_576.0;
        let estimated_gb = estimated_bytes as f32 / 1_073_741_824.0;

        if estimated_gb >= 1.0 {
            format!("{:.1}GB", estimated_gb)
        } else {
            format!("{:.0}MB", estimated_mb)
        }
    }

    /// Check if the system likely has sufficient resources for a model
    pub fn has_sufficient_resources(model_info: &ModelInfo) -> bool {
        if let Ok(available_memory) = Self::get_available_memory() {
            let required_memory = Self::estimate_model_memory_usage(model_info);
            available_memory >= required_memory
        } else {
            // If we can't determine available memory, assume it's sufficient
            true
        }
    }

    /// Get recommendations for model usage
    pub fn get_recommendations(model_info: &ModelInfo) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Memory recommendations
        if !Self::has_sufficient_resources(model_info) {
            recommendations
                .push("Consider using a smaller model or increasing system memory".to_string());
        }

        // Backend-specific recommendations
        match model_info.backend {
            ModelBackend::Candle => {
                recommendations.push(
                    "Consider GPU acceleration for better GGUF model performance".to_string(),
                );
            }
            ModelBackend::Custom => {
                if model_info.name.as_str().contains("Qwen3") {
                    recommendations.push(
                        "Use instruction-based embeddings with --instruction flag for optimal results".to_string()
                    );
                }
            }
            _ => {}
        }

        // Size-based recommendations
        if Self::is_large_model(model_info) {
            recommendations.push(
                "Large model download may take significant time on slower connections".to_string(),
            );
        }

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModelName, ModelType};

    fn create_test_model(name: &str, size_bytes: u64, backend: ModelBackend) -> ModelInfo {
        ModelInfo {
            name: ModelName::from(name),
            backend,
            model_type: ModelType::SentenceTransformer,
            dimensions: 384,
            size_bytes,
            description: "Test model".to_string(),
            download_url: Some("https://example.com".to_string()),
            local_path: None,
        }
    }

    #[test]
    fn test_estimate_memory_usage() {
        let fastembed_model = create_test_model("test", 100_000_000, ModelBackend::FastEmbed);
        let gguf_model = create_test_model("test", 100_000_000, ModelBackend::Candle);
        let hf_model = create_test_model("test", 100_000_000, ModelBackend::Custom);

        assert_eq!(
            ResourceWarnings::estimate_model_memory_usage(&fastembed_model),
            200_000_000
        );
        assert_eq!(
            ResourceWarnings::estimate_model_memory_usage(&gguf_model),
            300_000_000
        );
        assert_eq!(
            ResourceWarnings::estimate_model_memory_usage(&hf_model),
            400_000_000
        );
    }

    #[test]
    fn test_is_large_model() {
        let small_model = create_test_model("small", 500_000_000, ModelBackend::FastEmbed); // 500MB
        let large_model = create_test_model("large", 2_000_000_000, ModelBackend::FastEmbed); // 2GB

        assert!(!ResourceWarnings::is_large_model(&small_model));
        assert!(ResourceWarnings::is_large_model(&large_model));
    }

    #[test]
    fn test_describe_memory_requirements() {
        let small_model = create_test_model("small", 50_000_000, ModelBackend::FastEmbed); // 50MB -> 100MB required
        let large_model = create_test_model("large", 600_000_000, ModelBackend::FastEmbed); // 600MB -> 1200MB = 1.17GB required

        let small_desc = ResourceWarnings::describe_memory_requirements(&small_model);
        let large_desc = ResourceWarnings::describe_memory_requirements(&large_model);

        assert!(small_desc.contains("MB"));
        assert!(large_desc.contains("GB"));
    }

    #[test]
    fn test_get_recommendations() {
        let gguf_model = create_test_model("model.gguf", 100_000_000, ModelBackend::Candle);
        let qwen_model = create_test_model("Qwen3-test", 100_000_000, ModelBackend::Custom);
        let large_model = create_test_model("large", 2_000_000_000, ModelBackend::FastEmbed);

        let gguf_recs = ResourceWarnings::get_recommendations(&gguf_model);
        let qwen_recs = ResourceWarnings::get_recommendations(&qwen_model);
        let large_recs = ResourceWarnings::get_recommendations(&large_model);

        assert!(!gguf_recs.is_empty());
        assert!(!qwen_recs.is_empty());
        assert!(!large_recs.is_empty());

        // Check for specific recommendation types
        assert!(gguf_recs.iter().any(|r| r.contains("GPU acceleration")));
        assert!(qwen_recs.iter().any(|r| r.contains("instruction")));
        assert!(large_recs.iter().any(|r| r.contains("download")));
    }

    #[test]
    fn test_check_model_requirements() {
        let model = create_test_model("test-model", 100_000_000, ModelBackend::FastEmbed);

        // Should not panic
        ResourceWarnings::check_model_requirements(&model);
    }

    #[test]
    fn test_get_available_memory() {
        // Test that we can get memory info or handle errors gracefully
        match ResourceWarnings::get_available_memory() {
            Ok(memory) => {
                assert!(memory > 0);
            }
            Err(_) => {
                // It's OK if we can't get memory info on some systems
            }
        }
    }
}
