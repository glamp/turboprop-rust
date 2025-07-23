//! Similarity calculation utilities.
//!
//! This module provides functions for calculating similarity between vectors,
//! primarily used for semantic search operations.

/// Calculate cosine similarity between two vectors with robust error handling
///
/// Returns a value between -1.0 and 1.0, where:
/// - 1.0 indicates identical vectors (perfectly similar)
/// - 0.0 indicates orthogonal vectors (no similarity)
/// - -1.0 indicates opposite vectors
/// - 0.0 is returned for any error conditions (mismatched lengths, NaN/infinity values)
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    // Use checked arithmetic to prevent overflow
    let mut dot_product = 0.0_f32;
    let mut sum_a_squared = 0.0_f32;
    let mut sum_b_squared = 0.0_f32;

    for (x, y) in a.iter().zip(b.iter()) {
        // Check for infinity or NaN before operations
        if !x.is_finite() || !y.is_finite() {
            tracing::warn!("Non-finite values detected in cosine similarity calculation");
            return 0.0;
        }

        // Safely compute dot product with overflow checking
        let product = x * y;
        if !product.is_finite() {
            tracing::warn!("Overflow detected in dot product calculation");
            return 0.0;
        }
        dot_product += product;

        // Safely compute squared magnitudes with overflow checking
        let x_squared = x * x;
        let y_squared = y * y;
        if !x_squared.is_finite() || !y_squared.is_finite() {
            tracing::warn!("Overflow detected in magnitude calculation");
            return 0.0;
        }
        sum_a_squared += x_squared;
        sum_b_squared += y_squared;
    }

    // Check for valid intermediate results
    if !dot_product.is_finite() || !sum_a_squared.is_finite() || !sum_b_squared.is_finite() {
        return 0.0;
    }

    let magnitude_a = sum_a_squared.sqrt();
    let magnitude_b = sum_b_squared.sqrt();

    if magnitude_a == 0.0
        || magnitude_b == 0.0
        || !magnitude_a.is_finite()
        || !magnitude_b.is_finite()
    {
        return 0.0;
    }

    let result = dot_product / (magnitude_a * magnitude_b);

    // Final check for valid result
    if !result.is_finite() {
        tracing::warn!("Invalid result in cosine similarity calculation");
        return 0.0;
    }

    result
}
