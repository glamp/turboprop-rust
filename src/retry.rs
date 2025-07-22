//! Retry logic with exponential backoff for handling transient failures.
//!
//! This module provides configurable retry functionality with exponential backoff,
//! jitter, and the ability to determine which errors should be retried.

use rand::Rng;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, warn};

/// Configuration for retry behavior.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    max_attempts: u32,
    initial_delay: Duration,
    max_delay: Duration,
    backoff_multiplier: f64,
    jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: false,
        }
    }
}

impl RetryConfig {
    /// Create a new retry configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of retry attempts.
    pub fn with_max_attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = max_attempts;
        self
    }

    /// Set the initial delay before the first retry.
    pub fn with_initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = delay;
        self
    }

    /// Set the maximum delay between retries.
    pub fn with_max_delay(mut self, max_delay: Duration) -> Self {
        self.max_delay = max_delay;
        self
    }

    /// Set the backoff multiplier for exponential backoff.
    pub fn with_backoff_multiplier(mut self, multiplier: f64) -> Self {
        self.backoff_multiplier = multiplier;
        self
    }

    /// Enable or disable jitter to add randomness to delays.
    pub fn with_jitter(mut self, jitter: bool) -> Self {
        self.jitter = jitter;
        self
    }

    /// Get the maximum number of attempts.
    pub fn max_attempts(&self) -> u32 {
        self.max_attempts
    }

    /// Get the initial delay.
    pub fn initial_delay(&self) -> Duration {
        self.initial_delay
    }

    /// Get the maximum delay.
    pub fn max_delay(&self) -> Duration {
        self.max_delay
    }

    /// Get the backoff multiplier.
    pub fn backoff_multiplier(&self) -> f64 {
        self.backoff_multiplier
    }

    /// Check if jitter is enabled.
    pub fn jitter_enabled(&self) -> bool {
        self.jitter
    }

    /// Execute a retryable operation with the configured retry policy.
    pub async fn execute<T, E, Op>(&self, operation: &mut Op) -> Result<T, E>
    where
        Op: RetryableOperation<T, Error = E>,
        E: std::fmt::Debug,
    {
        let backoff = ExponentialBackoff::new(self);
        let mut last_error = None;

        for attempt in 1..=self.max_attempts {
            debug!("Attempt {} of {}", attempt, self.max_attempts);

            match operation.execute() {
                Ok(result) => {
                    if attempt > 1 {
                        debug!("Operation succeeded after {} attempts", attempt);
                    }
                    return Ok(result);
                }
                Err(error) => {
                    debug!("Attempt {} failed: {:?}", attempt, error);

                    // Check if this error should be retried
                    if !operation.is_retryable(&error) {
                        debug!("Error is not retryable, failing immediately");
                        return Err(error);
                    }

                    last_error = Some(error);

                    // Don't sleep after the last attempt
                    if attempt < self.max_attempts {
                        let delay = backoff.delay_for_attempt(attempt);
                        debug!("Waiting {:?} before retry", delay);
                        sleep(delay).await;
                    }
                }
            }
        }

        warn!("Operation failed after {} attempts", self.max_attempts);
        Err(last_error.unwrap())
    }
}

/// Trait for operations that can be retried.
pub trait RetryableOperation<T> {
    type Error;

    /// Execute the operation.
    fn execute(&mut self) -> Result<T, Self::Error>;

    /// Determine if an error should trigger a retry.
    fn is_retryable(&self, error: &Self::Error) -> bool;
}

/// Exponential backoff calculator.
pub struct ExponentialBackoff {
    config: RetryConfig,
}

impl ExponentialBackoff {
    /// Create a new exponential backoff calculator.
    pub fn new(config: &RetryConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Calculate the delay for a given attempt number (1-based).
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::from_millis(0);
        }

        let base_delay_ms = self.config.initial_delay.as_millis() as f64;
        let multiplier = self.config.backoff_multiplier;

        // Calculate exponential delay: initial_delay * multiplier^(attempt-1)
        let delay_ms = base_delay_ms * multiplier.powi((attempt - 1) as i32);
        let delay = Duration::from_millis(delay_ms as u64);

        // Cap at max_delay
        let capped_delay = std::cmp::min(delay, self.config.max_delay);

        // Add jitter if enabled
        if self.config.jitter {
            self.add_jitter(capped_delay)
        } else {
            capped_delay
        }
    }

    /// Add jitter to a delay (±25% randomization).
    fn add_jitter(&self, delay: Duration) -> Duration {
        let delay_ms = delay.as_millis() as f64;
        let jitter_range = delay_ms * 0.25; // ±25%

        let mut rng = rand::thread_rng();
        let jitter: f64 = rng.gen_range(-jitter_range..=jitter_range);

        let final_delay_ms = (delay_ms + jitter).max(0.0) as u64;
        Duration::from_millis(final_delay_ms)
    }
}

/// Convenience functions for common retry patterns.
impl RetryConfig {
    /// Configuration optimized for network operations.
    pub fn network_operations() -> Self {
        Self::new()
            .with_max_attempts(3)
            .with_initial_delay(Duration::from_millis(500))
            .with_max_delay(Duration::from_secs(10))
            .with_backoff_multiplier(2.0)
            .with_jitter(true)
    }

    /// Configuration optimized for file operations.
    pub fn file_operations() -> Self {
        Self::new()
            .with_max_attempts(2)
            .with_initial_delay(Duration::from_millis(100))
            .with_max_delay(Duration::from_secs(5))
            .with_backoff_multiplier(2.0)
            .with_jitter(false)
    }

    /// Configuration optimized for model downloads.
    pub fn model_downloads() -> Self {
        Self::new()
            .with_max_attempts(5)
            .with_initial_delay(Duration::from_secs(1))
            .with_max_delay(Duration::from_secs(60))
            .with_backoff_multiplier(1.5)
            .with_jitter(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_calculation() {
        let config = RetryConfig::new()
            .with_initial_delay(Duration::from_millis(100))
            .with_backoff_multiplier(2.0);

        let backoff = ExponentialBackoff::new(&config);

        assert_eq!(backoff.delay_for_attempt(0), Duration::from_millis(0));
        assert_eq!(backoff.delay_for_attempt(1), Duration::from_millis(100));
        assert_eq!(backoff.delay_for_attempt(2), Duration::from_millis(200));
        assert_eq!(backoff.delay_for_attempt(3), Duration::from_millis(400));
    }

    #[test]
    fn test_max_delay_cap() {
        let config = RetryConfig::new()
            .with_initial_delay(Duration::from_millis(100))
            .with_max_delay(Duration::from_millis(250))
            .with_backoff_multiplier(2.0);

        let backoff = ExponentialBackoff::new(&config);

        assert_eq!(backoff.delay_for_attempt(1), Duration::from_millis(100));
        assert_eq!(backoff.delay_for_attempt(2), Duration::from_millis(200));
        assert_eq!(backoff.delay_for_attempt(3), Duration::from_millis(250)); // Capped
        assert_eq!(backoff.delay_for_attempt(4), Duration::from_millis(250)); // Still capped
    }

    #[test]
    fn test_preset_configurations() {
        let network_config = RetryConfig::network_operations();
        assert_eq!(network_config.max_attempts(), 3);
        assert!(network_config.jitter_enabled());

        let file_config = RetryConfig::file_operations();
        assert_eq!(file_config.max_attempts(), 2);
        assert!(!file_config.jitter_enabled());

        let model_config = RetryConfig::model_downloads();
        assert_eq!(model_config.max_attempts(), 5);
        assert!(model_config.jitter_enabled());
    }
}
