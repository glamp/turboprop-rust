use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tp::retry::{RetryConfig, RetryableOperation, ExponentialBackoff};
use tp::error::{TurboPropError, TurboPropResult};

struct FailingOperation {
    attempts: Arc<Mutex<u32>>,
    succeed_after: u32,
}

impl FailingOperation {
    fn new(succeed_after: u32) -> Self {
        Self {
            attempts: Arc::new(Mutex::new(0)),
            succeed_after,
        }
    }

    fn get_attempts(&self) -> u32 {
        *self.attempts.lock().unwrap()
    }
}

impl RetryableOperation<String> for FailingOperation {
    type Error = TurboPropError;

    fn execute(&mut self) -> Result<String, Self::Error> {
        let mut attempts = self.attempts.lock().unwrap();
        *attempts += 1;
        
        if *attempts <= self.succeed_after {
            Err(TurboPropError::network("Temporary failure", None))
        } else {
            Ok(format!("Success after {} attempts", *attempts))
        }
    }

    fn is_retryable(&self, error: &Self::Error) -> bool {
        matches!(error, TurboPropError::NetworkError { .. })
    }
}

#[tokio::test]
async fn test_retry_config_default() {
    let config = RetryConfig::default();
    assert_eq!(config.max_attempts(), 3);
    assert_eq!(config.initial_delay(), Duration::from_millis(100));
    assert_eq!(config.max_delay(), Duration::from_secs(30));
    assert_eq!(config.backoff_multiplier(), 2.0);
}

#[tokio::test]
async fn test_retry_config_builder() {
    let config = RetryConfig::new()
        .with_max_attempts(5)
        .with_initial_delay(Duration::from_millis(250))
        .with_max_delay(Duration::from_secs(60))
        .with_backoff_multiplier(1.5);

    assert_eq!(config.max_attempts(), 5);
    assert_eq!(config.initial_delay(), Duration::from_millis(250));
    assert_eq!(config.max_delay(), Duration::from_secs(60));
    assert_eq!(config.backoff_multiplier(), 1.5);
}

#[tokio::test]
async fn test_exponential_backoff_first_delay() {
    let config = RetryConfig::new()
        .with_initial_delay(Duration::from_millis(100))
        .with_backoff_multiplier(2.0);
    
    let backoff = ExponentialBackoff::new(&config);
    
    assert_eq!(backoff.delay_for_attempt(1), Duration::from_millis(100));
    assert_eq!(backoff.delay_for_attempt(2), Duration::from_millis(200));
    assert_eq!(backoff.delay_for_attempt(3), Duration::from_millis(400));
}

#[tokio::test]
async fn test_exponential_backoff_max_delay() {
    let config = RetryConfig::new()
        .with_initial_delay(Duration::from_millis(100))
        .with_max_delay(Duration::from_millis(250))
        .with_backoff_multiplier(2.0);
    
    let backoff = ExponentialBackoff::new(&config);
    
    // Should cap at max_delay
    assert_eq!(backoff.delay_for_attempt(1), Duration::from_millis(100));
    assert_eq!(backoff.delay_for_attempt(2), Duration::from_millis(200));
    assert_eq!(backoff.delay_for_attempt(3), Duration::from_millis(250)); // Capped
    assert_eq!(backoff.delay_for_attempt(4), Duration::from_millis(250)); // Still capped
}

#[tokio::test]
async fn test_successful_operation_no_retries() {
    let config = RetryConfig::default();
    let mut operation = FailingOperation::new(0); // Succeed immediately
    
    let result = config.execute(&mut operation).await;
    
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "Success after 1 attempts");
    assert_eq!(operation.get_attempts(), 1);
}

#[tokio::test]
async fn test_operation_succeeds_after_retries() {
    let config = RetryConfig::new().with_max_attempts(5);
    let mut operation = FailingOperation::new(2); // Fail twice, then succeed
    
    let start = Instant::now();
    let result = config.execute(&mut operation).await;
    let elapsed = start.elapsed();
    
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "Success after 3 attempts");
    assert_eq!(operation.get_attempts(), 3);
    
    // Should have waited some time for retries
    assert!(elapsed >= Duration::from_millis(300)); // 100ms + 200ms
}

#[tokio::test]
async fn test_operation_exceeds_max_attempts() {
    let config = RetryConfig::new().with_max_attempts(3);
    let mut operation = FailingOperation::new(5); // Will never succeed within 3 attempts
    
    let result = config.execute(&mut operation).await;
    
    assert!(result.is_err());
    assert_eq!(operation.get_attempts(), 3); // Should only try max_attempts times
    
    match result.unwrap_err() {
        TurboPropError::NetworkError { message, .. } => {
            assert!(message.contains("Temporary failure"));
        }
        _ => panic!("Expected NetworkError"),
    }
}

#[tokio::test]
async fn test_non_retryable_error_no_retry() {
    let config = RetryConfig::default();
    
    struct NonRetryableOperation;
    impl RetryableOperation<String> for NonRetryableOperation {
        type Error = TurboPropError;
        
        fn execute(&mut self) -> Result<String, Self::Error> {
            Err(TurboPropError::config_validation("field", "invalid", "valid"))
        }
        
        fn is_retryable(&self, error: &Self::Error) -> bool {
            // Configuration errors should not be retried
            !matches!(error, TurboPropError::ConfigurationValidationError { .. })
        }
    }
    
    let mut operation = NonRetryableOperation;
    let result = config.execute(&mut operation).await;
    
    assert!(result.is_err());
    match result.unwrap_err() {
        TurboPropError::ConfigurationValidationError { .. } => {
            // Expected - should fail immediately without retry
        }
        _ => panic!("Expected ConfigurationValidationError"),
    }
}

#[tokio::test]
async fn test_jitter_adds_randomness() {
    let config = RetryConfig::new()
        .with_initial_delay(Duration::from_millis(100))
        .with_jitter(true);
    
    let backoff = ExponentialBackoff::new(&config);
    
    // Get delays for same attempt multiple times
    let delay1 = backoff.delay_for_attempt(2);
    let delay2 = backoff.delay_for_attempt(2);
    let delay3 = backoff.delay_for_attempt(2);
    
    // With jitter, delays should vary (though they might occasionally be the same)
    let delays = vec![delay1, delay2, delay3];
    let base_delay = Duration::from_millis(200); // 100ms * 2
    
    // All delays should be within reasonable bounds (50% to 150% of base)
    for delay in delays {
        assert!(delay >= Duration::from_millis(100)); // At least 50% of base
        assert!(delay <= Duration::from_millis(300)); // At most 150% of base
    }
}

#[tokio::test] 
async fn test_retry_network_operations() {
    // Test helper for network-like operations
    let config = RetryConfig::new()
        .with_max_attempts(3)
        .with_initial_delay(Duration::from_millis(50));

    struct NetworkOperation {
        attempts: Arc<Mutex<u32>>,
    }

    impl RetryableOperation<String> for NetworkOperation {
        type Error = TurboPropError;

        fn execute(&mut self) -> Result<String, Self::Error> {
            let mut attempts = self.attempts.lock().unwrap();
            *attempts += 1;

            if *attempts < 3 {
                Err(TurboPropError::network_timeout("model download", 30))
            } else {
                Ok("Downloaded successfully".to_string())
            }
        }

        fn is_retryable(&self, error: &Self::Error) -> bool {
            matches!(error, 
                TurboPropError::NetworkError { .. } | 
                TurboPropError::NetworkTimeout { .. }
            )
        }
    }

    let mut operation = NetworkOperation {
        attempts: Arc::new(Mutex::new(0)),
    };

    let result = config.execute(&mut operation).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "Downloaded successfully");
}