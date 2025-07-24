//! Lifecycle manager for MCP server - handles server lifecycle management
//!
//! Separates lifecycle concerns from protocol and business logic

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Server lifecycle states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerState {
    /// Server is being created
    Creating,
    /// Server is initializing
    Initializing,
    /// Server is ready to handle requests
    Ready,
    /// Server is running and processing requests
    Running,
    /// Server is shutting down
    ShuttingDown,
    /// Server is stopped
    Stopped,
    /// Server encountered a fatal error
    Error,
}

impl ServerState {
    /// Check if the server can accept requests in this state
    pub fn can_accept_requests(&self) -> bool {
        matches!(self, ServerState::Ready | ServerState::Running)
    }

    /// Check if the server is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(self, ServerState::Stopped | ServerState::Error)
    }
}

/// Manages server lifecycle and state transitions
pub struct LifecycleManager {
    /// Current server state
    state: Arc<RwLock<ServerState>>,
    /// Shutdown signal
    shutdown_requested: Arc<AtomicBool>,
    /// Graceful shutdown flag
    graceful_shutdown: Arc<AtomicBool>,
}

impl LifecycleManager {
    /// Create a new lifecycle manager
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(ServerState::Creating)),
            shutdown_requested: Arc::new(AtomicBool::new(false)),
            graceful_shutdown: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Get the current server state
    pub async fn get_state(&self) -> ServerState {
        let state_guard = self.state.read().await;
        *state_guard
    }

    /// Check if the server can accept requests
    pub async fn can_accept_requests(&self) -> bool {
        let state = self.get_state().await;
        state.can_accept_requests()
    }

    /// Check if shutdown has been requested
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested.load(Ordering::SeqCst)
    }

    /// Check if graceful shutdown is enabled
    pub fn is_graceful_shutdown(&self) -> bool {
        self.graceful_shutdown.load(Ordering::SeqCst)
    }

    /// Transition to initializing state
    pub async fn transition_to_initializing(&self) -> Result<(), LifecycleError> {
        let mut state_guard = self.state.write().await;
        match *state_guard {
            ServerState::Creating => {
                *state_guard = ServerState::Initializing;
                info!("Server state transitioned to Initializing");
                Ok(())
            }
            current_state => {
                warn!(
                    "Invalid state transition from {:?} to Initializing",
                    current_state
                );
                Err(LifecycleError::InvalidStateTransition {
                    from: current_state,
                    to: ServerState::Initializing,
                })
            }
        }
    }

    /// Transition to ready state
    pub async fn transition_to_ready(&self) -> Result<(), LifecycleError> {
        let mut state_guard = self.state.write().await;
        match *state_guard {
            ServerState::Initializing => {
                *state_guard = ServerState::Ready;
                info!("Server state transitioned to Ready");
                Ok(())
            }
            current_state => {
                warn!("Invalid state transition from {:?} to Ready", current_state);
                Err(LifecycleError::InvalidStateTransition {
                    from: current_state,
                    to: ServerState::Ready,
                })
            }
        }
    }

    /// Transition to running state
    pub async fn transition_to_running(&self) -> Result<(), LifecycleError> {
        let mut state_guard = self.state.write().await;
        match *state_guard {
            ServerState::Ready => {
                *state_guard = ServerState::Running;
                info!("Server state transitioned to Running");
                Ok(())
            }
            current_state => {
                warn!("Invalid state transition from {:?} to Running", current_state);
                Err(LifecycleError::InvalidStateTransition {
                    from: current_state,
                    to: ServerState::Running,
                })
            }
        }
    }

    /// Transition to shutting down state
    pub async fn transition_to_shutting_down(&self) -> Result<(), LifecycleError> {
        let mut state_guard = self.state.write().await;
        match *state_guard {
            ServerState::Ready | ServerState::Running => {
                *state_guard = ServerState::ShuttingDown;
                info!("Server state transitioned to ShuttingDown");
                Ok(())
            }
            current_state => {
                debug!(
                    "State transition from {:?} to ShuttingDown allowed in emergency",
                    current_state
                );
                *state_guard = ServerState::ShuttingDown;
                Ok(())
            }
        }
    }

    /// Transition to stopped state
    pub async fn transition_to_stopped(&self) -> Result<(), LifecycleError> {
        let mut state_guard = self.state.write().await;
        match *state_guard {
            ServerState::ShuttingDown => {
                *state_guard = ServerState::Stopped;
                info!("Server state transitioned to Stopped");
                Ok(())
            }
            current_state => {
                warn!("Invalid state transition from {:?} to Stopped", current_state);
                Err(LifecycleError::InvalidStateTransition {
                    from: current_state,
                    to: ServerState::Stopped,
                })
            }
        }
    }

    /// Transition to error state
    pub async fn transition_to_error(&self, error_reason: String) -> Result<(), LifecycleError> {
        let mut state_guard = self.state.write().await;
        let current_state = *state_guard;
        *state_guard = ServerState::Error;
        warn!(
            "Server state transitioned to Error from {:?}: {}",
            current_state, error_reason
        );
        Ok(())
    }

    /// Request server shutdown
    pub fn request_shutdown(&self, graceful: bool) {
        self.shutdown_requested.store(true, Ordering::SeqCst);
        self.graceful_shutdown.store(graceful, Ordering::SeqCst);
        
        if graceful {
            info!("Graceful shutdown requested");
        } else {
            warn!("Immediate shutdown requested");
        }
    }

    /// Reset the lifecycle manager (for tests)
    pub async fn reset(&self) {
        let mut state_guard = self.state.write().await;
        *state_guard = ServerState::Creating;
        self.shutdown_requested.store(false, Ordering::SeqCst);
        self.graceful_shutdown.store(true, Ordering::SeqCst);
        debug!("Lifecycle manager reset");
    }

    /// Get a summary of the current lifecycle state
    pub async fn get_summary(&self) -> LifecycleSummary {
        let state = self.get_state().await;
        LifecycleSummary {
            state,
            shutdown_requested: self.is_shutdown_requested(),
            graceful_shutdown: self.is_graceful_shutdown(),
            can_accept_requests: state.can_accept_requests(),
            is_terminal: state.is_terminal(),
        }
    }

    /// Handle STDIN closure (triggers shutdown)
    pub async fn handle_stdin_closed(&self) {
        info!("STDIN closed, initiating graceful shutdown");
        self.request_shutdown(true);
        let _ = self.transition_to_shutting_down().await;
    }

    /// Handle fatal error (transitions to error state)
    pub async fn handle_fatal_error(&self, error: String) {
        warn!("Fatal error occurred: {}", error);
        let _ = self.transition_to_error(error).await;
    }
}

impl Default for LifecycleManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Lifecycle manager summary
#[derive(Debug, Clone)]
pub struct LifecycleSummary {
    pub state: ServerState,
    pub shutdown_requested: bool,
    pub graceful_shutdown: bool,
    pub can_accept_requests: bool,
    pub is_terminal: bool,
}

/// Lifecycle manager errors
#[derive(Debug, thiserror::Error)]
pub enum LifecycleError {
    #[error("Invalid state transition from {from:?} to {to:?}")]
    InvalidStateTransition { from: ServerState, to: ServerState },

    #[error("Server is in terminal state: {state:?}")]
    TerminalState { state: ServerState },

    #[error("Operation not allowed in current state: {state:?}")]
    OperationNotAllowed { state: ServerState },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lifecycle_manager_creation() {
        let manager = LifecycleManager::new();
        let state = manager.get_state().await;
        assert_eq!(state, ServerState::Creating);
        assert!(!manager.is_shutdown_requested());
        assert!(manager.is_graceful_shutdown());
    }

    #[tokio::test]
    async fn test_state_transitions() {
        let manager = LifecycleManager::new();
        
        // Creating -> Initializing
        assert!(manager.transition_to_initializing().await.is_ok());
        assert_eq!(manager.get_state().await, ServerState::Initializing);
        
        // Initializing -> Ready
        assert!(manager.transition_to_ready().await.is_ok());
        assert_eq!(manager.get_state().await, ServerState::Ready);
        
        // Ready -> Running
        assert!(manager.transition_to_running().await.is_ok());
        assert_eq!(manager.get_state().await, ServerState::Running);
        
        // Running -> ShuttingDown
        assert!(manager.transition_to_shutting_down().await.is_ok());
        assert_eq!(manager.get_state().await, ServerState::ShuttingDown);
        
        // ShuttingDown -> Stopped
        assert!(manager.transition_to_stopped().await.is_ok());
        assert_eq!(manager.get_state().await, ServerState::Stopped);
    }

    #[tokio::test]
    async fn test_invalid_state_transition() {
        let manager = LifecycleManager::new();
        
        // Try to go directly from Creating to Running (should fail)
        let result = manager.transition_to_running().await;
        assert!(result.is_err());
        matches!(result.unwrap_err(), LifecycleError::InvalidStateTransition { .. });
    }

    #[tokio::test]
    async fn test_can_accept_requests() {
        let manager = LifecycleManager::new();
        
        // Creating state - cannot accept requests
        assert!(!manager.can_accept_requests().await);
        
        // Transition to Ready - can accept requests
        assert!(manager.transition_to_initializing().await.is_ok());
        assert!(manager.transition_to_ready().await.is_ok());
        assert!(manager.can_accept_requests().await);
        
        // Transition to Running - can accept requests
        assert!(manager.transition_to_running().await.is_ok());
        assert!(manager.can_accept_requests().await);
        
        // Transition to ShuttingDown - cannot accept requests
        assert!(manager.transition_to_shutting_down().await.is_ok());
        assert!(!manager.can_accept_requests().await);
    }

    #[tokio::test]
    async fn test_shutdown_request() {
        let manager = LifecycleManager::new();
        
        assert!(!manager.is_shutdown_requested());
        
        manager.request_shutdown(true);
        assert!(manager.is_shutdown_requested());
        assert!(manager.is_graceful_shutdown());
        
        manager.request_shutdown(false);
        assert!(manager.is_shutdown_requested());
        assert!(!manager.is_graceful_shutdown());
    }

    #[tokio::test]
    async fn test_error_state_transition() {
        let manager = LifecycleManager::new();
        
        let result = manager.transition_to_error("Test error".to_string()).await;
        assert!(result.is_ok());
        assert_eq!(manager.get_state().await, ServerState::Error);
    }

    #[tokio::test]
    async fn test_lifecycle_summary() {
        let manager = LifecycleManager::new();
        
        let summary = manager.get_summary().await;
        assert_eq!(summary.state, ServerState::Creating);
        assert!(!summary.shutdown_requested);
        assert!(summary.graceful_shutdown);
        assert!(!summary.can_accept_requests);
        assert!(!summary.is_terminal);
    }

    #[tokio::test]
    async fn test_stdin_closed_handling() {
        let manager = LifecycleManager::new();
        
        // Set up server in running state
        assert!(manager.transition_to_initializing().await.is_ok());
        assert!(manager.transition_to_ready().await.is_ok());
        assert!(manager.transition_to_running().await.is_ok());
        
        // Handle STDIN closed
        manager.handle_stdin_closed().await;
        
        assert!(manager.is_shutdown_requested());
        assert_eq!(manager.get_state().await, ServerState::ShuttingDown);
    }

    #[tokio::test]
    async fn test_reset() {
        let manager = LifecycleManager::new();
        
        // Change state and request shutdown
        assert!(manager.transition_to_initializing().await.is_ok());
        manager.request_shutdown(false);
        
        // Reset
        manager.reset().await;
        
        // Verify reset state
        assert_eq!(manager.get_state().await, ServerState::Creating);
        assert!(!manager.is_shutdown_requested());
        assert!(manager.is_graceful_shutdown());
    }
}