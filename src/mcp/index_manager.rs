//! Index manager for MCP server - handles index lifecycle and operations
//!
//! Separates index management concerns from protocol and server logic

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, error, info, warn};

use crate::config::TurboPropConfig;
use crate::index::PersistentChunkIndex;

// Index initialization timeouts
const INDEX_INITIALIZATION_TIMEOUT_SECS: u64 = 300; // 5 minutes
const INDEX_LOAD_TIMEOUT_SECS: u64 = 60; // 1 minute

/// Manages index lifecycle and operations
pub struct IndexManager {
    /// Current index (wrapped for thread safety)
    index: Arc<RwLock<Option<PersistentChunkIndex>>>,
    /// Initialization state
    initialized: Arc<RwLock<bool>>,
    /// Atomic flag to prevent concurrent initialization
    initializing: Arc<AtomicBool>,
    /// Semaphore to limit background index operations
    background_task_semaphore: Arc<Semaphore>,
}

impl IndexManager {
    /// Create a new index manager
    pub fn new() -> Self {
        Self {
            index: Arc::new(RwLock::new(None)),
            initialized: Arc::new(RwLock::new(false)),
            initializing: Arc::new(AtomicBool::new(false)),
            background_task_semaphore: Arc::new(Semaphore::new(5)), // Max 5 concurrent background tasks
        }
    }

    /// Get the current index (for read operations)
    pub fn get_index(&self) -> Arc<RwLock<Option<PersistentChunkIndex>>> {
        Arc::clone(&self.index)
    }

    /// Check if the index is initialized
    pub async fn is_initialized(&self) -> bool {
        let initialized_guard = self.initialized.read().await;
        *initialized_guard
    }

    /// Check if index initialization is in progress
    pub fn is_initializing(&self) -> bool {
        self.initializing.load(Ordering::SeqCst)
    }

    /// Start index initialization in background
    pub async fn start_initialization(
        &self,
        repo_path: PathBuf,
        config: TurboPropConfig,
    ) -> Result<(), IndexManagerError> {
        // Check if already initialized
        if self.is_initialized().await {
            info!("Index already initialized");
            return Ok(());
        }

        // Use atomic flag to prevent concurrent initialization
        if self
            .initializing
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            info!("Index initialization already in progress");
            return Ok(());
        }

        // Acquire semaphore permit for background task
        let background_permit = match self.background_task_semaphore.try_acquire() {
            Ok(permit) => permit,
            Err(_) => {
                warn!("Too many background tasks, deferring index initialization");
                // Reset initializing flag and return error
                self.initializing.store(false, Ordering::SeqCst);
                return Err(IndexManagerError::ResourcesExhausted);
            }
        };

        // Clone necessary data for background task
        let index_clone = Arc::clone(&self.index);
        let initialized_flag = Arc::clone(&self.initialized);
        let initializing_flag = Arc::clone(&self.initializing);

        // Spawn background initialization task
        tokio::spawn(async move {
            let _permit = background_permit; // Keep permit alive for the duration of the task

            info!("Starting background index initialization");
            match Self::initialize_index_with_fallback(&repo_path, &config).await {
                Ok(index) => {
                    {
                        let mut index_guard = index_clone.write().await;
                        *index_guard = Some(index);
                    }
                    {
                        let mut initialized_guard = initialized_flag.write().await;
                        *initialized_guard = true;
                    }
                    // Reset initializing flag
                    initializing_flag.store(false, Ordering::SeqCst);
                    info!("Index initialization completed successfully");
                }
                Err(e) => {
                    error!("Failed to initialize index: {}", e);
                    // Reset initializing flag even on failure
                    initializing_flag.store(false, Ordering::SeqCst);
                    // Note: Server continues to run but search will return errors
                }
            }
        });

        Ok(())
    }

    /// Force synchronous index initialization (for tests)
    pub async fn initialize_sync(
        &self,
        repo_path: &Path,
        config: &TurboPropConfig,
    ) -> Result<(), IndexManagerError> {
        info!("Starting synchronous index initialization");

        match Self::initialize_index_with_fallback(repo_path, config).await {
            Ok(index) => {
                {
                    let mut index_guard = self.index.write().await;
                    *index_guard = Some(index);
                }
                {
                    let mut initialized_guard = self.initialized.write().await;
                    *initialized_guard = true;
                }
                info!("Synchronous index initialization completed");
                Ok(())
            }
            Err(e) => {
                error!("Synchronous index initialization failed: {}", e);
                Err(IndexManagerError::InitializationFailed(e.to_string()))
            }
        }
    }

    /// Initialize index with fallback mechanisms and timeout handling
    async fn initialize_index_with_fallback(
        repo_path: &Path,
        config: &TurboPropConfig,
    ) -> Result<PersistentChunkIndex> {
        debug!("Attempting to initialize index with fallback");

        // Try loading existing index first with timeout
        if let Ok(index) = Self::try_load_existing_index(repo_path, config).await {
            return Ok(index);
        }

        info!("No existing index found or failed to load, building new index");

        // Fallback to building new index with timeout
        match tokio::time::timeout(
            Duration::from_secs(INDEX_INITIALIZATION_TIMEOUT_SECS),
            Self::build_new_index(repo_path, config),
        )
        .await
        {
            Ok(Ok(index)) => {
                info!("Successfully built new index with {} chunks", index.len());
                Ok(index)
            }
            Ok(Err(e)) => {
                error!("Failed to build new index: {}", e);
                Err(e)
            }
            Err(_) => {
                let error = anyhow::anyhow!(
                    "Index initialization timed out after {} seconds",
                    INDEX_INITIALIZATION_TIMEOUT_SECS
                );
                error!("{}", error);
                Err(error)
            }
        }
    }

    /// Try to load existing index with timeout and error handling
    async fn try_load_existing_index(
        repo_path: &Path,
        _config: &TurboPropConfig,
    ) -> Result<PersistentChunkIndex> {
        info!(
            "Attempting to load existing index from {}",
            repo_path.display()
        );

        match tokio::time::timeout(
            Duration::from_secs(INDEX_LOAD_TIMEOUT_SECS),
            PersistentChunkIndex::load(repo_path),
        )
        .await
        {
            Ok(Ok(index)) => {
                info!(
                    "Successfully loaded existing index with {} chunks",
                    index.len()
                );
                Ok(index)
            }
            Ok(Err(e)) => {
                info!("Failed to load existing index: {}", e);
                Err(e.into())
            }
            Err(_) => {
                let error = anyhow::anyhow!(
                    "Loading existing index timed out after {} seconds",
                    INDEX_LOAD_TIMEOUT_SECS
                );
                info!("{}", error);
                Err(error)
            }
        }
    }

    /// Build a new index from scratch
    async fn build_new_index(
        repo_path: &Path,
        config: &TurboPropConfig,
    ) -> Result<PersistentChunkIndex> {
        info!("Building new index for {}", repo_path.display());

        // Use existing TurboProp indexing logic
        let index = crate::commands::index::build_index(repo_path, config)
            .await
            .context("Failed to build initial index")?;

        Ok(index)
    }

    /// Get index statistics
    pub async fn get_stats(&self) -> IndexStats {
        let index_guard = self.index.read().await;
        let initialized = self.is_initialized().await;
        let initializing = self.is_initializing();

        match index_guard.as_ref() {
            Some(index) => IndexStats {
                initialized,
                initializing,
                chunk_count: Some(index.len()),
                index_size: None, // Could be computed if needed
            },
            None => IndexStats {
                initialized,
                initializing,
                chunk_count: None,
                index_size: None,
            },
        }
    }

    /// Reset the index manager (for tests)
    pub async fn reset(&self) {
        let mut index_guard = self.index.write().await;
        *index_guard = None;

        let mut initialized_guard = self.initialized.write().await;
        *initialized_guard = false;

        self.initializing.store(false, Ordering::SeqCst);

        debug!("Index manager reset");
    }
}

impl Default for IndexManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Index manager statistics
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub initialized: bool,
    pub initializing: bool,
    pub chunk_count: Option<usize>,
    pub index_size: Option<u64>,
}

/// Index manager errors
#[derive(Debug, thiserror::Error)]
pub enum IndexManagerError {
    #[error("Index initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Resources exhausted, cannot start new background task")]
    ResourcesExhausted,

    #[error("Index not ready")]
    NotReady,

    #[error("Index operation timed out")]
    Timeout,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_index_manager_creation() {
        let manager = IndexManager::new();
        assert!(!manager.is_initialized().await);
        assert!(!manager.is_initializing());
    }

    #[tokio::test]
    async fn test_index_manager_stats() {
        let manager = IndexManager::new();
        let stats = manager.get_stats().await;
        
        assert!(!stats.initialized);
        assert!(!stats.initializing);
        assert!(stats.chunk_count.is_none());
    }

    #[tokio::test]
    async fn test_index_manager_reset() {
        let manager = IndexManager::new();
        manager.reset().await;
        
        assert!(!manager.is_initialized().await);
        assert!(!manager.is_initializing());
    }
}