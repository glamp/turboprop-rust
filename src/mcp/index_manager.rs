//! Index management for MCP server
//!
//! Handles file watching, incremental updates, and index maintenance
//! while the MCP server is running.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, watch};
use tokio::time::{interval, Instant};
use tracing::{debug, error, info, warn};

use crate::config::TurboPropConfig;
use crate::git::GitignoreFilter;
use crate::incremental::IncrementalUpdater;
use crate::index::PersistentChunkIndex;
use crate::watcher::{FileWatcher, WatchEventBatch, WatcherConfig};

/// Manages index updates for the MCP server
pub struct IndexManager {
    /// Repository path
    #[allow(dead_code)]
    repo_path: PathBuf,
    /// Configuration
    #[allow(dead_code)]
    config: TurboPropConfig,
    /// Current index (wrapped for thread-safe access)
    index: Arc<RwLock<Option<PersistentChunkIndex>>>,
    /// Incremental updater
    updater: Arc<RwLock<Option<IncrementalUpdater>>>,
    /// File watcher
    file_watcher: Option<FileWatcher>,
    /// Update statistics
    stats: Arc<RwLock<IndexStats>>,
    /// Shutdown signal
    shutdown_tx: watch::Sender<bool>,
    shutdown_rx: watch::Receiver<bool>,
}

/// Index statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    /// Total number of chunks in index
    pub total_chunks: usize,
    /// Total number of files indexed
    pub total_files: usize,
    /// Last update timestamp
    pub last_update: Option<Instant>,
    /// Number of updates processed
    pub updates_processed: u64,
    /// Number of files added
    pub files_added: u64,
    /// Number of files updated
    pub files_updated: u64,
    /// Number of files removed
    pub files_removed: u64,
    /// Number of update errors
    pub update_errors: u64,
}

impl IndexManager {
    /// Create a new index manager
    pub async fn new(
        repo_path: &Path,
        config: &TurboPropConfig,
        initial_index: Option<PersistentChunkIndex>,
    ) -> Result<Self> {
        info!("Initializing index manager for {}", repo_path.display());
        
        // Create gitignore filter
        let gitignore_filter = GitignoreFilter::new(repo_path)
            .context("Failed to create gitignore filter")?;
        
        // Create file watcher with custom config
        let watcher_config = WatcherConfig {
            debounce_duration: Duration::from_millis(500),
            max_batch_size: 100,
        };
        let file_watcher = FileWatcher::with_config(repo_path, gitignore_filter, watcher_config)
            .context("Failed to create file watcher")?;
        
        // Create incremental updater
        let updater = IncrementalUpdater::new(config.clone(), repo_path)
            .await
            .context("Failed to create incremental updater")?;
        
        // Initialize index stats
        let stats = if let Some(ref index) = initial_index {
            IndexStats {
                total_chunks: index.len(),
                total_files: 0, // We'll estimate this later
                last_update: Some(Instant::now()),
                ..Default::default()
            }
        } else {
            IndexStats::default()
        };
        
        // Create shutdown channel
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        
        Ok(Self {
            repo_path: repo_path.to_path_buf(),
            config: config.clone(),
            index: Arc::new(RwLock::new(initial_index)),
            updater: Arc::new(RwLock::new(Some(updater))),
            file_watcher: Some(file_watcher),
            stats: Arc::new(RwLock::new(stats)),
            shutdown_tx,
            shutdown_rx,
        })
    }
    
    /// Set the initial index
    pub async fn set_index(&self, index: PersistentChunkIndex) {
        let mut index_guard = self.index.write().await;
        *index_guard = Some(index);
        
        // Update stats
        if let Some(ref index) = *index_guard {
            let mut stats_guard = self.stats.write().await;
            stats_guard.total_chunks = index.len();
            stats_guard.last_update = Some(Instant::now());
        }
        
        info!("Index set successfully");
    }
    
    /// Get a read-only reference to the current index
    pub async fn get_index(&self) -> Arc<RwLock<Option<PersistentChunkIndex>>> {
        Arc::clone(&self.index)
    }
    
    /// Get current index statistics
    pub async fn get_stats(&self) -> IndexStats {
        let stats_guard = self.stats.read().await;
        stats_guard.clone()
    }
    
    /// Start the index management background tasks
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting index manager background tasks");
        
        // Spawn file change processing task
        let index_clone = Arc::clone(&self.index);
        let stats_clone = Arc::clone(&self.stats);
        let updater_clone = Arc::clone(&self.updater);
        let mut file_watcher = self.file_watcher.take().unwrap();
        let mut shutdown_rx_clone = self.shutdown_rx.clone();
        
        tokio::spawn(async move {
            Self::process_file_changes(
                index_clone,
                stats_clone,
                updater_clone,
                &mut file_watcher,
                &mut shutdown_rx_clone,
            ).await;
        });
        
        // Spawn periodic maintenance task
        let index_clone = Arc::clone(&self.index);
        let stats_clone = Arc::clone(&self.stats);
        let mut shutdown_rx_clone = self.shutdown_rx.clone();
        
        tokio::spawn(async move {
            Self::periodic_maintenance(
                index_clone,
                stats_clone,
                &mut shutdown_rx_clone,
            ).await;
        });
        
        info!("Index manager started successfully");
        Ok(())
    }
    
    /// Stop the index manager
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping index manager");
        
        // Send shutdown signal
        let _ = self.shutdown_tx.send(true);
        
        info!("Index manager stopped");
        Ok(())
    }
    
    /// Background task for processing file changes
    async fn process_file_changes(
        index: Arc<RwLock<Option<PersistentChunkIndex>>>,
        stats: Arc<RwLock<IndexStats>>,
        updater: Arc<RwLock<Option<IncrementalUpdater>>>,
        file_watcher: &mut FileWatcher,
        shutdown_rx: &mut watch::Receiver<bool>,
    ) {
        info!("Starting file change processing task");
        
        let mut update_interval = interval(Duration::from_millis(1000)); // Check every second
        
        loop {
            tokio::select! {
                // Check for shutdown signal
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        info!("File change processing task shutting down");
                        break;
                    }
                }
                
                // Process file change batches
                _ = update_interval.tick() => {
                    if let Some(batch) = file_watcher.next_batch().await {
                        let batch_size = batch.events.len();
                        
                        match Self::handle_file_batch(
                            &index,
                            &stats,
                            &updater,
                            batch,
                        ).await {
                            Ok(()) => {
                                // Reset error count on successful processing
                                let mut stats_guard = stats.write().await;
                                if stats_guard.update_errors > 0 {
                                    info!("File processing recovered after {} errors", stats_guard.update_errors);
                                    stats_guard.update_errors = 0;
                                }
                            }
                            Err(e) => {
                                error!("Failed to process file batch: {}", e);
                                
                                // Update error count and implement error recovery
                                let mut stats_guard = stats.write().await;
                                stats_guard.update_errors += 1;
                                
                                // Implement exponential backoff for error recovery
                                let error_count = stats_guard.update_errors;
                                drop(stats_guard);
                                
                                if error_count >= 5 {
                                    warn!("Multiple consecutive errors ({}), implementing recovery delay", error_count);
                                    let delay_ms = std::cmp::min(1000 * 2_u64.pow((error_count - 5) as u32), 30000);
                                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                                }
                                
                                // Log detailed error information for debugging
                                debug!("Error context: batch contained {} events", batch_size);
                            }
                        }
                    }
                }
            }
        }
    }
    
    /// Handle a batch of file changes
    async fn handle_file_batch(
        index: &Arc<RwLock<Option<PersistentChunkIndex>>>,
        stats: &Arc<RwLock<IndexStats>>,
        _updater: &Arc<RwLock<Option<IncrementalUpdater>>>,
        batch: WatchEventBatch,
    ) -> Result<()> {
        if batch.events.is_empty() {
            return Ok(());
        }
        
        debug!("Processing file batch with {} events", batch.events.len());
        
        // Check if there are any file changes that require index updates
        let has_file_changes = batch.events.iter().any(|event| event.is_file_event());
        
        if !has_file_changes {
            return Ok(());
        }
        
        // For now, we'll use a simple approach: mark that files have changed
        // and update statistics. In the future, this could be enhanced to use
        // the IncrementalUpdater for more sophisticated updates.
        
        // Update statistics
        {
            let mut stats_guard = stats.write().await;
            stats_guard.updates_processed += 1;
            stats_guard.last_update = Some(Instant::now());
            
            // Estimate changes based on event types
            let (modified_files, created_files, deleted_files) = batch.group_by_type();
            stats_guard.files_added += created_files.len() as u64;
            stats_guard.files_updated += modified_files.len() as u64;
            stats_guard.files_removed += deleted_files.len() as u64;
            
            // Update total files/chunks from current index
            if let Some(ref current_index) = *index.read().await {
                stats_guard.total_chunks = current_index.len();
            }
        }
        
        info!(
            "File changes detected: {} events processed, statistics updated",
            batch.events.len()
        );
        
        Ok(())
    }
    
    /// Periodic maintenance task
    async fn periodic_maintenance(
        index: Arc<RwLock<Option<PersistentChunkIndex>>>,
        stats: Arc<RwLock<IndexStats>>,
        shutdown_rx: &mut watch::Receiver<bool>,
    ) {
        info!("Starting periodic maintenance task");
        
        let mut maintenance_interval = interval(Duration::from_secs(300)); // 5 minutes
        
        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        info!("Periodic maintenance task shutting down");
                        break;
                    }
                }
                
                _ = maintenance_interval.tick() => {
                    Self::perform_maintenance(&index, &stats).await;
                }
            }
        }
    }
    
    /// Perform periodic maintenance
    async fn perform_maintenance(
        index: &Arc<RwLock<Option<PersistentChunkIndex>>>,
        stats: &Arc<RwLock<IndexStats>>,
    ) {
        debug!("Performing periodic maintenance");
        
        // Log statistics
        {
            let stats_guard = stats.read().await;
            if stats_guard.updates_processed > 0 {
                info!(
                    "Index stats: {} chunks, {} updates processed",
                    stats_guard.total_chunks,
                    stats_guard.updates_processed
                );
                
                if stats_guard.update_errors > 0 {
                    warn!("Update errors encountered: {}", stats_guard.update_errors);
                }
            }
        }
        
        // Persist index to disk with retry logic
        {
            let index_guard = index.read().await;
            if let Some(ref current_index) = *index_guard {
                const MAX_SAVE_RETRIES: u32 = 3;
                let mut save_attempt = 0;
                
                loop {
                    match current_index.save() {
                        Ok(()) => {
                            debug!("Index saved successfully during maintenance");
                            break;
                        }
                        Err(e) => {
                            save_attempt += 1;
                            if save_attempt >= MAX_SAVE_RETRIES {
                                error!("Failed to save index after {} attempts: {}", MAX_SAVE_RETRIES, e);
                                
                                // Update error statistics
                                let mut stats_guard = stats.write().await;
                                stats_guard.update_errors += 1;
                                break;
                            } else {
                                warn!("Failed to save index (attempt {}), retrying: {}", save_attempt, e);
                                tokio::time::sleep(Duration::from_millis(1000 * save_attempt as u64)).await;
                            }
                        }
                    }
                }
            }
        }
        
        debug!("Maintenance cycle completed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_index_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();
        
        let manager = IndexManager::new(temp_dir.path(), &config, None).await;
        assert!(manager.is_ok());
    }
    
    #[tokio::test]
    async fn test_index_stats() {
        let temp_dir = TempDir::new().unwrap();
        let config = TurboPropConfig::default();
        
        let manager = IndexManager::new(temp_dir.path(), &config, None).await.unwrap();
        let stats = manager.get_stats().await;
        
        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.updates_processed, 0);
    }
}