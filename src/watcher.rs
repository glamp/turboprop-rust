//! File system watching implementation for incremental index updates.
//!
//! This module provides comprehensive file system monitoring capabilities with
//! debouncing, batch processing, and gitignore support for efficient incremental
//! index updates.

use anyhow::{Context, Result};
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};
use tokio::sync::mpsc as tokio_mpsc;
use tracing::{debug, error, info};

use crate::git::GitignoreFilter;

// Bounds for WatcherConfig validation
const MIN_DEBOUNCE_DURATION_MS: u64 = 50;  // 50ms minimum
const MAX_DEBOUNCE_DURATION_MS: u64 = 10000; // 10 seconds maximum
const MIN_BATCH_SIZE: usize = 1;
const MAX_BATCH_SIZE: usize = 10000;

/// Configuration for file watcher behavior
#[derive(Debug, Clone)]
pub struct WatcherConfig {
    /// Debounce duration for file change events to batch rapid changes
    pub debounce_duration: Duration,
    /// Maximum number of events to batch in a single update
    pub max_batch_size: usize,
}

impl Default for WatcherConfig {
    fn default() -> Self {
        Self {
            debounce_duration: Duration::from_millis(500),
            max_batch_size: 100,
        }
    }
}

impl WatcherConfig {
    /// Create a new WatcherConfig with validation
    pub fn new(debounce_duration: Duration, max_batch_size: usize) -> Result<Self> {
        let config = Self {
            debounce_duration,
            max_batch_size,
        };
        config.validate()?;
        Ok(config)
    }
    
    /// Validate the configuration parameters
    pub fn validate(&self) -> Result<()> {
        // Validate debounce duration bounds
        let debounce_ms = self.debounce_duration.as_millis() as u64;
        if debounce_ms < MIN_DEBOUNCE_DURATION_MS {
            return Err(anyhow::anyhow!(
                "Debounce duration {}ms is below minimum {}ms", 
                debounce_ms, 
                MIN_DEBOUNCE_DURATION_MS
            ));
        }
        if debounce_ms > MAX_DEBOUNCE_DURATION_MS {
            return Err(anyhow::anyhow!(
                "Debounce duration {}ms exceeds maximum {}ms", 
                debounce_ms, 
                MAX_DEBOUNCE_DURATION_MS
            ));
        }
        
        // Validate batch size bounds  
        if self.max_batch_size < MIN_BATCH_SIZE {
            return Err(anyhow::anyhow!(
                "Max batch size {} is below minimum {}", 
                self.max_batch_size, 
                MIN_BATCH_SIZE
            ));
        }
        if self.max_batch_size > MAX_BATCH_SIZE {
            return Err(anyhow::anyhow!(
                "Max batch size {} exceeds maximum {}", 
                self.max_batch_size, 
                MAX_BATCH_SIZE
            ));
        }
        
        Ok(())
    }
}

/// Events that trigger index updates
#[derive(Debug, Clone)]
pub enum WatchEvent {
    /// File was modified
    Modified(PathBuf),
    /// File was created
    Created(PathBuf),
    /// File was deleted
    Deleted(PathBuf),
    /// File was renamed
    Renamed { from: PathBuf, to: PathBuf },
    /// Directory was created (may contain new files)
    DirectoryCreated(PathBuf),
    /// Directory was deleted (files removed)
    DirectoryDeleted(PathBuf),
}

impl WatchEvent {
    /// Get the path associated with this event
    pub fn path(&self) -> &Path {
        match self {
            WatchEvent::Modified(path) => path,
            WatchEvent::Created(path) => path,
            WatchEvent::Deleted(path) => path,
            WatchEvent::Renamed { to, .. } => to,
            WatchEvent::DirectoryCreated(path) => path,
            WatchEvent::DirectoryDeleted(path) => path,
        }
    }

    /// Check if this is a file event (as opposed to directory event)
    pub fn is_file_event(&self) -> bool {
        matches!(
            self,
            WatchEvent::Modified(_) | WatchEvent::Created(_) | WatchEvent::Deleted(_) | WatchEvent::Renamed { .. }
        )
    }
}

/// Batched file change events ready for processing
#[derive(Debug)]
pub struct WatchEventBatch {
    /// Events in this batch
    pub events: Vec<WatchEvent>,
    /// Timestamp when the batch was created
    pub timestamp: Instant,
}

impl WatchEventBatch {
    /// Create a new event batch
    pub fn new(events: Vec<WatchEvent>) -> Self {
        Self {
            events,
            timestamp: Instant::now(),
        }
    }

    /// Get unique file paths from all events in this batch
    /// Uses "last event wins" deduplication to preserve the most recent event for each path
    pub fn unique_paths(&self) -> Vec<PathBuf> {
        // Use HashMap to store the last event index for each path
        // This ensures "last event wins" semantics and is more efficient than HashSet
        let mut path_to_index = std::collections::HashMap::new();
        
        // Iterate through events and store the latest index for each path
        for (index, event) in self.events.iter().enumerate() {
            let path = event.path().to_path_buf();
            path_to_index.insert(path, index);
        }
        
        // Collect unique paths in the order they last appeared
        let mut indexed_paths: Vec<(usize, PathBuf)> = path_to_index
            .into_iter()
            .map(|(path, index)| (index, path))
            .collect();
        
        // Sort by index to preserve event ordering (last event wins)
        indexed_paths.sort_by_key(|(index, _)| *index);
        
        // Extract just the paths
        indexed_paths.into_iter().map(|(_, path)| path).collect()
    }

    /// Group events by type for efficient processing
    pub fn group_by_type(&self) -> (Vec<PathBuf>, Vec<PathBuf>, Vec<PathBuf>) {
        let mut modified = Vec::new();
        let mut created = Vec::new();
        let mut deleted = Vec::new();

        for event in &self.events {
            match event {
                WatchEvent::Modified(path) => modified.push(path.clone()),
                WatchEvent::Created(path) | WatchEvent::DirectoryCreated(path) => {
                    created.push(path.clone())
                }
                WatchEvent::Deleted(path) | WatchEvent::DirectoryDeleted(path) => {
                    deleted.push(path.clone())
                }
                WatchEvent::Renamed { from, to } => {
                    // Treat rename as deletion of old path and creation of new path
                    deleted.push(from.clone());
                    created.push(to.clone());
                }
            }
        }

        (modified, created, deleted)
    }
}

/// File system watcher with debouncing and gitignore filtering
pub struct FileWatcher {
    /// The notify watcher instance
    _watcher: RecommendedWatcher,
    /// Receiver for batched events
    event_receiver: tokio_mpsc::Receiver<WatchEventBatch>,
    /// Gitignore filter for respecting .gitignore rules
    gitignore_filter: GitignoreFilter,
    /// Root path being watched
    root_path: PathBuf,
}

impl FileWatcher {
    /// Create a new file watcher for the specified path with default configuration
    ///
    /// # Arguments
    /// * `path` - Root directory to watch
    /// * `gitignore_filter` - Filter to respect .gitignore rules
    ///
    /// # Returns
    /// * `Result<Self>` - The file watcher instance or error
    pub fn new(path: &Path, gitignore_filter: GitignoreFilter) -> Result<Self> {
        Self::with_config(path, gitignore_filter, WatcherConfig::default())
    }

    /// Create a new file watcher for the specified path with custom configuration
    ///
    /// # Arguments
    /// * `path` - Root directory to watch
    /// * `gitignore_filter` - Filter to respect .gitignore rules
    /// * `config` - Configuration for watcher behavior
    ///
    /// # Returns
    /// * `Result<Self>` - The file watcher instance or error
    pub fn with_config(
        path: &Path,
        gitignore_filter: GitignoreFilter,
        config: WatcherConfig,
    ) -> Result<Self> {
        let (event_sender, event_receiver) = tokio_mpsc::channel(1000);
        let (tx, rx) = mpsc::channel();

        // Create the debouncer in a separate thread
        let debouncer = EventDebouncer::new(event_sender, config);
        let _debouncer_handle = debouncer.spawn(rx);

        // Create the file system watcher
        let mut watcher = RecommendedWatcher::new(
            tx,
            Config::default().with_poll_interval(Duration::from_millis(100)),
        )
        .context("Failed to create file system watcher")?;

        // Start watching the path recursively
        watcher
            .watch(path, RecursiveMode::Recursive)
            .with_context(|| format!("Failed to watch path: {}", path.display()))?;

        info!("File watcher initialized for path: {}", path.display());

        Ok(Self {
            _watcher: watcher,
            event_receiver,
            gitignore_filter,
            root_path: path.to_path_buf(),
        })
    }

    /// Wait for the next batch of file change events
    ///
    /// # Returns
    /// * `Option<WatchEventBatch>` - Next batch of events, or None if watcher is closed
    pub async fn next_batch(&mut self) -> Option<WatchEventBatch> {
        loop {
            match self.event_receiver.recv().await {
                Some(batch) => {
                    // Filter events through gitignore
                    let filtered_events = batch
                        .events
                        .into_iter()
                        .filter(|event| self.should_include_path(event.path()))
                        .collect::<Vec<_>>();

                    if filtered_events.is_empty() {
                        debug!("All events in batch were filtered out by gitignore");
                        // Continue the loop to wait for next batch
                        continue;
                    } else {
                        debug!("Received batch with {} events", filtered_events.len());
                        return Some(WatchEventBatch::new(filtered_events));
                    }
                }
                None => {
                    info!("File watcher event channel closed");
                    return None;
                }
            }
        }
    }

    /// Check if a path should be included based on gitignore rules
    fn should_include_path(&self, path: &Path) -> bool {
        // Convert to relative path from root
        let relative_path = match path.strip_prefix(&self.root_path) {
            Ok(rel) => rel,
            Err(_) => {
                debug!("Path outside watch root: {}", path.display());
                return false;
            }
        };

        // Check gitignore rules
        let should_include = self.gitignore_filter.should_include(relative_path);
        if !should_include {
            debug!("Path filtered by gitignore: {}", relative_path.display());
        }

        should_include
    }

    /// Get the root path being watched
    pub fn root_path(&self) -> &Path {
        &self.root_path
    }
}

/// Event debouncer to batch rapid file changes
struct EventDebouncer {
    /// Sender for batched events
    event_sender: tokio_mpsc::Sender<WatchEventBatch>,
    /// Configuration for debouncing behavior
    config: WatcherConfig,
}

impl EventDebouncer {
    /// Create a new event debouncer
    fn new(event_sender: tokio_mpsc::Sender<WatchEventBatch>, config: WatcherConfig) -> Self {
        Self {
            event_sender,
            config,
        }
    }

    /// Spawn the debouncer in a separate thread
    fn spawn(self, event_receiver: Receiver<notify::Result<Event>>) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
            rt.block_on(async {
                self.run(event_receiver).await;
            });
        })
    }

    /// Main debouncer loop
    async fn run(self, event_receiver: Receiver<notify::Result<Event>>) {
        let mut pending_events: HashMap<PathBuf, (WatchEvent, Instant)> = HashMap::new();
        let mut last_batch_time = Instant::now();

        // Use a 50ms interval for efficient polling instead of busy waiting
        let mut interval = tokio::time::interval(Duration::from_millis(50));

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    // Check for new events (non-blocking)
                    while let Ok(event_result) = event_receiver.try_recv() {
                        match event_result {
                            Ok(event) => {
                                if let Some(watch_event) = self.convert_notify_event(event) {
                                    let path = watch_event.path().to_path_buf();
                                    pending_events.insert(path, (watch_event, Instant::now()));
                                }
                            }
                            Err(e) => {
                                error!("File watcher error: {}", e);
                            }
                        }
                    }

                    // Check if we should send a batch
                    let now = Instant::now();
                    let should_send_batch = !pending_events.is_empty()
                        && (now.duration_since(last_batch_time) >= self.config.debounce_duration
                            || pending_events.len() >= self.config.max_batch_size);

                    if should_send_batch {
                        // Collect events that are ready (older than debounce duration)
                        let mut ready_events = Vec::new();
                        let mut keys_to_remove = Vec::new();

                        for (path, (event, timestamp)) in &pending_events {
                            if now.duration_since(*timestamp) >= self.config.debounce_duration {
                                ready_events.push(event.clone());
                                keys_to_remove.push(path.clone());
                            }
                        }

                        // Remove the processed events
                        for key in keys_to_remove {
                            pending_events.remove(&key);
                        }

                        if !ready_events.is_empty() {
                            let batch = WatchEventBatch::new(ready_events);
                            debug!("Sending batch with {} events", batch.events.len());

                            if let Err(e) = self.event_sender.send(batch).await {
                                error!("Failed to send event batch: {}", e);
                                break;
                            }

                            last_batch_time = now;
                        }
                    }
                }
            }
        }
    }

    /// Convert notify event to our watch event
    fn convert_notify_event(&self, event: Event) -> Option<WatchEvent> {
        if event.paths.is_empty() {
            return None;
        }

        let path = event.paths[0].clone();

        match event.kind {
            EventKind::Create(_) => {
                if path.is_dir() {
                    Some(WatchEvent::DirectoryCreated(path))
                } else {
                    Some(WatchEvent::Created(path))
                }
            }
            EventKind::Modify(_) => Some(WatchEvent::Modified(path)),
            EventKind::Remove(_) => {
                // We can't easily tell if removed path was file or directory
                // so we'll treat it as file removal (most common case)
                Some(WatchEvent::Deleted(path))
            }
            _ => {
                debug!("Ignoring event kind: {:?}", event.kind);
                None
            }
        }
    }
}

/// Signal handler for graceful shutdown
pub struct SignalHandler {
    /// Receiver for shutdown signals
    shutdown_receiver: tokio::sync::oneshot::Receiver<()>,
}

impl SignalHandler {
    /// Create a new signal handler
    pub fn new() -> Result<Self> {
        let (shutdown_sender, shutdown_receiver) = tokio::sync::oneshot::channel();

        // Handle SIGINT (Ctrl+C)
        let shutdown_sender = std::sync::Arc::new(std::sync::Mutex::new(Some(shutdown_sender)));
        let shutdown_sender_clone = shutdown_sender.clone();

        signal_hook::flag::register(
            signal_hook::consts::SIGINT,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        )
        .context("Failed to register SIGINT handler")?;

        // Spawn a task to handle the signal
        tokio::spawn(async move {
            tokio::signal::ctrl_c()
                .await
                .expect("Failed to listen for ctrl-c");
            if let Some(sender) = shutdown_sender_clone.lock().unwrap().take() {
                let _ = sender.send(());
            }
        });

        Ok(Self { shutdown_receiver })
    }

    /// Wait for shutdown signal
    pub async fn wait_for_shutdown(self) {
        let _ = self.shutdown_receiver.await;
        info!("Shutdown signal received");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_watch_event_path() {
        let path = PathBuf::from("/test/path");
        let event = WatchEvent::Modified(path.clone());
        assert_eq!(event.path(), path);
    }

    #[test]
    fn test_watch_event_is_file_event() {
        assert!(WatchEvent::Modified(PathBuf::new()).is_file_event());
        assert!(WatchEvent::Created(PathBuf::new()).is_file_event());
        assert!(WatchEvent::Deleted(PathBuf::new()).is_file_event());
        assert!(!WatchEvent::DirectoryCreated(PathBuf::new()).is_file_event());
        assert!(!WatchEvent::DirectoryDeleted(PathBuf::new()).is_file_event());
    }

    #[test]
    fn test_event_batch_unique_paths() {
        let events = vec![
            WatchEvent::Modified(PathBuf::from("/path1")),
            WatchEvent::Created(PathBuf::from("/path2")),
            WatchEvent::Modified(PathBuf::from("/path1")), // Duplicate
        ];

        let batch = WatchEventBatch::new(events);
        let unique_paths = batch.unique_paths();

        assert_eq!(unique_paths.len(), 2);
        assert!(unique_paths.contains(&PathBuf::from("/path1")));
        assert!(unique_paths.contains(&PathBuf::from("/path2")));
    }

    #[test]
    fn test_event_batch_group_by_type() {
        let events = vec![
            WatchEvent::Modified(PathBuf::from("/modified")),
            WatchEvent::Created(PathBuf::from("/created")),
            WatchEvent::Deleted(PathBuf::from("/deleted")),
            WatchEvent::DirectoryCreated(PathBuf::from("/dir_created")),
        ];

        let batch = WatchEventBatch::new(events);
        let (modified, created, deleted) = batch.group_by_type();

        assert_eq!(modified, vec![PathBuf::from("/modified")]);
        assert_eq!(
            created,
            vec![PathBuf::from("/created"), PathBuf::from("/dir_created")]
        );
        assert_eq!(deleted, vec![PathBuf::from("/deleted")]);
    }

    #[tokio::test]
    async fn test_file_watcher_creation() {
        let temp_dir = TempDir::new().unwrap();
        let gitignore_filter = GitignoreFilter::new(temp_dir.path()).unwrap();

        // Creating a file watcher should succeed
        let result = FileWatcher::new(temp_dir.path(), gitignore_filter);
        assert!(result.is_ok());

        let watcher = result.unwrap();
        assert_eq!(watcher.root_path(), temp_dir.path());
    }
}
