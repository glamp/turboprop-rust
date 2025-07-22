//! Progress reporting utilities for indexing operations.
//!
//! This module provides progress bars, spinners, and statistics tracking
//! for long-running indexing operations to give users clear feedback.

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle, ProgressState};
use std::fmt::Write;
use std::path::Path;
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// Progress tracker for the complete indexing pipeline
pub struct IndexingProgress {
    /// Start time of the indexing operation
    start_time: Instant,
    /// Progress bar for file discovery phase
    discovery_spinner: Option<ProgressBar>,
    /// Progress bar for file processing phase
    processing_bar: Option<ProgressBar>,
    /// Statistics about the indexing operation
    stats: IndexingStats,
    /// Whether progress bars are enabled (false for testing/quiet mode)
    enabled: bool,
}

/// Statistics collected during indexing
#[derive(Debug, Clone, Default)]
pub struct IndexingStats {
    /// Total number of files discovered
    pub files_discovered: usize,
    /// Number of files successfully processed
    pub files_processed: usize,
    /// Number of files that failed to process
    pub files_failed: usize,
    /// Total number of chunks created
    pub chunks_created: usize,
    /// Total number of embeddings generated
    pub embeddings_generated: usize,
    /// List of files that failed to process
    pub failed_files: Vec<String>,
}

impl IndexingProgress {
    /// Create a new progress tracker
    pub fn new(enabled: bool) -> Self {
        Self {
            start_time: Instant::now(),
            discovery_spinner: None,
            processing_bar: None,
            stats: IndexingStats::default(),
            enabled,
        }
    }

    /// Start the file discovery phase with a spinner
    pub fn start_discovery(&mut self) -> Result<()> {
        if !self.enabled {
            info!("Starting file discovery...");
            return Ok(());
        }

        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::with_template("{spinner:.green} {msg}")?
                .tick_strings(&[
                    "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
                ]),
        );
        spinner.set_message("Discovering files...");
        spinner.enable_steady_tick(Duration::from_millis(100));

        self.discovery_spinner = Some(spinner);
        Ok(())
    }

    /// Finish file discovery and show results
    pub fn finish_discovery(&mut self, file_count: usize) -> Result<()> {
        self.stats.files_discovered = file_count;

        if let Some(spinner) = &self.discovery_spinner {
            spinner.finish_with_message(format!("✓ Found {} files", file_count));
        } else {
            info!("Found {} files", file_count);
        }

        self.discovery_spinner = None;
        Ok(())
    }

    /// Start the file processing phase with a progress bar
    pub fn start_processing(&mut self, total_files: usize) -> Result<()> {
        if !self.enabled {
            info!("Starting file processing ({} files)...", total_files);
            return Ok(());
        }

        let bar = ProgressBar::new(total_files as u64);
        bar.set_style(
            ProgressStyle::with_template(
                "Processing files... [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} files {msg}"
            )?
            .with_key("eta", |state: &ProgressState, w: &mut dyn Write| {
                write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
            })
            .progress_chars("█▉▊▋▌▍▎▏  "),
        );

        self.processing_bar = Some(bar);
        Ok(())
    }

    /// Update progress for the current file being processed
    pub fn update_processing(&mut self, current_file: &Path) -> Result<()> {
        if let Some(bar) = &self.processing_bar {
            let file_name = current_file
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("<unknown>");
            bar.set_message(format!("({})", file_name));
            bar.inc(1);
        }

        Ok(())
    }

    /// Record successful processing of a file
    pub fn record_file_success(&mut self, chunks_created: usize, embeddings_generated: usize) {
        self.stats.files_processed += 1;
        self.stats.chunks_created += chunks_created;
        self.stats.embeddings_generated += embeddings_generated;
    }

    /// Record failed processing of a file
    pub fn record_file_failure(&mut self, file_path: &Path, error: &anyhow::Error) {
        self.stats.files_failed += 1;
        self.stats.failed_files.push(format!("{}: {}", file_path.display(), error));
        debug!("File processing failed: {} - {}", file_path.display(), error);
    }

    /// Finish processing and show final summary
    pub fn finish_processing(&mut self, index_path: &Path) -> Result<()> {
        if let Some(bar) = &self.processing_bar {
            bar.finish_and_clear();
        }

        self.processing_bar = None;
        self.show_summary(index_path)?;
        Ok(())
    }

    /// Display final summary statistics
    pub fn show_summary(&self, index_path: &Path) -> Result<()> {
        let duration = self.start_time.elapsed();
        let duration_str = humantime::format_duration(duration);

        println!("Generated {} chunks from {} files in {}", 
            self.stats.chunks_created, 
            self.stats.files_processed, 
            duration_str
        );

        if self.stats.files_failed > 0 {
            println!("⚠️  {} files failed to process:", self.stats.files_failed);
            for failure in &self.stats.failed_files {
                println!("   {}", failure);
            }
        }

        println!("Index saved to {}", index_path.display());

        // Log detailed statistics
        info!("Indexing completed - Files discovered: {}, processed: {}, failed: {}, chunks: {}, embeddings: {}, duration: {:?}",
            self.stats.files_discovered,
            self.stats.files_processed, 
            self.stats.files_failed,
            self.stats.chunks_created,
            self.stats.embeddings_generated,
            duration
        );

        Ok(())
    }

    /// Get current statistics
    pub fn stats(&self) -> &IndexingStats {
        &self.stats
    }

    /// Get elapsed time since start
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Check if any files failed to process
    pub fn has_failures(&self) -> bool {
        self.stats.files_failed > 0
    }
}

impl IndexingStats {
    /// Calculate success rate as a percentage
    pub fn success_rate(&self) -> f64 {
        if self.files_discovered == 0 {
            return 100.0;
        }
        (self.files_processed as f64 / self.files_discovered as f64) * 100.0
    }

    /// Get total files that were attempted for processing
    pub fn total_attempted(&self) -> usize {
        self.files_processed + self.files_failed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_indexing_progress_creation() {
        let progress = IndexingProgress::new(true);
        assert_eq!(progress.stats.files_discovered, 0);
        assert_eq!(progress.stats.files_processed, 0);
        assert_eq!(progress.stats.chunks_created, 0);
        assert!(progress.enabled);
    }

    #[test]
    fn test_stats_calculation() {
        let mut stats = IndexingStats::default();
        stats.files_discovered = 10;
        stats.files_processed = 8;
        stats.files_failed = 2;

        assert_eq!(stats.success_rate(), 80.0);
        assert_eq!(stats.total_attempted(), 10);
    }

    #[test]
    fn test_record_operations() {
        let mut progress = IndexingProgress::new(false);
        
        progress.record_file_success(5, 5);
        assert_eq!(progress.stats.files_processed, 1);
        assert_eq!(progress.stats.chunks_created, 5);
        assert_eq!(progress.stats.embeddings_generated, 5);

        let error = anyhow::anyhow!("Test error");
        progress.record_file_failure(&PathBuf::from("test.txt"), &error);
        assert_eq!(progress.stats.files_failed, 1);
        assert_eq!(progress.stats.failed_files.len(), 1);
        assert!(progress.has_failures());
    }

    #[test]
    fn test_disabled_progress() {
        let mut progress = IndexingProgress::new(false);
        
        // These should not panic even when disabled
        assert!(progress.start_discovery().is_ok());
        assert!(progress.finish_discovery(5).is_ok());
        assert!(progress.start_processing(5).is_ok());
        assert!(progress.update_processing(&PathBuf::from("test.txt")).is_ok());
        assert!(progress.finish_processing(&PathBuf::from(".turboprop")).is_ok());
    }
}