//! Processing pipeline coordination for indexing operations.
//!
//! This module coordinates the complete indexing pipeline from file discovery
//! through chunking, embedding generation, and persistent storage, with
//! comprehensive progress reporting and error handling.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

use crate::chunking::ChunkingStrategy;
use crate::config::TurboPropConfig;
use crate::embeddings::EmbeddingGenerator;
use crate::error_utils::{DirectoryErrorContext, ProcessingErrorContext};
use crate::files::FileDiscovery;
use crate::index::PersistentChunkIndex;
use crate::progress::IndexingProgress;
use crate::types::{ChunkIndex, FileMetadata, IndexedChunk};

/// Configuration for the indexing pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Whether to show progress bars and interactive output
    pub show_progress: bool,
    /// Whether to continue processing after individual file errors
    pub continue_on_error: bool,
    /// Batch size for embedding generation
    pub batch_size: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            show_progress: true,
            continue_on_error: true,
            batch_size: 32,
        }
    }
}

impl PipelineConfig {
    pub fn with_progress(mut self, show_progress: bool) -> Self {
        self.show_progress = show_progress;
        self
    }

    pub fn with_error_handling(mut self, continue_on_error: bool) -> Self {
        self.continue_on_error = continue_on_error;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

/// Result of a pipeline execution
#[derive(Debug)]
pub struct PipelineResult {
    /// The resulting chunk index
    pub index: ChunkIndex,
    /// Path where the index was saved (if persistent storage was used)
    pub index_path: Option<PathBuf>,
    /// Statistics from the pipeline execution
    pub stats: crate::progress::IndexingStats,
}

/// Coordinates the complete indexing pipeline with progress tracking
pub struct IndexingPipeline {
    config: TurboPropConfig,
    pipeline_config: PipelineConfig,
    progress: IndexingProgress,
}

impl IndexingPipeline {
    /// Create a new indexing pipeline
    pub fn new(config: TurboPropConfig, pipeline_config: PipelineConfig) -> Self {
        let progress = IndexingProgress::new(pipeline_config.show_progress);

        Self {
            config,
            pipeline_config,
            progress,
        }
    }

    /// Execute the complete indexing pipeline for the specified path
    pub async fn execute(&mut self, path: &Path) -> Result<PipelineResult> {
        info!("Starting indexing pipeline for: {}", path.display());

        // Validate input path
        self.validate_path(path)?;

        // Phase 1: File Discovery
        self.progress.start_discovery()?;
        let files = self.discover_files(path)?;
        self.progress.finish_discovery(files.len())?;

        if files.is_empty() {
            anyhow::bail!("No files found to index in: {}", path.display());
        }

        // Phase 2: Processing Pipeline
        self.progress.start_processing(files.len())?;
        let index = self.process_files(&files).await?;

        // Phase 3: Persistent Storage (if using persistent index)
        let index_path = self.save_persistent_index(path, &index).await?;

        self.progress
            .finish_processing(index_path.as_ref().unwrap_or(&PathBuf::from(".turboprop")))?;

        Ok(PipelineResult {
            index,
            index_path,
            stats: self.progress.stats().clone(),
        })
    }

    /// Execute pipeline and return a persistent index
    pub async fn execute_persistent(
        &mut self,
        path: &Path,
    ) -> Result<(PersistentChunkIndex, crate::progress::IndexingStats)> {
        info!(
            "Starting persistent indexing pipeline for: {}",
            path.display()
        );

        // Validate input path
        self.validate_path(path)?;

        // Phase 1: File Discovery
        self.progress.start_discovery()?;
        let files = self.discover_files(path)?;
        self.progress.finish_discovery(files.len())?;

        if files.is_empty() {
            anyhow::bail!("No files found to index in: {}", path.display());
        }

        // Phase 2: Build Persistent Index with Progress Tracking
        self.progress.start_processing(files.len())?;
        let persistent_index = self
            .build_persistent_index_with_progress(path, &files)
            .await?;

        let index_path = persistent_index.storage_path();
        self.progress.finish_processing(index_path)?;

        Ok((persistent_index, self.progress.stats().clone()))
    }

    /// Validate that the input path is suitable for indexing
    fn validate_path(&self, path: &Path) -> Result<()> {
        if !path.exists() {
            anyhow::bail!("Path does not exist: {}", path.display());
        }

        if !path.is_dir() {
            anyhow::bail!("Path is not a directory: {}", path.display());
        }

        Ok(())
    }

    /// Discover files to be indexed
    fn discover_files(&self, path: &Path) -> Result<Vec<FileMetadata>> {
        let discovery = FileDiscovery::new(self.config.file_discovery.clone());
        let files = discovery.discover_files(path).with_dir_read_context(path)?;

        debug!("Discovered {} files for indexing", files.len());
        Ok(files)
    }

    /// Process all discovered files through the chunking and embedding pipeline
    async fn process_files(&mut self, files: &[FileMetadata]) -> Result<ChunkIndex> {
        let (embedding_generator, chunking_strategy) =
            self.initialize_processing_components().await?;
        self.process_files_to_index(files, &chunking_strategy, embedding_generator)
            .await
    }

    /// Initialize components for file processing
    async fn initialize_processing_components(
        &self,
    ) -> Result<(EmbeddingGenerator, ChunkingStrategy)> {
        let embedding_generator = EmbeddingGenerator::new(self.config.embedding.clone())
            .await
            .context("Failed to initialize embedding generator")?;

        let chunking_strategy = ChunkingStrategy::new(self.config.chunking.clone());

        Ok((embedding_generator, chunking_strategy))
    }

    /// Process files and build in-memory chunk index
    async fn process_files_to_index(
        &mut self,
        files: &[FileMetadata],
        chunking_strategy: &ChunkingStrategy,
        mut embedding_generator: EmbeddingGenerator,
    ) -> Result<ChunkIndex> {
        let mut chunk_index = ChunkIndex::new();

        for file in files {
            let result = self
                .process_single_file(file, chunking_strategy, &mut embedding_generator)
                .await;

            match result {
                Ok((chunks, embeddings)) => {
                    self.handle_successful_file_processing_for_index(
                        file,
                        chunks,
                        embeddings,
                        &mut chunk_index,
                    );
                }
                Err(e) => {
                    self.handle_file_processing_error(file, e)?;
                }
            }

            // Update progress
            self.progress.update_processing(&file.path)?;
        }

        Ok(chunk_index)
    }

    /// Handle successful file processing for in-memory index
    fn handle_successful_file_processing_for_index(
        &mut self,
        file: &FileMetadata,
        chunks: Vec<crate::types::ContentChunk>,
        embeddings: Vec<Vec<f32>>,
        chunk_index: &mut ChunkIndex,
    ) {
        // Record success
        self.progress
            .record_file_success(chunks.len(), embeddings.len());

        // Add to index
        chunk_index.add_chunks(chunks, embeddings);

        debug!(
            "Successfully processed: {} ({} chunks)",
            file.path.display(),
            chunk_index.len()
        );
    }

    /// Process a single file through the chunking and embedding pipeline
    async fn process_single_file(
        &self,
        file: &FileMetadata,
        chunking_strategy: &ChunkingStrategy,
        embedding_generator: &mut EmbeddingGenerator,
    ) -> Result<(Vec<crate::types::ContentChunk>, Vec<Vec<f32>>)> {
        // Generate chunks
        let chunks = chunking_strategy
            .chunk_file(&file.path)
            .with_chunking_context(&file.path)?;

        if chunks.is_empty() {
            debug!("No chunks generated for: {}", file.path.display());
            return Ok((chunks, vec![]));
        }

        // Prepare text for embedding
        let chunk_texts: Vec<String> = chunks.iter().map(|chunk| chunk.content.clone()).collect();

        // Generate embeddings
        let embeddings = embedding_generator
            .embed_batch(&chunk_texts)
            .with_embedding_context(&file.path)?;

        Ok((chunks, embeddings))
    }

    /// Build a persistent index with integrated progress tracking
    async fn build_persistent_index_with_progress(
        &mut self,
        path: &Path,
        files: &[FileMetadata],
    ) -> Result<PersistentChunkIndex> {
        let (embedding_generator, chunking_strategy, mut persistent_index) =
            self.initialize_indexing_components(path).await?;

        let all_indexed_chunks = self
            .process_files_to_chunks(files, &chunking_strategy, embedding_generator)
            .await?;

        if !all_indexed_chunks.is_empty() {
            persistent_index = self.build_and_save_index(path, &all_indexed_chunks).await?;
        }

        info!(
            "Built persistent index with {} chunks",
            persistent_index.len()
        );
        Ok(persistent_index)
    }

    /// Initialize all components needed for indexing
    async fn initialize_indexing_components(
        &self,
        path: &Path,
    ) -> Result<(EmbeddingGenerator, ChunkingStrategy, PersistentChunkIndex)> {
        let embedding_generator = EmbeddingGenerator::new(self.config.embedding.clone())
            .await
            .context("Failed to initialize embedding generator")?;

        let chunking_strategy = ChunkingStrategy::new(self.config.chunking.clone());
        let persistent_index =
            PersistentChunkIndex::new(path).context("Failed to create persistent index")?;

        Ok((embedding_generator, chunking_strategy, persistent_index))
    }

    /// Process all files and collect indexed chunks
    async fn process_files_to_chunks(
        &mut self,
        files: &[FileMetadata],
        chunking_strategy: &ChunkingStrategy,
        mut embedding_generator: EmbeddingGenerator,
    ) -> Result<Vec<IndexedChunk>> {
        let mut all_indexed_chunks = Vec::new();

        for file in files {
            let result = self
                .process_single_file(file, chunking_strategy, &mut embedding_generator)
                .await;

            match result {
                Ok((chunks, embeddings)) => {
                    self.handle_successful_file_processing(
                        file,
                        chunks,
                        embeddings,
                        &mut all_indexed_chunks,
                    );
                }
                Err(e) => {
                    self.handle_file_processing_error(file, e)?;
                }
            }

            // Update progress
            self.progress.update_processing(&file.path)?;
        }

        Ok(all_indexed_chunks)
    }

    /// Handle successful file processing
    fn handle_successful_file_processing(
        &mut self,
        file: &FileMetadata,
        chunks: Vec<crate::types::ContentChunk>,
        embeddings: Vec<Vec<f32>>,
        all_indexed_chunks: &mut Vec<IndexedChunk>,
    ) {
        // Record success
        self.progress
            .record_file_success(chunks.len(), embeddings.len());

        // Create indexed chunks
        let indexed_chunks: Vec<IndexedChunk> = chunks
            .into_iter()
            .zip(embeddings)
            .map(|(chunk, embedding)| IndexedChunk { chunk, embedding })
            .collect();

        all_indexed_chunks.extend(indexed_chunks);

        debug!(
            "Successfully processed: {} ({} total chunks)",
            file.path.display(),
            all_indexed_chunks.len()
        );
    }

    /// Handle file processing errors
    fn handle_file_processing_error(
        &mut self,
        file: &FileMetadata,
        e: anyhow::Error,
    ) -> Result<()> {
        // Record failure
        self.progress.record_file_failure(&file.path, &e);

        if !self.pipeline_config.continue_on_error {
            return Err(e)
                .with_context(|| format!("Failed to process file: {}", file.path.display()));
        }

        warn!(
            "Skipping file due to error: {} - {}",
            file.path.display(),
            e
        );

        Ok(())
    }

    /// Build index configuration and save to storage
    async fn build_and_save_index(
        &self,
        path: &Path,
        all_indexed_chunks: &[IndexedChunk],
    ) -> Result<PersistentChunkIndex> {
        // Build the index configuration
        let index_config = crate::storage::IndexConfig {
            model_name: self.config.embedding.model_name.clone(),
            embedding_dimensions: self.config.embedding.embedding_dimensions,
            batch_size: self.config.embedding.batch_size,
            respect_gitignore: self.config.file_discovery.respect_gitignore,
            include_untracked: self.config.file_discovery.include_untracked,
        };

        // Save to storage
        let storage = crate::storage::IndexStorage::new(path)?;
        storage.save_index(
            all_indexed_chunks,
            &index_config,
            &self.config.general.storage_version,
        )?;

        // Load the saved index to return a properly initialized PersistentChunkIndex
        PersistentChunkIndex::load(path)
    }

    /// Save the in-memory index to persistent storage
    async fn save_persistent_index(
        &self,
        _path: &Path,
        index: &ChunkIndex,
    ) -> Result<Option<PathBuf>> {
        // For now, we don't save in-memory indexes to persistent storage
        // This would require converting ChunkIndex to the format expected by IndexStorage
        // The persistent path is handled by the execute_persistent method
        debug!("In-memory index contains {} chunks", index.len());
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TurboPropConfig;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_file(dir: &Path, name: &str, content: &str) -> PathBuf {
        let file_path = dir.join(name);
        fs::write(&file_path, content).unwrap();
        file_path
    }

    #[test]
    fn test_pipeline_config() {
        let config = PipelineConfig::default()
            .with_progress(false)
            .with_error_handling(false)
            .with_batch_size(16);

        assert!(!config.show_progress);
        assert!(!config.continue_on_error);
        assert_eq!(config.batch_size, 16);
    }

    #[test]
    fn test_path_validation() {
        let temp_dir = TempDir::new().unwrap();
        let pipeline = IndexingPipeline::new(
            TurboPropConfig::default(),
            PipelineConfig::default().with_progress(false),
        );

        // Valid directory should pass
        assert!(pipeline.validate_path(temp_dir.path()).is_ok());

        // Non-existent path should fail
        let non_existent = temp_dir.path().join("non_existent");
        assert!(pipeline.validate_path(&non_existent).is_err());

        // File (not directory) should fail
        let file_path = create_test_file(temp_dir.path(), "test.txt", "content");
        assert!(pipeline.validate_path(&file_path).is_err());
    }

    #[tokio::test]
    async fn test_file_discovery() {
        let temp_dir = TempDir::new().unwrap();

        // Create test files
        create_test_file(temp_dir.path(), "test1.txt", "Hello world");
        create_test_file(temp_dir.path(), "test2.rs", "fn main() {}");

        let pipeline = IndexingPipeline::new(
            TurboPropConfig::default(),
            PipelineConfig::default().with_progress(false),
        );

        let files = pipeline.discover_files(temp_dir.path()).unwrap();
        assert!(files.len() >= 2); // May discover additional files depending on git ignore settings
    }

    #[test]
    fn test_pipeline_result() {
        let index = ChunkIndex::new();
        let stats = crate::progress::IndexingStats::default();

        let result = PipelineResult {
            index,
            index_path: Some(PathBuf::from(".turboprop")),
            stats,
        };

        assert!(result.index_path.is_some());
        assert_eq!(result.stats.files_processed, 0);
    }
}
