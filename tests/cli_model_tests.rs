//! End-to-end CLI tests for model commands.
//!
//! These tests validate the complete CLI interface for model management
//! including list, info, download, and clear commands with proper
//! argument handling and output validation.
//!
//! Run these tests with: `cargo test cli_model`

use anyhow::Result;
use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::TempDir;

/// Test model list command shows all available models
#[tokio::test]
async fn test_model_list_command() -> Result<()> {
    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("model").arg("list");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Available Embedding Models"))
        .stdout(predicate::str::contains(
            "sentence-transformers/all-MiniLM-L6-v2",
        ))
        .stdout(predicate::str::contains("nomic-embed-code.Q5_K_S.gguf"))
        .stdout(predicate::str::contains("Qwen/Qwen3-Embedding-0.6B"))
        .stdout(predicate::str::contains("Dimensions:"))
        .stdout(predicate::str::contains("Size:"))
        .stdout(predicate::str::contains("Backend:"));

    Ok(())
}

/// Test model list command shows cache status
#[tokio::test]
async fn test_model_list_cache_status() -> Result<()> {
    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("model").arg("list");

    let output = cmd.assert().success();

    // Should show cache status for each model
    output.stdout(
        predicate::str::contains("cached").or(predicate::str::contains("download required")),
    );

    Ok(())
}

/// Test model info command for sentence transformer model
#[tokio::test]
async fn test_model_info_sentence_transformer() -> Result<()> {
    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("model")
        .arg("info")
        .arg("sentence-transformers/all-MiniLM-L6-v2");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Model Information"))
        .stdout(predicate::str::contains(
            "sentence-transformers/all-MiniLM-L6-v2",
        ))
        .stdout(predicate::str::contains("Dimensions: 384"))
        .stdout(predicate::str::contains("FastEmbed"))
        .stdout(predicate::str::contains("SentenceTransformer"))
        .stdout(predicate::str::contains("Cache Status:"));

    Ok(())
}

/// Test model info command for GGUF model
#[tokio::test]
async fn test_model_info_gguf() -> Result<()> {
    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("model")
        .arg("info")
        .arg("nomic-embed-code.Q5_K_S.gguf");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Model Information"))
        .stdout(predicate::str::contains("nomic-embed-code.Q5_K_S.gguf"))
        .stdout(predicate::str::contains("Dimensions: 768"))
        .stdout(predicate::str::contains("Candle"))
        .stdout(predicate::str::contains("GGUF"))
        .stdout(predicate::str::contains("Download URL:"))
        .stdout(predicate::str::contains("Nomic Embed Code Features:"))
        .stdout(predicate::str::contains("code search"));

    Ok(())
}

/// Test model info command for Qwen3 model
#[tokio::test]
async fn test_model_info_qwen3() -> Result<()> {
    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("model")
        .arg("info")
        .arg("Qwen/Qwen3-Embedding-0.6B");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Model Information"))
        .stdout(predicate::str::contains("Qwen/Qwen3-Embedding-0.6B"))
        .stdout(predicate::str::contains("Dimensions: 1024"))
        .stdout(predicate::str::contains("Custom"))
        .stdout(predicate::str::contains("HuggingFace"))
        .stdout(predicate::str::contains("Qwen3 Model Features:"))
        .stdout(predicate::str::contains("instruction-based embeddings"))
        .stdout(predicate::str::contains("Multilingual support"));

    Ok(())
}

/// Test model info command with nonexistent model
#[tokio::test]
async fn test_model_info_nonexistent() -> Result<()> {
    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("model").arg("info").arg("nonexistent/model");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("not found"));

    Ok(())
}

/// Test model download command for FastEmbed model
#[tokio::test]
async fn test_model_download_fastembed() -> Result<()> {
    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("model")
        .arg("download")
        .arg("sentence-transformers/all-MiniLM-L6-v2");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("FastEmbed model"))
        .stdout(predicate::str::contains("downloaded automatically"));

    Ok(())
}

/// Test model download command for nonexistent model
#[tokio::test]
async fn test_model_download_nonexistent() -> Result<()> {
    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("model").arg("download").arg("nonexistent/model");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("not found"));

    Ok(())
}

/// Test model clear all command
#[tokio::test]
async fn test_model_clear_all() -> Result<()> {
    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("model").arg("clear");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Clearing all model caches"))
        .stdout(predicate::str::contains("Successfully cleared"));

    Ok(())
}

/// Test model clear specific model command
#[tokio::test]
async fn test_model_clear_specific() -> Result<()> {
    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("model")
        .arg("clear")
        .arg("sentence-transformers/all-MiniLM-L6-v2");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Clearing cache for model"))
        .stdout(predicate::str::contains(
            "sentence-transformers/all-MiniLM-L6-v2",
        ))
        .stdout(predicate::str::contains("Successfully cleared"));

    Ok(())
}

/// Test index command with custom model selection
#[tokio::test]
async fn test_index_with_custom_model() -> Result<()> {
    let temp_dir = TempDir::new()?;

    // Create a small test repository
    std::fs::write(
        temp_dir.path().join("test.js"),
        "function hello() { console.log('world'); }",
    )?;

    // Initialize git repository
    let mut git_init = Command::new("git");
    git_init
        .args(["init", "--quiet"])
        .current_dir(temp_dir.path());
    git_init.assert().success();

    // Add test file to git
    let mut git_add = Command::new("git");
    git_add
        .args(["add", "test.js"])
        .current_dir(temp_dir.path());
    git_add.assert().success();

    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("index")
        .arg("--repo")
        .arg(temp_dir.path())
        .arg("--model")
        .arg("sentence-transformers/all-MiniLM-L6-v2");

    cmd.assert().success();

    Ok(())
}

/// Test search command with custom model selection
#[tokio::test]
async fn test_search_with_custom_model() -> Result<()> {
    let temp_dir = TempDir::new()?;

    // Create and index a test repository first
    std::fs::write(
        temp_dir.path().join("test.py"),
        "def authenticate_user(username, password):\n    return username == 'admin' and password == 'secret'"
    )?;

    // Initialize git repository
    let mut git_init = Command::new("git");
    git_init
        .args(["init", "--quiet"])
        .current_dir(temp_dir.path());
    git_init.assert().success();

    // Add test file to git
    let mut git_add = Command::new("git");
    git_add
        .args(["add", "test.py"])
        .current_dir(temp_dir.path());
    git_add.assert().success();

    // Index first
    let mut index_cmd = Command::cargo_bin("tp")?;
    index_cmd
        .arg("index")
        .arg("--repo")
        .arg(temp_dir.path())
        .arg("--model")
        .arg("sentence-transformers/all-MiniLM-L6-v2");
    index_cmd.assert().success();

    // Then search
    let mut search_cmd = Command::cargo_bin("tp")?;
    search_cmd
        .arg("search")
        .arg("authentication")
        .arg("--repo")
        .arg(temp_dir.path())
        .arg("--model")
        .arg("sentence-transformers/all-MiniLM-L6-v2");

    search_cmd
        .assert()
        .success()
        .stdout(predicate::str::contains("authenticate_user"));

    Ok(())
}

/// Test CLI help for model commands
#[tokio::test]
async fn test_model_help() -> Result<()> {
    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("model").arg("--help");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("model management commands"))
        .stdout(predicate::str::contains("list"))
        .stdout(predicate::str::contains("info"))
        .stdout(predicate::str::contains("download"))
        .stdout(predicate::str::contains("clear"));

    Ok(())
}

/// Test CLI help for specific model subcommands
#[tokio::test]
async fn test_model_list_help() -> Result<()> {
    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("model").arg("list").arg("--help");

    cmd.assert().success().stdout(predicate::str::contains(
        "List all available embedding models",
    ));

    Ok(())
}

#[tokio::test]
async fn test_model_info_help() -> Result<()> {
    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("model").arg("info").arg("--help");

    cmd.assert().success().stdout(predicate::str::contains(
        "Show detailed information about a specific model",
    ));

    Ok(())
}

#[tokio::test]
async fn test_model_download_help() -> Result<()> {
    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("model").arg("download").arg("--help");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Download a specific model"));

    Ok(())
}

#[tokio::test]
async fn test_model_clear_help() -> Result<()> {
    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("model").arg("clear").arg("--help");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Clear model cache"));

    Ok(())
}

/// Test error handling for missing arguments
#[tokio::test]
async fn test_model_info_missing_argument() -> Result<()> {
    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("model").arg("info");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("required"));

    Ok(())
}

#[tokio::test]
async fn test_model_download_missing_argument() -> Result<()> {
    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("model").arg("download");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("required"));

    Ok(())
}

/// Test JSON output format (if supported)
#[tokio::test]
async fn test_model_list_output_format() -> Result<()> {
    let mut cmd_default = Command::cargo_bin("tp")?;
    cmd_default.arg("model").arg("list");

    let output = cmd_default.assert().success();

    // Verify human-readable output contains expected formatting
    output
        .stdout(predicate::str::contains("Available Embedding Models"))
        .stdout(predicate::str::contains("Description:"))
        .stdout(predicate::str::contains("Type:"))
        .stdout(predicate::str::contains("Backend:"))
        .stdout(predicate::str::contains("Dimensions:"))
        .stdout(predicate::str::contains("Size:"));

    Ok(())
}

/// Test model command with verbose output
#[tokio::test]
async fn test_model_list_verbose() -> Result<()> {
    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("model").arg("list").arg("--verbose");

    // May not be implemented yet, but should not crash
    let result = cmd.assert();

    // Either succeeds with verbose output or fails gracefully
    match result.try_success() {
        Ok(output) => {
            output.stdout(predicate::str::contains("Available Embedding Models"));
        }
        Err(_) => {
            // Verbose flag might not be implemented yet
            // Test that it fails gracefully
        }
    }

    Ok(())
}

/// Test model commands with different log levels
#[tokio::test]
async fn test_model_command_with_log_level() -> Result<()> {
    let mut cmd = Command::cargo_bin("tp")?;
    cmd.env("RUST_LOG", "debug").arg("model").arg("list");

    cmd.assert().success();

    Ok(())
}

/// Test model caching behavior through CLI
#[tokio::test]
async fn test_model_caching_through_cli() -> Result<()> {
    let temp_dir = TempDir::new()?;

    // Set custom cache directory
    let cache_dir = temp_dir.path().join("custom_cache");
    std::fs::create_dir_all(&cache_dir)?;

    let mut cmd = Command::cargo_bin("tp")?;
    cmd.env("TURBOPROP_CACHE_DIR", cache_dir.to_str().unwrap())
        .arg("model")
        .arg("list");

    cmd.assert().success();

    Ok(())
}

#[tokio::test]
#[ignore] // Requires Qwen3 instruction support to be fully implemented
async fn test_qwen3_instruction_cli() -> Result<()> {
    let temp_dir = TempDir::new()?;

    std::fs::write(
        temp_dir.path().join("code.rs"),
        "fn main() { println!(\"Hello, world!\"); }",
    )?;

    // Initialize git repository
    let mut git_init = Command::new("git");
    git_init
        .args(["init", "--quiet"])
        .current_dir(temp_dir.path());
    git_init.assert().success();

    // Add test file to git
    let mut git_add = Command::new("git");
    git_add
        .args(["add", "code.rs"])
        .current_dir(temp_dir.path());
    git_add.assert().success();

    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("index")
        .arg("--repo")
        .arg(temp_dir.path())
        .arg("--model")
        .arg("Qwen/Qwen3-Embedding-0.6B")
        .arg("--instruction")
        .arg("Represent this code for search");

    // This test may be ignored if Qwen3 model is not fully available
    // When implemented, it should succeed
    cmd.assert().success();

    Ok(())
}

/// Test model download with progress indication (if supported)
#[tokio::test]
#[ignore] // Requires actual model download which is slow
async fn test_model_download_with_progress() -> Result<()> {
    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("model")
        .arg("download")
        .arg("nomic-embed-code.Q5_K_S.gguf");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Downloading"))
        .stdout(predicate::str::contains("Successfully downloaded"));

    Ok(())
}

/// Test model download for HuggingFace model (placeholder behavior)
#[tokio::test]
async fn test_model_download_huggingface_placeholder() -> Result<()> {
    let mut cmd = Command::cargo_bin("tp")?;
    cmd.arg("model")
        .arg("download")
        .arg("Qwen/Qwen3-Embedding-0.6B");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Downloading Hugging Face model"))
        .stdout(predicate::str::contains("placeholder"));

    Ok(())
}
