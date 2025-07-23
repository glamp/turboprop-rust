//! Tests for CLI argument parsing and model command functionality.
//!
//! These tests verify that the CLI correctly parses new model-related arguments
//! and commands without requiring binary execution.

use clap::Parser;
use turboprop::cli::{Cli, Commands, ModelCommands};

#[test]
fn test_cli_global_model_option() {
    let args = vec!["tp", "--model", "test-model", "index", "--repo", "."];
    let cli = Cli::try_parse_from(args).expect("Failed to parse CLI args");

    assert_eq!(cli.model, Some("test-model".to_string()));
    assert!(matches!(cli.command, Commands::Index { .. }));
}

#[test]
fn test_cli_global_instruction_option() {
    let args = vec![
        "tp",
        "--instruction",
        "test instruction",
        "search",
        "test query",
    ];
    let cli = Cli::try_parse_from(args).expect("Failed to parse CLI args");

    assert_eq!(cli.instruction, Some("test instruction".to_string()));
    assert!(matches!(cli.command, Commands::Search { .. }));
}

#[test]
fn test_cli_model_list_command() {
    let args = vec!["tp", "model", "list"];
    let cli = Cli::try_parse_from(args).expect("Failed to parse CLI args");

    match cli.command {
        Commands::Model { action } => {
            assert!(matches!(action, ModelCommands::List));
        }
        _ => panic!("Expected Model command"),
    }
}

#[test]
fn test_cli_model_download_command() {
    let args = vec!["tp", "model", "download", "test-model"];
    let cli = Cli::try_parse_from(args).expect("Failed to parse CLI args");

    match cli.command {
        Commands::Model { action } => match action {
            ModelCommands::Download { model } => {
                assert_eq!(model, "test-model");
            }
            _ => panic!("Expected Download subcommand"),
        },
        _ => panic!("Expected Model command"),
    }
}

#[test]
fn test_cli_model_info_command() {
    let args = vec!["tp", "model", "info", "test-model"];
    let cli = Cli::try_parse_from(args).expect("Failed to parse CLI args");

    match cli.command {
        Commands::Model { action } => match action {
            ModelCommands::Info { model } => {
                assert_eq!(model, "test-model");
            }
            _ => panic!("Expected Info subcommand"),
        },
        _ => panic!("Expected Model command"),
    }
}

#[test]
fn test_cli_model_clear_command() {
    let args = vec!["tp", "model", "clear"];
    let cli = Cli::try_parse_from(args).expect("Failed to parse CLI args");

    match cli.command {
        Commands::Model { action } => {
            assert!(matches!(action, ModelCommands::Clear { model: None }));
        }
        _ => panic!("Expected Model command"),
    }
}

#[test]
fn test_cli_model_clear_specific_command() {
    let args = vec!["tp", "model", "clear", "test-model"];
    let cli = Cli::try_parse_from(args).expect("Failed to parse CLI args");

    match cli.command {
        Commands::Model { action } => match action {
            ModelCommands::Clear { model } => {
                assert_eq!(model, Some("test-model".to_string()));
            }
            _ => panic!("Expected Clear subcommand"),
        },
        _ => panic!("Expected Model command"),
    }
}

#[test]
fn test_cli_index_with_model_and_instruction() {
    let args = vec![
        "tp",
        "index",
        "--repo",
        ".",
        "--model",
        "Qwen/Qwen3-Embedding-0.6B",
        "--instruction",
        "Represent this code for search",
    ];
    let cli = Cli::try_parse_from(args).expect("Failed to parse CLI args");

    match cli.command {
        Commands::Index {
            model, instruction, ..
        } => {
            assert_eq!(model, Some("Qwen/Qwen3-Embedding-0.6B".to_string()));
            assert_eq!(
                instruction,
                Some("Represent this code for search".to_string())
            );
        }
        _ => panic!("Expected Index command"),
    }
}

#[test]
fn test_cli_search_with_model_and_instruction() {
    let args = vec![
        "tp",
        "search",
        "test query",
        "--model",
        "nomic-embed-code.Q5_K_S.gguf",
        "--instruction",
        "Encode this text for similarity search",
    ];
    let cli = Cli::try_parse_from(args).expect("Failed to parse CLI args");

    match cli.command {
        Commands::Search {
            model, instruction, ..
        } => {
            assert_eq!(model, Some("nomic-embed-code.Q5_K_S.gguf".to_string()));
            assert_eq!(
                instruction,
                Some("Encode this text for similarity search".to_string())
            );
        }
        _ => panic!("Expected Search command"),
    }
}

#[test]
fn test_cli_help_shows_model_information() {
    // This test ensures that help text includes model information
    let args = vec!["tp", "index", "--help"];
    let result = Cli::try_parse_from(args);

    // We expect this to fail with a help message that contains model info
    assert!(result.is_err());
    let error = result.unwrap_err();
    let help_text = error.to_string();

    // Check that help includes model-related information
    assert!(help_text.contains("--model"));
    assert!(help_text.contains("Qwen"));
    assert!(help_text.contains("all-MiniLM"));
}
