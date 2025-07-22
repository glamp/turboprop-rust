# TurboProp (tp)

Fast code search and indexing tool written in Rust.

## Build

```bash
cargo build
```

## Usage

### Show help
```bash
cargo run -- --help
```

### Index files
```bash
cargo run -- index --path /path/to/code
```

### Search files
```bash
cargo run -- search "your query"
```

## Testing

```bash
cargo test
```

## Development

This project uses:
- `clap` for CLI parsing
- `tokio` for async runtime
- `serde` for serialization
- `anyhow` for error handling
- `tracing` for logging