# TurboProp MCP Server User Guide

## Overview

### What is MCP?

MCP (Model Context Protocol) is a standard way for AI coding agents to access external tools and data sources. Think of it as a bridge that allows your AI assistant to interact with tools beyond their built-in capabilities.

**Before MCP**: Your coding agent could only work with information you directly provide or share with it.
**With MCP**: Your coding agent can actively search, query, and interact with your entire codebase in real-time.

### Why TurboProp's MCP Server?

TurboProp's MCP server transforms your coding agent into a semantic code search expert. Instead of just reading the files you show it, your agent can:

- 🔍 **Search your entire codebase** using natural language queries
- 🎯 **Find relevant code instantly** based on meaning, not just keywords  
- 🔄 **Stay synchronized** with your changes through real-time file watching
- 🚀 **Work across any programming language** or file type in your project

**Example**: Ask your agent *"Find the JWT authentication implementation"* and it will search through your entire codebase, locate the relevant code, and explain how it works - all without you having to find and share the files manually.

### How It Works

The MCP server acts like a librarian for your codebase:

1. **📚 Catalogs Your Code**: Indexes all files in your repository with semantic embeddings
2. **👀 Watches for Changes**: Monitors file changes and updates the index automatically  
3. **🔍 Provides Search Tool**: Exposes a `search` tool that agents can use via MCP protocol
4. **⚡ Returns Smart Results**: Finds code based on meaning and context, not just keyword matching

This guide covers everything you need to know to set up, configure, and use the MCP server effectively with your preferred coding agent.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Basic Usage](#basic-usage)
3. [Agent Integration](#agent-integration)
4. [Configuration](#configuration)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)
7. [Performance Optimization](#performance-optimization)

## Installation and Setup

### Prerequisites

- TurboProp installed and available in PATH
- A git repository or directory with code files
- A compatible coding agent (Claude Code, Cursor, GitHub Copilot, etc.)

### Quick Setup

1. **Navigate to your project directory:**
   ```bash
   cd /path/to/your/project
   ```

2. **Start the MCP server:**
   ```bash
   tp mcp --repo .
   ```

3. **Configure your coding agent** (see [Agent Integration](#agent-integration))

4. **Start coding!** Your agent can now search your codebase semantically.

## Basic Usage

### Starting the Server

```bash
# Basic usage - index current directory
tp mcp --repo .

# Specify a different directory
tp mcp --repo /path/to/project

# Use custom model
tp mcp --repo . --model sentence-transformers/all-MiniLM-L12-v2

# Set file size limit
tp mcp --repo . --max-filesize 5mb

# Force rebuild of index
tp mcp --repo . --force-rebuild

# Enable verbose logging
tp mcp --repo . --verbose
```

### Server Lifecycle

When you start the MCP server:

1. **Initial Indexing**: Scans all files and builds semantic index
2. **File Watching**: Monitors directory for changes
3. **Real-time Updates**: Updates index when files change
4. **Ready State**: Accepts search requests from agents

The server runs until you stop it (Ctrl+C) or the agent disconnects.

### Understanding the Output

```bash
$ tp mcp --repo .
🚀 TurboProp MCP Server Started
────────────────────────────────
Repository: /Users/you/project
Configuration summary:
  Repository: /Users/you/project
  Model: sentence-transformers/all-MiniLM-L6-v2
  Max file size: 2mb
  Batch size: 32

MCP server ready and listening on stdio...
```

## Agent Integration

### Claude Code

**Method 1: Global Configuration** (applies to all projects)

Add to `~/.claude/mcp.json`:
```json
{
  "mcpServers": {
    "turboprop": {
      "command": "tp",
      "args": ["mcp", "--repo", "."],
      "env": {}
    }
  }
}
```

**Method 2: Project-specific Configuration** (recommended)

Add to your project's `.claude.json`:
```json
{
  "mcpServers": {
    "turboprop": {
      "command": "tp",
      "args": ["mcp", "--repo", "."]
    }
  }
}
```

**✓ Verify Setup**: Restart Claude Code and ask: *"Search for error handling code in this project"*

### Cursor

**Standard Configuration**

Add to `.cursor/mcp.json` in your project root:
```json
{
  "mcpServers": {
    "turboprop": {
      "command": "tp",
      "args": ["mcp", "--repo", "."],
      "cwd": "."
    }
  }
}
```

**Advanced Configuration** (with custom settings):
```json
{
  "mcpServers": {
    "turboprop": {
      "command": "tp",
      "args": ["mcp", "--repo", ".", "--verbose", "--max-filesize", "3mb"],
      "cwd": ".",
      "env": {
        "RUST_LOG": "info"
      }
    }
  }
}
```

**✓ Verify Setup**: Use Cursor's command palette and ask the AI to search your codebase

### GitHub Copilot

**VS Code Configuration**

Add to your VS Code `settings.json`:
```json
{
  "github.copilot.mcp.servers": [
    {
      "name": "turboprop",
      "command": "tp",
      "args": ["mcp", "--repo", "."],
      "env": {}
    }
  ]
}
```

**JetBrains IDEs** (IntelliJ, PyCharm, WebStorm)

Add to your IDE's MCP configuration:
```json
{
  "mcpServers": {
    "turboprop": {
      "command": "tp",
      "args": ["mcp", "--repo", "."]
    }
  }
}
```

### Windsurf

**Project Configuration**

Add to your Windsurf project configuration:
```json
{
  "mcp": {
    "servers": {
      "turboprop": {
        "command": "tp",
        "args": ["mcp", "--repo", "."]
      }
    }
  }
}
```

### Replit Agent

**Replit Configuration**

Add to your `.replit` file:
```toml
[mcp]
servers.turboprop.command = "tp"
servers.turboprop.args = ["mcp", "--repo", "."]
```

### Aider

**Standard Configuration**

Aider supports MCP through its `--mcp-server` flag. Start Aider with TurboProp MCP server:

```bash
# Basic setup - run from your project root
aider --mcp-server "tp mcp --repo ."
```

**Advanced Configuration** (with custom settings):

```bash
# With custom file size limits and verbose output
aider --mcp-server "tp mcp --repo . --verbose --max-filesize 5mb"
```

**Configuration File** (`.aider.conf.yml`):

```yaml
mcp-servers:
  - name: "turboprop"
    command: "tp"
    args: ["mcp", "--repo", "."]
    description: "Semantic code search for your project"
```

**Usage Examples**:
- *"Use the turboprop tool to find authentication code in this project"*
- *"Search for error handling patterns and show me the most relevant examples"*
- *"Find database connection logic and explain how it works"*

**✓ Verify Setup**: Ask Aider to search your codebase: *"Search for configuration files and explain the project structure"*

### Continue.dev

**VS Code Extension Configuration**

Add to your Continue configuration file (`~/.continue/config.json`):

```json
{
  "models": [
    {
      "title": "GPT-4",
      "provider": "openai",
      "model": "gpt-4"
    }
  ],
  "mcpServers": {
    "turboprop": {
      "command": "tp",
      "args": ["mcp", "--repo", "."],
      "env": {}
    }
  }
}
```

**JetBrains Plugin Configuration**

Add to your Continue settings in JetBrains IDEs:

```json
{
  "mcpServers": {
    "turboprop": {
      "command": "tp", 
      "args": ["mcp", "--repo", ".", "--verbose"],
      "workingDirectory": "."
    }
  }
}
```

**Project-specific Configuration** (`.continue/config.json` in project root):

```json
{
  "mcpServers": {
    "turboprop": {
      "command": "tp",
      "args": ["mcp", "--repo", ".", "--max-filesize", "3mb"],
      "description": "Semantic search for this specific project"
    }
  }
}
```

**Usage Examples**:
- Right-click in editor → Continue → *"Search this codebase for similar functions"*  
- Chat: *"Use turboprop to find all tests related to user authentication"*
- Chat: *"Search for TODO comments and prioritize them by complexity"*

**✓ Verify Setup**: Use Continue's chat and ask: *"Search the codebase for API endpoints and list them"*

### Development Environment Integration

#### VS Code (Multiple Agents)

For VS Code with various AI extensions, add to `settings.json`:
```json
{
  "mcp.servers": {
    "turboprop": {
      "command": "tp",
      "args": ["mcp", "--repo", "."],
      "description": "TurboProp semantic code search"
    }
  }
}
```

#### Neovim

For Neovim with MCP-compatible plugins:
```lua
-- In your Neovim config
require('mcp').setup({
  servers = {
    turboprop = {
      command = 'tp',
      args = {'mcp', '--repo', '.'},
      filetypes = {'*'}
    }
  }
})
```

#### Emacs

For Emacs with LSP/MCP integration:
```elisp
;; In your Emacs config
(setq mcp-servers
  '((turboprop . (:command "tp"
                  :args ("mcp" "--repo" ".")))))
```

### Team Configuration Templates

#### Shared Team Config (`.turboprop-mcp.json`)

Create a shared configuration file for your team:
```json
{
  "mcpServers": {
    "turboprop": {
      "command": "tp",
      "args": ["mcp", "--repo", ".", "--model", "sentence-transformers/all-MiniLM-L12-v2"],
      "env": {
        "TURBOPROP_CACHE_DIR": "./.turboprop-cache"
      }
    }
  }
}
```

Then team members can reference it in their agent configs:
```json
{
  "mcpServers": {
    "$ref": "./.turboprop-mcp.json#/mcpServers"
  }
}
```

### Troubleshooting Agent Integration

**Problem**: Agent can't find the `tp` command
- **Solution**: Ensure TurboProp is in your PATH: `which tp`
- **Alternative**: Use absolute path: `"/usr/local/bin/tp"`

**Problem**: Server starts but agent can't connect
- **Solution**: Check agent logs for specific MCP protocol errors
- **Alternative**: Try starting server manually: `tp mcp --repo . --verbose`

**Problem**: Configuration file not loading
- **Solution**: Verify JSON syntax and file permissions
- **Alternative**: Test with minimal configuration first

## Configuration

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--repo` | Repository path to index | `--repo /path/to/project` |
| `--model` | Embedding model to use | `--model sentence-transformers/all-MiniLM-L12-v2` |
| `--max-filesize` | Maximum file size to index | `--max-filesize 5mb` |
| `--filter` | Glob pattern for files to include | `--filter "src/**/*.rs"` |
| `--filetype` | Filter by file type | `--filetype rust` |
| `--force-rebuild` | Force rebuild of existing index | `--force-rebuild` |
| `--verbose` | Enable verbose logging | `--verbose` |
| `--debug` | Enable debug logging | `--debug` |

### Configuration Hierarchy

TurboProp follows this configuration priority order (highest to lowest):

1. **Command line arguments** (highest priority)
2. **Environment variables**
3. **Project `.turboprop.yml`** (in repository root)
4. **Global config** (`~/.turboprop/config.yml`)
5. **Built-in defaults** (lowest priority)

Example:
```bash
# Command line overrides config file
tp mcp --repo . --model "sentence-transformers/all-MiniLM-L12-v2"  # Uses L12 model
# Even if .turboprop.yml specifies L6 model
```

### Configuration File Examples

#### Basic Configuration (`.turboprop.yml`)

```yaml
# Model configuration
model: "sentence-transformers/all-MiniLM-L6-v2"
cache_dir: "~/.turboprop-cache"

# File discovery settings
max_filesize: "2mb"
file_discovery:
  include_patterns:
    - "**/*.rs"
    - "**/*.js" 
    - "**/*.ts"
    - "**/*.py"
    - "**/*.java"
    - "**/*.go"
    - "**/*.md"
  exclude_patterns:
    - "target/**"
    - "node_modules/**"
    - ".git/**"
    - "*.log"
  
# Search settings  
search:
  default_limit: 10
  similarity_threshold: 0.3

# Embedding settings
embedding:
  batch_size: 32
  cache_embeddings: true

# MCP server settings
mcp:
  enable_file_watching: true
  update_debounce_ms: 500
  max_concurrent_searches: 10
```

#### Monorepo Configuration

For large monorepos with multiple services/packages:

```yaml
# Monorepo-optimized configuration
model: "sentence-transformers/all-MiniLM-L12-v2"
max_filesize: "3mb"

# Service-based file discovery
file_discovery:
  include_patterns:
    - "services/*/src/**/*.{rs,py,js,ts}"
    - "libs/*/src/**/*.{rs,py,js,ts}"
    - "packages/*/src/**/*.{js,ts,tsx,jsx}"
    - "apps/*/src/**/*.{js,ts,tsx,jsx}"
    - "docs/**/*.md"
  exclude_patterns:
    - "**/target/**"
    - "**/node_modules/**"
    - "**/dist/**"
    - "**/build/**"
    - "**/*.test.*"
    - "**/*.spec.*"
    - "**/coverage/**"

# Performance optimization for large repos
embedding:
  batch_size: 16  # Reduce for stability
  worker_threads: 8
  cache_embeddings: true

mcp:
  update_debounce_ms: 1000  # Reduce update frequency
  max_concurrent_searches: 5
  
search:
  similarity_threshold: 0.4  # Higher threshold for better precision
```

#### Language-Specific Configurations

**Rust Project**:
```yaml
model: "nomic-embed-code.Q5_K_S.gguf"  # Code-specific model
max_filesize: "5mb"

file_discovery:
  include_patterns:
    - "src/**/*.rs"
    - "lib/**/*.rs"
    - "benches/**/*.rs"
    - "examples/**/*.rs"
    - "Cargo.toml"
    - "README.md"
  exclude_patterns:
    - "target/**"
    - "**/*_generated.rs"

search:
  similarity_threshold: 0.35
```

**JavaScript/TypeScript Project**:
```yaml
model: "sentence-transformers/all-MiniLM-L12-v2"
max_filesize: "2mb"

file_discovery:
  include_patterns:
    - "src/**/*.{js,ts,jsx,tsx}"
    - "lib/**/*.{js,ts}"
    - "pages/**/*.{js,ts,jsx,tsx}"
    - "components/**/*.{js,ts,jsx,tsx}"
    - "package.json"
    - "tsconfig.json"
    - "*.md"
  exclude_patterns:
    - "node_modules/**"
    - "dist/**"
    - "build/**"
    - "**/*.min.js"
    - "**/*.bundle.js"
    - "coverage/**"
```

**Python Project**:
```yaml
model: "sentence-transformers/all-MiniLM-L6-v2"
max_filesize: "3mb"

file_discovery:
  include_patterns:
    - "src/**/*.py"
    - "lib/**/*.py"
    - "scripts/**/*.py"
    - "tests/**/*.py"
    - "requirements*.txt"
    - "pyproject.toml"
    - "setup.py"
    - "*.md"
  exclude_patterns:
    - "__pycache__/**"
    - "*.pyc"
    - "venv/**"
    - ".env/**"
    - "build/**"
    - "dist/**"
```

#### Performance Profiles

**High Performance** (for fast machines):
```yaml
embedding:
  batch_size: 64
  worker_threads: 12
  cache_embeddings: true

mcp:
  update_debounce_ms: 200
  max_concurrent_searches: 15
```

**Memory Constrained** (for limited resources):
```yaml
embedding:
  batch_size: 8
  worker_threads: 2
  cache_embeddings: false

file_discovery:
  max_filesize: "500kb"
  
mcp:
  update_debounce_ms: 2000
  max_concurrent_searches: 3
```

#### Team Configuration

**Shared Team Standards**:
```yaml
# Team configuration - commit this file
model: "sentence-transformers/all-MiniLM-L12-v2"  # Team standard
cache_dir: "./.turboprop-cache"  # Project-local cache
max_filesize: "2mb"

file_discovery:
  include_patterns:
    - "src/**/*.{rs,py,js,ts}"
    - "docs/**/*.md"
  exclude_patterns:
    - "target/**"
    - "node_modules/**"
    - "**/*.generated.*"
    - "**/test-fixtures/**"

search:
  default_limit: 10
  similarity_threshold: 0.35

# Consistent performance settings
embedding:
  batch_size: 32
  cache_embeddings: true

mcp:
  enable_file_watching: true
  update_debounce_ms: 500
  max_concurrent_searches: 8
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TURBOPROP_CACHE_DIR` | Cache directory for models | `~/.turboprop-cache` |
| `TURBOPROP_LOG_LEVEL` | Log level (error, warn, info, debug) | `info` |
| `RUST_LOG` | Rust logging configuration | - |

## Advanced Features

### Model Selection

Choose the right model for your use case:

```bash
# Fast, lightweight (default)
tp mcp --repo . --model sentence-transformers/all-MiniLM-L6-v2

# Better accuracy
tp mcp --repo . --model sentence-transformers/all-MiniLM-L12-v2

# Code-specific model (when available)
tp mcp --repo . --model nomic-embed-code.Q5_K_S.gguf

# Multilingual support (when available)
tp mcp --repo . --model Qwen/Qwen3-Embedding-0.6B
```

### File Filtering

Control which files are indexed:

```bash
# Only Rust files
tp mcp --repo . --filter "**/*.rs"

# Multiple patterns in config file
# .turboprop.yml:
file_discovery:
  include_patterns:
    - "src/**/*.rs"
    - "lib/**/*.rs" 
    - "tests/**/*.rs"
  exclude_patterns:
    - "target/**"
    - "**/*_generated.rs"
```

### Performance Tuning

For large repositories:

```yaml
# .turboprop.yml
embedding:
  batch_size: 16  # Reduce for lower memory usage
  
file_discovery:
  max_filesize: "1mb"  # Skip very large files
  
mcp:
  update_debounce_ms: 1000  # Reduce update frequency
```

### Docker Environment Setup

Deploy TurboProp MCP server in Docker for consistent team development or production use.

#### Basic Docker Setup

**Dockerfile** for TurboProp MCP server:

```dockerfile
FROM rust:1.75 as builder

# Install TurboProp
RUN cargo install turboprop --version latest

FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy TurboProp binary from builder stage
COPY --from=builder /usr/local/cargo/bin/tp /usr/local/bin/tp

# Set working directory
WORKDIR /workspace

# Default command runs MCP server
CMD ["tp", "mcp", "--repo", "/workspace", "--verbose"]
```

**Build and run**:

```bash
# Build the image
docker build -t turboprop-mcp .

# Run with your code mounted
docker run -it --rm \
  -v "$(pwd):/workspace" \
  -p 3000:3000 \
  turboprop-mcp
```

#### Docker Compose Setup

**docker-compose.yml** for development environment:

```yaml
version: '3.8'
services:
  turboprop-mcp:
    build: .
    volumes:
      - .:/workspace:ro
      - turboprop-cache:/root/.turboprop-cache
    environment:
      - RUST_LOG=info
      - TURBOPROP_CACHE_DIR=/root/.turboprop-cache
    command: tp mcp --repo /workspace --max-filesize 5mb --verbose
    restart: unless-stopped
    
  # Optional: Development agent that uses the MCP server
  dev-agent:
    image: cursor/dev:latest
    depends_on:
      - turboprop-mcp
    volumes:
      - .:/workspace
    environment:
      - MCP_SERVER_URL=http://turboprop-mcp:3000
      
volumes:
  turboprop-cache:
```

**Start the environment**:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f turboprop-mcp

# Stop services
docker-compose down
```

#### Production Docker Setup

**Multi-stage production Dockerfile**:

```dockerfile
FROM rust:1.75-alpine as builder

# Install dependencies for building
RUN apk add --no-cache musl-dev git

# Install TurboProp
RUN cargo install turboprop --version latest

FROM alpine:latest

# Install runtime dependencies
RUN apk add --no-cache git ca-certificates

# Create non-root user for security
RUN addgroup -g 1000 turboprop && \
    adduser -D -s /bin/sh -u 1000 -G turboprop turboprop

# Copy binary and set ownership
COPY --from=builder /usr/local/cargo/bin/tp /usr/local/bin/tp
RUN chown turboprop:turboprop /usr/local/bin/tp

# Switch to non-root user
USER turboprop
WORKDIR /workspace

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD tp --version || exit 1

# Expose port for HTTP API (if enabled)
EXPOSE 3000

# Run MCP server
ENTRYPOINT ["tp", "mcp"]
CMD ["--repo", "/workspace", "--verbose"]
```

#### Kubernetes Deployment

**turboprop-mcp-deployment.yaml**:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: turboprop-mcp
spec:
  replicas: 1
  selector:
    matchLabels:
      app: turboprop-mcp
  template:
    metadata:
      labels:
        app: turboprop-mcp
    spec:
      containers:
      - name: turboprop-mcp
        image: your-registry/turboprop-mcp:latest
        ports:
        - containerPort: 3000
        env:
        - name: RUST_LOG
          value: "info"
        - name: TURBOPROP_CACHE_DIR
          value: "/cache"
        volumeMounts:
        - name: source-code
          mountPath: /workspace
          readOnly: true
        - name: cache-volume
          mountPath: /cache
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          exec:
            command:
            - tp
            - --version
          initialDelaySeconds: 30
          periodSeconds: 30
      volumes:
      - name: source-code
        persistentVolumeClaim:
          claimName: source-code-pvc
      - name: cache-volume
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: turboprop-mcp-service
spec:
  selector:
    app: turboprop-mcp
  ports:
  - protocol: TCP
    port: 3000
    targetPort: 3000
```

**Deploy to Kubernetes**:

```bash
# Apply the deployment
kubectl apply -f turboprop-mcp-deployment.yaml

# Check status
kubectl get pods -l app=turboprop-mcp

# View logs
kubectl logs -l app=turboprop-mcp -f
```

#### Docker Environment Tips

**Performance Optimization**:

```bash
# Use bind mounts for better I/O performance
docker run -it --rm \
  --mount type=bind,source="$(pwd)",target=/workspace,readonly \
  --mount type=volume,source=turboprop-cache,target=/root/.turboprop-cache \
  turboprop-mcp

# Allocate more memory for large repositories
docker run -it --rm \
  --memory=2g \
  --cpus=2 \
  -v "$(pwd):/workspace" \
  turboprop-mcp tp mcp --repo /workspace --max-filesize 10mb
```

**Development Workflow**:

```bash
# Create a development alias
alias turboprop-dev='docker run -it --rm -v "$(pwd):/workspace" turboprop-mcp'

# Use in any project
cd /path/to/your/project
turboprop-dev tp mcp --repo /workspace
```

**Multi-Project Setup**:

```bash
# Run multiple instances for different projects
docker run -d --name project1-mcp \
  -v "/path/to/project1:/workspace" \
  turboprop-mcp

docker run -d --name project2-mcp \
  -v "/path/to/project2:/workspace" \
  turboprop-mcp tp mcp --repo /workspace --model all-MiniLM-L12-v2
```

### CI/CD Pipeline Integration

Integrate TurboProp indexing and search capabilities into your continuous integration workflows.

#### GitHub Actions

**Basic CI Pipeline** (`.github/workflows/turboprop-ci.yml`):

```yaml
name: TurboProp CI Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  turboprop-analysis:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Full history for better analysis
        
    - name: Install TurboProp
      run: |
        curl -L https://github.com/your-org/turboprop/releases/latest/download/turboprop-linux-x64.tar.gz | tar xz
        sudo mv tp /usr/local/bin/
        tp --version
        
    - name: Cache TurboProp embeddings
      uses: actions/cache@v3
      with:
        path: ~/.turboprop-cache
        key: turboprop-${{ runner.os }}-${{ hashFiles('**/*.rs', '**/*.py', '**/*.js') }}
        restore-keys: |
          turboprop-${{ runner.os }}-
          
    - name: Build semantic index
      run: |
        tp index --repo . --max-filesize 5mb --verbose
        
    - name: Run code analysis
      run: |
        # Search for common code quality issues
        tp search "TODO" --limit 50 > analysis/todos.txt
        tp search "FIXME" --limit 50 > analysis/fixmes.txt
        tp search "deprecated" --limit 20 > analysis/deprecated.txt
        
        # Find security-related code
        tp search "password" --limit 10 > analysis/security-review.txt
        tp search "secret key" --limit 10 >> analysis/security-review.txt
        
    - name: Generate code insights
      run: |
        # Find test coverage gaps
        tp search "untested" --limit 30 > analysis/test-gaps.txt
        
        # Analyze error handling patterns
        tp search "error handling" --limit 20 > analysis/error-patterns.txt
        
    - name: Upload analysis artifacts
      uses: actions/upload-artifact@v3
      with:
        name: turboprop-analysis
        path: analysis/
        retention-days: 30
        
    - name: Comment on PR (if applicable)
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const todoCount = fs.readFileSync('analysis/todos.txt', 'utf8').split('\n').length - 1;
          const fixmeCount = fs.readFileSync('analysis/fixmes.txt', 'utf8').split('\n').length - 1;
          
          const comment = `## 🔍 TurboProp Code Analysis
          
          - **TODO items**: ${todoCount}
          - **FIXME items**: ${fixmeCount}
          
          [View detailed analysis artifacts](https://github.com/${{ github.repository }}/actions/runs/${{ github.run_id }})`;
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: comment
          });
```

**Advanced GitHub Actions** with semantic diff analysis:

```yaml
name: Semantic Code Change Analysis

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  semantic-diff:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout PR head
      uses: actions/checkout@v4
      with:
        ref: ${{ github.event.pull_request.head.sha }}
        
    - name: Checkout PR base
      uses: actions/checkout@v4
      with:
        ref: ${{ github.event.pull_request.base.sha }}
        path: base-branch
        
    - name: Install TurboProp
      run: |
        wget https://github.com/your-org/turboprop/releases/latest/download/turboprop-linux-x64.tar.gz
        tar -xzf turboprop-linux-x64.tar.gz
        sudo mv tp /usr/local/bin/
        
    - name: Index both versions
      run: |
        # Index current PR
        tp index --repo . --output-dir .turboprop-head
        
        # Index base branch
        cd base-branch
        tp index --repo . --output-dir ../.turboprop-base
        cd ..
        
    - name: Semantic change detection
      run: |
        # Find functions that changed semantically
        tp search "function" --repo . --index-dir .turboprop-head > head-functions.txt
        tp search "function" --repo base-branch --index-dir .turboprop-base > base-functions.txt
        
        # Compare and generate insights
        python scripts/compare-semantic-changes.py head-functions.txt base-functions.txt > semantic-diff-report.md
        
    - name: Post semantic diff report
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const report = fs.readFileSync('semantic-diff-report.md', 'utf8');
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: `## 🧠 Semantic Change Analysis\n\n${report}`
          });
```

#### GitLab CI

**GitLab CI Pipeline** (`.gitlab-ci.yml`):

```yaml
stages:
  - setup
  - analysis
  - report

variables:
  TURBOPROP_CACHE_DIR: ".turboprop-cache"

cache:
  paths:
    - .turboprop-cache/
    - .turboprop/

install-turboprop:
  stage: setup
  image: ubuntu:22.04
  before_script:
    - apt-get update && apt-get install -y curl wget
  script:
    - curl -L https://github.com/your-org/turboprop/releases/latest/download/turboprop-linux-x64.tar.gz | tar xz
    - mv tp /usr/local/bin/
    - tp --version
  artifacts:
    paths:
      - /usr/local/bin/tp
    expire_in: 1 hour

semantic-analysis:
  stage: analysis
  image: ubuntu:22.04
  dependencies:
    - install-turboprop
  script:
    - tp index --repo . --verbose
    
    # Code quality analysis
    - mkdir -p reports
    - tp search "TODO" --limit 100 --output-format json > reports/todos.json
    - tp search "FIXME" --limit 50 --output-format json > reports/fixmes.json
    - tp search "hack" --limit 20 --output-format json > reports/hacks.json
    
    # Security analysis
    - tp search "password" --limit 10 --output-format json > reports/security.json
    - tp search "hardcoded" --limit 15 --output-format json >> reports/security.json
    
    # Architecture analysis
    - tp search "dependency injection" --limit 30 > reports/architecture.txt
    - tp search "design pattern" --limit 25 > reports/patterns.txt
    
  artifacts:
    paths:
      - reports/
    expire_in: 1 week
    reports:
      junit: reports/turboprop-junit.xml

merge-request-analysis:
  stage: analysis
  image: ubuntu:22.04
  dependencies:
    - install-turboprop
  only:
    - merge_requests
  script:
    # Analyze only changed files
    - git diff --name-only $CI_MERGE_REQUEST_TARGET_BRANCH_SHA HEAD > changed-files.txt
    
    # Build focused index on changed files
    - tp index --repo . --filter-file changed-files.txt --verbose
    
    # Analyze changes
    - tp search "new feature" --limit 10 > reports/new-features.txt
    - tp search "breaking change" --limit 5 > reports/breaking-changes.txt
    - tp search "performance" --limit 15 > reports/performance-impact.txt
    
  artifacts:
    paths:
      - reports/
    expire_in: 3 days

generate-report:
  stage: report
  image: python:3.9
  dependencies:
    - semantic-analysis
  script:
    - pip install jinja2 json2html
    - python scripts/generate-turboprop-report.py reports/ > turboprop-report.html
    
  artifacts:
    paths:
      - turboprop-report.html
    expire_in: 1 month
  only:
    - main
    - develop
```

#### Jenkins Pipeline

**Jenkinsfile** for TurboProp integration:

```groovy
pipeline {
    agent any
    
    environment {
        TURBOPROP_CACHE_DIR = "${WORKSPACE}/.turboprop-cache"
    }
    
    stages {
        stage('Setup') {
            steps {
                script {
                    // Install TurboProp
                    sh '''
                        if ! command -v tp &> /dev/null; then
                            curl -L https://github.com/your-org/turboprop/releases/latest/download/turboprop-linux-x64.tar.gz | tar xz
                            sudo mv tp /usr/local/bin/
                        fi
                        tp --version
                    '''
                }
            }
        }
        
        stage('Semantic Indexing') {
            steps {
                // Cache embeddings between builds
                cache(maxCacheSize: 250, caches: [
                    arbitraryFileCache(
                        path: '.turboprop-cache',
                        includes: '**/*',
                        fingerprinting: true
                    )
                ]) {
                    sh 'tp index --repo . --max-filesize 5mb --verbose'
                }
            }
        }
        
        stage('Code Analysis') {
            parallel {
                stage('Quality Analysis') {
                    steps {
                        sh '''
                            mkdir -p analysis
                            tp search "TODO" --limit 100 > analysis/todos.txt
                            tp search "FIXME" --limit 50 > analysis/fixmes.txt
                            tp search "deprecated" --limit 25 > analysis/deprecated.txt
                        '''
                    }
                }
                
                stage('Security Scan') {
                    steps {
                        sh '''
                            mkdir -p analysis
                            tp search "password" --limit 10 > analysis/security-concerns.txt
                            tp search "secret" --limit 10 >> analysis/security-concerns.txt
                            tp search "hardcoded" --limit 15 >> analysis/security-concerns.txt
                        '''
                    }
                }
                
                stage('Architecture Review') {
                    steps {
                        sh '''
                            mkdir -p analysis
                            tp search "architecture" --limit 20 > analysis/architecture.txt
                            tp search "design decision" --limit 15 > analysis/design-decisions.txt
                            tp search "technical debt" --limit 30 > analysis/tech-debt.txt
                        '''
                    }
                }
            }
        }
        
        stage('Generate Reports') {
            steps {
                script {
                    // Generate summary report
                    sh '''
                        python3 << EOF
import os
import json

def count_lines(filename):
    try:
        with open(filename, 'r') as f:
            return len([line for line in f if line.strip()])
    except:
        return 0

report = {
    "todos": count_lines("analysis/todos.txt"),
    "fixmes": count_lines("analysis/fixmes.txt"),
    "deprecated": count_lines("analysis/deprecated.txt"),
    "security_concerns": count_lines("analysis/security-concerns.txt"),
    "tech_debt": count_lines("analysis/tech-debt.txt")
}

with open("analysis/summary.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"Analysis Summary:")
print(f"- TODOs: {report['todos']}")
print(f"- FIXMEs: {report['fixmes']}")  
print(f"- Deprecated: {report['deprecated']}")
print(f"- Security concerns: {report['security_concerns']}")
print(f"- Technical debt items: {report['tech_debt']}")
EOF
                    '''
                }
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'analysis/**/*', fingerprint: true
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: 'analysis',
                reportFiles: '*.txt,*.json',
                reportName: 'TurboProp Analysis Report'
            ])
        }
        
        success {
            echo 'TurboProp analysis completed successfully!'
        }
        
        failure {
            emailext (
                subject: "TurboProp Analysis Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "The TurboProp semantic analysis failed. Check ${env.BUILD_URL} for details.",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
```

#### Azure DevOps Pipeline

**azure-pipelines.yml**:

```yaml
trigger:
  branches:
    include:
      - main
      - develop
      - feature/*

pool:
  vmImage: 'ubuntu-latest'

variables:
  TURBOPROP_CACHE_DIR: '$(Pipeline.Workspace)/.turboprop-cache'

stages:
- stage: Setup
  displayName: 'Setup TurboProp'
  jobs:
  - job: InstallTurboProp
    displayName: 'Install TurboProp'
    steps:
    - script: |
        curl -L https://github.com/your-org/turboprop/releases/latest/download/turboprop-linux-x64.tar.gz | tar xz
        sudo mv tp /usr/local/bin/
        tp --version
      displayName: 'Install TurboProp binary'
      
    - task: Cache@2
      inputs:
        key: 'turboprop | "$(Agent.OS)" | $(Build.SourcesDirectory)/**/*.rs'
        restoreKeys: |
          turboprop | "$(Agent.OS)"
        path: $(TURBOPROP_CACHE_DIR)
      displayName: 'Cache TurboProp embeddings'

- stage: Analysis
  displayName: 'Semantic Code Analysis'
  dependsOn: Setup
  jobs:
  - job: SemanticAnalysis
    displayName: 'Run Semantic Analysis'
    steps:
    - script: |
        tp index --repo . --max-filesize 5mb --verbose
      displayName: 'Build semantic index'
      
    - script: |
        mkdir -p $(Build.ArtifactStagingDirectory)/analysis
        
        # Code quality analysis
        tp search "TODO" --limit 100 > $(Build.ArtifactStagingDirectory)/analysis/todos.txt
        tp search "FIXME" --limit 50 > $(Build.ArtifactStagingDirectory)/analysis/fixmes.txt
        tp search "hack" --limit 25 > $(Build.ArtifactStagingDirectory)/analysis/hacks.txt
        
        # Security analysis
        tp search "password" --limit 10 > $(Build.ArtifactStagingDirectory)/analysis/security.txt
        tp search "secret" --limit 10 >> $(Build.ArtifactStagingDirectory)/analysis/security.txt
        
        # Generate summary
        echo "# TurboProp Analysis Summary" > $(Build.ArtifactStagingDirectory)/analysis/README.md
        echo "Generated on: $(date)" >> $(Build.ArtifactStagingDirectory)/analysis/README.md
        echo "" >> $(Build.ArtifactStagingDirectory)/analysis/README.md
        echo "## Statistics" >> $(Build.ArtifactStagingDirectory)/analysis/README.md
        echo "- TODOs: $(wc -l < $(Build.ArtifactStagingDirectory)/analysis/todos.txt)" >> $(Build.ArtifactStagingDirectory)/analysis/README.md
        echo "- FIXMEs: $(wc -l < $(Build.ArtifactStagingDirectory)/analysis/fixmes.txt)" >> $(Build.ArtifactStagingDirectory)/analysis/README.md
        echo "- Security concerns: $(wc -l < $(Build.ArtifactStagingDirectory)/analysis/security.txt)" >> $(Build.ArtifactStagingDirectory)/analysis/README.md
      displayName: 'Run semantic code analysis'
      
    - task: PublishBuildArtifacts@1
      inputs:
        pathToPublish: '$(Build.ArtifactStagingDirectory)/analysis'
        artifactName: 'turboprop-analysis'
        publishLocation: 'Container'
      displayName: 'Publish analysis artifacts'

- stage: Report
  displayName: 'Generate Reports'
  dependsOn: Analysis
  condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
  jobs:
  - job: GenerateReport
    displayName: 'Generate HTML Report'
    steps:
    - task: DownloadBuildArtifacts@0
      inputs:
        buildType: 'current'
        downloadType: 'single'
        artifactName: 'turboprop-analysis'
        downloadPath: '$(System.ArtifactsDirectory)'
        
    - script: |
        python3 -c "
import os
import glob

analysis_dir = '$(System.ArtifactsDirectory)/turboprop-analysis'
html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>TurboProp Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .section { margin: 20px 0; }
        .count { font-weight: bold; color: #d73a49; }
        pre { background: #f6f8fa; padding: 15px; border-radius: 6px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>🔍 TurboProp Semantic Analysis Report</h1>
    <p>Build: $(Build.BuildNumber) | Date: $(date)</p>
'''

for filename in glob.glob(f'{analysis_dir}/*.txt'):
    name = os.path.basename(filename).replace('.txt', '').replace('-', ' ').title()
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        count = len(lines)
        
    html_content += f'''
    <div class='section'>
        <h2>{name} <span class='count'>({count} items)</span></h2>
        <details>
            <summary>View details</summary>
            <pre>{''.join(lines[:20])}</pre>
        </details>
    </div>
'''

html_content += '''
</body>
</html>
'''

with open('$(Build.ArtifactStagingDirectory)/turboprop-report.html', 'w') as f:
    f.write(html_content)
        "
      displayName: 'Generate HTML report'
      
    - task: PublishBuildArtifacts@1
      inputs:
        pathToPublish: '$(Build.ArtifactStagingDirectory)/turboprop-report.html'
        artifactName: 'turboprop-html-report'
        publishLocation: 'Container'
      displayName: 'Publish HTML report'
```

#### Integration Tips

**Best Practices for CI/CD Integration**:

1. **Caching Strategy**:
   ```bash
   # Cache embeddings based on code content hash
   cache_key="turboprop-$(find . -name '*.rs' -o -name '*.py' | xargs sha256sum | sha256sum | cut -d' ' -f1)"
   ```

2. **Incremental Analysis**:
   ```bash
   # Only analyze changed files in PRs
   git diff --name-only origin/main..HEAD | xargs tp index --files-from-stdin
   ```

3. **Parallel Processing**:
   ```bash
   # Split large repositories for parallel analysis
   tp index --repo . --worker-threads 4 --batch-size 16
   ```

4. **Quality Gates**:
   ```bash
   # Fail builds based on analysis results
   TODO_COUNT=$(tp search "TODO" --count-only)
   if [ "$TODO_COUNT" -gt 100 ]; then
     echo "Too many TODOs ($TODO_COUNT). Please address some before merging."
     exit 1
   fi
   ```

## Troubleshooting

### Quick Diagnostics (5-minute health check)

Run these commands to quickly identify issues:

```bash
# 1. Check if TurboProp is installed and accessible
tp --version

# 2. Test basic functionality
tp mcp --repo . --verbose --debug 2>debug.log &
MCP_PID=$!

# 3. Wait a moment, then check if server is running
sleep 5
ps aux | grep "tp mcp"

# 4. Stop the test server
kill $MCP_PID

# 5. Check the debug log for errors
grep -i error debug.log
```

### Troubleshooting by Symptom

#### 🚫 "My agent says MCP server not available"

**What you're seeing**:
- Agent reports "MCP server not available"
- "Connection refused" errors
- "Server timeout" messages
- Agent can't find TurboProp tool

**Diagnostic steps**:

1. **Verify server is running**:
   ```bash
   ps aux | grep "tp mcp"
   ```
   - **If nothing found**: Server isn't running, start it with `tp mcp --repo .`
   - **If found**: Server is running, check agent configuration

2. **Test server manually**:
   ```bash
   tp mcp --repo . --verbose
   ```
   - **Look for**: "MCP server ready and listening on stdio..."
   - **If error**: Fix the server startup issue first

3. **Check agent configuration**:
   - **JSON syntax**: Validate with a JSON validator
   - **Command path**: Use absolute path if needed: `"/usr/local/bin/tp"`
   - **Working directory**: Make sure `"cwd": "."` is set if needed

4. **Verify PATH**:
   ```bash
   which tp
   echo $PATH
   ```
   - **If not found**: Add TurboProp installation directory to PATH

#### 🔍 "Search returns no results" 

**What you're seeing**:
- Agent searches return empty results
- "No matches found" even for known code
- Search works but results seem irrelevant

**Diagnostic steps**:

1. **Wait for indexing to complete**:
   ```bash
   tp mcp --repo . --verbose
   ```
   - **Look for**: "Index initialization completed" message
   - **If indexing**: Wait for completion (can take 1-5 minutes)

2. **Test with known content**:
   - **Search for exact strings** that exist in your code
   - **Try broader queries**: "function" instead of "specific function name"
   - **Check different file types**: Search in files you know exist

3. **Check file filters**:
   ```bash
   tp mcp --repo . --debug 2>debug.log
   # Check debug.log for file discovery logs
   grep -i "discovered.*files" debug.log
   ```
   - **If few files discovered**: Check include/exclude patterns in `.turboprop.yml`
   - **Try without filters**: `tp mcp --repo . --force-rebuild`

4. **Verify search parameters**:
   - **Lower similarity threshold**: Try `threshold: 0.2` in config
   - **Increase result limit**: Ask agent to search with higher limits
   - **Different model**: Try `sentence-transformers/all-MiniLM-L12-v2`

#### 💾 "Server uses too much memory/CPU"

**What you're seeing**:
- Server uses > 2GB RAM
- High CPU usage (> 80%)
- System becomes slow or unresponsive
- Out of memory errors

**Diagnostic steps**:

1. **Monitor resource usage**:
   ```bash
   # Check current memory usage
   ps aux | grep "tp mcp"
   
   # Monitor over time
   while true; do
     ps -o pid,rss,vsz,pcpu,comm -p $(pgrep "tp")
     sleep 5
   done
   ```

2. **Reduce batch size** (most effective):
   ```yaml
   # .turboprop.yml
   embedding:
     batch_size: 8  # Default is 32, try 8 or 16
     worker_threads: 2  # Reduce if needed
   ```

3. **Limit file size and count**:
   ```bash
   tp mcp --repo . --max-filesize 500kb --filter "src/**/*.rs"
   ```

4. **Use restrictive patterns**:
   ```yaml
   file_discovery:
     include_patterns:
       - "src/**/*.{rs,py,js,ts}"  # Only source files
     exclude_patterns:
       - "target/**"    # Build artifacts
       - "node_modules/**"  # Dependencies
       - "**/*.log"     # Log files
       - "**/*.min.*"   # Minified files
   ```

#### ⚡ "Search is very slow"

**What you're seeing**:
- Search takes > 5 seconds to respond
- Agent seems to "hang" during searches
- High CPU usage during searches

**Diagnostic steps**:

1. **Use smaller/faster model**:
   ```bash
   tp mcp --repo . --model sentence-transformers/all-MiniLM-L6-v2
   ```

2. **Limit concurrent searches**:
   ```yaml
   # .turboprop.yml
   mcp:
     max_concurrent_searches: 3  # Default is 10
   ```

3. **Optimize queries** (advise your agent):
   - **Use specific terms**: "JWT authentication" not just "auth"
   - **Set result limits**: Ask for 5 results instead of 10+
   - **Filter by file type**: Only search `.rs` files when looking for Rust code

4. **Check system resources**:
   ```bash
   # Ensure adequate resources
   free -h  # Memory
   df -h    # Disk space
   ```

#### 📁 "Index doesn't update when I change files"

**What you're seeing**:
- Changes to code don't appear in search results
- Agent finds old versions of code
- File watching seems broken

**Diagnostic steps**:

1. **Check file watching logs**:
   ```bash
   tp mcp --repo . --debug 2>&1 | grep -i watch
   ```
   - **Look for**: "File change detected" messages
   - **If none**: File watching may be disabled or failing

2. **Test file changes**:
   - Make a simple change to a tracked file
   - Check if server logs show the change
   - Wait 1-2 seconds for index update

3. **Verify file patterns**:
   - Ensure changed files match include patterns
   - Check that files aren't excluded by exclude patterns
   - Try with minimal filters first

4. **Manual index rebuild**:
   ```bash
   tp mcp --repo . --force-rebuild
   ```

#### 🔧 "Server won't start at all"

**What you're seeing**:
- `tp mcp` command fails immediately
- Error messages about paths, permissions, or dependencies
- Server never reaches "ready" state

**Diagnostic steps**:

1. **Check repository path**:
   ```bash
   ls -la /path/to/project  # Verify path exists
   ls -ld /path/to/project  # Check permissions
   ```
   - **Use absolute paths** if relative paths fail
   - **Check permissions**: Ensure read access to directory

2. **Check file descriptor limits**:
   ```bash
   ulimit -n  # Should be > 1024
   ```
   - **If low**: Increase with `ulimit -n 4096`
   - **On macOS**: Check system limits with `launchctl limit maxfiles`

3. **Test with minimal configuration**:
   ```bash
   tp mcp --repo . --verbose  # Check detailed output
   ```
   - **Look for specific error messages**
   - **Try different repository**: Test with empty directory

4. **Check dependencies**:
   ```bash
   tp --version  # Verify installation
   rustc --version  # Check Rust installation if building from source
   ```

### Debug Mode

Enable detailed logging:

```bash
# Enable debug logging
tp mcp --repo . --debug

# Enable Rust debug logging
RUST_LOG=debug tp mcp --repo .

# Log to file (stderr only, stdout is used for MCP)
tp mcp --repo . --verbose 2> mcp-server.log
```

### Performance Issues

**Symptoms**:
- Slow search responses
- High CPU usage
- Frequent index updates

**Solutions**:
1. **Optimize file patterns**:
   ```yaml
   file_discovery:
     exclude_patterns:
       - "target/**"      # Build artifacts
       - "node_modules/**" # Dependencies
       - "*.log"          # Log files
       - "**/*.min.js"    # Minified files
   ```

2. **Adjust update settings**:
   ```yaml
   mcp:
     update_debounce_ms: 2000  # Wait longer before updates
     max_concurrent_searches: 5  # Limit concurrent operations
   ```

3. **Use appropriate model**:
   - Smaller models for faster responses
   - Larger models for better accuracy

### Getting Help

If you encounter issues not covered here:

1. **Check server logs** with `--verbose` or `--debug`
2. **Search existing issues** on GitHub
3. **Create a new issue** with:
   - TurboProp version (`tp --version`)
   - Operating system
   - Agent type and version
   - Configuration files
   - Server logs
   - Steps to reproduce

## Performance Optimization

### Repository Size Guidelines

| Repository Size | Recommended Settings |
|----------------|---------------------|
| Small (< 1k files) | Default settings |
| Medium (1k-10k files) | `batch_size: 16`, `max_filesize: "1mb"` |
| Large (10k+ files) | `batch_size: 8`, aggressive filtering, consider watchman |

### Memory Usage

Monitor memory usage and adjust:

```bash
# Monitor memory usage
ps aux | grep "tp mcp"

# Reduce memory usage
tp mcp --repo . --max-filesize 500kb --filter "src/**/*.rs"
```

### Search Performance

Optimize search performance:

1. **Set appropriate similarity thresholds**
2. **Use specific file type filters**
3. **Limit result counts**
4. **Use targeted queries**

Example optimized search parameters:
```json
{
  "query": "authentication JWT validation",
  "limit": 5,
  "threshold": 0.6,
  "filetype": ".rs",
  "filter": "src/**/*.rs"
}
```

## Best Practices

### 1. Repository Organization

- Keep code organized in clear directory structures
- Use meaningful file names and comments
- Avoid deeply nested directory structures

### 2. Configuration Management

- Use `.turboprop.yml` for project-specific settings
- Version control your configuration
- Document any special configuration requirements

### 3. Agent Integration

- Test integration with a simple query first
- Use descriptive search queries for better results
- Take advantage of file type and pattern filtering

### 4. Maintenance

- Regularly update TurboProp to get latest improvements
- Monitor server performance and adjust configuration as needed
- Keep an eye on disk usage for cache and index files

### 5. Team Usage

- Share configuration files with team members
- Document any special setup requirements
- Consider using consistent model choices across team

## FAQ

**Q: Can I run multiple MCP servers for different projects?**
A: Yes, each server operates independently. Start each with its own `--repo` path.

**Q: Does the server work with remote repositories?**
A: The server indexes local files only. Clone the repository locally first.

**Q: Can I exclude certain file types?**
A: Yes, use `exclude_patterns` in your configuration file.

**Q: How often does the index update?**
A: The server watches for file changes and updates incrementally, typically within 1-2 seconds.

**Q: What happens if I modify a file while the server is running?**
A: The server automatically detects changes and updates the relevant parts of the index.

**Q: Can I use custom models?**
A: Currently, TurboProp supports specific pre-trained models. Check the model documentation for available options.

**Q: Is there a limit to repository size?**
A: No hard limit, but performance depends on your system resources. Use filtering for very large repositories.

For more questions, see the [main TurboProp documentation](../README.md) or [file an issue](https://github.com/turboprop-org/turboprop/issues).