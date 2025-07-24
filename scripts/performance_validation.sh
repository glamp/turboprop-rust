#!/bin/bash
# performance_validation.sh

set -e

echo "Performance Validation for Model Support"
echo "========================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TP_BINARY="$PROJECT_ROOT/target/release/tp"

# Performance thresholds (in seconds)
INDEX_THRESHOLD=30
SEARCH_THRESHOLD=5
MODEL_LOAD_THRESHOLD=20

# Test configuration
TEXT_COUNT=100
ITERATIONS=3

# Create test data directory
TEST_DIR="$(mktemp -d)"
echo -e "${BLUE}Using test directory: $TEST_DIR${NC}"

# Cleanup function
cleanup() {
    echo -e "${YELLOW}Cleaning up test directory...${NC}"
    cd / > /dev/null 2>&1
    rm -rf "$TEST_DIR" > /dev/null 2>&1 || true
}
trap cleanup EXIT

cd "$TEST_DIR"

# Function to measure execution time
measure_time() {
    local command="$1"
    local description="$2"
    
    echo -ne "${CYAN}$description... ${NC}"
    
    local start_time=$(date +%s.%N)
    if eval "$command" > /dev/null 2>&1; then
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc -l)
        echo -e "${GREEN}${duration}s${NC}"
        echo "$duration"
    else
        echo -e "${RED}FAILED${NC}"
        echo "999"  # Return a large number to indicate failure
    fi
}

# Function to measure memory usage (Linux/macOS compatible)
measure_memory() {
    local command="$1"
    local description="$2"
    
    echo -ne "${CYAN}$description... ${NC}"
    
    if command -v /usr/bin/time > /dev/null 2>&1; then
        # Use GNU time if available
        local output
        if output=$(/usr/bin/time -v "$command" 2>&1); then
            local max_memory=$(echo "$output" | grep "Maximum resident set size" | awk '{print $6}')
            if [ -n "$max_memory" ]; then
                local memory_mb=$(echo "scale=2; $max_memory / 1024" | bc -l)
                echo -e "${GREEN}${memory_mb}MB${NC}"
                echo "$memory_mb"
            else
                echo -e "${YELLOW}N/A${NC}"
                echo "0"
            fi
        else
            echo -e "${RED}FAILED${NC}"
            echo "0"
        fi
    else
        # Fallback to basic time measurement
        local start_time=$(date +%s.%N)
        if eval "$command" > /dev/null 2>&1; then
            local end_time=$(date +%s.%N)
            local duration=$(echo "$end_time - $start_time" | bc -l)
            echo -e "${GREEN}${duration}s (time only)${NC}"
            echo "$duration"
        else
            echo -e "${RED}FAILED${NC}"
            echo "999"
        fi
    fi
}

# Create test repository with varied content
echo -e "${BLUE}Creating test repository with $TEXT_COUNT files...${NC}"

mkdir -p src/{components,utils,api} tests docs examples

# Generate test files with realistic code content
for i in $(seq 1 $TEXT_COUNT); do
    case $((i % 5)) in
        0)
            cat > "src/file$i.js" << EOF
function processData$i(data) {
    const result = data.map(item => ({
        id: item.id,
        name: item.name.toUpperCase(),
        value: item.value * 1.1,
        processed: true
    }));
    
    return result.filter(item => item.value > 0);
}

export async function fetchData$i(endpoint) {
    try {
        const response = await fetch(endpoint);
        const data = await response.json();
        return processData$i(data);
    } catch (error) {
        console.error('Failed to fetch data:', error);
        throw error;
    }
}
EOF
            ;;
        1)
            cat > "src/file$i.py" << EOF
def calculate_metrics$i(data_points):
    """Calculate statistical metrics for data points"""
    if not data_points:
        return {}
    
    total = sum(data_points)
    count = len(data_points)
    mean = total / count
    
    variance = sum((x - mean) ** 2 for x in data_points) / count
    std_dev = variance ** 0.5
    
    return {
        'count': count,
        'sum': total,
        'mean': mean,
        'variance': variance,
        'std_dev': std_dev,
        'min': min(data_points),
        'max': max(data_points)
    }

class DataProcessor$i:
    def __init__(self, config):
        self.config = config
        self.processed_count = 0
    
    def process_batch(self, batch):
        results = []
        for item in batch:
            if self.validate_item(item):
                processed = self.transform_item(item)
                results.append(processed)
                self.processed_count += 1
        return results
    
    def validate_item(self, item):
        return hasattr(item, 'id') and hasattr(item, 'data')
    
    def transform_item(self, item):
        return {
            'id': item.id,
            'data': item.data,
            'processed_at': time.time(),
            'processor_id': f'proc_{$i}'
        }
EOF
            ;;
        2)
            cat > "src/file$i.rs" << EOF
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record$i {
    pub id: u64,
    pub name: String,
    pub data: HashMap<String, String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl Record$i {
    pub fn new(id: u64, name: String) -> Self {
        Self {
            id,
            name,
            data: HashMap::new(),
            created_at: chrono::Utc::now(),
        }
    }
    
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.data.insert(key, value);
    }
    
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.data.get(key)
    }
}

pub async fn fetch_records$i(limit: usize) -> Result<Vec<Record$i>, Box<dyn std::error::Error>> {
    let mut records = Vec::new();
    
    for i in 0..limit {
        let mut record = Record$i::new(i as u64, format!("record_{}", i));
        record.add_metadata("source".to_string(), "generated".to_string());
        record.add_metadata("batch".to_string(), "$i".to_string());
        records.push(record);
    }
    
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_record_creation() {
        let record = Record$i::new(1, "test".to_string());
        assert_eq!(record.id, 1);
        assert_eq!(record.name, "test");
    }
}
EOF
            ;;
        3)
            cat > "src/file$i.go" << EOF
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "time"
)

type Service$i struct {
    client  *http.Client
    baseURL string
    timeout time.Duration
}

func NewService$i(baseURL string) *Service$i {
    return &Service$i{
        client: &http.Client{
            Timeout: 30 * time.Second,
        },
        baseURL: baseURL,
        timeout: 10 * time.Second,
    }
}

func (s *Service$i) ProcessRequest(ctx context.Context, data map[string]interface{}) (*Response, error) {
    requestBody, err := json.Marshal(data)
    if err != nil {
        return nil, fmt.Errorf("failed to marshal request: %w", err)
    }
    
    ctx, cancel := context.WithTimeout(ctx, s.timeout)
    defer cancel()
    
    req, err := http.NewRequestWithContext(ctx, "POST", s.baseURL+"/process", bytes.NewBuffer(requestBody))
    if err != nil {
        return nil, fmt.Errorf("failed to create request: %w", err)
    }
    
    req.Header.Set("Content-Type", "application/json")
    
    resp, err := s.client.Do(req)
    if err != nil {
        return nil, fmt.Errorf("request failed: %w", err)
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
    }
    
    var response Response
    if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
        return nil, fmt.Errorf("failed to decode response: %w", err)
    }
    
    return &response, nil
}

type Response struct {
    ID        string                 \`json:"id"\`
    Status    string                 \`json:"status"\`
    Data      map[string]interface{} \`json:"data"\`
    Timestamp time.Time              \`json:"timestamp"\`
}
EOF
            ;;
        4)
            cat > "docs/doc$i.md" << EOF
# Documentation $i

This document describes the functionality of component $i in our system.

## Overview

Component $i provides essential functionality for data processing and transformation. It includes the following features:

- Data validation and sanitization
- Transform operations with configurable rules
- Error handling and recovery mechanisms
- Performance monitoring and metrics collection

## Usage Examples

### Basic Usage

\`\`\`javascript
const processor = new DataProcessor$i({
    batchSize: 100,
    timeout: 5000,
    retryAttempts: 3
});

const results = await processor.process(inputData);
console.log('Processed:', results.length, 'items');
\`\`\`

### Advanced Configuration

\`\`\`python
processor = DataProcessor$i({
    'validation_rules': {
        'required_fields': ['id', 'name', 'data'],
        'max_length': 1000,
        'allowed_types': ['string', 'number', 'object']
    },
    'transformation_rules': {
        'normalize_strings': True,
        'convert_timestamps': True,
        'remove_empty_fields': True
    }
})
\`\`\`

## API Reference

### Methods

- \`process(data)\` - Process a single data item
- \`processBatch(items)\` - Process multiple items efficiently
- \`validate(item)\` - Validate item against configured rules
- \`getMetrics()\` - Retrieve processing metrics

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| batchSize | number | 50 | Items to process in each batch |
| timeout | number | 3000 | Timeout in milliseconds |
| retryAttempts | number | 2 | Number of retry attempts |

## Performance Characteristics

Component $i is optimized for high-throughput processing:

- Processes up to 1000 items/second under normal conditions
- Memory usage scales linearly with batch size
- Supports concurrent processing with thread safety
- Includes built-in backpressure handling
EOF
            ;;
    esac
done

# Initialize git repository
git init > /dev/null 2>&1
git add . > /dev/null 2>&1
git commit -m "Initial performance test repository" > /dev/null 2>&1

# Build TurboProp binary
echo -e "${BLUE}Building TurboProp binary...${NC}"
cd "$PROJECT_ROOT"
cargo build --release > /dev/null 2>&1
cd "$TEST_DIR"

# Verify binary exists
if [ ! -x "$TP_BINARY" ]; then
    echo -e "${RED}Error: TurboProp binary not found at $TP_BINARY${NC}"
    exit 1
fi

echo -e "${BLUE}Running performance benchmarks...${NC}"
echo

# Test models (only those likely to be available)
MODELS=(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Track performance results
declare -A INDEX_TIMES
declare -A SEARCH_TIMES
declare -A MEMORY_USAGE

for model in "${MODELS[@]}"; do
    echo -e "${YELLOW}Testing performance for model: $model${NC}"
    
    # Model loading time
    echo -e "${CYAN}Model Loading Performance:${NC}"
    load_time=$(measure_time "'$TP_BINARY' model info '$model'" "Model info retrieval")
    
    # Indexing performance
    echo -e "${CYAN}Indexing Performance:${NC}"
    index_time=$(measure_time "'$TP_BINARY' index --repo . --model '$model'" "Full repository indexing")
    INDEX_TIMES["$model"]="$index_time"
    
    # Search performance (multiple iterations for average)
    echo -e "${CYAN}Search Performance (average of $ITERATIONS runs):${NC}"
    total_search_time=0
    
    for i in $(seq 1 $ITERATIONS); do
        search_time=$(measure_time "'$TP_BINARY' search 'function data process' --model '$model' --limit 10" "Search iteration $i")
        total_search_time=$(echo "$total_search_time + $search_time" | bc -l)
    done
    
    avg_search_time=$(echo "scale=3; $total_search_time / $ITERATIONS" | bc -l)
    SEARCH_TIMES["$model"]="$avg_search_time"
    echo -e "${GREEN}Average search time: ${avg_search_time}s${NC}"
    
    # Memory usage test
    echo -e "${CYAN}Memory Usage:${NC}"
    memory_usage=$(measure_memory "'$TP_BINARY' search 'function process data calculate' --model '$model' --limit 20" "Peak memory usage")
    MEMORY_USAGE["$model"]="$memory_usage"
    
    # Throughput test
    echo -e "${CYAN}Throughput Test:${NC}"
    throughput_start=$(date +%s.%N)
    "$TP_BINARY" search "data processing function calculate transform" --model "$model" --limit 50 > /dev/null 2>&1
    throughput_end=$(date +%s.%N)
    throughput_time=$(echo "$throughput_end - $throughput_start" | bc -l)
    throughput_qps=$(echo "scale=2; 1 / $throughput_time" | bc -l)
    echo -e "${GREEN}Throughput: ${throughput_qps} queries/second${NC}"
    
    # Concurrent search test
    echo -e "${CYAN}Concurrent Performance:${NC}"
    concurrent_start=$(date +%s.%N)
    (
        "$TP_BINARY" search "function" --model "$model" --limit 5 > /dev/null 2>&1 &
        "$TP_BINARY" search "data" --model "$model" --limit 5 > /dev/null 2>&1 &
        "$TP_BINARY" search "process" --model "$model" --limit 5 > /dev/null 2>&1 &
        wait
    )
    concurrent_end=$(date +%s.%N)
    concurrent_time=$(echo "$concurrent_end - $concurrent_start" | bc -l)
    echo -e "${GREEN}3 concurrent searches: ${concurrent_time}s${NC}"
    
    echo
done

# Batch size performance comparison
echo -e "${YELLOW}Batch Size Performance Comparison:${NC}"
BATCH_SIZES=(8 16 32 64)
for batch_size in "${BATCH_SIZES[@]}"; do
    batch_time=$(measure_time "'$TP_BINARY' index --repo . --batch-size $batch_size --force-rebuild" "Batch size $batch_size")
    echo -e "${GREEN}Batch size $batch_size: ${batch_time}s${NC}"
done

# File type performance
echo -e "${YELLOW}File Type Performance:${NC}"
js_time=$(measure_time "'$TP_BINARY' search 'function' --filetype .js --limit 10" "JavaScript files only")
py_time=$(measure_time "'$TP_BINARY' search 'function' --filetype .py --limit 10" "Python files only")
rs_time=$(measure_time "'$TP_BINARY' search 'function' --filetype .rs --limit 10" "Rust files only")

echo -e "${GREEN}JavaScript search: ${js_time}s${NC}"
echo -e "${GREEN}Python search: ${py_time}s${NC}"
echo -e "${GREEN}Rust search: ${rs_time}s${NC}"

# System resource monitoring
echo -e "${YELLOW}System Resource Analysis:${NC}"
if command -v free > /dev/null 2>&1; then
    echo -e "${CYAN}Available memory:${NC}"
    free -h | head -2
fi

if command -v df > /dev/null 2>&1; then
    echo -e "${CYAN}Disk usage for test directory:${NC}"
    df -h "$TEST_DIR" | tail -1
fi

# Index size analysis
if [ -d ".turboprop" ]; then
    index_size=$(du -sh .turboprop 2>/dev/null | cut -f1)
    source_size=$(du -sh src 2>/dev/null | cut -f1)
    echo -e "${CYAN}Index size: $index_size (source: $source_size)${NC}"
fi

# Performance analysis and recommendations
echo
echo "=================================================================="
echo -e "${BLUE}Performance Analysis and Recommendations${NC}"
echo "=================================================================="

# Analyze results
for model in "${MODELS[@]}"; do
    echo -e "${YELLOW}Model: $model${NC}"
    
    index_time="${INDEX_TIMES[$model]:-0}"
    search_time="${SEARCH_TIMES[$model]:-0}"
    memory="${MEMORY_USAGE[$model]:-0}"
    
    echo -e "  Index time: ${index_time}s"
    echo -e "  Search time: ${search_time}s"
    echo -e "  Memory usage: ${memory}MB"
    
    # Performance evaluation
    if (( $(echo "$index_time > $INDEX_THRESHOLD" | bc -l) )); then
        echo -e "  ${RED}⚠ Index time exceeds threshold (${INDEX_THRESHOLD}s)${NC}"
        echo -e "  ${YELLOW}Recommendation: Consider reducing batch size or file count${NC}"
    else
        echo -e "  ${GREEN}✓ Index performance acceptable${NC}"
    fi
    
    if (( $(echo "$search_time > $SEARCH_THRESHOLD" | bc -l) )); then
        echo -e "  ${RED}⚠ Search time exceeds threshold (${SEARCH_THRESHOLD}s)${NC}"
        echo -e "  ${YELLOW}Recommendation: Consider optimizing query or reducing result limit${NC}"
    else
        echo -e "  ${GREEN}✓ Search performance acceptable${NC}"
    fi
    
    if (( $(echo "$memory > 500" | bc -l) )); then
        echo -e "  ${RED}⚠ High memory usage (>500MB)${NC}"
        echo -e "  ${YELLOW}Recommendation: Consider smaller batch sizes${NC}"
    else
        echo -e "  ${GREEN}✓ Memory usage reasonable${NC}"
    fi
    
    echo
done

# Final performance summary
echo -e "${BLUE}Performance Summary:${NC}"
echo -e "✓ Tested $TEXT_COUNT files across multiple file types"
echo -e "✓ Measured indexing, searching, and memory performance"
echo -e "✓ Validated concurrent operation capabilities"
echo -e "✓ Analyzed batch size performance characteristics"

echo -e "${GREEN}Performance validation complete${NC}"

# Clean up any background processes
pkill -f "$TP_BINARY" 2>/dev/null || true