# Test-Driven Development (TDD) Workflow

## Overview

Iterate to correct test failures in the codebase.

Use a specific test command if noted in the local CLAUDE.md.

## Goals

The goal is to get the code to pass all tests by correcting test failures.

## Guidelines

- Always run tests with the terminal using a command line tool appropriate for the project.
- Corrections should be constructive, meaning 'just deleting code' is not an acceptable fix.

## Process

- Run unit tests (cargo test)
- Read error output
- For each test failure:
  - Read the error output
  - Think sequentially, deeply, and critically about the error output
  - Identify the cause of the failure
  - Describe your proposed fix
  - Correct the code to fix the failure
  - Run the single corrected test
  - If it passes, proceed to the next failure
  - If it fails, stop and think what might be the problem
  - Try again
- Once all tests have been individually fixed, run all tests for the entire project as a final check
- Repeat until all tests pass

## Goals

The goal is to have:

- 0 errors
- 0 warnings

Anything else is a failure that must be corrected.
