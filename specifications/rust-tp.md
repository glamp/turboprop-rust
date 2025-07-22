# Rust Turboprop

A simple command line tool for semantic code search.

- Optimize for a basic working version. Do not add extra features.
- Expected number of files will be between 50-10,000
- To start, don't worry about performance. Focus on getting something simple to work.
- For chunking and embedding, these should go hand-in-hand. The chunk size and the dimensions of the model used for embedding should be aware/related. For example, sentence-transformers/all-MiniLM-L6-v2 has dimension of 384. Chunk size should be appropriate based on that. I don't want ot configure the chunk size. I want the model to be configurable. Again, optimize for getting something simple working well.
- Do not have language specific capabilities yet. Just index each file the same way.
- Do not have class or function level capabilities, just index the files.

## Capabilities

- Can index your codebase and then watch for changes and reindex it - Uses a small LLM model to index files. Handles chunks, filenames, etc. - Respects the `git ls` / `.gitignore` so only indexes files under source control - `--max-filesize` to filter out files that are to big to index - index should be located in a `${--repo}/.turboprop/` folder - .turboprop/.turboprop.yml for optional configuration of command line options - Assume there will be multiple processes interacting with the index
- Let's you search using an index inside a repository - Uses the index created by the `index` to perform cosine vector similarity search on your codebase - Returns results in a format easily digestable by LLMs - Can filter by filetype

## API

```bash
tp index --repo . --max-filesize 2mb
<indexes the repository and displays a progress bar while doing so>

tp index --watch --repo .
<indexes the repository and displays a progress bar while doing so>
<after done, watches for any file changes. updates the indexes accordingly>

tp search "jwt authentication" --repo .
<line delimited json results>

tp search --filetype .js "jwt authentication" --repo .
<line delimited json results but only .js files>
tp search --filetype .js "jwt authentication" --repo . --output text
<outputs a nice human readable version>
```

## Guiding Principles

- Keep things simple. Do not add more requirements than we have here.
- Write tests first using the API here. Do not mock. This guide should be the tests.
- Use the sample-codebases/poker codebase for testing
- Use TDD to develop the functionality
- Create a simple README explaining how to run this on a sample.
