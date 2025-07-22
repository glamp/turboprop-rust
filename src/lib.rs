pub mod cli;

use anyhow::Result;

pub fn index_files(path: &str) -> Result<()> {
    println!("Indexing files in: {}", path);
    // TODO: Implement indexing logic
    Ok(())
}

pub fn search_files(query: &str) -> Result<()> {
    println!("Searching for: {}", query);
    // TODO: Implement search logic
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_files() {
        let result = index_files(".");
        assert!(result.is_ok());
    }

    #[test]
    fn test_search_files() {
        let result = search_files("test query");
        assert!(result.is_ok());
    }
}