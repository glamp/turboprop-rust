use clap::CommandFactory;
use clap_mangen::Man;
use std::fs;
use std::io::Error;

include!("src/cli.rs");

fn main() -> Result<(), Error> {
    println!("cargo:rerun-if-changed=src/cli.rs");

    let out_dir = match std::env::var_os("OUT_DIR") {
        None => return Ok(()),
        Some(out_dir) => out_dir,
    };

    let out_dir = std::path::PathBuf::from(out_dir);

    // Create man directory
    let man_dir = out_dir.join("man");
    fs::create_dir_all(&man_dir)?;

    // Generate main command man page
    let cmd = Cli::command();
    let man = Man::new(cmd);
    let mut buffer: Vec<u8> = vec![];
    man.render(&mut buffer)?;

    fs::write(man_dir.join("tp.1"), buffer)?;

    println!("Generated man pages in {:?}", man_dir);

    Ok(())
}