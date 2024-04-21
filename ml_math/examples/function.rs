//! Example on how to interact with a deployed `stylus-hello-world` program using defaults.
//! This example uses ethers-rs to instantiate the program using a Solidity ABI.
//! Then, it attempts to check the current counter value, increment it via a tx,
//! and check the value again. The deployed program is fully written in Rust and compiled to WASM
//! but with Stylus, it is accessible just as a normal Solidity smart contract is via an ABI.

use ethers::{
    middleware::SignerMiddleware,
    prelude::abigen,
    providers::{Http, Middleware, Provider},
    signers::{LocalWallet, Signer},
    types::Address,
};
use eyre::eyre;
use std::io::{BufRead, BufReader};
use std::str::FromStr;
use std::sync::Arc;

/// Your private key file path.
const PRIV_KEY_PATH: &str = "./../scripts/.env";

/// Stylus RPC endpoint url.
const RPC_URL: &str = "https://stylus-testnet.arbitrum.io/rpc";

/// Deployed pragram address.
const STYLUS_PROGRAM_ADDRESS: &str = "0xfc7e317841fd75DC965EFab1F6D58447cd627088";

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let priv_key_path =
        std::env::var(PRIV_KEY_PATH).map_err(|_| eyre!("No {} env var set", PRIV_KEY_PATH))?;
    let rpc_url = std::env::var(RPC_URL).map_err(|_| eyre!("No {} env var set", RPC_URL))?;
    let program_address = std::env::var(STYLUS_PROGRAM_ADDRESS)
        .map_err(|_| eyre!("No {} env var set", STYLUS_PROGRAM_ADDRESS))?;
    abigen!(
        MlMath,
        r#"[
            function reluDerive(int128[][] memory m) external pure returns (int128[][] memory)
            function elementSumRow(int128[][] memory z) external pure returns (int128[][] memory)
            function scalarDiv(int128[][] memory mat, int128 scalar) external pure returns (int128[][] memory)
            function elementwiseMul(int128[][] memory m1, int128[][] memory m2) external pure returns (int128[][] memory)
            function transpose(int128[][] memory ori) external pure returns (int128[][] memory)
            function softmax(int128[][] memory z) external pure returns (int128[][] memory)
            function relu(int128[][] memory z) external pure returns (int128[][] memory)
            function dotProduct(int128[][] memory m1, int128[][] memory m2) external pure returns (int128[][] memory)
            function sum(int128[][] memory m1, int128[][] memory m2) external pure returns (int128[][] memory)
            function oneHot(int128[][] memory y) external pure returns (int128[][] memory)
            function scalarMul(int128[][] memory mat, int128 scalar) external pure returns (int128[][] memory)
        ]"#
    );

    let provider = Provider::<Http>::try_from(rpc_url)?;
    let address: Address = program_address.parse()?;

    let privkey = read_secret_from_file(&priv_key_path)?;
    let wallet = LocalWallet::from_str(&privkey)?;
    let chain_id = provider.get_chainid().await?.as_u64();
    let client = Arc::new(SignerMiddleware::new(
        provider,
        wallet.clone().with_chain_id(chain_id),
    ));

    let ml_math = MlMath::new(address, client);

    //data
    let x = vec![vec![21; 11]; 10];
    let scalar: i128 = 2;
    let _ = ml_math.scalar_mul(x, scalar).send().await?.await?;
    println!("Successfully incremented counter via a tx");
    Ok(())
}

fn read_secret_from_file(fpath: &str) -> eyre::Result<String> {
    let f = std::fs::File::open(fpath)?;
    let mut buf_reader = BufReader::new(f);
    let mut secret = String::new();
    buf_reader.read_line(&mut secret)?;
    Ok(secret.trim().to_string())
}
