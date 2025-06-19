mod logic;
use anyhow::{Result, anyhow};
use clap::Parser;
use once_cell::sync::Lazy;
use logic::{OnnxState, EmbeddingInput};

static GLOBAL_STATE: Lazy<OnnxState> = Lazy::new(|| OnnxState {
    session: std::sync::Mutex::new(None),
});

#[derive(Parser)]
struct Args {
    /// Text to be encoded
    #[arg(default_value = "")]
    text: String,
    /// Path to tokenizer.json
    #[clap(long, default_value = "models/tokenizer.json")]
    tokenizer: String,
    /// Path to ONNX model
    #[clap(long, default_value = "models/model.onnx")]
    model: String,
    /// Maximum sequence length
    #[clap(long, default_value_t = 256)]
    max_len: usize,
}

fn main() -> Result<()> {
    ort::init_from("models/onnxruntime.dll").commit()?;

    let args = Args::parse();

    /* 1. Initialize/reuse ONNX Session */
    logic::init_onnx_session(args.model.clone(), &GLOBAL_STATE)
        .map_err(|e| anyhow!(e))?;

    /* 2. Tokenization */
    let tok = logic::tokenize_batch(
        vec![args.text],
        args.tokenizer,
        args.max_len,
    ).map_err(|e| anyhow!(e))?;

    /* 3. Generate embedding */
    let emb = logic::generate_embedding(
        EmbeddingInput {
            input_ids: tok.input_ids.iter().map(|v| v.iter().map(|&x| x as i64).collect()).collect(),
            attention_mask: tok.attention_mask.iter().map(|v| v.iter().map(|&x| x as i64).collect()).collect(),
            token_type_ids: tok.token_type_ids.iter().map(|v| v.iter().map(|&x| x as i64).collect()).collect(),
        },
        &GLOBAL_STATE,
    ).map_err(|e| anyhow!(e))?;

    println!("Embedding dim = {}", emb[0].len());
    println!("first 10 dims = {:?}", &emb[0][..10]);
    Ok(())
}