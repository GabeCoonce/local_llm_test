//! Local LLM inference proof-of-concept using Candle + Llama 3.1 8B Instruct.
//!
//! First run downloads ~16 GB to `~/.cache/huggingface/hub/`.
//! Requires a HuggingFace token with access to meta-llama/Meta-Llama-3.1-8B-Instruct.
//! Set it via: echo "hf_yourtoken" > ~/.cache/huggingface/token
//!
//! Usage:
//!   cargo run --release --features metal -- --chat --prompt "What is the Rust borrow checker?"
//!   cargo run --release --features metal -- --prompt "The capital of France is"
//!   cargo run --release --features metal -- --help

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::{Cache, Config, Llama, LlamaConfig};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;
use tokenizers::Tokenizer;

const MODEL_ID: &str = "meta-llama/Meta-Llama-3.1-8B-Instruct";

#[derive(Parser, Debug)]
#[command(about = "Local LLM inference with Llama 3.1 8B Instruct + Candle")]
struct Args {
    /// Text prompt
    #[arg(short, long, default_value = "The meaning of life is")]
    prompt: String,

    /// Wrap prompt in the Llama 3.1 chat template (system / user / assistant)
    #[arg(long)]
    chat: bool,

    /// Maximum number of new tokens to generate
    #[arg(long, default_value_t = 800)]
    max_new_tokens: usize,

    /// Sampling temperature — 0.0 = greedy, higher = more creative
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Top-p nucleus sampling threshold (0.0–1.0)
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    /// Force CPU even when a GPU backend is available
    #[arg(long)]
    cpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = select_device(args.cpu)?;

    // ── 1. Download / retrieve cached model files ─────────────────────────
    eprintln!("Model : {MODEL_ID}");
    eprintln!("Device: {device:?}");
    eprintln!("Fetching model files (cached after first download) …");

    let api = Api::new()?;
    let repo = api.repo(Repo::new(MODEL_ID.to_string(), RepoType::Model));

    let tokenizer_path = repo.get("tokenizer.json")?;
    let config_path = repo.get("config.json")?;

    // Llama 3.1 8B is sharded across 4 safetensors files.
    // The index.json maps weight names to shard filenames — we parse it to
    // get the unique set of shards, then download each one.
    let index_path = repo.get("model.safetensors.index.json")?;
    let index_json = std::fs::read_to_string(&index_path)?;
    let index: serde_json::Value = serde_json::from_str(&index_json)?;

    let mut shard_files: Vec<String> = index["weight_map"]
        .as_object()
        .ok_or_else(|| E::msg("invalid index.json"))?
        .values()
        .map(|v| v.as_str().unwrap_or("").to_string())
        .collect();
    shard_files.sort();
    shard_files.dedup();

    eprintln!("Downloading {} weight shards …", shard_files.len());
    let weight_paths: Vec<_> = shard_files
        .iter()
        .map(|f| repo.get(f))
        .collect::<Result<_, _>>()?;

    // ── 2. Tokenizer ──────────────────────────────────────────────────────
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;
    // Llama 3.1 uses <|eot_id|> (end-of-turn) as the generation stop token.
    let eos_token_id = tokenizer.token_to_id("<|eot_id|>").unwrap_or(128001);

    // ── 3. Model config ───────────────────────────────────────────────────
    let raw_cfg: LlamaConfig = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
    let config: Config = raw_cfg.into_config(false);

    // ── 4. Load weights ───────────────────────────────────────────────────
    eprintln!("Loading weights …");
    // BF16 is natively supported on Metal and halves memory vs F32 (~16 GB total).
    let dtype = DType::BF16;
    // SAFETY: files are read-only and outlive the program.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_paths, dtype, &device)? };
    let model = Llama::load(vb, &config)?;
    let mut cache = Cache::new(true, dtype, &config, &device)?;

    // ── 5. Build input text ───────────────────────────────────────────────
    // Llama 3.1 chat template. Without --chat the model does raw completion.
    let input_text = if args.chat {
        format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\
             You are a helpful assistant. Users will provide a prompt, often but not always containing a question, and you'll aim to provide a useful response.\n
             You aim to keep a respectful and conversational tone, but don't shy away from being a bit fun or witty when appropriate. Prompts may be general questions, requests for advice, code review, or anything else.<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\
             {}<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n",
            args.prompt.trim()
        )
    } else {
        args.prompt.clone()
    };

    // ── 6. Tokenize ───────────────────────────────────────────────────────
    let encoding = tokenizer
        .encode(input_text.as_str(), true)
        .map_err(E::msg)?;
    let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();

    let decoded_prompt = tokenizer.decode(&token_ids, true).unwrap_or_default();
    let mut decoded_len = decoded_prompt.len();

    // ── 7. Print header ───────────────────────────────────────────────────
    eprintln!("\n─── output ──────────────────────────────────────────────────");
    if args.chat {
        print!("[user]      {}\n[assistant] ", args.prompt.trim());
    } else {
        print!("{decoded_prompt}");
    }
    std::io::stdout().flush()?;

    // ── 8. Generation loop ────────────────────────────────────────────────
    let mut logits_proc = LogitsProcessor::new(42, Some(args.temperature), Some(args.top_p));

    for step in 0..args.max_new_tokens {
        let (ctx, pos) = if step == 0 {
            (token_ids.as_slice(), 0usize)
        } else {
            let n = token_ids.len();
            (&token_ids[n - 1..], n - 1)
        };

        let input = Tensor::new(ctx, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, pos, &mut cache)?;
        let logits = logits.squeeze(0)?;

        let next_token = logits_proc.sample(&logits)?;
        token_ids.push(next_token);

        if next_token == eos_token_id {
            break;
        }

        if let Ok(full_text) = tokenizer.decode(&token_ids, true) {
            if full_text.len() > decoded_len {
                print!("{}", &full_text[decoded_len..]);
                std::io::stdout().flush()?;
                decoded_len = full_text.len();
            }
        }
    }

    println!();
    eprintln!("─────────────────────────────────────────────────────────────");
    Ok(())
}

/// Pick the best available compute device.
fn select_device(force_cpu: bool) -> Result<Device> {
    if force_cpu {
        return Ok(Device::Cpu);
    }

    #[cfg(feature = "metal")]
    match Device::new_metal(0) {
        Ok(d) => {
            eprintln!("Metal GPU available — using it.");
            return Ok(d);
        }
        Err(e) => eprintln!("Metal init failed ({e}), falling back to CPU."),
    }

    #[cfg(not(feature = "metal"))]
    eprintln!("Tip: compile with `--features metal` to use your Apple GPU.");

    Ok(Device::Cpu)
}
