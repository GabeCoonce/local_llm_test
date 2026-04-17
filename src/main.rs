//! Local LLM inference proof-of-concept using Candle + Llama 3.1 8B Instruct.
//!
//! First run downloads ~16 GB to `~/.cache/huggingface/hub/`.
//! Requires a HuggingFace token: echo "hf_yourtoken" > ~/.cache/huggingface/token
//!
//! Usage:
//!   cargo run --release --features metal -- --chat --prompt "What is Rust?"
//!   cargo run --release --features metal -- -i
//!   cargo run --release --features metal -- -s --port 3000
//!   cargo run --release --features metal -- --help

use anyhow::{Error as E, Result};
use axum::{Json, extract::{ConnectInfo, State}, http::StatusCode, routing::post, Router};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::{Cache, Config, Llama, LlamaConfig};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;

const MODEL_ID: &str = "meta-llama/Meta-Llama-3.1-8B-Instruct";

const MAX_INPUT_CHARS: usize = 8192;

const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant. Users will provide a prompt, \
    often but not always containing a question, and you'll aim to provide a useful response.\n\n\
    You aim to keep a respectful and conversational tone, but don't shy away from being a bit \
    fun or witty when appropriate. Prompts may be general questions, requests for advice, \
    code review, or anything else.";

// ── CLI ────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(about = "Local LLM inference with Llama 3.1 8B Instruct + Candle")]
struct Args {
    /// Text prompt (single-turn mode)
    #[arg(short, long, default_value = "The meaning of life is")]
    prompt: String,

    /// Wrap prompt in the Llama 3.1 chat template (single-turn)
    #[arg(long)]
    chat: bool,

    /// Interactive multi-turn chat mode
    #[arg(short, long)]
    interactive: bool,

    /// HTTP server mode
    #[arg(short, long)]
    server: bool,

    /// Port to listen on in server mode
    #[arg(long, default_value_t = 3000)]
    port: u16,

    /// Bind to all interfaces (0.0.0.0) instead of localhost only
    #[arg(long)]
    public: bool,

    /// Maximum number of new tokens to generate per turn
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

// ── Shared model state ─────────────────────────────────────────────────────

struct ModelState {
    model:          Llama,
    tokenizer:      Tokenizer,
    eos_token_id:   u32,
    config:         Config,
    dtype:          DType,
    device:         Device,
    max_new_tokens: usize,
    temperature:    f64,
    top_p:          f64,
}

// ── HTTP types ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct InferRequest {
    prompt: String,
    /// Overrides the default system prompt. Pass "" for no system prompt.
    system_prompt: Option<String>,
}

#[derive(Serialize)]
struct InferResponse {
    response: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// ── main ───────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();
    let device = select_device(args.cpu)?;

    eprintln!("Model : {MODEL_ID}");
    eprintln!("Device: {device:?}");
    eprintln!("Fetching model files (cached after first download) …");

    let api = Api::new()?;
    let repo = api.repo(Repo::new(MODEL_ID.to_string(), RepoType::Model));

    let tokenizer_path = repo.get("tokenizer.json")?;
    let config_path = repo.get("config.json")?;

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

    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;
    let eos_token_id = tokenizer.token_to_id("<|eot_id|>").unwrap_or(128001);

    let raw_cfg: LlamaConfig =
        serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
    let config: Config = raw_cfg.into_config(false);

    eprintln!("Loading weights …");
    let dtype = DType::BF16;
    // SAFETY: files are read-only and outlive the program.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&weight_paths, dtype, &device)?
    };
    let model = Llama::load(vb, &config)?;

    let state = ModelState {
        model,
        tokenizer,
        eos_token_id,
        config,
        dtype,
        device,
        max_new_tokens: args.max_new_tokens,
        temperature:    args.temperature,
        top_p:          args.top_p,
    };

    if args.server {
        run_server(state, args.port, args.public)?;
    } else if args.interactive {
        run_interactive(state)?;
    } else {
        run_single(&args, state)?;
    }

    Ok(())
}

// ── Server mode ────────────────────────────────────────────────────────────

fn run_server(state: ModelState, port: u16, public: bool) -> Result<()> {
    let shared = Arc::new(Mutex::new(state));
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let app = Router::new()
            .route("/infer", post(infer_handler))
            .with_state(shared);

        let host = if public { "0.0.0.0" } else { "127.0.0.1" };
        let addr = format!("{host}:{port}");
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        eprintln!("Server listening on http://{addr}");
        eprintln!("POST /infer  {{\"prompt\": \"...\", \"system_prompt\": \"...\"}}");

        // into_make_service_with_connect_info propagates the client socket
        // address so handlers can extract it via ConnectInfo<SocketAddr>.
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        ).await?;
        Ok::<(), anyhow::Error>(())
    })?;
    Ok(())
}

// ── Request handler ────────────────────────────────────────────────────────

async fn infer_handler(
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    State(shared): State<Arc<Mutex<ModelState>>>,
    Json(req): Json<InferRequest>,
) -> Result<Json<InferResponse>, (StatusCode, Json<ErrorResponse>)> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    eprintln!(
        "[{timestamp}] POST /infer  from={}  prompt={:?}  system_prompt={:?}",
        client_addr,
        req.prompt,
        req.system_prompt.as_deref().unwrap_or("<default>"),
    );

    let state = shared.try_lock().map_err(|_| {
        eprintln!("[{timestamp}] 503 inference already in progress");
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse { error: "inference already in progress".into() }),
        )
    })?;

    let system_prompt = req.system_prompt
        .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string());

    let combined_len = req.prompt.len() + system_prompt.len();
    if combined_len > MAX_INPUT_CHARS {
        eprintln!("[{timestamp}] 400  input too large ({combined_len} chars)");
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!(
                    "combined prompt and system_prompt exceeds {MAX_INPUT_CHARS} characters ({combined_len} sent)"
                ),
            }),
        ));
    }

    let input_text = build_chat_prompt(&req.prompt, &system_prompt);

    let mut cache = Cache::new(true, state.dtype, &state.config, &state.device)
        .map_err(|e| server_error(e.to_string()))?;
    let mut logits_proc =
        LogitsProcessor::new(42, Some(state.temperature), Some(state.top_p));

    let encoding = state.tokenizer
        .encode(input_text.as_str(), false)
        .map_err(|e| server_error(e.to_string()))?;
    let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();

    // Snapshot the prompt length before generation — generate_turn advances
    // decoded_len as it streams, so we need the original offset to slice out
    // just the generated response afterwards.
    let prompt_decoded_len = state.tokenizer
        .decode(&token_ids, true)
        .unwrap_or_default()
        .len();
    let mut decoded_len = prompt_decoded_len;

    generate_turn(
        &state.model, &mut cache, &state.tokenizer,
        &mut token_ids, 0,
        state.eos_token_id, state.max_new_tokens,
        &mut logits_proc, &state.device,
        &mut decoded_len,
        true, // silent — don't stream to stdout in server mode
    ).map_err(|e| server_error(e.to_string()))?;

    let full_text = state.tokenizer
        .decode(&token_ids, true)
        .unwrap_or_default();
    let response = full_text[prompt_decoded_len..].trim().to_string();

    eprintln!("[{timestamp}] 200  tokens_generated={}", token_ids.len());

    Ok(Json(InferResponse { response }))
}

fn server_error(msg: String) -> (StatusCode, Json<ErrorResponse>) {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: msg }))
}

// ── Single-turn mode ───────────────────────────────────────────────────────

fn run_single(args: &Args, state: ModelState) -> Result<()> {
    let input_text = if args.chat {
        build_chat_prompt(&args.prompt, DEFAULT_SYSTEM_PROMPT)
    } else {
        args.prompt.clone()
    };

    let encoding = state.tokenizer.encode(input_text.as_str(), false).map_err(E::msg)?;
    let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let mut cache = Cache::new(true, state.dtype, &state.config, &state.device)?;
    let mut logits_proc =
        LogitsProcessor::new(42, Some(state.temperature), Some(state.top_p));

    eprintln!("\n─── output ──────────────────────────────────────────────────");
    if args.chat {
        print!("[user]      {}\n[assistant] ", args.prompt.trim());
    } else {
        let decoded = state.tokenizer.decode(&token_ids, true).unwrap_or_default();
        print!("{decoded}");
    }
    std::io::stdout().flush()?;

    let mut decoded_len = state.tokenizer.decode(&token_ids, true).unwrap_or_default().len();
    generate_turn(&state.model, &mut cache, &state.tokenizer, &mut token_ids, 0,
                  state.eos_token_id, state.max_new_tokens, &mut logits_proc, &state.device,
                  &mut decoded_len, false)?;

    println!();
    eprintln!("─────────────────────────────────────────────────────────────");
    Ok(())
}

// ── Interactive multi-turn mode ────────────────────────────────────────────

fn run_interactive(state: ModelState) -> Result<()> {
    eprintln!("\n─── interactive chat ────────────────────────────────────────");
    eprintln!("Type your message and press Enter. Empty line or Ctrl+C to quit.");
    eprintln!("─────────────────────────────────────────────────────────────");

    let mut cache = Cache::new(true, state.dtype, &state.config, &state.device)?;
    let mut logits_proc =
        LogitsProcessor::new(42, Some(state.temperature), Some(state.top_p));

    let system_text = format!(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\
         {DEFAULT_SYSTEM_PROMPT}<|eot_id|>"
    );
    let encoding = state.tokenizer.encode(system_text.as_str(), false).map_err(E::msg)?;
    let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let mut first_turn = true;

    loop {
        print!("\n[user] ");
        std::io::stdout().flush()?;

        let mut user_input = String::new();
        if std::io::stdin().read_line(&mut user_input)? == 0 {
            break;
        }
        let user_input = user_input.trim();
        if user_input.is_empty() {
            break;
        }

        let prefill_start = if first_turn {
            first_turn = false;
            0
        } else {
            token_ids.len()
        };

        let user_turn = format!(
            "<|start_header_id|>user<|end_header_id|>\n\
             {user_input}<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n"
        );
        let turn_enc = state.tokenizer.encode(user_turn.as_str(), false).map_err(E::msg)?;
        token_ids.extend_from_slice(turn_enc.get_ids());

        print!("[assistant] ");
        std::io::stdout().flush()?;

        let mut decoded_len = state.tokenizer.decode(&token_ids, true).unwrap_or_default().len();
        generate_turn(&state.model, &mut cache, &state.tokenizer, &mut token_ids, prefill_start,
                      state.eos_token_id, state.max_new_tokens, &mut logits_proc, &state.device,
                      &mut decoded_len, false)?;

        println!();
    }

    eprintln!("\n─────────────────────────────────────────────────────────────");
    Ok(())
}

// ── Core generation ────────────────────────────────────────────────────────

/// Runs prefill then autoregressive decode.
///
/// `silent` suppresses stdout streaming — used in server mode where the
/// response is returned over HTTP rather than printed to the terminal.
fn generate_turn(
    model: &Llama,
    cache: &mut Cache,
    tokenizer: &Tokenizer,
    token_ids: &mut Vec<u32>,
    prefill_start: usize,
    eos_token_id: u32,
    max_new_tokens: usize,
    logits_proc: &mut LogitsProcessor,
    device: &Device,
    decoded_len: &mut usize,
    silent: bool,
) -> Result<()> {
    let prefill_end = token_ids.len();

    let mut logits = if prefill_start == 0 {
        let input = Tensor::new(&token_ids[..prefill_end], device)?.unsqueeze(0)?;
        model.forward(&input, 0, cache)?.squeeze(0)?
    } else {
        let mut last = None;
        for i in prefill_start..prefill_end {
            let input = Tensor::new(&[token_ids[i]], device)?.unsqueeze(0)?;
            last = Some(model.forward(&input, i, cache)?.squeeze(0)?);
        }
        last.ok_or_else(|| E::msg("empty prefill range"))?
    };

    for _ in 0..max_new_tokens {
        let next_token = logits_proc.sample(&logits)?;
        token_ids.push(next_token);

        if next_token == eos_token_id {
            break;
        }

        if !silent {
            if let Ok(full_text) = tokenizer.decode(token_ids, true) {
                if full_text.len() > *decoded_len {
                    print!("{}", &full_text[*decoded_len..]);
                    std::io::stdout().flush()?;
                    *decoded_len = full_text.len();
                }
            }
        }

        let n = token_ids.len();
        let input = Tensor::new(&[token_ids[n - 1]], device)?.unsqueeze(0)?;
        logits = model.forward(&input, n - 1, cache)?.squeeze(0)?;
    }

    Ok(())
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn build_chat_prompt(prompt: &str, system_prompt: &str) -> String {
    format!(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\
         {system_prompt}<|eot_id|>\
         <|start_header_id|>user<|end_header_id|>\n\
         {}<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n",
        prompt.trim()
    )
}

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
