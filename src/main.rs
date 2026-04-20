//! Local LLM inference proof-of-concept using Candle.
//!
//! Supports:
//!   --model llama     Llama 3.1 8B Instruct (BF16 safetensors, ~16 GB, requires HF token)
//!   --model deepseek  DeepSeek-R1-Distill-Qwen-32B (GGUF Q4_K_M, ~18 GB, MIT licensed)
//!
//! Usage:
//!   cargo run --release --features metal -- --model llama --chat --prompt "What is Rust?"
//!   cargo run --release --features metal -- --model deepseek -i
//!   cargo run --release --features metal -- -s --port 3000
//!   cargo run --release --features metal -- --help

use anyhow::{Error as E, Result};
use axum::{Json, extract::{ConnectInfo, State}, http::StatusCode, routing::post, Router};
use candle_core::{DType, Device, Tensor, quantized::gguf_file};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::{
    Cache as LlamaCache, Config as LlamaModelConfig, Llama, LlamaConfig as LlamaJsonConfig,
};
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2Weights;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;

// ── Model identifiers ──────────────────────────────────────────────────────

const LLAMA_MODEL_ID: &str = "meta-llama/Meta-Llama-3.1-8B-Instruct";

const DEEPSEEK_MODEL_REPO: &str  = "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF";
const DEEPSEEK_GGUF_FILE: &str   = "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf";
const DEEPSEEK_TOKENIZER_REPO: &str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B";

const MAX_INPUT_CHARS: usize = 8192;

const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant. Users will provide a prompt, \
    often but not always containing a question, and you'll aim to provide a useful response.\n\n\
    You aim to keep a respectful and conversational tone, but don't shy away from being a bit \
    fun or witty when appropriate. Prompts may be general questions, requests for advice, \
    code review, or anything else.";

// ── Model selection ────────────────────────────────────────────────────────

#[derive(clap::ValueEnum, Clone, Debug, Default, PartialEq)]
enum ModelKind {
    /// Llama 3.1 8B Instruct — BF16 safetensors, ~16 GB (requires HF token)
    #[default]
    Llama,
    /// DeepSeek-R1-Distill-Qwen-32B — GGUF Q4_K_M, ~18 GB (MIT licensed)
    #[value(name = "deepseek")]
    DeepSeek,
}

// ── Model enum ─────────────────────────────────────────────────────────────

enum Model {
    Llama {
        inner:  Llama,
        cache:  LlamaCache,
        config: LlamaModelConfig, // kept here so reset_cache can recreate it
        dtype:  DType,
    },
    Qwen2(Qwen2Weights),
}

impl Model {
    /// Resets the KV cache for a fresh context.
    ///
    /// Llama: recreates the Cache struct (external to the model).
    /// Qwen2: no-op — passing index_pos = 0 to forward() discards the
    ///        internal cache automatically.
    fn reset_cache(&mut self, device: &Device) -> Result<()> {
        if let Model::Llama { cache, config, dtype, .. } = self {
            *cache = LlamaCache::new(true, *dtype, config, device)?;
        }
        Ok(())
    }

    /// Builds a complete single-turn chat prompt (system + user + assistant header).
    fn build_chat_prompt(&self, prompt: &str, system_prompt: &str) -> String {
        match self {
            Model::Llama { .. } => format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\
                 {system_prompt}<|eot_id|>\
                 <|start_header_id|>user<|end_header_id|>\n\
                 {}<|eot_id|>\
                 <|start_header_id|>assistant<|end_header_id|>\n",
                prompt.trim()
            ),
            Model::Qwen2(_) => format!(
                "<|im_start|>system\n{system_prompt}<|im_end|>\n\
                 <|im_start|>user\n{}<|im_end|>\n\
                 <|im_start|>assistant\n",
                prompt.trim()
            ),
        }
    }

    /// Returns just the system-turn prefix, used to seed interactive mode.
    fn system_prefix(&self, system_prompt: &str) -> String {
        match self {
            Model::Llama { .. } => format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\
                 {system_prompt}<|eot_id|>"
            ),
            Model::Qwen2(_) => format!("<|im_start|>system\n{system_prompt}<|im_end|>\n"),
        }
    }

    /// Returns a formatted user turn including the assistant reply header.
    fn user_turn_text(&self, user_input: &str) -> String {
        match self {
            Model::Llama { .. } => format!(
                "<|start_header_id|>user<|end_header_id|>\n\
                 {user_input}<|eot_id|>\
                 <|start_header_id|>assistant<|end_header_id|>\n"
            ),
            Model::Qwen2(_) => format!(
                "<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
            ),
        }
    }
}

// ── CLI ────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(about = "Local LLM inference with Candle (Llama 3.1 8B or DeepSeek-R1-Distill-Qwen-32B)")]
struct Args {
    /// Which model to load
    #[arg(long, value_enum, default_value_t = ModelKind::Llama)]
    model: ModelKind,

    /// Text prompt (single-turn mode)
    #[arg(short, long, default_value = "The meaning of life is")]
    prompt: String,

    /// Wrap prompt in the model's chat template (single-turn)
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
    model:          Model,
    tokenizer:      Tokenizer,
    eos_token_id:   u32,
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

    eprintln!("Device: {device:?}");
    eprintln!("Fetching model files (cached after first download) …");

    let api = Api::new()?;

    let state = match args.model {
        ModelKind::Llama => {
            eprintln!("Model : {LLAMA_MODEL_ID}");
            let repo = api.repo(Repo::new(LLAMA_MODEL_ID.to_string(), RepoType::Model));

            let tokenizer_path = repo.get("tokenizer.json")?;
            let config_path    = repo.get("config.json")?;
            let index_path     = repo.get("model.safetensors.index.json")?;

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

            let tokenizer     = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;
            let eos_token_id  = tokenizer.token_to_id("<|eot_id|>").unwrap_or(128001);

            let raw_cfg: LlamaJsonConfig =
                serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
            let config: LlamaModelConfig = raw_cfg.into_config(false);

            eprintln!("Loading weights …");
            let dtype = DType::BF16;
            // SAFETY: files are read-only and outlive the program.
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&weight_paths, dtype, &device)?
            };
            let inner = Llama::load(vb, &config)?;
            let cache = LlamaCache::new(true, dtype, &config, &device)?;

            ModelState {
                model: Model::Llama { inner, cache, config, dtype },
                tokenizer,
                eos_token_id,
                device,
                max_new_tokens: args.max_new_tokens,
                temperature:    args.temperature,
                top_p:          args.top_p,
            }
        }

        ModelKind::DeepSeek => {
            eprintln!("Model : {DEEPSEEK_MODEL_REPO} / {DEEPSEEK_GGUF_FILE}");

            let tok_repo = api.repo(Repo::new(DEEPSEEK_TOKENIZER_REPO.to_string(), RepoType::Model));
            let tokenizer_path = tok_repo.get("tokenizer.json")?;

            let model_repo = api.repo(Repo::new(DEEPSEEK_MODEL_REPO.to_string(), RepoType::Model));
            eprintln!("Downloading GGUF weights (~18 GB, cached after first run) …");
            let gguf_path = model_repo.get(DEEPSEEK_GGUF_FILE)?;

            let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

            eprintln!("Loading GGUF weights into {device:?} …");
            let mut model_file  = std::fs::File::open(&gguf_path)?;
            let gguf_content    = gguf_file::Content::read(&mut model_file)?;

            // Prefer the EOS token ID from GGUF metadata — it is authoritative
            // for the quantized model's vocab indices.
            let eos_token_id: u32 = gguf_content
                .metadata
                .get("tokenizer.ggml.eos_token_id")
                .and_then(|v| v.to_u32().ok())
                .or_else(|| tokenizer.token_to_id("<|im_end|>"))
                .unwrap_or(151645);
            eprintln!("EOS token id: {eos_token_id}");

            let model = Qwen2Weights::from_gguf(gguf_content, &mut model_file, &device)?;

            ModelState {
                model: Model::Qwen2(model),
                tokenizer,
                eos_token_id,
                device,
                max_new_tokens: args.max_new_tokens,
                temperature:    args.temperature,
                top_p:          args.top_p,
            }
        }
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

    let mut state_guard = shared.try_lock().map_err(|_| {
        eprintln!("[{timestamp}] 503 inference already in progress");
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse { error: "inference already in progress".into() }),
        )
    })?;
    // Reborrow as a plain &mut so the borrow checker can split fields
    // (e.g. &mut model alongside &tokenizer and &device in the same call).
    let state: &mut ModelState = &mut *state_guard;

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

    // Each request gets a fresh context: Llama recreates its Cache struct;
    // Qwen2 resets automatically when generate_turn calls forward() at index_pos = 0.
    state.model.reset_cache(&state.device)
        .map_err(|e| server_error(e.to_string()))?;

    let input_text = state.model.build_chat_prompt(&req.prompt, &system_prompt);

    let encoding = state.tokenizer
        .encode(input_text.as_str(), false)
        .map_err(|e| server_error(e.to_string()))?;
    let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();

    let mut logits_proc = LogitsProcessor::new(42, Some(state.temperature), Some(state.top_p));

    let prompt_decoded_len = state.tokenizer
        .decode(&token_ids, true)
        .unwrap_or_default()
        .len();
    let mut decoded_len = prompt_decoded_len;

    generate_turn(
        &mut state.model, &state.tokenizer,
        &mut token_ids, 0,
        state.eos_token_id, state.max_new_tokens,
        &mut logits_proc, &state.device,
        &mut decoded_len,
        true,
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

fn run_single(args: &Args, mut state: ModelState) -> Result<()> {
    let input_text = if args.chat {
        state.model.build_chat_prompt(&args.prompt, DEFAULT_SYSTEM_PROMPT)
    } else {
        args.prompt.clone()
    };

    let encoding = state.tokenizer.encode(input_text.as_str(), false).map_err(E::msg)?;
    let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let mut logits_proc = LogitsProcessor::new(42, Some(state.temperature), Some(state.top_p));

    eprintln!("\n─── output ──────────────────────────────────────────────────");
    if args.chat {
        print!("[user]      {}\n[assistant] ", args.prompt.trim());
    } else {
        let decoded = state.tokenizer.decode(&token_ids, true).unwrap_or_default();
        print!("{decoded}");
    }
    std::io::stdout().flush()?;

    let mut decoded_len = state.tokenizer.decode(&token_ids, true).unwrap_or_default().len();
    generate_turn(
        &mut state.model, &state.tokenizer,
        &mut token_ids, 0,
        state.eos_token_id, state.max_new_tokens,
        &mut logits_proc, &state.device,
        &mut decoded_len, false,
    )?;

    println!();
    eprintln!("─────────────────────────────────────────────────────────────");
    Ok(())
}

// ── Interactive multi-turn mode ────────────────────────────────────────────

fn run_interactive(mut state: ModelState) -> Result<()> {
    eprintln!("\n─── interactive chat ────────────────────────────────────────");
    eprintln!("Type your message and press Enter. Empty line or Ctrl+C to quit.");
    eprintln!("─────────────────────────────────────────────────────────────");

    let mut logits_proc = LogitsProcessor::new(42, Some(state.temperature), Some(state.top_p));

    let system_text = state.model.system_prefix(DEFAULT_SYSTEM_PROMPT);
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

        // First turn: batch-prefill the whole context (system + user message)
        // from position 0.  Subsequent turns: only prefill the new tokens and
        // let the KV cache supply the prior context.
        // Note: Llama requires sequential single-token prefill when
        // prefill_start > 0 due to an attention-mask shape constraint —
        // generate_turn handles that internally.
        let prefill_start = if first_turn {
            first_turn = false;
            0
        } else {
            token_ids.len()
        };

        let user_turn = state.model.user_turn_text(user_input);
        let turn_enc = state.tokenizer.encode(user_turn.as_str(), false).map_err(E::msg)?;
        token_ids.extend_from_slice(turn_enc.get_ids());

        print!("[assistant] ");
        std::io::stdout().flush()?;

        let mut decoded_len = state.tokenizer.decode(&token_ids, true).unwrap_or_default().len();
        generate_turn(
            &mut state.model, &state.tokenizer,
            &mut token_ids, prefill_start,
            state.eos_token_id, state.max_new_tokens,
            &mut logits_proc, &state.device,
            &mut decoded_len, false,
        )?;

        println!();
    }

    eprintln!("\n─────────────────────────────────────────────────────────────");
    Ok(())
}

// ── Core generation ────────────────────────────────────────────────────────

/// Batch-prefills `token_ids[prefill_start..]` then autoregressively decodes.
///
/// Llama: batch prefill only works from position 0 due to an attention-mask
/// shape constraint in candle.  For subsequent turns (prefill_start > 0) it
/// falls back to sequential single-token prefill.
/// Qwen2: batch prefill works at any position; the internal KV cache resets
/// automatically when called with index_pos = 0.
///
/// `silent` suppresses stdout — used in server mode where the response is
/// returned over HTTP rather than printed to the terminal.
fn generate_turn(
    model: &mut Model,
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

    // Prefill — produces logits for the last token of the prompt.
    let mut logits = match model {
        Model::Llama { inner, cache, .. } => {
            if prefill_start == 0 {
                let input = Tensor::new(&token_ids[..prefill_end], device)?.unsqueeze(0)?;
                inner.forward(&input, 0, cache)?.squeeze(0)?
            } else {
                let mut last = None;
                for i in prefill_start..prefill_end {
                    let input = Tensor::new(&[token_ids[i]], device)?.unsqueeze(0)?;
                    last = Some(inner.forward(&input, i, cache)?.squeeze(0)?);
                }
                last.ok_or_else(|| E::msg("empty prefill range"))?
            }
        }
        Model::Qwen2(weights) => {
            let input =
                Tensor::new(&token_ids[prefill_start..prefill_end], device)?.unsqueeze(0)?;
            weights.forward(&input, prefill_start)?.squeeze(0)?
        }
    };

    // Decode — sample one token at a time until EOS or max_new_tokens.
    for _ in 0..max_new_tokens {
        let next_token = logits_proc.sample(&logits)?;
        token_ids.push(next_token);

        if next_token == eos_token_id {
            break;
        }

        if !silent {
            if let Ok(full_text) = tokenizer.decode(token_ids, true) {
                if full_text.len() > *decoded_len {
                    // BPE decoding can shift byte offsets when new tokens are
                    // appended (space insertion, multi-byte codepoints).  Walk
                    // forward to the nearest valid char boundary so we never
                    // slice inside a multi-byte character.
                    let mut start = *decoded_len;
                    while start < full_text.len() && !full_text.is_char_boundary(start) {
                        start += 1;
                    }
                    print!("{}", &full_text[start..]);
                    std::io::stdout().flush()?;
                    *decoded_len = full_text.len();
                }
            }
        }

        let n = token_ids.len();
        let input = Tensor::new(&[token_ids[n - 1]], device)?.unsqueeze(0)?;
        logits = match model {
            Model::Llama { inner, cache, .. } => {
                inner.forward(&input, n - 1, cache)?.squeeze(0)?
            }
            Model::Qwen2(weights) => weights.forward(&input, n - 1)?.squeeze(0)?,
        };
    }

    Ok(())
}

// ── Helpers ────────────────────────────────────────────────────────────────

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
