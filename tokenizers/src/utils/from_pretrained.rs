use crate::Result;
use cached_path::CacheBuilder;
use itertools::Itertools;
use reqwest::{blocking::Client, header};
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;

/// Returns a directory to be used as cache.
///
/// If the `TOKENIZERS_CACHE` environment variable is set, we just return it. It is the
/// responsibility of the user to make sure this path is correct.
///
/// Otherwise, we try to use the default cache directory as defined for each OS:
///     - macOS: `/Users/{user}/Library/Caches/huggingface/tokenizers`
///     - linux: `/home/{user}/.cache/huggingface/tokenizers`
///     - windows: `C:\Users\{user}\AppData\Local\huggingface\tokenizers`
/// If the default cache directory cannot be found (if the user HOME folder is not defined),
/// then we fall back on a temporary directory
fn cache_dir() -> PathBuf {
    if let Ok(path) = std::env::var("TOKENIZERS_CACHE") {
        PathBuf::from(path)
    } else {
        let mut dir = dirs::cache_dir().unwrap_or_else(std::env::temp_dir);
        dir.push("huggingface");
        dir.push("tokenizers");
        dir
    }
}

/// Returns a directory to be used as cache, creating it if it doesn't exist
///
/// Cf `cache_dir()` to understand how the cache dir is selected.
fn ensure_cache_dir(path: PathBuf) -> std::io::Result<PathBuf> {
    if !path.exists() {
        std::fs::create_dir_all(&path)?;
    }
    Ok(path)
}

/// Sanitize a key or value to be used inside the user_agent
/// The user_agent uses `/` and `;` to format the key-values, so we
/// replace them by `-`
fn sanitize_user_agent(item: &str) -> Cow<str> {
    let mut sanitized = Cow::Borrowed(item);
    if sanitized.contains('/') {
        sanitized = Cow::Owned(sanitized.replace('/', "-"));
    }
    if sanitized.contains(';') {
        sanitized = Cow::Owned(sanitized.replace(';', "-"));
    }
    sanitized
}

const VERSION: &str = env!("CARGO_PKG_VERSION");

// We allow unstable name collisions in this case because we don't care if it
// starts using the new stable feature when it will be stable. This feature is
// supposed to be a copy of the one we use anyway.
// cf https://github.com/rust-lang/rust/issues/79524
#[allow(unstable_name_collisions)]
fn user_agent(additional_info: HashMap<String, String>) -> String {
    let additional_str: String = additional_info
        .iter()
        .map(|(k, v)| format!("{}/{}", sanitize_user_agent(k), sanitize_user_agent(v)))
        .intersperse("; ".to_string())
        .collect();

    let user_agent = format!(
        "tokenizers/{}{}",
        VERSION,
        if !additional_str.is_empty() {
            format!("; {}", additional_str)
        } else {
            String::new()
        }
    );

    user_agent
}

/// Defines the aditional parameters available for the `from_pretrained` function
#[derive(Debug, Clone)]
pub struct FromPretrainedParameters {
    pub path: Option<String>,
    pub revision: String,
    pub user_agent: HashMap<String, String>,
    pub auth_token: Option<String>,
}

impl Default for FromPretrainedParameters {
    fn default() -> Self {
        Self {
            path: None,
            revision: "main".into(),
            user_agent: HashMap::new(),
            auth_token: None,
        }
    }
}

/// Downloads and cache the identified tokenizer if it exists on
/// the Hugging Face Hub, and returns a local path to the file
pub fn from_pretrained<S: AsRef<str>>(
    identifier: S,
    params: Option<FromPretrainedParameters>,
) -> Result<PathBuf> {
    let params = params.unwrap_or_default();
    let cache_dir =
        ensure_cache_dir(params.path.map(PathBuf::from).unwrap_or_else(|| cache_dir()))?;

    // Build a custom HTTP Client using our user-agent and custom headers
    let mut headers = header::HeaderMap::new();
    if let Some(ref token) = params.auth_token {
        headers.insert(
            "Authorization",
            header::HeaderValue::from_str(&format!("Bearer {}", token))?,
        );
    }
    let client_builder =
        Client::builder().user_agent(user_agent(params.user_agent)).default_headers(headers);

    let url_to_download = format!(
        "https://huggingface.co/{}/resolve/{}/tokenizer.json",
        identifier.as_ref(),
        params.revision,
    );

    let tokenizer: String = client_builder
        .build()
        .unwrap()
        .get(url_to_download)
        .send()
        .expect(&format!(
            "Model \"{}\" on the Hub doesn't have a tokenizer",
            identifier.as_ref(),
        ))
        .text()
        .unwrap();

    let path = cache_dir.as_path().join("tokenizer");

    std::fs::write(&path, tokenizer).unwrap();

    Ok(path)
}
