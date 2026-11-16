use crate::error::ToolError;

/// A SHA-256 digest as 64 lowercase hex characters.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sha256Hex(String);

/// [`Sha256Hex::parse`] failure: the input is not 64 hex characters.
/// A library-level parse error with its own type, so parsing a digest
/// never surfaces a CLI-flavoured variant; callers on the operation
/// path convert it with `From` into [`ToolError::ParseDigest`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseDigestError {
    detail: String,
}

impl std::fmt::Display for ParseDigestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.detail)
    }
}

impl std::error::Error for ParseDigestError {}

impl From<ParseDigestError> for ToolError {
    fn from(e: ParseDigestError) -> Self {
        Self::ParseDigest { detail: e.detail }
    }
}

impl Sha256Hex {
    /// Parses 64 ASCII hex characters, normalized to lowercase.
    pub fn parse(s: &str) -> Result<Self, ParseDigestError> {
        let ok = s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit());
        if ok {
            Ok(Self(s.to_ascii_lowercase()))
        } else {
            Err(ParseDigestError {
                detail: format!("expected 64 hex characters of SHA-256, got {s:?}"),
            })
        }
    }

    pub(crate) fn from_digest(bytes: &[u8]) -> Self {
        Self(bytes.iter().map(|b| format!("{b:02x}")).collect())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Sha256Hex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
