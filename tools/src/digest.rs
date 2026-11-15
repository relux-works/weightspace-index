use crate::error::ToolError;

/// A SHA-256 digest as 64 lowercase hex characters.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sha256Hex(String);

impl Sha256Hex {
    /// Parses 64 ASCII hex characters, normalized to lowercase.
    pub fn parse(s: &str) -> Result<Self, ToolError> {
        let ok = s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit());
        if ok {
            Ok(Self(s.to_ascii_lowercase()))
        } else {
            Err(ToolError::InvalidArgument {
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
