//! Checks for the Weightspace Index: every JSON record against
//! `schema/records.schema.json`, and the layout rules a single record
//! cannot express. CI and the intake tool share these checks.
//!
//! Nothing here touches the network or executes repository content.

pub mod generate;
pub mod layout;
pub mod records;
