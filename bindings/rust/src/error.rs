// Copyright 2026 TerseTS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! The [`Error`] and [`Result`] types used throughout the TerseTS library.

use std::error::Error;
use std::ffi::NulError;
use std::fmt::Result as FmtResult;
use std::fmt::{Display, Formatter};
use std::result::Result as StdResult;

/// Result type used throughout TerseTS.
pub type Result<T> = StdResult<T, TerseTSError>;

/// Error type used throughout TerseTS.
#[derive(Debug)]
pub enum TerseTSError {
    /// Error returned by TerseTS.
    TerseTS(i32),
    /// Error returned by [`CString`](std::ffi::CString).
    Nul(NulError),
}

impl Display for TerseTSError {
    /// Convert this [`TerseTSError`] to a string and write it to `f` for printing.
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        match self {
            Self::TerseTS(reason) => write!(f, "TerseTS Error: {reason}"),
            Self::Nul(reason) => write!(f, "Nul Error: {reason}"),
        }
    }
}

impl Error for TerseTSError {
    /// Return any [`Error`] that caused this [`TerseTSError`].
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::TerseTS(_reason) => None,
            Self::Nul(reason) => Some(reason),
        }
    }
}

impl From<NulError> for TerseTSError {
    /// Automatically convert [`NulError`] to [`TerseTSError`], e.g, with the question mark operator.
    fn from(error: NulError) -> Self {
        Self::Nul(error)
    }
}
