use std::cmp::Ord;
use std::fmt;
use std::rc::Rc;
// Use Rc for cheap sharing of filename

#[derive(Clone, PartialEq, Eq, Hash, Ord, PartialOrd)] // Remove Copy for now due to Rc
pub struct Location {
    // Using Rc makes cloning cheap and avoids lifetime issues if filename stored elsewhere
    pub filename: Rc<String>,
    pub line: usize,
    pub col: usize,
}

// Custom Debug impl to avoid printing the whole Rc internals
impl fmt::Debug for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.filename, self.line, self.col)
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.filename, self.line, self.col)
    }
}

// Implement default manually as Rc<String> doesn't derive Default easily
impl Default for Location {
    fn default() -> Self {
        Location {
            filename: Rc::new("?".to_string()), // Default filename
            line: 1,
            col: 1,
        }
    }
}

// Span represents a range in the source code
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Span {
    pub start: Location,
    pub end: Location,
}

impl Span {
    // Combine two spans to create a larger one encompassing both
    pub fn combine(&self, other: &Self) -> Self {
        // Basic combination: take the earliest start and latest end.
        // Assumes they share the same file. Add checks if needed.
        Span {
            start: if self.start <= other.start {
                self.start.clone()
            } else {
                other.start.clone()
            },
            // Or min(self.start, other.start) if order matters (needs Ord)
            end: if self.end >= other.end {
                self.end.clone()
            } else {
                other.end.clone()
            },
            // Or max(self.end, other.end)
        }
    }

    // Create a span from two locations
    pub fn from_locations(start: Location, end: Location) -> Self {
        Span { start, end }
    }

    // Create a span covering a single location (e.g., for a single char token)
    pub fn single(loc: Location) -> Self {
        // End column is exclusive, so maybe +1? Let's keep it simple for now.
        let mut end_loc = loc.clone();
        end_loc.col += 1;
        Span {
            start: loc,
            end: end_loc,
        }
    }
}

// Display for printing spans in errors
impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Show only start location for simplicity, or range if needed
        write!(
            f,
            "{}:{}:{}",
            self.start.filename, self.start.line, self.start.col
        )
        // write!(f, "{}:{}:{}-{}:{}", self.start.filename, self.start.line, self.start.col, self.end.line, self.end.col)
    }
}
