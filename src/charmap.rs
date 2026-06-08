//! Unicode general-category charmap, native (no Python `unicodedata`).
//!
//! Builds, once, a map from each 2-letter general-category code to the codepoint
//! intervals in that category (scanning 0..=0x10FFFF via the `unicode-general-
//! category` crate; surrogates U+D800..U+DFFF are category "Cs"). Used by
//! `characters()` to construct the alphabet IntervalSet for category filters.

use std::collections::HashMap;
use std::sync::OnceLock;
use unicode_general_category::{get_general_category, GeneralCategory};

const MAX_CP: i64 = 0x10FFFF;

fn code(g: GeneralCategory) -> &'static str {
    use GeneralCategory::*;
    match g {
        UppercaseLetter => "Lu",
        LowercaseLetter => "Ll",
        TitlecaseLetter => "Lt",
        ModifierLetter => "Lm",
        OtherLetter => "Lo",
        NonspacingMark => "Mn",
        SpacingMark => "Mc",
        EnclosingMark => "Me",
        DecimalNumber => "Nd",
        LetterNumber => "Nl",
        OtherNumber => "No",
        ConnectorPunctuation => "Pc",
        DashPunctuation => "Pd",
        OpenPunctuation => "Ps",
        ClosePunctuation => "Pe",
        InitialPunctuation => "Pi",
        FinalPunctuation => "Pf",
        OtherPunctuation => "Po",
        MathSymbol => "Sm",
        CurrencySymbol => "Sc",
        ModifierSymbol => "Sk",
        OtherSymbol => "So",
        SpaceSeparator => "Zs",
        LineSeparator => "Zl",
        ParagraphSeparator => "Zp",
        Control => "Cc",
        Format => "Cf",
        Surrogate => "Cs",
        PrivateUse => "Co",
        Unassigned => "Cn",
        _ => "Cn",
    }
}

fn category_of(cp: i64) -> &'static str {
    if (0xD800..=0xDFFF).contains(&cp) {
        return "Cs";
    }
    match u32::try_from(cp).ok().and_then(char::from_u32) {
        Some(c) => code(get_general_category(c)),
        None => "Cn",
    }
}

/// code -> sorted, merged intervals of codepoints in that category.
fn charmap() -> &'static HashMap<&'static str, Vec<(i64, i64)>> {
    static MAP: OnceLock<HashMap<&'static str, Vec<(i64, i64)>>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut map: HashMap<&'static str, Vec<(i64, i64)>> = HashMap::new();
        let mut cur_code = category_of(0);
        let mut start = 0i64;
        for cp in 1..=MAX_CP + 1 {
            let c = if cp <= MAX_CP { category_of(cp) } else { "" };
            if c != cur_code {
                map.entry(cur_code).or_default().push((start, cp - 1));
                cur_code = c;
                start = cp;
            }
        }
        map
    })
}

pub(crate) const ALL_CATEGORIES: [&str; 30] = [
    "Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me", "Nd", "Nl", "No", "Pc", "Pd", "Ps", "Pe",
    "Pi", "Pf", "Po", "Sm", "Sc", "Sk", "So", "Zs", "Zl", "Zp", "Cc", "Cf", "Cs", "Co", "Cn",
];

/// Merged intervals for the union of the given category codes.
pub(crate) fn intervals_for_categories(allowed: &[String]) -> Vec<(i64, i64)> {
    let map = charmap();
    let mut all: Vec<(i64, i64)> = Vec::new();
    for cat in allowed {
        if let Some(v) = map.get(cat.as_str()) {
            all.extend_from_slice(v);
        }
    }
    all.sort_unstable();
    // merge adjacent/overlapping
    let mut merged: Vec<(i64, i64)> = Vec::new();
    for (lo, hi) in all {
        if let Some(last) = merged.last_mut() {
            if lo <= last.1 + 1 {
                last.1 = last.1.max(hi);
                continue;
            }
        }
        merged.push((lo, hi));
    }
    merged
}
