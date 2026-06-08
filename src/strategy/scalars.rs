//! Scalar/base constructors: integers/booleans/floats/none/just/nothing/sampled_from/
//! one_of/tuples/lists/slices/fractions/decimals + shared validation helpers.
#![allow(clippy::wildcard_imports)]
use super::*;

/// Coerce an integers() bound to a BigInt, matching upstream validation: an int passes
/// through; an integral float/Decimal (1.0, Decimal('2')) is accepted; a non-integral one
/// (0.1, Decimal('1.5')) raises "cannot be exactly represented as an integer"; NaN/inf/non-real
/// raise "Invalid end point". Without this the `Option<BigInt>` signature rejected float/Decimal
/// bounds with a bare PyO3 TypeError, not the InvalidArgument upstream raises.
pub(crate) fn coerce_integer_bound(py: Python<'_>, v: &Bound<'_, PyAny>, name: &str) -> PyResult<BigInt> {
    if v.is_instance_of::<pyo3::types::PyInt>() {
        return v.extract::<BigInt>();
    }
    let repr = v.repr()?.extract::<String>()?;
    // NaN (and non-real types) are not valid endpoints. NaN != NaN, so self-inequality flags it.
    if v.ne(v).unwrap_or(true) {
        return Err(invalid_argument(py, format!("Invalid end point {name}={repr}")));
    }
    // int(v) truncates; inf/non-real raise here → also "Invalid end point".
    let as_int = match py.import("builtins")?.getattr("int")?.call1((v,)) {
        Ok(i) => i,
        Err(_) => return Err(invalid_argument(py, format!("Invalid end point {name}={repr}"))),
    };
    // value != int(value) ⇒ it has a fractional part, so it can't be exactly an integer.
    if v.ne(&as_int)? {
        return Err(invalid_argument(
            py,
            format!(
                "{name}={repr} of type {} cannot be exactly represented as an integer.",
                v.get_type().name()?,
            ),
        ));
    }
    as_int.extract::<BigInt>()
}

/// Coerce a floats() bound to f64, matching upstream: the value must be a real number
/// (numbers.Real) — a complex or string raises "must be a real number" (InvalidArgument),
/// not the bare TypeError the old `Option<f64>` signature produced. int/float/Decimal/Fraction
/// all convert through float(value).
pub(crate) fn coerce_float_bound(py: Python<'_>, v: &Bound<'_, PyAny>, name: &str) -> PyResult<f64> {
    let real = py.import("numbers")?.getattr("Real")?;
    if !v.is_instance(&real)? {
        return Err(invalid_argument(
            py,
            format!("{name}={} must be a real number.", v.repr()?.extract::<String>()?),
        ));
    }
    py.import("builtins")?.getattr("float")?.call1((v,))?.extract::<f64>()
}

#[pyfunction]
#[pyo3(name = "integers", signature = (min_value=None, max_value=None))]
pub(crate) fn integers(
    py: Python<'_>,
    min_value: Option<Bound<'_, PyAny>>,
    max_value: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let min_value = match min_value {
        Some(v) => Some(coerce_integer_bound(py, &v, "min_value")?),
        None => None,
    };
    let max_value = match max_value {
        Some(v) => Some(coerce_integer_bound(py, &v, "max_value")?),
        None => None,
    };
    if let (Some(mn), Some(mx)) = (&min_value, &max_value) {
        if mn > mx {
            return deferred_invalid(
                py,
                format!("min_value={mn} cannot be greater than max_value={mx}"),
            );
        }
    }
    SearchStrategy::wrap(py, StrategyNode::Integers { min: min_value, max: max_value })
}

#[pyfunction]
#[pyo3(name = "booleans")]
pub(crate) fn booleans(py: Python<'_>) -> PyResult<Py<PyAny>> {
    SearchStrategy::wrap(py, StrategyNode::Booleans)
}

pub(crate) fn fmt_float_opt(py: Python<'_>, v: Option<f64>) -> String {
    match v {
        None => "None".to_string(),
        Some(f) => PyFloat::new(py, f)
            .repr()
            .map(|r| r.to_string())
            .unwrap_or_else(|_| format!("{f}")),
    }
}

#[pyfunction]
#[pyo3(name = "floats", signature = (min_value=None, max_value=None, *, allow_nan=None, allow_infinity=None, allow_subnormal=None, width=64, exclude_min=None, exclude_max=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn floats(
    py: Python<'_>,
    // Accept any value so a non-real bound (complex, string) raises InvalidArgument via
    // coerce_float_bound, not the bare TypeError the old `Option<f64>` produced.
    min_value: Option<Bound<'_, PyAny>>,
    max_value: Option<Bound<'_, PyAny>>,
    // Accept any value (not just bool): upstream treats these flags by truthiness, so e.g.
    // allow_infinity=0 is valid (== False). Coerced below.
    allow_nan: Option<Bound<'_, PyAny>>,
    allow_infinity: Option<Bound<'_, PyAny>>,
    allow_subnormal: Option<Bound<'_, PyAny>>,
    width: u32,
    // exclude_min/max are strict bools (exclude_min=None must raise InvalidArgument, not a
    // PyO3 arg-parse TypeError); validated via check_bool_flag below.
    exclude_min: Option<Bound<'_, PyAny>>,
    exclude_max: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let min_value = match min_value {
        Some(v) => Some(coerce_float_bound(py, &v, "min_value")?),
        None => None,
    };
    let max_value = match max_value {
        Some(v) => Some(coerce_float_bound(py, &v, "max_value")?),
        None => None,
    };
    let exclude_min = match exclude_min {
        Some(v) => check_bool_flag(py, &v, "exclude_min")?,
        None => false,
    };
    let exclude_max = match exclude_max {
        Some(v) => check_bool_flag(py, &v, "exclude_max")?,
        None => false,
    };
    let allow_nan = match allow_nan {
        Some(v) => Some(v.is_truthy()?),
        None => None,
    };
    let allow_infinity = match allow_infinity {
        Some(v) => Some(v.is_truthy()?),
        None => None,
    };
    let allow_subnormal = match allow_subnormal {
        Some(v) => Some(v.is_truthy()?),
        None => None,
    };
    // A finite bound that rounds to infinity at the target width is out of range —
    // raise OverflowError (mirrors struct.pack / float_of). The width=64 case where a
    // huge Python int won't fit in f64 already raises OverflowError at extraction.
    if matches!(width, 16 | 32) {
        for v in [min_value, max_value].into_iter().flatten() {
            if v.is_finite() && crate::floats::narrow_to_width(v, width).is_infinite() {
                return Err(pyo3::exceptions::PyOverflowError::new_err(format!(
                    "min_value/max_value out of range for a float of width {width}"
                )));
            }
        }
    }
    // Other validation errors are DEFERRED (raised at validate()/draw, not
    // construction), matching hypothesis's lazy strategy validation.
    match floats_params(
        py, min_value, max_value, allow_nan, allow_infinity, allow_subnormal, width, exclude_min,
        exclude_max,
    ) {
        Ok((min, max, allow_nan, allow_inf, snm)) => SearchStrategy::wrap(
            py,
            StrategyNode::Floats { min, max, allow_nan, allow_inf, snm, width },
        ),
        Err(msg) => deferred_invalid(py, msg),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn floats_params(
    py: Python<'_>,
    min_value: Option<f64>,
    max_value: Option<f64>,
    allow_nan: Option<bool>,
    allow_infinity: Option<bool>,
    allow_subnormal: Option<bool>,
    width: u32,
    exclude_min: bool,
    exclude_max: bool,
) -> Result<(f64, f64, bool, bool, f64), String> {
    if !matches!(width, 16 | 32 | 64) {
        return Err(format!(
            "Got width={width}, but the only valid values are the integers 16, 32, and 64."
        ));
    }
    if min_value.map_or(false, |v| v.is_nan()) {
        return Err("min_value=nan is not a valid float bound (cannot be NaN)".to_string());
    }
    if max_value.map_or(false, |v| v.is_nan()) {
        return Err("max_value=nan is not a valid float bound (cannot be NaN)".to_string());
    }
    // A finite bound that doesn't round-trip exactly through the target width is not
    // exactly representable (the out-of-range case is raised as OverflowError before
    // we get here). Mirrors upstream's `min_value != float_of(min_value, width)` check.
    if width != 64 {
        for (v, name) in [(min_value, "min_value"), (max_value, "max_value")] {
            if let Some(v) = v {
                let n = crate::floats::narrow_to_width(v, width);
                if n.is_finite() && n != v {
                    return Err(format!(
                        "{name}={} cannot be exactly represented as a float of width {} - use {name}={} instead.",
                        fmt_float_opt(py, Some(v)),
                        width,
                        fmt_float_opt(py, Some(n))
                    ));
                }
            }
        }
    }
    let allow_nan = match allow_nan {
        None => min_value.is_none() && max_value.is_none(),
        Some(true) => {
            if min_value.is_some() || max_value.is_some() {
                return Err("Cannot have allow_nan=True, with min_value or max_value".to_string());
            }
            true
        }
        Some(false) => false,
    };

    let min_arg = min_value;
    let max_arg = max_value;
    if exclude_min && (min_value.is_none() || min_value == Some(f64::INFINITY)) {
        return Err(format!("Cannot exclude min_value={}", fmt_float_opt(py, min_value)));
    }
    if exclude_max && (max_value.is_none() || max_value == Some(f64::NEG_INFINITY)) {
        return Err(format!("Cannot exclude max_value={}", fmt_float_opt(py, max_value)));
    }

    let mut min_value = min_value;
    let mut max_value = max_value;
    // Excluding an endpoint moves it to the adjacent float. A single next_up/next_down
    // across zero only flips the sign (e.g. next_down(0.0) == -0.0, still numerically
    // equal to the bound), so step again in that case — mirrors the signed-zero handling
    // in upstream floats() (next_up_normal/next_down_normal double-step).
    // Excluding an endpoint moves it to the adjacent value representable at `width`. At width
    // 16/32 this MUST step in the width's own ULP: a float64 next_up of e.g. 0.0 rounds back to
    // 0.0 when narrowed, and stepping float64-ULP-by-ULP up to the smallest narrow subnormal is
    // ~10^300 iterations (an effective hang). next_up_width/next_down_width step at the width.
    if exclude_min {
        min_value = Some(crate::floats::next_up_width(min_value.unwrap(), width));
    }
    if exclude_max {
        max_value = Some(crate::floats::next_down_width(max_value.unwrap(), width));
    }
    if min_value == Some(f64::NEG_INFINITY) {
        min_value = None;
    }
    if max_value == Some(f64::INFINITY) {
        max_value = None;
    }

    if let (Some(mn), Some(mx)) = (min_value, max_value) {
        // `min=+0.0, max=-0.0` is an empty (inverted) zero interval even though
        // +0.0 == -0.0 numerically (upstream's bad_zero_bounds).
        let bad_zero_bounds =
            mn == 0.0 && mx == 0.0 && mx.is_sign_negative() && !mn.is_sign_negative();
        if mn > mx || bad_zero_bounds {
            let mut msg = format!(
                "There are no {width}-bit floating-point values between min_value={} and max_value={}",
                fmt_float_opt(py, min_arg),
                fmt_float_opt(py, max_arg)
            );
            if exclude_min || exclude_max {
                msg += &format!(
                    ", exclude_min={} and exclude_max={}",
                    if exclude_min { "True" } else { "False" },
                    if exclude_max { "True" } else { "False" }
                );
            }
            return Err(msg);
        }
    }

    let allow_infinity = match allow_infinity {
        None => min_value.is_none() || max_value.is_none(),
        Some(true) => {
            if min_value.is_some() && max_value.is_some() {
                return Err(
                    "Cannot have allow_infinity=True, with both min_value and max_value".to_string(),
                );
            }
            true
        }
        Some(false) => {
            if min_value == Some(f64::INFINITY) {
                if min_arg == Some(f64::INFINITY) {
                    return Err("allow_infinity=False excludes min_value=inf".to_string());
                }
                return Err(format!(
                    "exclude_min=True turns min_value={} into inf, but allow_infinity=False",
                    fmt_float_opt(py, min_arg)
                ));
            }
            if max_value == Some(f64::NEG_INFINITY) {
                if max_arg == Some(f64::NEG_INFINITY) {
                    return Err("allow_infinity=False excludes max_value=-inf".to_string());
                }
                return Err(format!(
                    "exclude_max=True turns max_value={} into -inf, but allow_infinity=False",
                    fmt_float_opt(py, max_arg)
                ));
            }
            false
        }
    };

    let smallest_normal: f64 = match width {
        16 => 2f64.powi(-14),
        32 => 2f64.powi(-126),
        _ => f64::MIN_POSITIVE,
    };
    let allow_subnormal = match allow_subnormal {
        None => match (min_value, max_value) {
            (Some(mn), Some(mx)) => {
                if mn == mx {
                    -smallest_normal < mn && mn < smallest_normal
                } else {
                    mn < smallest_normal && mx > -smallest_normal
                }
            }
            (Some(mn), None) => mn < smallest_normal,
            (None, Some(mx)) => mx > -smallest_normal,
            (None, None) => true,
        },
        Some(b) => b,
    };
    if allow_subnormal {
        if let Some(mn) = min_value {
            if mn >= smallest_normal {
                return Err(format!(
                    "allow_subnormal=True, but minimum value {} excludes values below float{}'s smallest positive normal {}",
                    fmt_float_opt(py, Some(mn)), width, fmt_float_opt(py, Some(smallest_normal))
                ));
            }
        }
        if let Some(mx) = max_value {
            if mx <= -smallest_normal {
                return Err(format!(
                    "allow_subnormal=True, but maximum value {} excludes values above float{}'s smallest negative normal {}",
                    fmt_float_opt(py, Some(mx)), width, fmt_float_opt(py, Some(-smallest_normal))
                ));
            }
        }
    }
    let snm = if allow_subnormal { SMALLEST_SUBNORMAL } else { smallest_normal };

    Ok((
        min_value.unwrap_or(f64::NEG_INFINITY),
        max_value.unwrap_or(f64::INFINITY),
        allow_nan,
        allow_infinity,
        snm,
    ))
}

pub(crate) fn deferred_invalid(py: Python<'_>, msg: String) -> PyResult<Py<PyAny>> {
    SearchStrategy::wrap(py, StrategyNode::Invalid { msg, resolution_failed: false })
}

/// A deferred ResolutionFailed (a subclass of InvalidArgument) — raised at validate()/draw,
/// for a from_type resolution that produced an empty strategy (matches upstream `as_strategy`).
pub(crate) fn deferred_resolution_failed(py: Python<'_>, msg: String) -> PyResult<Py<PyAny>> {
    SearchStrategy::wrap(py, StrategyNode::Invalid { msg, resolution_failed: true })
}

/// Parse a collection size argument: a non-negative int or None (→ default).
/// Returns Err(message) for non-int / negative, so the caller can defer it.
pub(crate) fn parse_size(
    val: Option<Bound<'_, PyAny>>,
    name: &str,
    default: usize,
) -> Result<usize, String> {
    match val {
        None => Ok(default),
        Some(v) if v.is_none() => Ok(default),
        Some(v) => match v.extract::<i64>() {
            Ok(n) if n >= 0 => Ok(n as usize),
            Ok(n) => Err(format!("{name}={n} must be non-negative")),
            Err(_) => {
                let r = v.repr().map(|s| s.to_string()).unwrap_or_default();
                Err(format!("{name}={r} must be an integer or None"))
            }
        },
    }
}

pub(crate) fn strat_is_empty(py: Python<'_>, s: &Py<PyAny>) -> bool {
    s.bind(py)
        .downcast::<SearchStrategy>()
        .map(|ss| node_is_empty(&ss.borrow().node, py).unwrap_or(false))
        .unwrap_or(false)
}

/// Validate collection size + element constraints; returns (min, max) or Err(msg).
pub(crate) fn collection_sizes(
    py: Python<'_>,
    min_size: Option<Bound<'_, PyAny>>,
    max_size: Option<Bound<'_, PyAny>>,
    kind: &str,
    empty_elem: bool,
    elem_repr: &str,
) -> Result<(usize, usize), String> {
    let max_explicit = max_size.as_ref().is_some_and(|o| !o.is_none());
    let min = parse_size(min_size, "min_size", 0)?;
    let max = parse_size(max_size, "max_size", COLLECTION_DEFAULT_MAX_SIZE)?;
    if min > max {
        return Err(format!("min_size={min} cannot be greater than max_size={max}"));
    }
    // Upstream rejects min_size > BUFFER_SIZE (8 * 1024): such a collection "can never
    // generate an example" since the choice buffer can't hold that many elements. Without
    // this guard the engine would attempt to draw min_size elements and hang (the per-test
    // thread timeout can't interrupt the GIL-holding Rust draw loop).
    const BUFFER_SIZE: usize = 8 * 1024;
    if min > BUFFER_SIZE {
        return Err(format!(
            "Cannot create a {kind} of min_size={min}: it can never generate an example, \
             because min_size is larger than the maximum buffer size {BUFFER_SIZE}"
        ));
    }
    if min > 0 && empty_elem {
        return Err(format!(
            "Cannot create non-empty {kind} with elements drawn from strategy {elem_repr} because it has no values"
        ));
    }
    // An explicit positive max_size with an empty element strategy is unsatisfiable (you asked
    // for up to N>0 elements but none can be drawn). text() is exempt: an empty alphabet still
    // yields the empty string.
    if empty_elem && max_explicit && max > 0 && kind != "text" {
        return Err(format!(
            "Cannot create a {kind} of max_size={max}, because no elements can be drawn from \
             the element strategy {elem_repr}"
        ));
    }
    let _ = py;
    Ok((min, max))
}

#[pyfunction]
#[pyo3(name = "none")]
pub(crate) fn none(py: Python<'_>) -> PyResult<Py<PyAny>> {
    SearchStrategy::wrap(py, StrategyNode::NoneVal)
}

#[pyfunction]
#[pyo3(name = "just")]
pub(crate) fn just(py: Python<'_>, value: Py<PyAny>) -> PyResult<Py<PyAny>> {
    SearchStrategy::wrap(py, StrategyNode::Just(value))
}

#[pyfunction]
#[pyo3(name = "nothing")]
pub(crate) fn nothing(py: Python<'_>) -> PyResult<Py<PyAny>> {
    SearchStrategy::wrap(py, StrategyNode::Nothing)
}

pub(crate) fn invalid_argument(py: Python<'_>, msg: String) -> PyErr {
    match py
        .import("hypothesis_fast.errors")
        .and_then(|m| m.getattr("InvalidArgument"))
        .and_then(|c| c.call1((msg,)))
    {
        Ok(inst) => PyErr::from_value(inst),
        Err(e) => e,
    }
}

/// st.data() is not a first-class strategy: map/filter/flatmap/example on it are errors
/// (use @composite). Mirrors DataStrategy.__not_a_first_class_strategy.
pub(crate) fn not_a_first_class_strategy(py: Python<'_>, name: &str) -> PyErr {
    invalid_argument(
        py,
        format!(
            "Cannot call {name} on a DataStrategy. You should probably \
             be using @composite for whatever it is you're trying to do."
        ),
    )
}

pub(crate) fn resolution_failed_err(py: Python<'_>, msg: String) -> PyErr {
    match py
        .import("hypothesis_fast.errors")
        .and_then(|m| m.getattr("ResolutionFailed"))
        .and_then(|c| c.call1((msg,)))
    {
        Ok(inst) => PyErr::from_value(inst),
        Err(e) => e,
    }
}

/// Reject unordered collections (sets/dicts/generators), mirroring
/// conjecture.utils.check_sample — sampling must be reproducible across runs.
pub(crate) fn check_ordered_sample(py: Python<'_>, values: &Bound<'_, PyAny>, name: &str) -> PyResult<()> {
    let abc = py.import("collections.abc")?;
    let is_seq = values.is_instance(&abc.getattr("Sequence")?)?;
    let is_enum = values.is_instance(&py.import("enum")?.getattr("EnumMeta")?)?;
    let is_od = values.is_instance(&py.import("collections")?.getattr("OrderedDict")?)?;
    if is_seq || is_enum || is_od {
        return Ok(());
    }
    // A 1-D numpy ndarray is an ordered, reproducible sample (it isn't an abc.Sequence, but
    // upstream's check_sample special-cases it). Higher-dimensional arrays are ambiguous.
    if let Ok(modules) = py.import("sys").and_then(|s| s.getattr("modules")) {
        if modules.contains("numpy").unwrap_or(false) {
            let np = py.import("numpy")?;
            if values.is_instance(&np.getattr("ndarray")?)? {
                let ndim: usize = values.getattr("ndim")?.extract()?;
                if ndim == 1 {
                    return Ok(());
                }
                return Err(invalid_argument(
                    py,
                    format!(
                        "Only one-dimensional arrays are supported for sampling, and you \
                         passed a {ndim}-dimensional array."
                    ),
                ));
            }
        }
    }
    Err(invalid_argument(
        py,
        format!(
            "Cannot sample from {}, not an ordered collection. Hypothesis needs \
             stable results between runs for the {} strategy, ruling out sets/dicts \
             due to hash randomization. Use `sorted(values)` for a stable order.",
            values.repr().map(|r| r.to_string()).unwrap_or_default(),
            name,
        ),
    ))
}

#[pyfunction]
#[pyo3(name = "sampled_from")]
pub(crate) fn sampled_from(py: Python<'_>, elements: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    check_ordered_sample(py, &elements, "sampled_from")?;
    // A `range` is kept lazily (never materialised), so sampled_from(range(10**100)) is O(1)
    // to build and repr instead of trying to realise 10^100 ints — which would OOM-kill the
    // worker (test_sampled_repr_leaves_range_as_range). Matches upstream's check_sample.
    let range_ty = py.import("builtins")?.getattr("range")?;
    if elements.is_instance(&range_ty)? {
        return SearchStrategy::wrap(
            py,
            StrategyNode::SampledFromRange { range: elements.clone().unbind() },
        );
    }
    let is_tuple = elements.is_instance_of::<PyTuple>();
    let mut out = Vec::new();
    for it in elements.try_iter()? {
        out.push(it?.unbind());
    }
    if out.is_empty() {
        // An empty Flag enum (e.g. `Flag("X", {"a": 0})`) iterates to no members, but it is
        // still satisfiable: its only value is the zero/empty flag, so sample that
        // (test_can_sample_enums[EmptyFlag]).
        if let Ok(flag_cls) = py.import("enum").and_then(|m| m.getattr("Flag")) {
            if elements.is_instance(&py.import("builtins")?.getattr("type")?)?
                && elements
                    .downcast::<pyo3::types::PyType>()
                    .map(|t| t.is_subclass(&flag_cls).unwrap_or(false))
                    .unwrap_or(false)
            {
                if let Ok(zero) = elements.call1((0,)) {
                    return SearchStrategy::wrap(
                        py,
                        StrategyNode::SampledFrom { elements: vec![zero.unbind()], is_tuple: false },
                    );
                }
            }
        }
        // An empty Enum that nonetheless has annotations: the user probably wrote an enum
        // as if it were a dataclass (`a: int` instead of `a = ...`) — suggest that
        // (test_suggests_elements_instead_of_annotations).
        if let Ok(enum_cls) = py.import("enum").and_then(|m| m.getattr("Enum")) {
            let is_enum = elements
                .downcast::<pyo3::types::PyType>()
                .map(|t| t.is_subclass(&enum_cls).unwrap_or(false))
                .unwrap_or(false);
            if is_enum {
                let has_ann = py
                    .import("annotationlib")
                    .and_then(|al| al.getattr("get_annotations"))
                    .and_then(|f| f.call1((&elements,)))
                    .and_then(|a| a.is_truthy())
                    .or_else(|_| {
                        elements.getattr("__annotations__").and_then(|a| a.is_truthy())
                    })
                    .unwrap_or(false);
                if has_ann {
                    let module: String =
                        elements.getattr("__module__").and_then(|m| m.extract()).unwrap_or_default();
                    let name: String =
                        elements.getattr("__name__").and_then(|m| m.extract()).unwrap_or_default();
                    return deferred_invalid(
                        py,
                        format!(
                            "Cannot sample from {module}.{name} because it contains no \
                             elements.  It does however have annotations, so maybe you tried \
                             to write an enum as if it was a dataclass?"
                        ),
                    );
                }
            }
        }
        // Upstream sampled_from([]) raises InvalidArgument (lazily, at validate/draw),
        // rather than producing an empty strategy — e.g. from_type of an empty Enum.
        return deferred_invalid(py, "Cannot sample from a length-zero sequence.".to_string());
    }
    SearchStrategy::wrap(py, StrategyNode::SampledFrom { elements: out, is_tuple })
}

/// Whether `cb` is usable as a child strategy: a native SearchStrategy, or a foreign
/// (real-hypothesis) strategy — checked by `isinstance(real SearchStrategy)`, NOT by
/// `hasattr("do_draw")`. A decoy that merely defines `do_draw` (test's `Sneaky`) is therefore
/// rejected, matching upstream's check_strategy (test_data_explicitly_rejects_non_strategies).
pub(crate) fn is_strategy(cb: &Bound<'_, PyAny>) -> bool {
    if cb.downcast::<SearchStrategy>().is_ok() {
        return true;
    }
    let py = cb.py();
    if let Ok(real_ss) = py
        .import("hypothesis_fast.strategies")
        .and_then(|m| m.call_method0("_real_hypothesis"))
        .and_then(|h| h.getattr("strategies"))
        .and_then(|s| s.getattr("SearchStrategy"))
    {
        return cb.is_instance(&real_ss).unwrap_or(false);
    }
    false
}

/// Raise InvalidArgument if `cb` isn't a strategy — used by lists/tuples/dictionaries/... to
/// reject a non-strategy child (e.g. `lists('hi')`, `tuples(1)`) with InvalidArgument, not the
/// bare TypeError a downstream draw would produce. Mirrors upstream's check_strategy.
pub(crate) fn check_strategy(py: Python<'_>, cb: &Bound<'_, PyAny>, name: &str) -> PyResult<()> {
    if !is_strategy(cb) {
        return Err(invalid_argument(
            py,
            format!(
                "Expected a SearchStrategy but got {name}={} (type={})",
                cb.repr()?.extract::<String>()?,
                cb.get_type().name()?,
            ),
        ));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "one_of", signature = (*args))]
pub(crate) fn one_of(py: Python<'_>, args: Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    // accept either one_of(s1, s2, ...) or one_of([s1, s2, ...])
    let mut raw: Vec<Py<PyAny>> = Vec::new();
    // one_of([s1, s2, ...]) — a single ITERABLE non-strategy arg is unpacked. A single
    // non-iterable non-strategy (one_of(1)) is NOT unpacked; it stays a lone element so the
    // all-non-strategy check below raises InvalidArgument, rather than `1` raising a bare
    // TypeError from try_iter (test_validates_args).
    let single_iterable = args.len() == 1
        && args.get_item(0)?.downcast::<SearchStrategy>().is_err()
        && args.get_item(0)?.try_iter().is_ok();
    if single_iterable {
        for it in args.get_item(0)?.try_iter()? {
            raw.push(it?.unbind());
        }
    } else {
        for it in args.try_iter()? {
            raw.push(it?.unbind());
        }
    }
    let invalid = || -> PyResult<Bound<'_, PyAny>> {
        py.import("hypothesis_fast.errors")?.getattr("InvalidArgument")
    };
    // All-non-strategy args almost always mean sampled_from was intended.
    if !raw.is_empty() && raw.iter().all(|c| !is_strategy(&c.bind(py))) {
        let lst = PyList::new(py, raw.iter().map(|c| c.bind(py)))?;
        let msg = format!("Did you mean st.sampled_from({})?", lst.repr()?);
        return Err(PyErr::from_value(invalid()?.call1((msg,))?));
    }
    // Validate every element is a strategy. (No flattening here: one_of's repr shows
    // its args verbatim — `one_of(a, one_of(b, c))` stays nested. `__or__` flattens.)
    for c in &raw {
        let cb = c.bind(py);
        if !is_strategy(&cb) {
            let msg = format!(
                "Expected a SearchStrategy but got {} (type={})",
                cb.repr()?,
                cb.get_type().name()?
            );
            return Err(PyErr::from_value(invalid()?.call1((msg,))?));
        }
    }
    // A single strategy is a no-op: return it unchanged (so `one_of(s) is s`).
    if raw.len() == 1 {
        return Ok(raw.into_iter().next().unwrap());
    }
    SearchStrategy::wrap(py, StrategyNode::OneOf(raw))
}

#[pyfunction]
#[pyo3(name = "tuples", signature = (*args))]
pub(crate) fn tuples(py: Python<'_>, args: Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let mut children = Vec::new();
    for it in args.try_iter()? {
        let child = it?;
        check_strategy(py, &child, "tuples() argument")?;
        children.push(child.unbind());
    }
    SearchStrategy::wrap(py, StrategyNode::Tuples(children))
}

#[pyfunction]
#[pyo3(name = "lists", signature = (elements, *, min_size=None, max_size=None, unique_by=None, unique=false))]
pub(crate) fn lists(
    py: Python<'_>,
    elements: Py<PyAny>,
    min_size: Option<Bound<'_, PyAny>>,
    max_size: Option<Bound<'_, PyAny>>,
    unique_by: Option<Py<PyAny>>,
    unique: bool,
) -> PyResult<Py<PyAny>> {
    check_strategy(py, &elements.bind(py), "lists() elements")?;
    if unique && unique_by.is_some() {
        return deferred_invalid(
            py,
            "Cannot pass both unique and unique_by (you probably only want to use unique_by)"
                .to_string(),
        );
    }
    // unique_by must be a callable or a non-empty tuple of callables (upstream check):
    // unique_by=1, (), and (1,) are all rejected with InvalidArgument.
    if let Some(k) = &unique_by {
        let kb = k.bind(py);
        let ok = if let Ok(t) = kb.downcast::<PyTuple>() {
            !t.is_empty() && t.iter().all(|f| f.is_callable())
        } else {
            kb.is_callable()
        };
        if !ok {
            return Err(invalid_argument(
                py,
                format!(
                    "unique_by={} must be a callable or a (non-empty) tuple of callables.",
                    kb.repr()?.extract::<String>()?
                ),
            ));
        }
    }
    let er = elements.bind(py).repr().map(|s| s.to_string()).unwrap_or_default();
    let (min, max) = match collection_sizes(py, min_size, max_size, "lists", strat_is_empty(py, &elements), &er) {
        Ok(v) => v,
        Err(m) => return deferred_invalid(py, m),
    };
    // Plain `unique=True` over a small finite domain: sample WITHOUT replacement (pop
    // distinct values from a copy) instead of rejecting duplicates, so a near-exhaustive
    // unique list (e.g. 255 of 256 int8 values) doesn't hit the birthday paradox.
    let swap_domain: Option<Vec<Py<PyAny>>> = if unique && unique_by.is_none() {
        finite_unique_domain(py, &elements)?
    } else {
        None
    };
    let key = match unique_by {
        Some(k) => Some(k),
        None if unique => Some(py.eval(c"(lambda x: x)", None, None)?.unbind()),
        None => None,
    };
    // A unique list holds at most as many elements as the domain has distinct values
    // (hypothesis caps max_size to that count and rejects an impossible min_size up front).
    let card: Option<usize> = match &swap_domain {
        Some(d) => Some(d.len()),
        None if key.is_some() => sampled_from_len(py, &elements),
        None => None,
    };
    let max = match card {
        Some(n) => {
            if min > n {
                return deferred_invalid(
                    py,
                    format!(
                        "Cannot create a collection of min_size={min} unique elements with \
                         values drawn from only {n} distinct elements"
                    ),
                );
            }
            std::cmp::min(max, n)
        }
        None => max,
    };
    SearchStrategy::wrap(
        py,
        StrategyNode::Lists { elem: elements, min, max, unique_by: key, swap_domain },
    )
}

/// The explicit value domain for a plain `unique=True` list over a small finite element
/// strategy (or None). A bounded `integers()` (range <= 255, matching hypothesis) is
/// materialised, ordered by absolute value (small |x| first) for shrink-friendliness; an
/// explicit `sampled_from` uses its elements as-is. Lets us sample WITHOUT replacement.
fn finite_unique_domain(py: Python<'_>, elements: &Py<PyAny>) -> PyResult<Option<Vec<Py<PyAny>>>> {
    let bound = elements.bind(py);
    let ss = match bound.downcast::<SearchStrategy>() {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };
    let borrowed = ss.borrow();
    match &borrowed.node {
        StrategyNode::SampledFrom { elements, .. } => {
            Ok(Some(elements.iter().map(|e| e.clone_ref(py)).collect()))
        }
        StrategyNode::Integers { min: Some(a), max: Some(b) } if (b - a) <= BigInt::from(255u32) => {
            let ordered = ordered_int_domain(&a.clone(), &b.clone());
            drop(borrowed);
            let mut out: Vec<Py<PyAny>> = Vec::with_capacity(ordered.len());
            for v in ordered {
                out.push(v.into_pyobject(py)?.into_any().unbind());
            }
            Ok(Some(out))
        }
        _ => Ok(None),
    }
}

/// Values of `a..=b` ordered by absolute value (smallest |x| first), mirroring hypothesis's
/// integers->sampled_from rewrite so shrinking prefers small-magnitude values.
fn ordered_int_domain(a: &BigInt, b: &BigInt) -> Vec<BigInt> {
    let zero = BigInt::from(0);
    let one = BigInt::from(1);
    let mut out = Vec::new();
    if a > &zero || b < &zero {
        // all one sign: smallest |x| (the end nearest zero) first
        if a > &zero {
            let mut v = a.clone();
            while &v <= b {
                out.push(v.clone());
                v += &one;
            }
        } else {
            let mut v = b.clone();
            while &v >= a {
                out.push(v.clone());
                v -= &one;
            }
        }
    } else {
        // straddles zero: 0,1,..,b then -1,-2,..,a
        let mut v = zero.clone();
        while &v <= b {
            out.push(v.clone());
            v += &one;
        }
        let mut v = BigInt::from(-1);
        while &v >= a {
            out.push(v.clone());
            v -= &one;
        }
    }
    out
}

/// If `strat` is a `sampled_from` strategy, the number of distinct choices it offers.
pub(crate) fn sampled_from_len(py: Python<'_>, strat: &Py<PyAny>) -> Option<usize> {
    let bound = strat.bind(py);
    let ss = bound.downcast::<SearchStrategy>().ok()?;
    let borrowed = ss.borrow();
    match &borrowed.node {
        StrategyNode::SampledFrom { elements, .. } => Some(elements.len()),
        _ => None,
    }
}

#[pyfunction]
#[pyo3(name = "deferred")]
pub(crate) fn deferred(py: Python<'_>, definition: Py<PyAny>) -> PyResult<Py<PyAny>> {
    SearchStrategy::wrap(py, StrategyNode::Deferred { thunk: definition })
}

#[pyfunction]
#[pyo3(name = "slices", signature = (size))]
pub(crate) fn slices(py: Python<'_>, size: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // size must be a non-negative int: None/'chips' aren't ints, 2.3 is a float, -1 is negative.
    // The old `usize` signature rejected those with a bare PyO3 TypeError, not InvalidArgument.
    let n: usize = match size.extract::<i64>() {
        Ok(n) if n >= 0 && size.is_instance_of::<pyo3::types::PyInt>() => n as usize,
        _ => {
            return Err(invalid_argument(
                py,
                format!(
                    "Expected size to be a non-negative integer, but got {}",
                    size.repr()?.extract::<String>()?
                ),
            ));
        }
    };
    SearchStrategy::wrap(py, StrategyNode::Slices { size: n })
}

#[pyfunction]
#[pyo3(name = "fractions", signature = (min_value=None, max_value=None, *, max_denominator=None))]
pub(crate) fn fractions(
    py: Python<'_>,
    min_value: Option<Bound<'_, PyAny>>,
    max_value: Option<Bound<'_, PyAny>>,
    max_denominator: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let frac_cls = py.import("fractions")?.getattr("Fraction")?;
    // Whether max_denominator was explicitly given. Real hypothesis only validates the
    // bound denominators (and only then) when max_denominator is not None; with the
    // default (None) it generates unbounded denominators and accepts any bound. We still
    // cap *generation* at 65536 for the unbounded case, but must not reject a bound whose
    // denominator exceeds that cap (e.g. decimals(min_value=1e-100) → denom ~10**100).
    let bound_denom = matches!(&max_denominator, Some(md) if !md.is_none());
    // max_denominator must be a positive int (or None). The old `Option<i64>` rejected 1.5 with
    // a bare TypeError and silently clamped 0 → 1; upstream raises InvalidArgument for both.
    let max_denom: i64 = match &max_denominator {
        None => 65536,
        Some(md) if md.is_none() => 65536,
        Some(md) => {
            let n = match md.extract::<i64>() {
                Ok(n) if md.is_instance_of::<pyo3::types::PyInt>() => n,
                _ => {
                    return Err(invalid_argument(
                        py,
                        format!(
                            "max_denominator={} must be an integer.",
                            md.repr()?.extract::<String>()?
                        ),
                    ));
                }
            };
            if n < 1 {
                return Err(invalid_argument(
                    py,
                    format!("max_denominator={n} must be >= 1, but the smallest valid denominator is 1."),
                ));
            }
            n
        }
    };
    // Convert each bound to a Fraction (rejecting NaN/complex/unconvertible values), and check
    // its lowest-terms denominator fits within max_denominator (test fractions string bounds).
    let conv = |v: &Bound<'_, PyAny>, name: &str| -> PyResult<Py<PyAny>> {
        let f = match frac_cls.call1((v,)) {
            Ok(f) => f,
            Err(_) => {
                return Err(invalid_argument(
                    py,
                    format!(
                        "Cannot convert {name}={} to a Fraction.",
                        v.repr()?.extract::<String>()?
                    ),
                ));
            }
        };
        // Only reject an over-large bound denominator when max_denominator was given.
        // Compare via Python so a bigint denominator (e.g. 10**100) doesn't overflow i64.
        if bound_denom && f.getattr("denominator")?.gt(max_denom)? {
            return Err(invalid_argument(
                py,
                format!(
                    "The {name}={} has a denominator greater than the max_denominator={max_denom}",
                    f.repr()?.extract::<String>()?
                ),
            ));
        }
        Ok(f.unbind())
    };
    let min = match &min_value {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => Some(conv(v, "min_value")?),
    };
    let max = match &max_value {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => Some(conv(v, "max_value")?),
    };
    if let (Some(mn), Some(mx)) = (&min, &max) {
        if mn.bind(py).gt(mx.bind(py))? {
            return Err(invalid_argument(
                py,
                format!(
                    "min_value={} cannot be greater than max_value={}",
                    mn.bind(py).repr()?.extract::<String>()?,
                    mx.bind(py).repr()?.extract::<String>()?
                ),
            ));
        }
    }
    SearchStrategy::wrap(py, StrategyNode::Fractions { min, max, max_denom })
}

#[pyfunction]
#[pyo3(name = "decimals", signature = (min_value=None, max_value=None, *, allow_nan=None, allow_infinity=None, places=None))]
pub(crate) fn decimals(
    py: Python<'_>,
    min_value: Option<Py<PyAny>>,
    max_value: Option<Py<PyAny>>,
    allow_nan: Option<bool>,
    allow_infinity: Option<bool>,
    places: Option<i32>,
) -> PyResult<Py<PyAny>> {
    let _ = (allow_nan, allow_infinity);
    SearchStrategy::wrap(py, StrategyNode::Decimals { min: min_value, max: max_value, places })
}
