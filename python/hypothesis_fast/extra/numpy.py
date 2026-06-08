"""Strategies for :mod:`numpy` arrays, dtypes and indices.

Native port of ``hypothesis.extra.numpy``: identical public API and generation
logic, but built on the native ``hypothesis_fast`` strategies so user-supplied
``elements``/``fill`` strategies compose natively (no real-hypothesis
``ConjectureData`` interop boundary). The custom array/index strategies subclass
the native ``SearchStrategy`` and draw against the native ConjectureData via
``data.draw`` / ``data.draw_integer`` / ``cu.many``. Pure helpers
(``cu``/``check_type``/``check_function``/``proxies``/``note_deprecation``) are
reused from real hypothesis unchanged.
"""

from __future__ import annotations

import importlib
import math
import types
from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    overload,
)

import numpy as np

from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.coverage import check_function
from hypothesis.internal.reflection import proxies
from hypothesis.internal.validation import check_type
from hypothesis.utils.deprecation import note_deprecation

from hypothesis_fast import strategies as st
from hypothesis_fast.errors import HypothesisException, InvalidArgument
from hypothesis_fast.extra._array_helpers import (
    NDIM_MAX,
    BasicIndexStrategy,
    BroadcastableShapes,
    Shape,
    array_shapes,
    broadcastable_shapes,
    check_argument,
    check_valid_dims,
    mutually_broadcastable_shapes as _mutually_broadcastable_shapes,
    order_check,
    valid_tuple_axes as _valid_tuple_axes,
)
from hypothesis_fast.extra._lazy import defines_strategy
from hypothesis_fast.native_strategies import check_strategy

if TYPE_CHECKING:
    # For type-checking only, alias the public SearchStrategy name to the real hypothesis
    # generic class (which ships stubs) so `SearchStrategy[...]` annotations resolve; at
    # runtime SearchStrategy IS our native class (the compiled _engine has no stubs, so a
    # bare native annotation reads as a plain variable to pyright).
    from hypothesis.strategies._internal.strategies import SearchStrategy
else:
    from hypothesis_fast.native_strategies import SearchStrategy

# TypeVars used only in annotations (which are lazy strings under future-annotations).
Ex = TypeVar("Ex")
T = TypeVar("T")
Real = Union[int, float]


def _try_import(mod_name: str, attr_name: str) -> Any:
    assert "." not in attr_name
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr_name, None)
    except ImportError:
        return None


if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray
else:
    NDArray = _try_import("numpy.typing", "NDArray")

ArrayLike = _try_import("numpy.typing", "ArrayLike")
_NestedSequence = _try_import("numpy._typing._nested_sequence", "_NestedSequence")
_SupportsArray = _try_import("numpy._typing._array_like", "_SupportsArray")

__all__ = [
    "BroadcastableShapes",
    "array_dtypes",
    "array_shapes",
    "arrays",
    "basic_indices",
    "boolean_dtypes",
    "broadcastable_shapes",
    "byte_string_dtypes",
    "complex_number_dtypes",
    "datetime64_dtypes",
    "floating_dtypes",
    "from_dtype",
    "integer_array_indices",
    "integer_dtypes",
    "mutually_broadcastable_shapes",
    "nested_dtypes",
    "scalar_dtypes",
    "timedelta64_dtypes",
    "unicode_string_dtypes",
    "unsigned_integer_dtypes",
    "valid_tuple_axes",
]

TIME_RESOLUTIONS = ("Y", "M", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as")

# See https://github.com/HypothesisWorks/hypothesis/pull/3394 and linked discussion.
NP_FIXED_UNICODE = tuple(int(x) for x in np.__version__.split(".")[:2]) >= (1, 19)


@defines_strategy(force_reusable_values=True)
def from_dtype(
    dtype: np.dtype,
    *,
    alphabet: SearchStrategy | None = None,
    min_size: int = 0,
    max_size: int | None = None,
    min_value: int | float | None = None,
    max_value: int | float | None = None,
    allow_nan: bool | None = None,
    allow_infinity: bool | None = None,
    allow_subnormal: bool | None = None,
    exclude_min: bool | None = None,
    exclude_max: bool | None = None,
    min_magnitude: Real = 0,
    max_magnitude: Real | None = None,
) -> SearchStrategy:
    """Creates a strategy which can generate any value of the given dtype.

    Compatible parameters are passed to the inferred strategy function while
    inapplicable ones are ignored.
    """
    check_type(np.dtype, dtype, "dtype")
    kwargs = {k: v for k, v in locals().items() if k != "dtype" and v is not None}

    # Compound datatypes, eg 'f4,f4,f4'
    if dtype.names is not None and dtype.fields is not None:
        # mapping np.void.type over a strategy is nonsense, so return now.
        fields: Any = dtype.fields
        subs = [from_dtype(fields[name][0], **kwargs) for name in dtype.names]
        return st.tuples(*subs)

    # Subarray datatypes, eg '(2, 3)i4'
    if dtype.subdtype is not None:
        subtype, shape = dtype.subdtype
        return arrays(subtype, shape, elements=kwargs)

    def compat_kw(*args, **kw):
        """Update default args to the strategy with user-supplied keyword args."""
        assert {"min_value", "max_value", "max_size"}.issuperset(kw)
        for key in set(kwargs).intersection(kw):
            msg = f"dtype {dtype!r} requires {key}={kwargs[key]!r} to be %s {kw[key]!r}"
            if kw[key] is not None:
                if key.startswith("min_") and kw[key] > kwargs[key]:
                    raise InvalidArgument(msg % ("at least",))
                elif key.startswith("max_") and kw[key] < kwargs[key]:
                    raise InvalidArgument(msg % ("at most",))
        kw.update({k: v for k, v in kwargs.items() if k in args or k in kw})
        return kw

    # Scalar datatypes
    if dtype.kind == "b":
        result: SearchStrategy = st.booleans()
    elif dtype.kind == "f":
        result = st.floats(
            width=cast(Literal[16, 32, 64], min(8 * dtype.itemsize, 64)),
            **compat_kw(
                "min_value",
                "max_value",
                "allow_nan",
                "allow_infinity",
                "allow_subnormal",
                "exclude_min",
                "exclude_max",
            ),
        )
    elif dtype.kind == "c":
        result = st.complex_numbers(
            width=cast(
                Literal[32, 64, 128], min(8 * dtype.itemsize, 128)
            ),  # convert from bytes to bits
            **compat_kw(
                "min_magnitude",
                "max_magnitude",
                "allow_nan",
                "allow_infinity",
                "allow_subnormal",
            ),
        )
    elif dtype.kind in ("S", "a"):
        # Numpy strings are null-terminated; only allow round-trippable values.
        # `itemsize == 0` means 'fixed length determined at array creation'
        max_size = dtype.itemsize or None
        result = st.binary(**compat_kw("min_size", max_size=max_size)).filter(
            lambda b: b[-1:] != b"\0"
        )
    elif dtype.kind == "u":
        kw = compat_kw(min_value=0, max_value=2 ** (8 * dtype.itemsize) - 1)
        result = st.integers(**kw)
    elif dtype.kind == "i":
        overflow = 2 ** (8 * dtype.itemsize - 1)
        result = st.integers(**compat_kw(min_value=-overflow, max_value=overflow - 1))
    elif dtype.kind == "U":
        # Encoded in UTF-32 (four bytes/codepoint) and null-terminated
        max_size = (dtype.itemsize or 0) // 4 or None
        if NP_FIXED_UNICODE and "alphabet" not in kwargs:
            kwargs["alphabet"] = st.characters()
        result = st.text(**compat_kw("alphabet", "min_size", max_size=max_size)).filter(
            lambda b: b[-1:] != "\0"
        )
    elif dtype.kind in ("m", "M"):
        if "[" in dtype.str:
            res = st.just(dtype.str.split("[")[-1][:-1])
        else:
            # Note that this case isn't valid to pass to arrays(), but we support
            # it here because we'd have to guard against equivalents in arrays()
            # regardless and drawing scalars is a valid use-case.
            res = st.sampled_from(TIME_RESOLUTIONS)
        if allow_nan is not False:
            elems = st.integers(-(2**63), 2**63 - 1) | st.just("NaT")
        else:  # NEP-7 defines the NaT value as integer -(2**63)
            elems = st.integers(-(2**63) + 1, 2**63 - 1)
        result = st.builds(dtype.type, elems, res)
    elif dtype.kind == "O":
        return st.from_type(object)
    else:
        raise InvalidArgument(f"No strategy inference for {dtype}")
    return result.map(dtype.type)


class ArrayStrategy(SearchStrategy):
    def __init__(self, element_strategy, shape, dtype, fill, unique):
        super().__init__()
        self.shape = tuple(shape)
        self.fill = fill
        self.array_size = int(np.prod(shape))
        self.dtype = dtype
        self.element_strategy = element_strategy
        self.unique = unique
        self._check_elements = dtype.kind not in ("O", "V")

    def __repr__(self):
        return (
            f"ArrayStrategy({self.element_strategy!r}, shape={self.shape}, "
            f"dtype={self.dtype!r}, fill={self.fill!r}, unique={self.unique!r})"
        )

    def set_element(self, val, result, idx, *, fill=False):
        # `val` is either an arbitrary object (for dtype="O"), or otherwise an
        # instance of a numpy dtype.
        try:
            result[idx] = val
        except TypeError as err:
            raise InvalidArgument(
                f"Could not add element={val!r} of "
                f"{getattr(val, 'dtype', type(val))} to array of "
                f"{result.dtype!r} - possible mismatch of time units in dtypes?"
            ) from err

        try:
            elem_changed = self._check_elements and val != result[idx] and val == val
        except Exception as err:  # pragma: no cover
            # This branch only exists to help debug weird behaviour in Numpy,
            # such as the string problems we had a while back.
            raise HypothesisException(
                f"Internal error when checking element={val!r} of "
                f"{getattr(val, 'dtype', type(val))!r} to array of "
                f"{result.dtype!r}"
            ) from err

        if elem_changed:
            strategy = self.fill if fill else self.element_strategy
            if self.dtype.kind == "f":  # pragma: no cover
                # This logic doesn't trigger in our coverage tests under Numpy 1.24+,
                # with built-in checks for overflow, but we keep it for good error
                # messages and compatibility with older versions of Numpy.
                try:
                    is_subnormal = 0 < abs(val) < np.finfo(self.dtype).tiny
                except Exception:
                    # val may be a non-float that does not support the
                    # operations __lt__ and __abs__
                    is_subnormal = False
                if is_subnormal:
                    raise InvalidArgument(
                        f"Generated subnormal float {val} from strategy "
                        f"{strategy} resulted in {result[idx]!r}, probably "
                        "as a result of NumPy being built with flush-to-zero "
                        "compiler options. Consider passing "
                        "allow_subnormal=False."
                    )
            raise InvalidArgument(
                f"Generated array element {val!r} from {strategy!r} cannot be "
                f"represented as dtype {self.dtype!r} - instead it becomes "
                f"{result[idx]!r} (type {type(result[idx])!r}).  Consider using "
                "a more precise strategy, for example passing the `width` argument "
                "to `floats()`."
            )

    def do_draw(self, data):
        if 0 in self.shape:
            return np.zeros(dtype=self.dtype, shape=self.shape)

        # Because Numpy allocates memory for strings at array creation, if we have
        # an unsized string dtype we'll fill an object array and then cast it back.
        unsized_string_dtype = (
            self.dtype.kind in ("S", "a", "U") and self.dtype.itemsize == 0
        )

        # This could legitimately be a np.empty, but the performance gains for
        # that would be so marginal that there's really not much point risking
        # undefined behaviour shenanigans.
        result: Any = np.zeros(
            shape=self.array_size, dtype=object if unsized_string_dtype else self.dtype
        )

        if self.fill.is_empty:
            # We have no fill value (either because the user explicitly disabled it
            # or because the default behaviour was used and our elements strategy
            # does not produce reusable values), so we must generate a fully dense
            # array with a freshly drawn value for each entry.
            if self.unique:
                elems = st.lists(
                    self.element_strategy,
                    min_size=self.array_size,
                    max_size=self.array_size,
                    unique=True,
                )
                for i, v in enumerate(data.draw(elems)):
                    self.set_element(v, result, i)
            else:
                for i in range(len(result)):
                    self.set_element(data.draw(self.element_strategy), result, i)
        else:
            # We draw numpy arrays as "sparse with an offset". We draw a collection
            # of index assignments within the array and assign fresh values from our
            # elements strategy to those indices. If at the end we have not assigned
            # every element then we draw a single value from our fill strategy and
            # use that to populate the remaining positions.
            elements = cu.many(
                data,
                min_size=0,
                max_size=self.array_size,
                # sqrt isn't chosen for any particularly principled reason. It just
                # grows reasonably quickly but sublinearly, and for small arrays it
                # represents a decent fraction of the array size.
                average_size=min(
                    0.9 * self.array_size,  # ensure small arrays sometimes use fill
                    max(10, math.sqrt(self.array_size)),  # ...but *only* sometimes
                ),
            )

            needs_fill = np.full(self.array_size, True)
            seen = set()

            while elements.more():
                i = data.draw_integer(0, self.array_size - 1)
                if not needs_fill[i]:
                    elements.reject()
                    continue
                self.set_element(data.draw(self.element_strategy), result, i)
                if self.unique:
                    if result[i] in seen:
                        elements.reject()
                        continue
                    seen.add(result[i])

                needs_fill[i] = False
            if needs_fill.any():
                # We didn't fill all of the indices in the early loop, so we put a
                # fill value into the rest.

                # We have to do this hilarious little song and dance to work around
                # numpy's special handling of iterable values. If the value here were
                # e.g. a tuple then neither array creation nor putmask would do the
                # right thing. But by creating an array of size one and then assigning
                # the fill value as a single element, we both get an array with the
                # right value in it and putmask will do the right thing by repeating
                # the values of the array across the mask.
                one_element: Any = np.zeros(
                    shape=1, dtype=object if unsized_string_dtype else self.dtype
                )
                self.set_element(data.draw(self.fill), one_element, 0, fill=True)
                if unsized_string_dtype:
                    one_element = one_element.astype(self.dtype)
                fill_value = one_element[0]
                if self.unique:
                    try:
                        is_nan = np.isnan(fill_value)
                    except TypeError:
                        is_nan = False

                    if not is_nan:
                        raise InvalidArgument(
                            f"Cannot fill unique array with non-NaN value {fill_value!r}"
                        )

                np.putmask(result, needs_fill, one_element)

        if unsized_string_dtype:
            out = result.astype(self.dtype)
            mismatch = out != result
            if mismatch.any():
                raise InvalidArgument(
                    f"Array elements {result[mismatch]!r} cannot be represented "
                    f"as dtype {self.dtype!r} - instead they become "
                    f"{out[mismatch]!r}.  Use a more precise strategy, e.g. without "
                    "trailing null bytes, as this will be an error future versions."
                )
            result = out

        result = result.reshape(self.shape).copy()

        assert result.base is None

        return result


def fill_for(elements, unique, fill, name=""):
    if fill is None:
        if unique or not elements.has_reusable_values:
            fill = st.nothing()
        else:
            fill = elements
    else:
        check_strategy(fill, f"{name}.fill" if name else "fill")
    return fill


D = TypeVar("D", bound="DTypeLike")
G = TypeVar("G", bound="np.generic")


@overload
def arrays(
    dtype: Union["np.dtype[G]", SearchStrategy],
    shape: int | SearchStrategy | Shape | SearchStrategy,
    *,
    elements: SearchStrategy | Mapping[str, Any] | None = None,
    fill: SearchStrategy | None = None,
    unique: bool = False,
) -> SearchStrategy: ...


@overload
def arrays(
    dtype: D | SearchStrategy,
    shape: int | SearchStrategy | Shape | SearchStrategy,
    *,
    elements: SearchStrategy | Mapping[str, Any] | None = None,
    fill: SearchStrategy | None = None,
    unique: bool = False,
) -> SearchStrategy: ...


@defines_strategy(force_reusable_values=True)
def arrays(
    dtype: D | SearchStrategy,
    shape: int | SearchStrategy | Shape | SearchStrategy,
    *,
    elements: SearchStrategy | Mapping[str, Any] | None = None,
    fill: SearchStrategy | None = None,
    unique: bool = False,
) -> SearchStrategy:
    r"""Returns a strategy for generating :class:`numpy:numpy.ndarray`\ s.

    * ``dtype`` may be any valid input to :class:`~numpy:numpy.dtype`
      (this includes :class:`~numpy:numpy.dtype` objects), or a strategy that
      generates such values.
    * ``shape`` may be an integer >= 0, a tuple of such integers, or a
      strategy that generates such values.
    * ``elements`` is a strategy for generating values to put in the array.
      If it is None a suitable value will be inferred based on the dtype, which
      may give any legal value (including eg NaN for floats). If a mapping, it
      will be passed as ``**kwargs`` to ``from_dtype()``.
    * ``fill`` is a strategy that may be used to generate a single background
      value for the array. If None, a suitable default will be inferred based on
      the other arguments. If set to :func:`~hypothesis.strategies.nothing` then
      filling behaviour will be disabled entirely and every element will be
      generated independently.
    * ``unique`` specifies if the elements of the array should all be distinct
      from one another; see the upstream docs for details.
    """
    # Our dtype argument might be a union, e.g. `np.float64 | np.complex64`; we
    # handle that by turning it into a strategy up-front.
    if type(dtype) in (getattr(types, "UnionType", object()), Union):
        dtype = st.one_of(*(from_dtype(np.dtype(d)) for d in dtype.__args__))  # type: ignore

    # We support passing strategies as arguments for convenience, or at least for
    # legacy reasons, but don't want to pay the perf cost of a composite strategy
    # when it's not needed. So we get the best of both worlds by recursing with
    # flatmap, but only when it's actually needed.
    if isinstance(dtype, SearchStrategy):
        return dtype.flatmap(
            lambda d: arrays(d, shape, elements=elements, fill=fill, unique=unique)
        )
    if isinstance(shape, SearchStrategy):
        return shape.flatmap(
            lambda s: arrays(dtype, s, elements=elements, fill=fill, unique=unique)
        )
    # From here on, we're only dealing with values and it's relatively simple.
    dtype = np.dtype(dtype)  # type: ignore[arg-type]
    assert isinstance(dtype, np.dtype)  # help mypy out a bit...
    if elements is None or isinstance(elements, Mapping):
        if dtype.kind in ("m", "M") and "[" not in dtype.str:
            # For datetime and timedelta dtypes, we have a tricky situation - because
            # they *may or may not* specify a unit as part of the dtype. If not, we
            # flatmap over the various resolutions so that array elements have
            # consistent units but units may vary between arrays.
            return (
                st.sampled_from(TIME_RESOLUTIONS)
                .map((dtype.str + "[{}]").format)
                .flatmap(lambda d: arrays(d, shape=shape, fill=fill, unique=unique))
            )
        elements = from_dtype(dtype, **(elements or {}))
    check_strategy(elements, "elements")
    # Upstream strips a redundant `.map(dtype.type)` here (when elements is a real
    # MappedStrategy whose pack is dtype.type) to unlock fast unique sampled_from. Our
    # native strategies are not real MappedStrategy instances, so that fast-path never
    # applies — we draw the (already dtype-typed) elements directly, which is correct,
    # just without that micro-optimization.
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    check_argument(
        all(isinstance(s, int) for s in shape),
        "Array shape must be integer in each dimension, provided shape was {}",
        shape,
    )
    fill = fill_for(elements=elements, unique=unique, fill=fill)
    return ArrayStrategy(elements, shape, dtype, fill, unique)


@defines_strategy()
def scalar_dtypes() -> SearchStrategy:
    """Return a strategy that can return any non-flexible scalar dtype."""
    return st.one_of(
        boolean_dtypes(),
        integer_dtypes(),
        unsigned_integer_dtypes(),
        floating_dtypes(),
        complex_number_dtypes(),
        datetime64_dtypes(),
        timedelta64_dtypes(),
    )


def defines_dtype_strategy(strat: Any) -> Any:
    @defines_strategy()
    @proxies(strat)
    def inner(*args, **kwargs):
        return strat(*args, **kwargs).map(np.dtype)

    return inner


@defines_dtype_strategy
def boolean_dtypes() -> SearchStrategy:
    """Return a strategy for boolean dtypes."""
    return st.just("?")  # type: ignore[arg-type]


def dtype_factory(kind, sizes, valid_sizes, endianness):
    # Utility function, shared logic for most integer and string types
    valid_endian = ("?", "<", "=", ">")
    check_argument(
        endianness in valid_endian,
        "Unknown endianness: was {}, must be in {}",
        endianness,
        valid_endian,
    )
    if valid_sizes is not None:
        if isinstance(sizes, int):
            sizes = (sizes,)
        check_argument(sizes, "Dtype must have at least one possible size.")
        check_argument(
            all(s in valid_sizes for s in sizes),
            "Invalid sizes: was {} must be an item or sequence in {}",
            sizes,
            valid_sizes,
        )
        if all(isinstance(s, int) for s in sizes):
            sizes = sorted({s // 8 for s in sizes})
    strat = st.sampled_from(sizes)
    if "{}" not in kind:
        kind += "{}"
    if endianness == "?":
        return strat.map(("<" + kind).format) | strat.map((">" + kind).format)
    return strat.map((endianness + kind).format)


@overload
def unsigned_integer_dtypes(
    *, endianness: str = "?", sizes: Literal[8]
) -> SearchStrategy: ...


@overload
def unsigned_integer_dtypes(
    *, endianness: str = "?", sizes: Literal[16]
) -> SearchStrategy: ...


@overload
def unsigned_integer_dtypes(
    *, endianness: str = "?", sizes: Literal[32]
) -> SearchStrategy: ...


@overload
def unsigned_integer_dtypes(
    *, endianness: str = "?", sizes: Literal[64]
) -> SearchStrategy: ...


@overload
def unsigned_integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Sequence[Literal[8, 16, 32, 64]] = (8, 16, 32, 64),
) -> SearchStrategy: ...


@defines_dtype_strategy
def unsigned_integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[8, 16, 32, 64] | Sequence[Literal[8, 16, 32, 64]] = (8, 16, 32, 64),
) -> SearchStrategy:
    """Return a strategy for unsigned integer dtypes.

    endianness may be ``<`` for little-endian, ``>`` for big-endian, ``=`` for
    native byte order, or ``?`` to allow either byte order. This argument only
    applies to dtypes of more than one byte.

    sizes must be a collection of integer sizes in bits. The default
    (8, 16, 32, 64) covers the full range of sizes.
    """
    return dtype_factory("u", sizes, (8, 16, 32, 64), endianness)


@overload
def integer_dtypes(
    *, endianness: str = "?", sizes: Literal[8]
) -> SearchStrategy: ...


@overload
def integer_dtypes(
    *, endianness: str = "?", sizes: Literal[16]
) -> SearchStrategy: ...


@overload
def integer_dtypes(
    *, endianness: str = "?", sizes: Literal[32]
) -> SearchStrategy: ...


@overload
def integer_dtypes(
    *, endianness: str = "?", sizes: Literal[64]
) -> SearchStrategy: ...


@overload
def integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Sequence[Literal[8, 16, 32, 64]] = (8, 16, 32, 64),
) -> SearchStrategy: ...


@defines_dtype_strategy
def integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[8, 16, 32, 64] | Sequence[Literal[8, 16, 32, 64]] = (8, 16, 32, 64),
) -> SearchStrategy:
    """Return a strategy for signed integer dtypes.

    endianness and sizes are treated as for :func:`unsigned_integer_dtypes`.
    """
    return dtype_factory("i", sizes, (8, 16, 32, 64), endianness)


@overload
def floating_dtypes(
    *, endianness: str = "?", sizes: Literal[16]
) -> SearchStrategy: ...


@overload
def floating_dtypes(
    *, endianness: str = "?", sizes: Literal[32]
) -> SearchStrategy: ...


@overload
def floating_dtypes(
    *, endianness: str = "?", sizes: Literal[64]
) -> SearchStrategy: ...


@overload
def floating_dtypes(
    *, endianness: str = "?", sizes: Literal[128]
) -> SearchStrategy: ...


@overload
def floating_dtypes(
    *,
    endianness: str = "?",
    sizes: Sequence[Literal[16, 32, 64, 96, 128]] = (16, 32, 64),
) -> SearchStrategy: ...


@defines_dtype_strategy
def floating_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[16, 32, 64, 96, 128] | Sequence[Literal[16, 32, 64, 96, 128]] = (
        16,
        32,
        64,
    ),
) -> SearchStrategy:
    """Return a strategy for floating-point dtypes.

    sizes is the size in bits of floating-point number. Some machines support
    96- or 128-bit floats, but these are not generated by default.
    """
    return dtype_factory("f", sizes, (16, 32, 64, 96, 128), endianness)


@overload
def complex_number_dtypes(
    *, endianness: str = "?", sizes: Literal[64]
) -> SearchStrategy: ...


@overload
def complex_number_dtypes(
    *, endianness: str = "?", sizes: Literal[128]
) -> SearchStrategy: ...


@overload
def complex_number_dtypes(
    *, endianness: str = "?", sizes: Literal[256]
) -> SearchStrategy: ...


@overload
def complex_number_dtypes(
    *,
    endianness: str = "?",
    sizes: Sequence[Literal[64, 128, 192, 256]] = (64, 128),
) -> SearchStrategy: ...


@defines_dtype_strategy
def complex_number_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[64, 128, 192, 256] | Sequence[Literal[64, 128, 192, 256]] = (
        64,
        128,
    ),
) -> SearchStrategy:
    """Return a strategy for complex-number dtypes.

    sizes is the total size in bits of a complex number, which consists of two
    floats. Complex halves (a 16-bit real part) are not supported by numpy and
    will not be generated by this strategy.
    """
    return dtype_factory("c", sizes, (64, 128, 192, 256), endianness)


@check_function
def validate_time_slice(max_period, min_period):
    check_argument(
        max_period in TIME_RESOLUTIONS,
        "max_period {} must be a valid resolution in {}",
        max_period,
        TIME_RESOLUTIONS,
    )
    check_argument(
        min_period in TIME_RESOLUTIONS,
        "min_period {} must be a valid resolution in {}",
        min_period,
        TIME_RESOLUTIONS,
    )
    start = TIME_RESOLUTIONS.index(max_period)
    end = TIME_RESOLUTIONS.index(min_period) + 1
    check_argument(
        start < end,
        "max_period {} must be earlier in sequence {} than min_period {}",
        max_period,
        TIME_RESOLUTIONS,
        min_period,
    )
    return TIME_RESOLUTIONS[start:end]


@defines_dtype_strategy
def datetime64_dtypes(
    *, max_period: str = "Y", min_period: str = "ns", endianness: str = "?"
) -> SearchStrategy:
    """Return a strategy for datetime64 dtypes, with various precisions from
    year to attosecond."""
    return dtype_factory(
        "datetime64[{}]",
        validate_time_slice(max_period, min_period),
        TIME_RESOLUTIONS,
        endianness,
    )


@defines_dtype_strategy
def timedelta64_dtypes(
    *, max_period: str = "Y", min_period: str = "ns", endianness: str = "?"
) -> SearchStrategy:
    """Return a strategy for timedelta64 dtypes, with various precisions from
    year to attosecond."""
    return dtype_factory(
        "timedelta64[{}]",
        validate_time_slice(max_period, min_period),
        TIME_RESOLUTIONS,
        endianness,
    )


@defines_dtype_strategy
def byte_string_dtypes(
    *, endianness: str = "?", min_len: int = 1, max_len: int = 16
) -> SearchStrategy:
    """Return a strategy for generating bytestring dtypes, of various lengths
    and byteorder.

    While Hypothesis' string strategies can generate empty strings, string dtypes
    with length 0 indicate that size is still to be determined, so the minimum
    length for string dtypes is 1.
    """
    order_check("len", 1, min_len, max_len)
    return dtype_factory("S", list(range(min_len, max_len + 1)), None, endianness)


@defines_dtype_strategy
def unicode_string_dtypes(
    *, endianness: str = "?", min_len: int = 1, max_len: int = 16
) -> SearchStrategy:
    """Return a strategy for generating unicode string dtypes, of various lengths
    and byteorder.

    While Hypothesis' string strategies can generate empty strings, string dtypes
    with length 0 indicate that size is still to be determined, so the minimum
    length for string dtypes is 1.
    """
    order_check("len", 1, min_len, max_len)
    return dtype_factory("U", list(range(min_len, max_len + 1)), None, endianness)


def _no_title_is_name_of_a_titled_field(ls):
    seen = set()
    for title_and_name, *_ in ls:
        if isinstance(title_and_name, tuple):
            if seen.intersection(title_and_name):  # pragma: no cover
                # Our per-element filters below make this as rare as possible, so
                # it's not always covered.
                return False
            seen.update(title_and_name)
    return True


@defines_dtype_strategy
def array_dtypes(
    subtype_strategy: SearchStrategy = scalar_dtypes(),
    *,
    min_size: int = 1,
    max_size: int = 5,
    allow_subarrays: bool = False,
) -> SearchStrategy:
    """Return a strategy for generating array (compound) dtypes, with members
    drawn from the given subtype strategy."""
    order_check("size", 0, min_size, max_size)
    # The empty string is replaced by f{idx}; see #1963 for details. Much easier to
    # insist that field names be unique and just boost f{idx} strings manually.
    field_names = st.integers(0, 127).map("f{}".format) | st.text(min_size=1)
    name_titles = st.one_of(
        field_names,
        st.tuples(field_names, field_names).filter(lambda ns: ns[0] != ns[1]),
    )
    elements: SearchStrategy = st.tuples(name_titles, subtype_strategy)
    if allow_subarrays:
        elements |= st.tuples(
            name_titles, subtype_strategy, array_shapes(max_dims=2, max_side=2)
        )
    return st.lists(  # type: ignore[return-value]
        elements=elements,
        min_size=min_size,
        max_size=max_size,
        unique_by=(
            # Deduplicate by both name and title for efficiency before filtering.
            # (Field names must be unique, as must titles, and no intersections)
            lambda d: d[0] if isinstance(d[0], str) else d[0][0],
            lambda d: d[0] if isinstance(d[0], str) else d[0][1],
        ),
    ).filter(_no_title_is_name_of_a_titled_field)


@defines_strategy()
def nested_dtypes(
    subtype_strategy: SearchStrategy = scalar_dtypes(),
    *,
    max_leaves: int = 10,
    max_itemsize: int | None = None,
) -> SearchStrategy:
    """Return the most-general dtype strategy.

    Elements drawn from this strategy may be simple (from the subtype_strategy),
    or several such values drawn from :func:`array_dtypes` with
    ``allow_subarrays=True``. Subdtypes in an array dtype may be nested to any
    depth, subject to the max_leaves argument.
    """
    return st.recursive(
        subtype_strategy,
        lambda x: array_dtypes(x, allow_subarrays=True),
        max_leaves=max_leaves,
    ).filter(lambda d: max_itemsize is None or d.itemsize <= max_itemsize)


@proxies(_valid_tuple_axes)
def valid_tuple_axes(*args, **kwargs):
    return _valid_tuple_axes(*args, **kwargs)


valid_tuple_axes.__doc__ = """
    Return a strategy for generating permissible tuple-values for the ``axis``
    argument for a numpy sequential function (e.g. :func:`numpy:numpy.sum`), given
    an array of the specified dimensionality.
    """


@proxies(_mutually_broadcastable_shapes)
def mutually_broadcastable_shapes(*args, **kwargs):
    return _mutually_broadcastable_shapes(*args, **kwargs)


mutually_broadcastable_shapes.__doc__ = _mutually_broadcastable_shapes.__doc__


@overload
def basic_indices(
    shape: Shape,
    *,
    min_dims: int = 0,
    max_dims: int | None = None,
    allow_newaxis: Literal[False] = ...,
    allow_ellipsis: Literal[False],
) -> SearchStrategy: ...


@overload
def basic_indices(
    shape: Shape,
    *,
    min_dims: int = 0,
    max_dims: int | None = None,
    allow_newaxis: Literal[False] = ...,
    allow_ellipsis: Literal[True] = ...,
) -> SearchStrategy: ...


@overload
def basic_indices(
    shape: Shape,
    *,
    min_dims: int = 0,
    max_dims: int | None = None,
    allow_newaxis: Literal[True],
    allow_ellipsis: Literal[False],
) -> SearchStrategy: ...


@overload
def basic_indices(
    shape: Shape,
    *,
    min_dims: int = 0,
    max_dims: int | None = None,
    allow_newaxis: Literal[True],
    allow_ellipsis: Literal[True] = ...,
) -> SearchStrategy: ...


@defines_strategy()
def basic_indices(
    shape: Shape,
    *,
    min_dims: int = 0,
    max_dims: int | None = None,
    allow_newaxis: bool = False,
    allow_ellipsis: bool = True,
) -> SearchStrategy:
    """Return a strategy for :doc:`basic indexes <numpy:reference/routines.indexing>`
    of arrays with the specified shape, which may include dimensions of size zero.

    It generates tuples containing some mix of integers, :obj:`python:slice`
    objects, ``...`` (an ``Ellipsis``), and ``None``.

    * ``shape`` is the shape of the array that will be indexed, as a tuple of
      positive integers. This must be at least two-dimensional for a tuple to be
      a valid index; for one-dimensional arrays use
      :func:`~hypothesis.strategies.slices` instead.
    * ``min_dims`` is the minimum dimensionality of the resulting array.
    * ``max_dims`` is the maximum dimensionality of the resulting array.
    * ``allow_newaxis`` specifies whether ``None`` is allowed in the index.
    * ``allow_ellipsis`` specifies whether ``...`` is allowed in the index.
    """
    # Arguments to exclude scalars, zero-dim arrays, and dims of size zero were all
    # considered and rejected. We want users to explicitly consider those cases.
    check_type(tuple, shape, "shape")
    check_argument(
        all(isinstance(x, int) and x >= 0 for x in shape),
        f"{shape=}, but all dimensions must be non-negative integers.",
    )
    check_type(bool, allow_ellipsis, "allow_ellipsis")
    check_type(bool, allow_newaxis, "allow_newaxis")
    check_type(int, min_dims, "min_dims")
    if min_dims > len(shape) and not allow_newaxis:
        note_deprecation(
            f"min_dims={min_dims} is larger than len(shape)={len(shape)}, "
            "but allow_newaxis=False makes it impossible for an indexing "
            "operation to add dimensions.",
            since="2021-09-15",
            has_codemod=False,
        )
    check_valid_dims(min_dims, "min_dims")

    if max_dims is None:
        if allow_newaxis:
            max_dims = min(max(len(shape), min_dims) + 2, NDIM_MAX)
        else:
            max_dims = min(len(shape), NDIM_MAX)
    else:
        check_type(int, max_dims, "max_dims")
        if max_dims > len(shape) and not allow_newaxis:
            note_deprecation(
                f"max_dims={max_dims} is larger than len(shape)={len(shape)}, "
                "but allow_newaxis=False makes it impossible for an indexing "
                "operation to add dimensions.",
                since="2021-09-15",
                has_codemod=False,
            )
    check_valid_dims(max_dims, "max_dims")

    order_check("dims", 0, min_dims, max_dims)

    return BasicIndexStrategy(
        shape,
        min_dims=min_dims,
        max_dims=max_dims,
        allow_ellipsis=allow_ellipsis,
        allow_newaxis=allow_newaxis,
        allow_fewer_indices_than_dims=True,
    )


IntegerT = TypeVar("IntegerT", bound="np.integer")


@overload
def integer_array_indices(
    shape: Shape,
    *,
    result_shape: SearchStrategy = array_shapes(),
) -> SearchStrategy: ...


@overload
def integer_array_indices(
    shape: Shape,
    *,
    result_shape: SearchStrategy = array_shapes(),
    dtype: np.dtype[IntegerT],
) -> SearchStrategy: ...


@defines_strategy()
def integer_array_indices(
    shape: Shape,
    *,
    result_shape: SearchStrategy = array_shapes(),
    dtype: Any = np.dtype(int),
) -> SearchStrategy:
    """Return a search strategy for tuples of integer-arrays that, when used to
    index into an array of shape ``shape``, given an array whose shape was drawn
    from ``result_shape``.

    See the upstream documentation for the full description of the arguments and
    advanced-indexing semantics.
    """
    check_type(tuple, shape, "shape")
    check_argument(
        shape and all(isinstance(x, int) and x > 0 for x in shape),
        f"{shape=} must be a non-empty tuple of integers > 0",
    )
    check_strategy(result_shape, "result_shape")
    check_argument(
        np.issubdtype(dtype, np.integer), f"{dtype=} must be an integer dtype"
    )
    signed = np.issubdtype(dtype, np.signedinteger)

    def array_for(index_shape, size):
        return arrays(
            dtype=dtype,
            shape=index_shape,
            elements=st.integers(-size if signed else 0, size - 1),
        )

    return result_shape.flatmap(
        lambda index_shape: st.tuples(*(array_for(index_shape, size) for size in shape))
    )


def _unpack_dtype(dtype):
    dtype_args = getattr(dtype, "__args__", ())
    if dtype_args and type(dtype) not in (getattr(types, "UnionType", object()), Union):
        assert len(dtype_args) == 1
        if isinstance(dtype_args[0], TypeVar):
            # numpy.dtype[+ScalarType]
            assert dtype_args[0].__bound__ == np.generic
            dtype = Any
        else:
            # plain dtype
            dtype = dtype_args[0]
    return dtype


def _dtype_from_args(args):
    if len(args) <= 1:
        # Zero args: ndarray, _SupportsArray
        # One arg: ndarray[type], _SupportsArray[type]
        dtype = _unpack_dtype(args[0]) if args else Any
    else:
        # Two args: ndarray[shape, type], NDArray[*]
        assert len(args) == 2
        dtype = _unpack_dtype(args[1])

    if dtype is Any:
        return scalar_dtypes()
    elif type(dtype) in (getattr(types, "UnionType", object()), Union):
        return dtype
    return np.dtype(cast(Any, dtype))


def _from_type(thing: type[Ex]) -> SearchStrategy | None:
    """Called by st.from_type to try to infer a strategy for thing using numpy.

    If we can infer a numpy-specific strategy for thing, we return that; otherwise,
    we return None.
    """
    base_strats = st.one_of(
        [
            st.booleans(),
            st.integers(),
            st.floats(),
            st.complex_numbers(),
            st.text(),
            st.binary(),
        ]
    )
    # don't mix strings and non-ascii bytestrings (ex: ['', b'\x80']). See
    # https://github.com/numpy/numpy/issues/23899.
    base_strats_ascii = st.one_of(
        [
            st.booleans(),
            st.integers(),
            st.floats(),
            st.complex_numbers(),
            st.text(),
            st.binary().filter(bytes.isascii),
        ]
    )

    if thing == np.dtype:
        # Note: Parameterized dtypes and DTypeLike are not supported.
        return st.one_of(
            scalar_dtypes(),
            byte_string_dtypes(),
            unicode_string_dtypes(),
            array_dtypes(),
            nested_dtypes(),
        )

    if thing == ArrayLike:
        # We override the default type resolution to ensure the "coercible to
        # array" contract is honoured.
        return st.one_of(
            base_strats,
            st.recursive(st.lists(base_strats_ascii), extend=st.tuples),
            st.recursive(st.from_type(np.ndarray), extend=st.tuples),
        )

    if isinstance(thing, type) and issubclass(thing, np.generic):
        dtype = np.dtype(thing)
        return from_dtype(dtype) if dtype.kind not in "OV" else None

    origin = get_origin(thing)
    # if origin is not generic-like, get_origin returns None. Fall back to thing.
    if origin is None:
        origin = thing
    args = get_args(thing)

    if origin == _NestedSequence:
        # We have to override the default resolution to ensure sequences are of
        # equal length.
        assert len(args) <= 1
        base_strat = st.from_type(args[0]) if args else base_strats
        return st.one_of(
            st.lists(base_strat),
            st.recursive(st.tuples(), st.tuples),
            st.recursive(st.tuples(base_strat), st.tuples),
            st.recursive(st.tuples(base_strat, base_strat), st.tuples),
        )

    # note: get_origin(np.typing.NDArray[np.int64]) is np.ndarray in numpy < 2.5.0,
    # but is np.typing.NDArray in numpy >= 2.5.0. Support both here.
    if origin in [np.typing.NDArray, np.ndarray, _SupportsArray]:
        dtype = _dtype_from_args(args)
        return arrays(dtype, array_shapes(max_dims=2))  # type: ignore[return-value]

    # We didn't find a type to resolve, continue
    return None


def _register_from_type() -> None:
    """Hook numpy type resolution into the native ``from_type``, mirroring real
    hypothesis's numpy extra. The native registry resolves a request by *exact* type
    (otherwise falling back to ``builds()``, which for a numpy scalar yields only its zero
    value), so each concrete scalar type must be registered individually with its
    ``from_dtype`` strategy. ``np.dtype`` / ``np.ndarray`` / the typing aliases use the
    ``_from_type`` factory so parametrized requests (``NDArray[...]``) resolve too.
    Best-effort and idempotent."""
    # Every concrete numpy scalar type (leaves of the np.generic hierarchy with a real,
    # non-object/void dtype).
    seen: set = set()
    stack = list(np.generic.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        try:
            dt = np.dtype(cls)
        except Exception:  # noqa: BLE001 - abstract bases (np.signedinteger, ...) have no dtype
            continue
        if dt.kind in ("O", "V"):
            continue
        try:
            st.register_type_strategy(cls, from_dtype(dt))
        except Exception:  # noqa: BLE001 - best-effort per type
            pass
    # dtype, ndarray and the typing aliases: factory so parametrized forms resolve.
    targets = [np.dtype, np.ndarray]
    for opt in (ArrayLike, NDArray, _NestedSequence, _SupportsArray):
        if opt is not None:
            targets.append(opt)
    for target in targets:
        try:
            st.register_type_strategy(target, _from_type)
        except Exception:  # noqa: BLE001 - some targets are typing aliases, not types
            pass


def _unregister_from_type(snapshot_registry: dict, snapshot_user: set) -> None:
    """Restore the native type registry to a prior snapshot — the inverse of
    ``_register_from_type``. Used by the parity-suite fixture to keep numpy registration
    from leaking into other tests' ``from_type`` resolution."""
    from hypothesis_fast import _engine, native_strategies as _ns

    _ns._NATIVE_TYPE_REGISTRY.clear()
    _ns._NATIVE_TYPE_REGISTRY.update(snapshot_registry)
    _ns._USER_REGISTERED.clear()
    _ns._USER_REGISTERED.update(snapshot_user)
    _engine._clear_resolution_caches()
