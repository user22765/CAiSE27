"""Reported p-values and coefficients: parsing, merging, and evaluation bins."""
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

P_PATTERN = re.compile(r"([<>]=?|=)?(\d*\.?\d+(?:e[-+]?\d+)?)")
P_THRESHOLDS = ("0.001", "0.01", "0.05", "0.1")


def missing(value):
    """Explicitly missing: None or a placeholder string such as '', 'NA' or 'null'."""
    return value is None or (isinstance(value, str) and value.strip().casefold() in {"", "none", "null", "na", "n/a"})


def decimal_number(value):
    """Read a finite number without losing decimal precision."""
    if isinstance(value, bool):
        raise ValueError("A boolean is not a coefficient or p-value")
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"Invalid number: {value!r}") from error
    if not number.is_finite():
        raise ValueError("Numbers must be finite")
    return number


def normalize_p(value):
    """'p ≤ .05' -> '<=0.05', '0.030' -> '=0.03'; None if missing; ValueError if impossible."""
    if missing(value):
        return None
    if isinstance(value, bool):
        raise ValueError("A boolean is not a p-value")
    value = str(value).strip().casefold().replace("≤", "<=").replace("≥", ">=")
    value = value.replace(r"\leq", "<=").replace(r"\geq", ">=")
    value = value.replace(r"\lt", "<").replace(r"\gt", ">")
    value = re.sub(r"[\s${}]", "", value)
    value = re.sub(r"^p(?:-?value)?", "", value)
    match = P_PATTERN.fullmatch(value)
    if not match:
        raise ValueError(f"Unrecognized p-value {value!r}; annotate it explicitly")
    operator, number = match.groups()
    number = decimal_number(number)
    if not 0 <= number <= 1:
        raise ValueError(f"Invalid p-value {number}")
    if (operator == "<" and number == 0) or (operator == ">" and number == 1):
        raise ValueError("The p-value bound describes an empty range")
    return (operator or "=") + format(number.normalize(), "f")


def p_interval(value):
    """(low, high, low_closed, high_closed) of the range a reported p-value allows."""
    operator, number = re.fullmatch(r"([<>]=?|=)(.+)", normalize_p(value)).groups()
    number = Decimal(number)
    if operator == "=":
        return number, number, True, True
    if operator.startswith("<"):
        return Decimal(0), number, True, operator == "<="
    return number, Decimal(1), operator == ">=", True


def merge_p(first, second):
    """(True, narrower value) when one reported range contains the other, else (False, None)."""
    a, b = normalize_p(first), normalize_p(second)
    if a is None or b is None:
        value = a or b
        return True, value.removeprefix("=") if value else None

    def contained(inner, outer):
        lo, hi, lo_closed, hi_closed = inner
        outer_lo, outer_hi, outer_lo_closed, outer_hi_closed = outer
        lower_ok = lo > outer_lo or (lo == outer_lo and (not lo_closed or outer_lo_closed))
        upper_ok = hi < outer_hi or (hi == outer_hi and (not hi_closed or outer_hi_closed))
        return lower_ok and upper_ok

    if contained(p_interval(a), p_interval(b)):
        return True, a.removeprefix("=")
    if contained(p_interval(b), p_interval(a)):
        return True, b.removeprefix("=")
    return False, None


def p_level(value):
    """Evaluation bin of a p-value; inequality boundaries and unknown ranges are kept."""
    value = normalize_p(value)
    if value is None:
        return None
    operator, number = re.fullmatch(r"([<>]=?|=)(.+)", value).groups()
    number = Decimal(number)
    if operator == "=":
        return next(("<" + t for t in P_THRESHOLDS if number < Decimal(t)), ">=0.1")
    if operator in {"<", "<="}:
        for threshold in P_THRESHOLDS:
            boundary = Decimal(threshold)
            if number == boundary and operator == "<=":
                return "<=" + threshold
            if number <= boundary:
                return "<" + threshold
    return value


def rounded_beta(value):
    """Coefficient rounded half-up to two decimals, for evaluation."""
    if missing(value):
        return None
    try:
        return decimal_number(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except InvalidOperation as error:
        raise ValueError(f"Coefficient is too large to round: {value}") from error
