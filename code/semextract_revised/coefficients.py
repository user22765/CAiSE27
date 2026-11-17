"""Coefficient diagnostics shown to text reviews: possible omissions and values not in the article."""
import html
import math
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


_SYMBOL = r"(?P<symbol>\\?(?:beta|gamma)|[βϐγbB])"
_INDEX = r"(?:\s*_\s*(?:\{[^{}\n=]{1,40}\}|[A-Za-z0-9]+)|[₀₁₂₃₄₅₆₇₈₉]+|\d{1,3})?"
_SPACE = r"(?:\s|\\[,;!]|\\(?:quad|qquad)\b)*"
_ABBREVIATIONS = {"fig", "figs", "eq", "eqs", "tab", "dr", "mr", "mrs", "vs", "al", "e.g", "i.e"}


def coefficient_pattern(decimal_comma=False):
    decimal = r"[.,]" if decimal_comma else r"\."
    number = rf"[+−-]?\s*(?:\d+(?:{decimal}\d+)?|{decimal}\d+)(?:[eE][+−-]?\d+)?"
    return re.compile(
        rf"(?<![\w\\])(?:\\hat\s*\{{\s*)?{_SYMBOL}{_INDEX}"
        rf"(?:\s*\}})?{_INDEX}(?:\s*['′])?{_SPACE}(?:\$)?{_SPACE}"
        rf"(?P<operator>=|＝|≈|\\approx\b|\\simeq\b){_SPACE}(?:\$)?{_SPACE}"
        rf"(?P<number>{number})(?![\w]|[.,]\d)", re.IGNORECASE,
    )


def sentence_spans(text):
    """Conservative sentence boundaries: preserve decimals and common abbreviations."""
    start = 0
    for match in re.finditer(r"[.!?]+[\"'’”)\]}]*(?=\s|$)|\n\s*\n", text):
        boundary = match.group()
        if boundary.startswith("."):
            previous = re.search(r"([\w.]+)$", text[:match.start()])
            if previous and (previous[1].casefold() in _ABBREVIATIONS
                             or (len(previous[1]) == 1 and previous[1].isalpha())):
                continue
        end = match.end()
        left = start
        while left < end and text[left].isspace():
            left += 1
        if left < end:
            yield left, end
        start = end
    while start < len(text) and text[start].isspace():
        start += 1
    if start < len(text):
        yield start, len(text)


def coefficient_sentences(text, source_id="text", decimal_comma=False):
    """Return each matching sentence once, retaining every coefficient occurrence.

    Handles β/ϐ/γ, beta/gamma (also LaTeX), b/B, indices, signed/leading-dot
    decimals and approximate equalities. Decimal comma is
    opt-in to avoid confusing list punctuation with a decimal separator.
    """
    matches = list(coefficient_pattern(decimal_comma).finditer(text))
    output = []
    cursor = 0
    for start, end in sentence_spans(text):
        mentions = []
        while cursor < len(matches) and matches[cursor].start() < end:
            match = matches[cursor]
            cursor += 1
            raw_symbol = match["symbol"]
            symbol = raw_symbol.lstrip("\\").casefold()
            symbol = "beta" if symbol in {"beta", "β", "ϐ"} else "gamma" if symbol in {"gamma", "γ"} else "b"
            mentions.append({"symbol": symbol,
                             "number_text": match["number"],
                             "expression": match[0]})
        if mentions:
            output.append({"source_id": source_id, "sentence": text[start:end],
                           "span": (start, end), "end": end, "mentions": mentions})
    return output


def plain_text(text):
    """Read HTML table cells as text without discarding inequality signs."""
    text = re.sub(r"</?(?:table|tr|td|th|thead|tbody|tfoot|p|div|br)\b[^>]*>", " ", text, flags=re.I)
    text = re.sub(r"</?(?:span|em|strong|b|i|sub|sup)\b[^>]*>", "", text, flags=re.I)
    return html.unescape(text)


def round_beta(value):
    """Round half-up to two decimals as a float; None for missing or unreadable values."""
    if value is None or isinstance(value, bool) or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str) and value.strip().casefold() in {"", "none", "null", "na", "n/a", "nan"}:
        return None
    try:
        number = Decimal(str(value).strip().replace("−", "-"))
        if not number.is_finite():
            return None
        return float(number.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    except (InvalidOperation, OverflowError):
        return None


def decimal_value(value):
    try:
        value = Decimal(re.sub(r"\s", "", str(value)).replace("−", "-").replace(",", "."))
        return value if value.is_finite() else None
    except InvalidOperation:
        return None


def number_values(text, decimal_comma=False):
    separator = r"[.,]" if decimal_comma else r"\."
    punctuation = ".," if decimal_comma else "."
    pattern = rf"(?<![\w{punctuation}+−-])[+−-]?\s*(?:\d+(?:{separator}\d+)?|{separator}\d+)(?:[eE][+−-]?\d+)?(?![\w]|[{punctuation}]\d)"
    return {decimal_value(match[0]) for match in re.finditer(pattern, plain_text(text))}


def text_diagnostics(relations, sources, decimal_comma=False):
    """Two article-wide checks, shown to text reviews as cues.

    Omission matching compares signed values, not path identities. Repeated
    occurrences remain visible, but an equal coefficient on another path can
    conceal an omission. Flags are review cues, never automatic corrections.
    """
    texts = [(s.id, plain_text(s.text)) for s in sources if s.text]
    article_values = set().union(*(number_values(text, decimal_comma) for _, text in texts))
    article_values = {decimal_value(round_beta(v)) for v in article_values}
    predicted_values = {decimal_value(round_beta(r.get("beta"))) for r in relations if r.get("beta") is not None}
    possible_omissions = []
    for sid, text in texts:
        for sentence in coefficient_sentences(text, sid, decimal_comma):
            for mention in sentence["mentions"]:
                value = decimal_value(round_beta(decimal_value(mention["number_text"])))
                if value not in predicted_values:
                    possible_omissions.append({"source_id": sid, "value": str(value),
                        "expression": mention["expression"], "text": sentence["sentence"]})
    coefficients_not_in_article = [
        {"candidate_id": f"R{i + 1}", "cause": r.get("cause"), "effect": r.get("effect"), "beta": r["beta"]}
        for i, r in enumerate(relations)
        if r.get("beta") is not None and decimal_value(round_beta(r["beta"])) not in article_values
    ]
    return {"needs_revision": bool(possible_omissions or coefficients_not_in_article),
            "possible_omissions": possible_omissions,
            "coefficients_not_in_article": coefficients_not_in_article}
