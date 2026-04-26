import math
import re
from collections import Counter
from urllib.parse import urlparse

import tldextract


def get_registered_domain_length(ext):
    """Calcula la longitud del dominio registrable, incluyendo el TLD."""

    if ext.suffix:
        return len(f"{ext.domain}.{ext.suffix}")

    return len(ext.domain)


def get_tld_length(ext):
    """Calcula la longitud del TLD extraído."""

    return len(ext.suffix)


def get_subdomain_count(ext):
    """Cuenta cuántos niveles de subdominio tiene la URL."""

    if not ext.subdomain:
        return 0

    return len(ext.subdomain.split("."))


def get_entropy(url: str):
    """Calcula la entropía de caracteres de una URL."""

    if not url:
        return 0

    counter = Counter(url)
    length = len(url)

    entropy = 0
    for count in counter.values():
        p = count / length
        entropy -= p * math.log2(p)

    return entropy


def extract_url_features(url: str) -> dict:
    """Extrae las features usadas por el modelo a partir de una URL."""

    parsed = urlparse(url)
    ext = tldextract.extract(url)

    total_len = len(url)
    dom_len = get_registered_domain_length(ext)
    tld_len = get_tld_length(ext)
    subdom_cnt = get_subdomain_count(ext)
    is_ip = 1 if re.match(r"\d+\.\d+\.\d+\.\d+", parsed.netloc) else 0
    is_https = 1 if parsed.scheme == "https" else 0
    letter_cnt = sum(c.isalpha() for c in url)
    digit_cnt = sum(c.isdigit() for c in url)
    special_cnt = sum(not c.isalnum() for c in url)
    eq_cnt = url.count("=")
    qm_cnt = url.count("?")
    amp_cnt = url.count("&")
    dot_cnt = url.count(".")
    dash_cnt = url.count("-")
    under_cnt = url.count("_")
    slash_cnt = url.count("/")
    path_len = len(parsed.path)
    query_len = len(parsed.query)
    entropy = get_entropy(url)

    if total_len > 0:
        letter_ratio = letter_cnt / total_len
        digit_ratio = digit_cnt / total_len
        spec_ratio = special_cnt / total_len
    else:
        letter_ratio = digit_ratio = spec_ratio = 0

    return {
        "url_len": total_len,
        "dom_len": dom_len,
        "is_ip": is_ip,
        "tld_len": tld_len,
        "subdom_cnt": subdom_cnt,
        "letter_cnt": letter_cnt,
        "digit_cnt": digit_cnt,
        "special_cnt": special_cnt,
        "eq_cnt": eq_cnt,
        "qm_cnt": qm_cnt,
        "amp_cnt": amp_cnt,
        "dot_cnt": dot_cnt,
        "dash_cnt": dash_cnt,
        "under_cnt": under_cnt,
        "letter_ratio": letter_ratio,
        "digit_ratio": digit_ratio,
        "spec_ratio": spec_ratio,
        "is_https": is_https,
        "slash_cnt": slash_cnt,
        "entropy": entropy,
        "path_len": path_len,
        "query_len": query_len,
    }
