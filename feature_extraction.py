from urllib.parse import urlparse
import re


def extract_features(url):

    url_lower = url.lower()

    # Parsing URL
    parsed = urlparse(url_lower)

    domain = parsed.netloc
    path = parsed.path
    query = parsed.query

    # Jika URL tidak memiliki scheme
    if not domain:
        parsed = urlparse("http://" + url_lower)
        domain = parsed.netloc
        path = parsed.path
        query = parsed.query

    # Hapus port dari domain
    domain_without_port = domain.split(":")[0]

    # Hitung jumlah subdomain
    domain_parts = domain_without_port.split(".")
    num_subdomains = max(len(domain_parts) - 2, 0)

    # Keyword mencurigakan
    suspicious_keywords = [
        "judi",
        "judionline",
        "slot",
        "casino",
        "kasino",
        "togel",
        "jackpot",
        "bet",
        "bets",
        "betting",
        "poker",
        "game",
        "gacor",
        "pragmatic",
        "spin",
        "deposit",
        "withdraw",
        "bonus",
        "maxwin"
    ]

    keyword_count = sum(
        1 for keyword in suspicious_keywords
        if keyword in url_lower
    )

    # Cek apakah URL menggunakan IP
    has_ip = 1 if re.match(
        r"^(?:\d{1,3}\.){3}\d{1,3}$",
        domain_without_port
    ) else 0

    # Fitur URL
    features = {

        # Fitur dasar
        "url_length": len(url),
        "domain_length": len(domain_without_port),

        # Karakter
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": sum(
            not c.isalnum() for c in url
        ),

        # Struktur domain
        "num_subdomains": num_subdomains,

        # Protokol
        "has_https": 1 if parsed.scheme == "https" else 0,

        # IP dan simbol mencurigakan
        "has_ip": has_ip,
        "has_at_symbol": 1 if "@" in url else 0,
        "has_hyphen": 1 if "-" in domain_without_port else 0,

        # Keyword
        "has_suspicious_keyword": 1 if keyword_count > 0 else 0,
        "num_suspicious_keywords": keyword_count,

        # Query dan path
        "num_query_parameters": (
            len(query.split("&"))
            if query else 0
        ),

        "path_length": len(path),
        "query_length": len(query),

        # Karakter URL tambahan
        "num_dots": url.count("."),
        "num_slashes": url.count("/"),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_percent": url.count("%"),
        "num_equals": url.count("="),
        "num_question_marks": url.count("?"),
        "num_ampersands": url.count("&"),

        # Digit khusus pada domain
        "num_domain_digits": sum(
            c.isdigit()
            for c in domain_without_port
        )
    }

    return features


# ==========================================
# PENGUJIAN FITUR
# ==========================================

if __name__ == "__main__":

    url = "https://contoh-slot-gacor.com/bonus?deposit=123"

    features = extract_features(url)

    print("URL:")
    print(url)

    print("\nHasil ekstraksi fitur:")

    for key, value in features.items():
        print(f"{key}: {value}")