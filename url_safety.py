import os
import re
import ssl
import json
import socket
import ipaddress
import urllib.parse
import urllib.request
import urllib.error

HIGH_ABUSE_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "buzz", "top", "click", "fit", "rest",
    "country", "work", "link", "surf", "stream", "gdn", "mom", "kim"
}

KNOWN_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "buff.ly", "ow.ly",
    "cutt.ly", "rb.gy", "goo.gl", "rebrand.ly", "shorturl.at"
}

REPUTABLE_DOMAINS = {
    "google.com", "www.google.com", "github.com", "www.github.com",
    "microsoft.com", "www.microsoft.com", "apple.com", "www.apple.com",
    "zoom.us", "meet.google.com", "teams.microsoft.com", "webex.com",
    "slack.com", "linkedin.com", "www.linkedin.com", "youtube.com",
    "www.youtube.com", "wikipedia.org", "www.wikipedia.org", "amazon.com",
    "www.amazon.com", "stackoverflow.com"
}

BRAND_KEYWORDS = [
    "paypal", "google", "microsoft", "apple", "amazon", "netflix",
    "chase", "wellsfargo", "bankofamerica", "binance", "coinbase"
]

def extract_first_url(text):
    if not text or not isinstance(text, str):
        return None
    match = re.search(r'https?://[^\s<>"]+|www\.[^\s<>"]+', text, re.IGNORECASE)
    if match:
        found = match.group(0).rstrip('.,;!?:)')
        if found.lower().startswith('www.'):
            found = 'https://' + found
        return found
    return None

def _inconclusive_result(reason, raw_input=None):
    return {
        "url": raw_input,
        "safety_percentage": None,
        "risk_level": "Inconclusive",
        "explanation": f"Inconclusive: {reason}",
        "checks_performed": ["local_url_validation"],
        "findings": [reason]
    }

def assess_url_safety(raw_url):
    """
    Assesses a URL and returns a safety percentage (0-100%) and risk level.
    Higher percentage = lower estimated risk.
    Never returns 100% (an estimate of risk is not proof of safety).
    Uses 100% local, rule-based Python logic and safe validation checks.
    """
    if not raw_url or not isinstance(raw_url, str):
        return _inconclusive_result("No URL provided to assess.")

    url_str = raw_url.strip()
    if not url_str:
        return _inconclusive_result("Empty URL provided.")

    # Reject blatant non-URL inputs
    if " " in url_str or (len(url_str) < 4 and "." not in url_str):
        return _inconclusive_result("Provided input is not a valid web URL.", url_str)

    # Detect scheme
    scheme_match = re.match(r'^([a-zA-Z][a-zA-Z0-9+.-]*):', url_str)
    if scheme_match:
        scheme = scheme_match.group(1).lower()
        if scheme in ['javascript', 'data', 'vbscript', 'file', 'blob']:
            return {
                "url": url_str,
                "safety_percentage": 5,
                "risk_level": "High Risk",
                "explanation": f"High Risk: Dangerous URI scheme '{scheme}:' detected. This can execute malicious code or harvest browser credentials.",
                "checks_performed": ["scheme_validation"],
                "findings": [f"Dangerous URI scheme '{scheme}:'"]
            }
        if scheme not in ['http', 'https']:
            return _inconclusive_result(f"Unsupported URI scheme '{scheme}:'. Only standard web URLs (HTTP/HTTPS) can be assessed.", url_str)
    else:
        # Prepend https:// if format resembles a domain or host
        if '.' in url_str:
            url_str = 'https://' + url_str
            scheme = 'https'
        else:
            return _inconclusive_result("URL lacks a valid domain name or protocol scheme.", url_str)

    try:
        parsed = urllib.parse.urlparse(url_str)
    except Exception:
        return _inconclusive_result("Could not parse URL syntax.", url_str)

    host = (parsed.hostname or '').lower()
    if not host or ('.' not in host and host != 'localhost'):
        return _inconclusive_result("Could not identify a valid host or domain name in URL.", url_str)

    findings = []
    deductions = 0

    # 1. Base Score depending on initial protocol
    base_score = 85
    if scheme != 'https':
        findings.append("Unencrypted connection (HTTP instead of HTTPS)")
        deductions += 15

    # 2. Embedded credentials
    if parsed.username or parsed.password:
        deductions += 40
        findings.append("Contains embedded user authentication credentials (userinfo)")

    # 3. IP address check
    is_ip = False
    is_private_ip = False
    try:
        ip_obj = ipaddress.ip_address(host)
        is_ip = True
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_reserved:
            is_private_ip = True
            deductions += 65
            findings.append("Host points to a private or internal loopback IP address")
        else:
            deductions += 28
            findings.append("Uses raw public IP address instead of registered domain name")
    except ValueError:
        is_ip = False

    # 4. Punycode / Homograph attack check
    if "xn--" in host or any(ord(c) > 127 for c in host):
        deductions += 35
        findings.append("Contains Punycode or internationalized characters (possible domain spoofing)")

    # 5. High-risk TLD check
    parts = host.split('.')
    tld = parts[-1] if len(parts) > 1 else ''
    if tld in HIGH_ABUSE_TLDS:
        deductions += 20
        findings.append(f"Uses top-level domain '.{tld}' known for high abuse rates")

    # 6. Deep subdomain nesting
    if len(parts) > 4:
        deductions += 15
        findings.append(f"Excessive subdomain nesting ({len(parts) - 2} levels)")

    # 7. Brand spoofing / impersonation check
    for brand in BRAND_KEYWORDS:
        if brand in host:
            if not (host == f"{brand}.com" or host.endswith(f".{brand}.com")):
                deductions += 38
                findings.append(f"Potential brand impersonation: '{brand}' appears on an unverified domain")
                break

    # 8. Known URL Shortener
    is_shortener = host in KNOWN_SHORTENERS
    if is_shortener:
        deductions += 16
        findings.append("Uses URL shortening service which conceals destination address")

    # 9. Reputable domain baseline bonus
    is_reputable = False
    for rep in REPUTABLE_DOMAINS:
        if host == rep or host.endswith('.' + rep):
            is_reputable = True
            break

    if is_reputable and (not findings or (len(findings) == 1 and "Unencrypted" in findings[0])):
        base_score = 90

    # 10. DNS Resolution Check
    dns_resolved = False
    if not is_ip:
        try:
            socket.getaddrinfo(host, 80, socket.AF_UNSPEC, socket.SOCK_STREAM)
            dns_resolved = True
        except Exception:
            deductions += 25
            findings.append("Domain does not resolve to an active DNS host")

    # 11. Safe Redirect and TLS Inspection
    final_url = url_str
    if (dns_resolved or (is_ip and not is_private_ip)) and not is_private_ip:
        try:
            req = urllib.request.Request(
                url_str,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) SmartGatewaySafetyCheck/1.0'},
                method='HEAD'
            )
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=2.5, context=ctx) as resp:
                final_url = resp.geturl()
                final_parsed = urllib.parse.urlparse(final_url)
                if scheme == 'https' and final_parsed.scheme == 'http':
                    deductions += 25
                    findings.append("Redirect downgrades connection security from HTTPS to HTTP")
                if final_parsed.hostname and final_parsed.hostname.lower() != host:
                    if is_shortener:
                        findings.append(f"Resolves destination to '{final_parsed.hostname}'")
                    else:
                        findings.append(f"Redirects cross-domain to '{final_parsed.hostname}'")
        except urllib.error.HTTPError as e:
            try:
                e.close()
            except Exception:
                pass
            if e.code in [401, 403, 404]:
                findings.append(f"Server returned HTTP {e.code}")
        except urllib.error.URLError as e:
            err_msg = str(e).lower()
            if "certificate" in err_msg or "ssl" in err_msg:
                deductions += 30
                findings.append("SSL/TLS certificate verification failed or invalid")
            else:
                findings.append("Server connection could not be established")
        except Exception:
            findings.append("Redirect verification timed out")

    # 12. Calculate Final Safety Percentage
    # Local rule-based formula ensures 5 <= score <= 92 (Never claims 100% safe)
    score = base_score - deductions
    score = max(5, min(92, score))

    if score >= 70:
        risk_level = "Lower Risk"
        if not findings:
            explanation = "Lower Risk: No known threats detected across domain structure, protocol encryption, and redirect checks. Standard caution still advised when entering credentials."
        else:
            explanation = f"Lower Risk: General indicators appear healthy with minor observations: {'; '.join(findings)}."
    elif score >= 40:
        risk_level = "Suspicious"
        explanation = f"Suspicious: Warning signs detected ({'; '.join(findings)}). Exercise caution before proceeding or opening."
    else:
        risk_level = "High Risk"
        explanation = f"High Risk: Strong evidence of risk detected ({'; '.join(findings)}). It is not recommended to open this link."

    return {
        "url": url_str,
        "safety_percentage": score,
        "risk_level": risk_level,
        "explanation": explanation,
        "checks_performed": [
            "scheme_validation",
            "embedded_credentials_check",
            "ip_address_analysis",
            "punycode_homograph_check",
            "abuse_tld_check",
            "subdomain_nesting_check",
            "brand_impersonation_check",
            "shortener_detection",
            "dns_resolution",
            "safe_redirect_tls_inspection"
        ],
        "findings": findings
    }
