from __future__ import annotations

import re
from functools import lru_cache
from urllib.parse import urlparse, unquote
from typing import Dict, List, Tuple, Union, Optional

import requests


DEFAULT_INDEX_BLOB_URLS = [
    "https://github.com/kamyu104/LeetCode-Solutions/blob/master/0001-1000.md",
    "https://github.com/kamyu104/LeetCode-Solutions/blob/master/1001-2000.md",
    "https://github.com/kamyu104/LeetCode-Solutions/blob/master/README.md",
]


def _normalize_title(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def github_file_url_to_raw(url: str) -> str:
    """
    Convert GitHub file URL to raw URL.
    Supports:
      - https://github.com/<owner>/<repo>/blob/<branch>/<path>
      - https://github.com/<owner>/<repo>/raw/<branch>/<path>
      - https://raw.githubusercontent.com/<owner>/<repo>/<branch>/<path>
    """
    url = url.strip()
    p = urlparse(url)
    host = p.netloc.lower()
    path = unquote(p.path)

    if host == "raw.githubusercontent.com":
        return url

    if host != "github.com":
        raise ValueError(f"Unsupported host for GitHub raw conversion: {host}")

    parts = [x for x in path.split("/") if x]
    if len(parts) < 5:
        raise ValueError(f"Unrecognized GitHub file URL format: {url}")

    owner, repo, kind, branch = parts[0], parts[1], parts[2], parts[3]
    if kind not in ("blob", "raw"):
        raise ValueError(f"Unrecognized GitHub file URL kind '{kind}': {url}")

    file_path = "/".join(parts[4:])
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"


@lru_cache(maxsize=128)
def _fetch_text(url: str, timeout: int = 30) -> str:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "leetcode-solution-fetcher"})
    resp.raise_for_status()
    return resp.text


def _abs_raw_from_md(md_blob_url: str, href: str) -> str:
    """
    Resolve markdown href to a raw URL, relative to md_blob_url repo/branch if needed.
    """
    href = href.strip()

    if href.startswith("http://") or href.startswith("https://"):
        return github_file_url_to_raw(href)

    if href.startswith("/"):
        return github_file_url_to_raw("https://github.com" + href)

    base_raw = github_file_url_to_raw(md_blob_url)
    p = urlparse(base_raw)
    parts = [x for x in p.path.split("/") if x]  # [owner, repo, branch, ...]
    if len(parts) < 3:
        raise ValueError("Failed to parse base repo info from markdown URL")

    owner, repo, branch = parts[0], parts[1], parts[2]
    rel = href[2:] if href.startswith("./") else href
    rel = rel.lstrip("/")
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{rel}"


def _pick_python_href(links: List[Tuple[str, str]]) -> Optional[str]:
    """
    Prefer Python3 over Python; require .py target.
    links: list[(text, href)]
    """
    best = None
    for text, href in links:
        t = text.strip().lower()
        if "python" not in t:
            continue
        if ".py" not in href.lower():
            continue
        if "python3" in t:
            return href
        best = best or href
    return best


def _parse_index_markdown(md_text: str, md_blob_url: str) -> Tuple[Dict[int, Tuple[str, str]], Dict[str, Tuple[int, str]]]:
    """
    Parse a markdown index file and return:
      by_number[num] = (title, py_raw_url)
      by_title[norm_title] = (num, py_raw_url)

    Handles rows like:
      0001 | [Two Sum](...) | [C++](...) [Python](./Python/two-sum.py) | ...
    and also:
      | 0001 | [Two Sum](...) | ...
    """
    by_number: Dict[int, Tuple[str, str]] = {}
    by_title: Dict[str, Tuple[int, str]] = {}

    # capture first 3 cells: # | Title | Solution |
    row_re = re.compile(
        r"^\s*\|?\s*(\d{1,4})\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|",
        flags=re.I
    )
    md_link_re = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    for line in md_text.splitlines():
        ll = line.lower()
        if "python" not in ll or ".py" not in ll:
            continue

        m = row_re.match(line)
        if not m:
            continue

        num = int(m.group(1))
        title_cell = m.group(2).strip()
        solution_cell = m.group(3).strip()

        title_links = md_link_re.findall(title_cell)
        title = title_links[0][0].strip() if title_links else title_cell

        sol_links = md_link_re.findall(solution_cell)
        py_href = _pick_python_href([(txt, href) for (txt, href) in sol_links])
        if not py_href:
            continue

        py_raw_url = _abs_raw_from_md(md_blob_url, py_href)
        by_number[num] = (title, py_raw_url)
        by_title[_normalize_title(title)] = (num, py_raw_url)

    return by_number, by_title


@lru_cache(maxsize=16)
def _build_combined_index(index_blob_urls: Tuple[str, ...]) -> Tuple[Dict[int, Tuple[str, str]], Dict[str, Tuple[int, str]]]:
    """
    Build a combined index across multiple markdown files.
    First hit wins (so order of index_blob_urls matters).
    """
    combined_by_number: Dict[int, Tuple[str, str]] = {}
    combined_by_title: Dict[str, Tuple[int, str]] = {}

    for md_blob_url in index_blob_urls:
        md_raw_url = github_file_url_to_raw(md_blob_url)
        md_text = _fetch_text(md_raw_url)

        by_num, by_t = _parse_index_markdown(md_text, md_blob_url)

        for num, val in by_num.items():
            combined_by_number.setdefault(num, val)
        for tkey, val in by_t.items():
            combined_by_title.setdefault(tkey, val)

    return combined_by_number, combined_by_title


def fetch_leetcode_python_solution(
    problem_number_or_title: Union[int, str],
    index_blob_urls: Optional[List[str]] = None,
) -> str:
    """
    Fetch the Python solution code for a LeetCode problem by number (e.g. 1/"0001")
    or by title (case-insensitive), searching across multiple index markdown URLs.
    """
    if index_blob_urls is None:
        index_blob_urls = DEFAULT_INDEX_BLOB_URLS

    by_number, by_title = _build_combined_index(tuple(index_blob_urls))

    # Number lookup
    if isinstance(problem_number_or_title, int) or re.fullmatch(r"\s*\d{1,4}\s*", str(problem_number_or_title)):
        num = int(str(problem_number_or_title).strip())
        if num not in by_number:
            raise ValueError(f"Problem #{num} not found (or no Python .py link found) in provided indexes.")
        _, py_raw_url = by_number[num]
        return _fetch_text(py_raw_url)

    # Title lookup
    key = _normalize_title(str(problem_number_or_title))
    if key not in by_title:
        raise ValueError(f"Problem title '{problem_number_or_title}' not found (or no Python .py link found) in provided indexes.")
    _, py_raw_url = by_title[key]
    return _fetch_text(py_raw_url)

def get_leetcode_id_from_title(
    title: str,
    index_blob_urls: Optional[List[str]] = None,
) -> int:
    """
    Return the LeetCode problem id (number) for an exact-ish title match (case-insensitive).
    Raises ValueError if not found.
    """
    if index_blob_urls is None:
        index_blob_urls = DEFAULT_INDEX_BLOB_URLS

    _, by_title = _build_combined_index(tuple(index_blob_urls))

    key = _normalize_title(title)
    if key not in by_title:
        raise ValueError(f"Title '{title}' not found in provided indexes.")
    num, _ = by_title[key]
    return num


def get_leetcode_title_from_id(
    problem_id: Union[int, str],
    index_blob_urls: Optional[List[str]] = None,
) -> str:
    """
    Return the LeetCode problem title for a given problem id.
    Raises ValueError if not found.
    """
    if index_blob_urls is None:
        index_blob_urls = DEFAULT_INDEX_BLOB_URLS

    by_number, _ = _build_combined_index(tuple(index_blob_urls))

    if isinstance(problem_id, int) or re.fullmatch(r"\s*\d{1,4}\s*", str(problem_id)):
        num = int(str(problem_id).strip())
    else:
        raise ValueError(f"Invalid problem id: {problem_id!r}")

    if num not in by_number:
        raise ValueError(f"Problem id '{num}' not found in provided indexes.")
    title, _ = by_number[num]
    return title
