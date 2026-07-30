"""Extract an action graph from agent execution traces.

An agent that writes shell emits programs, not single actions, so there is no
granularity that obviously makes a good vocabulary. Too coarse and the graph
collapses into a near-clique; too fine and almost every atom is a singleton.

This module splits each compound command on its shell operators (`&&`, `;`, `|`)
— the boundaries the model itself wrote — and handles inline Python with a real
AST rather than regular expressions.

No data is bundled here: everything comes from the user's own traces.
"""

from __future__ import annotations

import ast
import glob
import json
import os
import re
import shlex
from dataclasses import dataclass, field
from typing import Iterator

# --------------------------------------------------------------------------
# Binary families (L2 level of the hierarchy)
# --------------------------------------------------------------------------

NETWORK = {"curl", "wget", "http", "nc", "ping", "dig", "ssh", "scp", "rsync"}
INSPECT = {"cat", "head", "tail", "ls", "find", "grep", "wc", "stat", "file",
           "du", "df", "tree", "jq", "sed", "awk", "diff"}
MUTATING = {"rm", "mv", "cp", "mkdir", "touch", "chmod", "chown", "ln", "tee"}
RUNTIME = {"python3", "python", "node", "deno", "bun", "npm", "pnpm", "tsc",
           "cargo", "go", "make"}
OPS = {"docker", "systemctl", "kill", "ps", "top", "journalctl", "crontab",
       "service", "pm2", "openclaw"}

# Language keywords: they appear at the start of a segment when an inline code
# block has been incorrectly split. Excluding them avoids phantom atoms such as
# `for`, `const`, `import`.
LANG_KEYWORDS = {
    "for", "if", "then", "fi", "do", "done", "else", "elif", "while", "until",
    "const", "let", "var", "import", "from", "with", "def", "class",
    "return", "in", "EOF", "PY", "EOSQL",
}

# Prefixes that wrap a command without being the action itself.
WRAPPERS = {"sudo", "time", "nohup", "exec", "env"}

_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?.*?^\s*\1\s*$", re.S | re.M)
_INLINE_FLAG = re.compile(r"(-c|-e|--eval)\s+(['\"])(?:(?!\2).)*\2", re.S)
_SHELL_VAR = re.compile(r"\$\{?\w+\}?")


def family(binary: str) -> str:
    """Classify a binary into a functional family (L2 level)."""
    if binary in ("git", "gh"):
        return "git_vcs"
    if binary in NETWORK:
        return "network_http"
    if binary in INSPECT:
        return "inspect_fs_text"
    if binary in MUTATING:
        return "mutating_write"
    if binary in RUNTIME:
        return "runtime_build"
    if binary in OPS:
        return "ops_process"
    return "other_shell"


# --------------------------------------------------------------------------
# Reading traces
# --------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A single tool call extracted from a session trace."""
    agent: str
    session: str
    timestamp: str
    tool: str
    args: dict


def iter_tool_calls(sessions_glob: str) -> Iterator[ToolCall]:
    """Iterate over JSONL sessions and yield each tool call.

    `sessions_glob` points to YOUR traces, for example:
        ~/.openclaw/agents/*/sessions/*.jsonl

    `.trajectory.` and `.checkpoint.` files are skipped: they are derived views
    that would duplicate the calls.
    """
    for path in glob.glob(os.path.expanduser(sessions_glob)):
        if ".trajectory." in path or ".checkpoint." in path:
            continue
        agent = path.split("/agents/")[1].split("/")[0] if "/agents/" in path else "unknown"
        session = os.path.basename(path).replace(".jsonl", "")
        with open(path, errors="ignore") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = event.get("message")
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "toolCall":
                        yield ToolCall(
                            agent=agent,
                            session=session,
                            timestamp=event.get("timestamp") or "",
                            tool=block.get("name") or "?",
                            args=block.get("arguments") or {},
                        )


def iter_shell_commands(sessions_glob: str) -> Iterator[tuple[ToolCall, str]]:
    """Yield only the calls that execute shell, paired with their command string."""
    for call in iter_tool_calls(sessions_glob):
        if call.tool not in ("exec", "bash", "shell"):
            continue
        command = call.args.get("command") or call.args.get("cmd") or ""
        if isinstance(command, list):
            command = " ".join(str(part) for part in command)
        if command.strip():
            yield call, command


# --------------------------------------------------------------------------
# Segmentation — the critical point
# --------------------------------------------------------------------------

def split_segments(command: str) -> list[str]:
    """Split a compound command on its shell operators.

    A regex split on `|` also cuts inside quoted strings: `grep -E 'a|b|c'`
    becomes three "commands", and each fragment produces a phantom atom
    (`awk.print1`, `awk.print2`, …) with no real command behind it.  On real
    corpora that inflates the vocabulary substantially.

    `shlex` with `punctuation_chars` respects quotes and avoids this.  Keep that
    property if you rewrite this function.
    """
    segments: list[str] = []
    for line in command.split("\n"):
        try:
            lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            # Unclosed quote: fall back to an approximate split rather than
            # dropping the entire command.
            segments.extend(re.split(r"&&|\|\||[;|]", line))
            continue
        current: list[str] = []
        for token in tokens:
            if token in ("&&", "||", ";", "|", "&"):
                if current:
                    segments.append(" ".join(current))
                    current = []
            else:
                current.append(token)
        if current:
            segments.append(" ".join(current))
    return segments


def mask_inline_code(command: str) -> str:
    """Neutralise inline code blocks before segmentation.

    A Python heredoc contains its own newlines: without this masking,
    segmentation splits it and collects `import`, `for`, `const` as if they
    were shell actions.
    """
    command = _HEREDOC.sub(" __INLINE__ ", command)
    return _INLINE_FLAG.sub(" -c __INLINE__ ", command)


def normalize_path(token: str, components: int = 2) -> str:
    """Reduce a path to its first N components.

    The number of components is the main granularity lever on the file side:
    too many and every file becomes a unique node; too few and the whole disk
    collapses into one.
    """
    token = token.strip("\"'").replace("~", "HOME")
    parts = [p for p in token.split("/") if p and p != "."]
    return "_".join(parts[:components]) if parts else ""


@dataclass
class Atom:
    """An atomic action: a binary and what it acts upon."""
    binary: str
    resource: str = ""

    @property
    def key(self) -> str:
        base = f"tool.exec.{family(self.binary)}.{self.binary}"
        return f"{base}.{self.resource}" if self.resource else base


def parse_segment(segment: str, url_segments: int = 3,
                  path_components: int = 2) -> Atom | None:
    """Turn a command segment into an atom, or None if it is not one.

    Resource normalisation is binary-specific: a URL, a file path and an SSH
    host have different structures and call for different treatment.
    """
    text = segment.strip().strip("()").rstrip("\\")
    if not text or text.startswith("#"):
        return None
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError:
        tokens = text.split()

    while tokens and (_ASSIGNMENT.match(tokens[0]) or tokens[0] in WRAPPERS):
        tokens.pop(0)
    if not tokens:
        return None

    binary = tokens[0].split("/")[-1]
    if binary in LANG_KEYWORDS or not re.match(r"^[A-Za-z_][\w.-]*$", binary):
        return None

    rest = tokens[1:]
    resource = ""

    if binary in ("curl", "wget"):
        for token in rest:
            match = re.match(r"https?://([^/]+)(/[^?\s]*)?", token)
            if match:
                host = match.group(1)
                parts = [p for p in (match.group(2) or "").split("/") if p]
                resource = "_".join([host, *parts[:url_segments]]).strip("_")
                break
    elif binary in RUNTIME:
        resource = "inline"
    elif binary == "ssh":
        skip = False
        for token in rest:
            if skip:
                skip = False
                continue
            if token in ("-o", "-i", "-p"):
                skip = True
                continue
            if not token.startswith("-"):
                resource = token.split("@")[-1]
                break
    elif binary in ("git", "gh"):
        subcommands = [t for t in rest if not t.startswith("-")]
        resource = "_".join(subcommands[:2])
    else:
        skip = False
        for token in rest:
            if skip:
                skip = False
                continue
            if token in ("-e", "-i", "-n"):
                skip = True
                continue
            if not token.startswith("-"):
                resource = normalize_path(token, path_components)
                break

    return Atom(binary, re.sub(r"[^A-Za-z0-9_.:-]", "", resource)[:30])


# --------------------------------------------------------------------------
# Inline Python — a real parser instead of regexes
# --------------------------------------------------------------------------

def extract_inline_python(command: str) -> list[str]:
    """Extract Python bodies passed via heredoc or via `-c`."""
    bodies: list[str] = []
    pattern = re.compile(
        r"(?:python3?|py)\s+[^\n]*?<<-?\s*['\"]?(\w+)['\"]?\s*\n(.*?)^\s*\1\s*$",
        re.S | re.M,
    )
    for match in pattern.finditer(command):
        bodies.append(match.group(2))
    for match in re.finditer(r"(?:python3?|py)\s+(?:-[A-Za-z]+\s+)*-c\s+(['\"])(.*?)\1",
                            command, re.S):
        bodies.append(match.group(2))
    return [b for b in bodies if b.strip()]


def python_atoms(source: str) -> set[str]:
    """Extract imports and calls from a Python block via its AST.

    Python has a parser; shell does not, so shell segmentation stays heuristic
    while this path works on a real syntax tree.

    Shell-interpolated variables (`$TOKEN`) are replaced before analysis;
    otherwise the block is not valid Python.
    """
    cleaned = _SHELL_VAR.sub("'__VAR__'", source).replace("\\n", "\n")
    try:
        tree = ast.parse(cleaned)
    except (SyntaxError, ValueError):
        return set()

    atoms: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                atoms.add(f"py.import.{alias.name.split('.')[0]}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                atoms.add(f"py.import.{node.module.split('.')[0]}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                chain = [func.attr]
                base = func.value
                while isinstance(base, ast.Attribute):
                    chain.append(base.attr)
                    base = base.value
                if isinstance(base, ast.Name):
                    chain.append(base.id)
                atoms.add("py.call." + ".".join(reversed(chain))[:44])
            elif isinstance(func, ast.Name):
                atoms.add(f"py.call.{func.id}")
    return atoms


# --------------------------------------------------------------------------
# Hyperedges
# --------------------------------------------------------------------------

@dataclass
class Hyperedge:
    """A compound command: a link that connects N actions at once.

    This is a true hyperedge, and it is irreducible to binary edges: pairs
    lose the information "these N actions formed a single block".
    """
    atoms: list[str]
    timestamp: str
    agent: str
    session: str
    command: str = field(repr=False, default="")


def build_hyperedges(sessions_glob: str, dedup: bool = True,
                     include_python: bool = True, **granularity) -> list[Hyperedge]:
    """Build hyperedges from your traces.

    `dedup` removes commands that replay an already-seen pattern. This matters:
    on a real corpus, more than half of all commands are repetitions of
    scheduled tasks. Without deduplication, the graph mostly measures the
    frequency of your crons.
    """
    edges: list[Hyperedge] = []
    seen: set[tuple[str, ...]] = set()

    for call, command in iter_shell_commands(sessions_glob):
        masked = mask_inline_code(command)
        atoms = {
            atom.key
            for atom in (parse_segment(s, **granularity) for s in split_segments(masked))
            if atom is not None
        }
        if include_python:
            for body in extract_inline_python(command):
                atoms |= python_atoms(body)
        if not atoms:
            continue

        signature = tuple(sorted(atoms))
        if dedup and signature in seen:
            continue
        seen.add(signature)
        edges.append(Hyperedge(
            atoms=sorted(atoms),
            timestamp=call.timestamp,
            agent=call.agent,
            session=call.session,
            command=command,
        ))
    return edges


# --------------------------------------------------------------------------
# Security
# --------------------------------------------------------------------------

_SECRET_PATTERNS = [
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{12,}", re.I),
    re.compile(r"([A-Za-z0-9_]*(?:TOKEN|SECRET|PASSWORD|PAT|API_?KEY)[A-Za-z0-9_]*\s*=\s*)\S+", re.I),
    re.compile(r"(gh[pousr]_)[A-Za-z0-9]{20,}"),
    re.compile(r"(sk-)[A-Za-z0-9]{20,}"),
]


def mask_secrets(text: str) -> str:
    """Mask secrets in a command before any display or sharing.

    Agent traces routinely carry Bearer tokens, API keys and exported secrets in
    plain text. Pass any command through this function before displaying it in a
    notebook meant for sharing.

    The patterns cover common forms only: this reduces the risk without
    eliminating it. Review output before publishing.
    """
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(r"\1***", text)
    return text
