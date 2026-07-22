#!/usr/bin/env python3
"""
chat.py — interactive chatPDB CLI with hybrid RAG + fine-tuned model.

Each turn:
  1. Corpus fast-path: an exact PDB/UniProt/CCD ID match is answered from the corpus directly,
     bypassing the LLM entirely (rag/corpus_lookup.py).
  2. Otherwise, retrieves the top-N most relevant corpus chunks for the question
     (rag/retrieve.py) and appends them to the user turn as grounded context — the same
     structural pattern chatPDB's own SFT data uses (see scripts/build_dataset.py's
     RAG-shaped-synthesis generator), not injected into the system message.
  3. Streams the response live from the fused Qwen3-32B model.
  4. Auto-executes any Biopython code blocks and prints the computed values (rag/tool_exec.py).

Usage:
    python scripts/chat.py
    python scripts/chat.py --model models/chatpdb_32b_v1
    python scripts/chat.py --no-rag              # model-only, no retrieval
    python scripts/chat.py --n-chunks 8          # retrieve more context
    python scripts/chat.py --max-tokens 1024     # longer responses

Slash commands:
    /help           show available commands
    /clear          clear screen and reprint banner
    /reset          clear conversation history (asks for confirmation)
    /history [n]    show last n conversation turns (default: all)
    /save           save conversation to a Markdown file
    /info           show model, corpus, and training configuration
    /retry          regenerate the previous response
"""

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

# Allow imports from project root regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt_toolkit import PromptSession
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text
from rich.theme import Theme

_THEME = Theme({
    "chatpdb.header":  "bold cyan",
    "chatpdb.section": "bold yellow",
    "chatpdb.label":   "dim",
    "chatpdb.ok":      "bold green",
    "chatpdb.warn":    "bold yellow",
})
_console = Console(theme=_THEME, highlight=False)

# Dark-teal/emerald palette, distinct from chem_sage's navy/bright-blue — ties to the teal used
# for chatPDB's own README badge (#00897B, the Qwen3-32B badge colour).
_PT_STYLE = Style.from_dict({
    "bottom-toolbar":      "bg:#0d2620 #3ddbb0",
    "bottom-toolbar.text": "bg:#0d2620 #3ddbb0",
    "prompt":              "bold #3ddbb0",
    "completion-menu.completion":          "bg:#1a4a3a #c0ffe8",
    "completion-menu.completion.current":  "bg:#3ddbb0 #0d2620 bold",
})

_SLASH_COMPLETER = WordCompleter(
    ["/help", "/clear", "/reset", "/history", "/save", "/info", "/retry"],
    sentence=True,
)

_DESKTOP = Path.home() / "Desktop"
_SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "config" / "system_prompt.txt"


# ---------------------------------------------------------------------------
# Toolbar
# ---------------------------------------------------------------------------

def _make_toolbar(history: list, model_name: str = "chatPDB", rag_active: bool = True):
    def _toolbar():
        turns = len(history) // 2
        ctx = min(turns, 3)
        state = f"  ·  turn {turns}  ·  ctx {ctx}/3" if turns else ""
        rag_tag = " · RAG" if rag_active else ""
        return HTML(
            f'<b fg="#3ddbb0"> ⬡ chatPDB </b>'
            f'<style fg="#2a6a58"> · {escape(model_name)} · QLoRA{rag_tag}{state}</style>'
            f'<style fg="#1a4a3a">    /help · /clear · /reset · /history · /save · /info · /retry</style>'
        )
    return _toolbar


# ---------------------------------------------------------------------------
# Progress spinner
# ---------------------------------------------------------------------------

def _progress() -> Progress:
    return Progress(
        SpinnerColumn(spinner_name="bouncingBar", style="bold cyan"),
        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
        TimeElapsedColumn(),
        console=_console,
        transient=True,
    )


# ---------------------------------------------------------------------------
# Silence third-party chatter during loading
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _quiet():
    """Suppress stdout, stderr, and Python logging <= WARNING for the block."""
    import logging
    devnull = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = devnull, devnull
    logging.disable(logging.WARNING)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)
        sys.stdout, sys.stderr = old_out, old_err
        devnull.close()


# ---------------------------------------------------------------------------
# Banner / Goodbye
# ---------------------------------------------------------------------------

def _figlet_art() -> str:
    try:
        import pyfiglet
        return pyfiglet.figlet_format("chatPDB", font="small_slant")
    except Exception:
        return "  CHATPDB"


def _print_banner() -> None:
    sys.stdout.write("\x1b]0;chatPDB\x07")
    sys.stdout.flush()

    _console.print()
    for line in _figlet_art().splitlines():
        _console.print(f"[chatpdb.header]{escape(line)}[/chatpdb.header]")

    _console.print()
    _console.print(
        "  [chatpdb.label]chatPDB[/chatpdb.label]"
        "  [dim]·[/dim]"
        "  [dim]Protein Structure AI[/dim]"
        "  [dim]·[/dim]"
        "  [dim]Qwen3-32B · QLoRA · RAG[/dim]"
    )
    _console.print(
        "  [dim]Built by "
        "[link=https://marcdeller.com]Marc C. Deller[/link]"
        "  ·  "
        "[link=mailto:marc@marcdeller.com]marc@marcdeller.com[/link]"
        "[/dim]"
    )
    _console.print()
    _console.rule(style="dim")
    _console.print()


def _print_goodbye() -> None:
    art_markup = "\n".join(
        f"[chatpdb.header]{escape(line)}[/chatpdb.header]"
        for line in _figlet_art().splitlines()
    )
    content = (
        art_markup
        + "\n\n"
        "  [dim]Thank you for using chatPDB.[/dim]\n\n"
        "  [dim]Built by "
        "[link=https://marcdeller.com]Marc C. Deller[/link]"
        "  ·  "
        "[link=mailto:marc@marcdeller.com]marc@marcdeller.com[/link]"
        "[/dim]"
    )
    _console.print()
    _console.print(Panel(content, border_style="dim cyan", padding=(1, 2)))
    _console.print()


# ---------------------------------------------------------------------------
# System prompt / context injection
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    _SYSTEM_PROMPT_PATH.read_text() if _SYSTEM_PROMPT_PATH.exists() else "You are chatPDB."
)


def _build_user_turn(user_input: str, context: str | None) -> str:
    """Append retrieved context to the USER message, matching chatPDB's real training data.

    Unlike chem_sage's chat.py (which injects context into the system message), chatPDB's own
    SFT examples that use retrieved context put it in the user turn under a literal
    "Retrieved context:" label (scripts/build_dataset.py's RAG-shaped-synthesis generator) —
    the system message is always the exact unmodified config/system_prompt.txt text in 100% of
    training examples, never augmented. Matching that structure keeps generation consistent with
    what the model actually saw in training.
    """
    if not context:
        return user_input
    return f"{user_input}\n\nRetrieved context:\n{context}"


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------

def _print_response() -> None:
    _console.print()
    _console.rule("[chatpdb.header] chatPDB [/chatpdb.header]", style="dim cyan")
    _console.print()


def _print_stats(n_tokens: int, elapsed: float) -> None:
    tok_per_s = n_tokens / elapsed if elapsed > 0 else 0
    _console.print(
        f"\n  [dim]⚡ {n_tokens} tok · {tok_per_s:.1f} tok/s · {elapsed:.1f}s[/dim]"
    )


def _print_tool_output(response: str, execute) -> None:
    exec_result = execute(response)
    if exec_result.startswith("[no executable"):
        return
    _console.print()
    _console.print(Panel(
        Markdown(f"```\n{exec_result}\n```", code_theme="monokai"),
        title="[chatpdb.section] Biopython Output [/chatpdb.section]",
        border_style="yellow",
        padding=(0, 1),
    ))


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

def _handle_slash_command(cmd: str, history: list, info: dict | None = None) -> None:
    parts = cmd.strip().split(None, 1)
    name = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if name == "/help":
        _console.print(Panel(
            "  [b]/help[/b]            show this message\n"
            "  [b]/clear[/b]           clear screen and reprint banner\n"
            "  [b]/reset[/b]           clear conversation history\n"
            "  [b]/history \\[n][/b]     show last n turns (default: all)\n"
            "  [b]/save[/b]            save conversation to Markdown\n"
            "  [b]/info[/b]            show model, corpus, and training details\n"
            "  [b]/retry[/b]           regenerate the previous response\n\n"
            "  [b]quit / exit / q[/b]  exit chatPDB\n\n"
            "  [dim]Built by [link=https://marcdeller.com]Marc C. Deller[/link]"
            "  ·  [link=mailto:marc@marcdeller.com]marc@marcdeller.com[/link][/dim]",
            title="[chatpdb.header] Commands [/chatpdb.header]",
            border_style="cyan",
            padding=(1, 2),
        ))
        return

    if name == "/clear":
        os.system("clear")
        _print_banner()
        return

    if name == "/reset":
        try:
            confirm = pt_prompt("  Clear conversation history? [y/N] → ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            _console.print()
            return
        if confirm == "y":
            history.clear()
            _console.print("\n  [chatpdb.ok]✓[/chatpdb.ok]  [dim]Conversation history cleared.[/dim]\n")
        else:
            _console.print("\n  [dim]Cancelled.[/dim]\n")
        return

    if name == "/history":
        pairs = list(zip(history[::2], history[1::2]))
        n = int(arg) if arg.isdigit() else len(pairs)
        shown = pairs[-n:]
        if not shown:
            _console.print("\n  [dim]No conversation history yet.[/dim]\n")
            return
        _console.print()
        offset = len(pairs) - len(shown) + 1
        for i, (u, a) in enumerate(shown, offset):
            _console.print(f"  [chatpdb.label][{i}] You:[/chatpdb.label]     {escape(u['content'][:120])}")
            _console.print(f"      [chatpdb.label]chatPDB:[/chatpdb.label] {escape(a['content'][:120])}")
            _console.print()
        return

    if name == "/save":
        _save_conversation(history)
        return

    if name == "/info":
        _print_info(info)
        return

    _console.print(
        f"\n  [chatpdb.warn]Unknown command:[/chatpdb.warn] [dim]{escape(cmd)}[/dim]"
        "  (type [b]/help[/b] for commands)\n"
    )


def _save_conversation(history: list) -> None:
    if not history:
        _console.print("\n  [dim]Nothing to save yet.[/dim]\n")
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    default = _DESKTOP / f"chatpdb_{ts}.md"
    try:
        raw = pt_prompt(f"  Save to [{default}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        _console.print("\n  [dim]Cancelled.[/dim]\n")
        return
    fpath = Path(raw).expanduser() if raw else default
    fpath.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# chatPDB Conversation\n",
        f"*{time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n",
        "Built by [Marc C. Deller](https://marcdeller.com) · marc@marcdeller.com\n\n",
        "---\n\n",
    ]
    for i, (u, a) in enumerate(zip(history[::2], history[1::2]), 1):
        lines.append(f"## Turn {i}\n\n")
        lines.append(f"**You:** {u['content']}\n\n")
        lines.append(f"**chatPDB:** {a['content']}\n\n")

    fpath.write_text("".join(lines))
    turns = len(history) // 2
    _console.print(
        f"\n  [chatpdb.ok]✓[/chatpdb.ok]  {turns} turn{'s' if turns != 1 else ''}"
        f" saved → [dim]{fpath}[/dim]\n"
    )


def _print_info(info: dict | None) -> None:
    if not info:
        _console.print("\n  [dim]Session info not available.[/dim]\n")
        return

    rag_status = (
        f"{info.get('chunk_count', 0):,} indexed chunks  ·  "
        f"{info.get('corpus_sources', '?')} source files"
        if info.get("rag")
        else "disabled (--no-rag)"
    )

    content = (
        f"  [chatpdb.label]Model   [/chatpdb.label]  {escape(info.get('model_name', '?'))}\n"
        f"  [chatpdb.label]Disk    [/chatpdb.label]  {info.get('model_gb', 0):.1f} GB"
        f"  ·  32B parameters  ·  4-bit quantised\n\n"
        f"  [chatpdb.label]Training[/chatpdb.label]  "
        f"{info.get('iters', '?'):,} iters  ·  "
        f"{info.get('ltype', 'RSLoRA')} rank {info.get('rank', 32)}"
        f"  ·  lr {info.get('lr', '?')}  ·  ctx {info.get('ctx', 1536):,}\n\n"
        f"  [chatpdb.label]Corpus  [/chatpdb.label]  {rag_status}\n"
        f"  [chatpdb.label]Embed   [/chatpdb.label]  BAAI/bge-base-en-v1.5  ·  ChromaDB\n\n"
        f"  [chatpdb.label]Built by[/chatpdb.label]  "
        "[link=https://marcdeller.com]Marc C. Deller[/link]"
        "  ·  "
        "[link=mailto:marc@marcdeller.com]marc@marcdeller.com[/link]"
    )
    _console.print()
    _console.print(Panel(
        content,
        title="[chatpdb.header] chatPDB — Session Info [/chatpdb.header]",
        border_style="cyan",
        padding=(1, 2),
    ))
    _console.print()


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

def chat_loop(
    model_path: str,
    use_rag: bool,
    n_chunks: int,
    chroma_store: str,
    max_tokens: int,
) -> None:
    # -- Initialise -------------------------------------------------------------
    with _progress() as bar:
        bar.add_task("Initialising…", total=None)
        from mlx_lm import load, generate
        try:
            from mlx_lm import stream_generate
        except ImportError:
            stream_generate = None
        from mlx_lm.sample_utils import make_sampler, make_logits_processors
        from rag.corpus_lookup import lookup as corpus_lookup
        from rag.retrieve import Retriever, format_context
        from rag.tool_exec import execute

    _print_banner()

    # -- Model --------------------------------------------------------------------
    with _progress() as bar:
        bar.add_task("Loading model…", total=None)
        with _quiet():
            model, tokenizer = load(model_path)
    model_name = Path(model_path).name
    _console.print(
        f"  [chatpdb.ok]✓[/chatpdb.ok]  [chatpdb.label]Model   [/chatpdb.label]  {escape(model_name)}"
    )
    model_gb = sum(p.stat().st_size for p in Path(model_path).rglob("*") if p.is_file()) / 1e9
    _console.print(
        f"  [chatpdb.ok]✓[/chatpdb.ok]  [chatpdb.label]Disk    [/chatpdb.label]  "
        f"{model_gb:.1f} GB  ·  32B parameters  ·  4-bit quantised"
    )

    # -- Retriever ------------------------------------------------------------------
    retriever: "Retriever | None" = None
    chunk_count = 0
    corpus_sources = 0
    if use_rag:
        with _progress() as bar:
            bar.add_task("Loading retriever…", total=None)
            with _quiet():
                try:
                    retriever = Retriever(store=chroma_store)
                    chunk_count = retriever._collection.count()
                except Exception:
                    pass
        if retriever:
            _corpus_dir = Path("data/corpus")
            corpus_sources = len(list(_corpus_dir.rglob("*.csv"))) if _corpus_dir.exists() else 0
            _console.print(
                f"  [chatpdb.ok]✓[/chatpdb.ok]  [chatpdb.label]Corpus  [/chatpdb.label]  "
                f"{chunk_count:,} indexed chunks  ·  {corpus_sources} source files"
            )
        else:
            _console.print(
                "  [chatpdb.warn]⚠[/chatpdb.warn]  "
                "[chatpdb.label]Retriever unavailable — model only[/chatpdb.label]"
            )

    # -- Training config -------------------------------------------------------------
    try:
        import yaml as _yaml
        _cfg_path = Path(__file__).parent.parent / "config" / "train_config.yaml"
        _cfg = _yaml.safe_load(_cfg_path.read_text())
        _iters = _cfg.get("iters", 820)
        _rank = _cfg.get("lora_parameters", {}).get("rank", 32)
        _lr = _cfg.get("learning_rate", 2e-5)
        _ctx = _cfg.get("max_seq_length", 1536)
        _ltype = "RSLoRA" if _cfg.get("use_rslora", False) else "LoRA"
    except Exception:
        _iters, _rank, _lr, _ctx, _ltype = 820, 32, 2e-5, 1536, "RSLoRA"
    _lr_str = f"{_lr:.0e}" if isinstance(_lr, float) else str(_lr)
    _console.print(
        f"  [chatpdb.ok]✓[/chatpdb.ok]  [chatpdb.label]Training[/chatpdb.label]  "
        f"{_iters:,} iters  ·  {_ltype} rank {_rank}  ·  lr {_lr_str}  ·  ctx {_ctx:,}"
    )

    # -- Session info dict (passed to /info) -----------------------------------------
    _session_info = {
        "model_name": model_name,
        "model_gb": model_gb,
        "chunk_count": chunk_count,
        "corpus_sources": corpus_sources,
        "rag": retriever is not None,
        "iters": _iters,
        "rank": _rank,
        "lr": _lr_str,
        "ctx": _ctx,
        "ltype": _ltype,
    }

    # -- Ready ------------------------------------------------------------------------
    _console.print()
    _console.rule(style="dim")
    _console.print()
    _console.print(
        "  [dim]Type your question and press Enter.  "
        "/help for commands.  Tab to complete.  'quit' to exit.[/dim]"
    )
    _console.print()

    history: list[dict] = []
    session = PromptSession(
        history=InMemoryHistory(),
        style=_PT_STYLE,
        bottom_toolbar=_make_toolbar(history, model_name, rag_active=retriever is not None),
        completer=_SLASH_COMPLETER,
        complete_while_typing=False,
    )

    # -- Generation helper (closure over model / tokenizer / imports) ---------------
    def _do_generate(messages: list, _max_tokens: int) -> tuple[str, float]:
        """Stream-generate a response; return (full_text, elapsed_seconds)."""
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,   # required for this Qwen3 base -- see PROJECT_PLAN.md Phase 1
        )

        _print_response()

        full_text = ""
        t0 = time.time()

        if stream_generate is not None:
            with Live(
                Text(""),
                console=_console,
                refresh_per_second=12,
                vertical_overflow="visible",
            ) as live:
                for chunk in stream_generate(
                    model, tokenizer, prompt=prompt_str,
                    max_tokens=_max_tokens,
                    sampler=make_sampler(temp=0.15),
                    logits_processors=make_logits_processors(repetition_penalty=1.15),
                ):
                    tok = chunk.text if hasattr(chunk, "text") else str(chunk)
                    if tok:
                        full_text += tok
                        live.update(Text(full_text))
                if full_text:
                    live.update(Markdown(full_text, code_theme="monokai"))
        else:
            # Fallback: blocking generate with spinner
            with _console.status(
                "[bold cyan]  Thinking…[/bold cyan]",
                spinner="dots", spinner_style="bold cyan",
            ):
                full_text = generate(
                    model, tokenizer, prompt=prompt_str,
                    max_tokens=_max_tokens,
                    sampler=make_sampler(temp=0.15),
                    logits_processors=make_logits_processors(repetition_penalty=1.15),
                )
            _console.print(Markdown(full_text, code_theme="monokai"))

        return full_text, time.time() - t0

    # State for /retry
    _last_messages: list | None = None

    while True:
        try:
            user_input = session.prompt(HTML("<b>You</b>: ")).strip()
        except (KeyboardInterrupt, EOFError):
            _print_goodbye()
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            _print_goodbye()
            break

        # -- /retry — handled here so it can access model internals -----------------
        if user_input.lower() == "/retry":
            if _last_messages is None:
                _console.print("\n  [dim]Nothing to retry yet.[/dim]\n")
                continue
            response, elapsed = _do_generate(_last_messages, max_tokens)
            try:
                n_tokens = len(tokenizer.encode(response))
            except Exception:
                n_tokens = len(response.split())
            _print_stats(n_tokens, elapsed)
            _print_tool_output(response, execute)
            if len(history) >= 2:
                history[-1] = {"role": "assistant", "content": response}
            continue

        if user_input.startswith("/"):
            _handle_slash_command(user_input, history, info=_session_info)
            continue

        # -- Corpus fast-path: exact ID match answered from the corpus, never LLM ---
        fast = corpus_lookup(user_input)
        if fast and not fast.startswith(("No PDB ID", "No corpus match")):
            _console.print()
            _console.print(Panel(
                fast,
                title="[chatpdb.header] Corpus Lookup [/chatpdb.header]",
                border_style="dim cyan",
                padding=(0, 1),
            ))
            summary = "Corpus lookup:\n" + fast
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": summary})
            continue

        # -- RAG retrieval ------------------------------------------------------------
        context: str | None = None
        if retriever:
            try:
                with _console.status(
                    "[dim]  🔍 Retrieving context…[/dim]",
                    spinner="dots", spinner_style="dim cyan",
                ):
                    chunks = retriever.retrieve(user_input, n=n_chunks)
                    context = format_context(chunks)
                _console.print(
                    f"  [dim]📚 {len(chunks)} chunk{'s' if len(chunks) != 1 else ''} retrieved[/dim]"
                )
            except Exception as exc:
                _console.print(Panel(
                    escape(str(exc)),
                    title="[bold red] Retrieval Error [/bold red]",
                    border_style="red",
                    padding=(0, 1),
                ))

        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.extend(history[-6:])
        messages.append({"role": "user", "content": _build_user_turn(user_input, context)})

        # Stash for /retry
        _last_messages = messages

        # -- Generate -------------------------------------------------------------------
        response, elapsed = _do_generate(messages, max_tokens)

        try:
            n_tokens = len(tokenizer.encode(response))
        except Exception:
            n_tokens = len(response.split())
        _print_stats(n_tokens, elapsed)

        # -- Execute Biopython code blocks -----------------------------------------------
        _print_tool_output(response, execute)

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="chatPDB — RAG-augmented protein-structure assistant")
    ap.add_argument("--model", default="models/chatpdb_32b_v1",
                    help="Path to fused model directory (default: models/chatpdb_32b_v1)")
    ap.add_argument("--chroma", default=".chroma",
                    help="ChromaDB store path (default: .chroma)")
    ap.add_argument("--no-rag", action="store_true",
                    help="Disable RAG retrieval — use fine-tuned model only")
    ap.add_argument("--n-chunks", type=int, default=5,
                    help="Number of corpus chunks to retrieve per query (default: 5)")
    ap.add_argument("--max-tokens", type=int, default=600,
                    help="Maximum tokens to generate per response (default: 600)")
    args = ap.parse_args()

    chat_loop(
        model_path=args.model,
        use_rag=not args.no_rag,
        n_chunks=args.n_chunks,
        chroma_store=args.chroma,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
