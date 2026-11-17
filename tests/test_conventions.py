import ast
import io
import pathlib
import subprocess
import tokenize

ROOT = pathlib.Path(__file__).resolve().parents[1]


def our_files():
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [f for f in listed if (ROOT / f).exists()]


def comment_lines(source):
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return [
        t.start[0] for t in tokens if t.type == tokenize.COMMENT and not (t.start[0] == 1 and t.string.startswith("#!"))
    ]


def docstring_lines(source):
    tree = ast.parse(source)
    reads_doc = any(isinstance(n, ast.Name) and n.id == "__doc__" for n in ast.walk(tree))
    printed = tree.body[0] if tree.body and reads_doc else None
    return [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Constant)
        and isinstance(n.value.value, str)
        and n is not printed
    ]


def test_no_comments_and_no_docstrings():
    found = []
    for f in our_files():
        source = (ROOT / f).read_text()
        found += [f"{f}:{line}" for line in comment_lines(source) + docstring_lines(source)]
    assert not found
