#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOKS_DIR="$REPO_ROOT/.githooks"

if [ ! -d "$HOOKS_DIR" ]; then
  echo "No .githooks directory found. Creating one with pre-commit hook."
  mkdir -p "$HOOKS_DIR"
fi

cat > "$HOOKS_DIR/pre-commit" <<'HOOK'
#!/usr/bin/env bash
set -euo pipefail

# Run notebook widget fixer on staged notebooks
STAGED_NOTEBOOKS=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.ipynb$' || true)
if [ -n "$STAGED_NOTEBOOKS" ]; then
  echo "Running notebook fixer on staged notebooks..."
  for nb in $STAGED_NOTEBOOKS; do
    python3 scripts/fix_notebook_widgets.py "$nb"
    git add "$nb"
  done
fi

exit 0
HOOK

chmod +x "$HOOKS_DIR/pre-commit"

echo "Installing git hooks from $HOOKS_DIR to .git/hooks/"
cp -v "$HOOKS_DIR"/* .git/hooks/
chmod +x .git/hooks/*

echo "Git hooks installed."
