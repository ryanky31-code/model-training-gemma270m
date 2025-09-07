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

echo "Configuring repository to use hooks from $HOOKS_DIR"
# Prefer setting core.hooksPath so hooks are tracked in the repo and we don't overwrite local .git/hooks samples
git config core.hooksPath "$HOOKS_DIR"
echo "Configured git to use hooks from $HOOKS_DIR (git config core.hooksPath)"

echo "Ensure hooks are executable"
chmod +x "$HOOKS_DIR"/* || true

echo "Git hooks configured. To revert, run: git config --unset core.hooksPath"
