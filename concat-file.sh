#!/bin/bash
# Usage: ./concat.sh txt md py

pattern=""
for ext in "$@"; do
    pattern="$pattern -o -name \"*.$ext\""
done
pattern="${pattern# -o }"

eval "find . -type f \( $pattern \) -print0" |
while IFS= read -r -d '' file; do
    echo "--- BEGIN \`$file\` ---"
    cat "$file"
    echo "--- END \`$file\` ---"
    echo
done > combined.txt