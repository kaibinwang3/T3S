#!/bin/bash

# Use read -r to treat backslashes literally
# Use -d '' to read until null byte (effectively reads everything until EOF marker)
read -r -d '' model_config <<'EOF'
This is line one.
    This is line two with $HOME literally.
    -aa
    "aaa"
    123
        2
    q
EOF

echo "--- Here Doc (read) ---"
echo "$model_config"
echo "-----------------------"
