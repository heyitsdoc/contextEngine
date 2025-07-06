#!/bin/bash
coproc CONTEXT_CLIENT { socat - UNIX-CONNECT:/tmp/context.sock; }

# Send a line
echo "How do I stop a process?" >&"${CONTEXT_CLIENT[1]}"

# Read response
read -r reply <&"${CONTEXT_CLIENT[0]}"
echo "📘 Got: $reply"


