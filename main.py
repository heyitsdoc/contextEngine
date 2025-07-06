from context_engine import ContextEngine
import sys

ctx = ContextEngine()

ctx.add_context("Kill process on port 3000")
ctx.add_context("Show current git branch")
ctx.add_context("List all running Docker containers")

print(str(sys.argv[1]))

query =  str(sys.argv[1])
results = ctx.retrieve(query)

print("\n🔍 Most Relevant Context:")
for text, score in results:
    print(f"→ \"{text}\" (distance: {score:.4f})")
