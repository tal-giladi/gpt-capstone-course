"""Merge curriculum/glossary-inbox/*.md into glossary.md (dedupe by term, sorted)."""
import glob, re
terms = {}
for f in ["glossary.md"] + sorted(glob.glob("curriculum/glossary-inbox/*.md")):
    for line in open(f, encoding="utf-8"):
        m = re.match(r"^- \*\*(.+?)\*\* — (.+)$", line.strip())
        if m and m.group(1).lower() not in {k.lower() for k in terms}:
            terms[m.group(1)] = m.group(2)
out = ["# Glossary", "", "Terms used across the capstone, in alphabetical order.", ""]
out += [f"- **{k}** — {v}" for k, v in sorted(terms.items(), key=lambda kv: kv[0].lower())]
open("glossary.md", "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print(len(terms), "terms")
