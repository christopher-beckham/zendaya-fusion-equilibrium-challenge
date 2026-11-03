"""args_to_cli.py TRAIN_PY ARGS_JSON [key=value ...]: print the train.py command-line flags that reproduce ARGS_JSON
(string values as recorded), with the given overrides. 'out' is dropped (run.sh passes --out)."""
import json, re, sys
train, aj = sys.argv[1], sys.argv[2]
import os
d = os.path.dirname(os.path.abspath(train))
src = "".join(open(os.path.join(d, f)).read() for f in os.listdir(d) if f.endswith(".py"))
flags = dict(re.findall(r'add_argument\(\s*"(--[\w-]+)"([^)]*)\)', src))
a = json.load(open(aj))
for kv in sys.argv[3:]:
    k, v = kv.split("=", 1); a[k] = v
out = []
for k, v in a.items():
    if k == "out":
        continue
    f = "--" + k.replace("_", "-")
    if f not in flags:
        sys.exit(f"unknown flag {f}")
    if "store_true" in flags[f]:
        if v == "True": out.append(f)
    elif "store_false" in flags[f]:
        if v == "False": out.append(f)
    elif v != "None":
        out += [f, v]
print(" ".join(out))
