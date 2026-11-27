"""Read a Latet workbook's literal cells inside the cold-clock window."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "audit"))
from workbooks import literal_json, read_model


def main():
    reference = Path(sys.argv[1])
    workbook = Path(os.environ["V34_XLSX"])
    engine = Path(os.environ["V34_WORKBOOK_ENGINE"])
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    expected = reference.read_bytes()
    model, _ = read_model(workbook.read_bytes(), json.loads(expected)["vehicles"])
    payload = (literal_json(model) + "\n").encode()
    if payload != expected:
        raise ValueError("literal workbook cells disagree with the registered canonical bytes")
    directory = out.with_name(out.name + ".xlsx-input")
    directory.mkdir(parents=True, exist_ok=False)
    generated = directory / reference.name
    generated.write_bytes(payload)
    os.execv(sys.executable, [sys.executable, "-B", str(engine), str(generated), *sys.argv[2:]])


if __name__ == "__main__":
    main()
