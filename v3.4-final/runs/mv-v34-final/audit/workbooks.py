"""Read the four authorized author workbooks from literal XLSX XML."""

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction as F
from io import BytesIO
import json
from pathlib import Path
import subprocess
from uuid import uuid4
from xml.etree import ElementTree as X
from zipfile import ZipFile

from proofs import PROJECT, endpoint_shortcuts, need, oracle, stdlib_only
from replay import checked_payload, decode, digest, encode


ARCHIVE = PROJECT / "data/raw/Datasets.rar"
ARCHIVE_SHA = "1ac5860e2f54c3a15ee48016b29070be910f314a6973a1b167e84faa038eb601"
CASES = {
    "latet-1908-k2": "LATET1908.xlsx",
    "latet-2209-k3": "LATET2209.xlsx",
    "latet-2809-k3": "LATET2809.xlsx",
    "latet-2808-k3": "LATET2808.xlsx",
}
FIELDS = ("capacity", "limit", "service", "supply", "population", "travel", "vehicles")
A = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
RELS = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def col_letters(number):
    col = ""
    while number:
        number, add = divmod(number - 1, 26)
        col = chr(65 + add) + col
    return col


def values(payload):
    with ZipFile(BytesIO(payload)) as book:
        workbook = X.fromstring(book.read("xl/workbook.xml"))
        sheet = workbook.find(f"{A}sheets/{A}sheet")
        relationships = X.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        target, = (line.get("Target") for line in relationships.findall(f"{RELS}Relationship")
                   if line.get("Id") == sheet.get(R))
        strings = X.fromstring(book.read("xl/sharedStrings.xml"))
        text = ["".join(node.text or "" for node in item.iter(f"{A}t"))
                for item in strings.findall(f"{A}si")]
        worksheet = X.fromstring(book.read(f"xl/{target}"))
    result = {}
    for cell in worksheet.iter(f"{A}c"):
        value = cell.find(f"{A}v")
        if value is None or value.text is None:
            continue
        kind = cell.get("t", "n")
        if kind == "s":
            result[cell.get("r")] = text[int(value.text)]
        elif kind == "n":
            result[cell.get("r")] = F(value.text)
    return sheet.get("name"), result


def read_model(payload, vehicles):
    worksheet, cells = values(payload)
    need(worksheet == "Sheet1" and tuple(cells[key] for key in ("E1", "F1", "G1", "I1"))
         == ("pi", "ni", "si", "C"), "workbook sheet or role labels differ")
    nodes = list(range(71))
    matrix = [col_letters(col) for col in range(14, 85)]
    need([cells[col + "1"] for col in matrix] == nodes
         and [cells[f"M{i + 2}"] for i in nodes] == nodes
         and [cells[f"D{i + 2}"] for i in nodes] == nodes, "workbook station labels disagree")
    service, population, supply = (
        [cells.get(f"{column}{i + 2}", F(0)) for i in nodes]
        for column in ("E", "F", "G")
    )
    active = [0, *(i for i in nodes[1:] if supply[i] or population[i])]
    model = {
        "vehicles": vehicles, "capacity": cells["J1"], "limit": cells["J2"],
        "service": [service[i] for i in active], "population": [population[i] for i in active],
        "supply": [supply[i] for i in active],
        "travel": [[cells[f"{matrix[j]}{i + 2}"] for j in active] for i in active],
    }
    oracle().Model.read(model)
    return model, active


def inactive_transits(payload, active):
    """Prove that the removed zero-quantity stations cannot shorten any arc."""
    _, cells = values(payload)
    nodes = list(range(71))
    excluded = [i for i in nodes if i not in active]
    proven_active = [0, *(i for i in nodes[1:]
                          if cells.get(f"F{i + 2}", 0) or cells.get(f"G{i + 2}", 0))]
    need(proven_active == active, "workbook active station list differs")
    service = [cells.get(f"E{i + 2}", F(0)) for i in nodes]
    columns = [col_letters(col) for col in range(14, 85)]
    direct = [[cells[f"{column}{i + 2}"] for column in columns] for i in nodes]
    need(min(*service, *(min(row) for row in direct)) >= 0, "negative original workbook time")
    best = [row.copy() for row in direct]
    # Floyd closure permits only the inactive stations as interior points.
    # Service is charged once for each such interior visit.
    for k in excluded:
        for i in nodes:
            head = best[i][k] + service[k]
            for j in nodes:
                best[i][j] = min(best[i][j], head + best[k][j])
    pairs = [(i, j) for i in active for j in active if i != j]
    shortcuts = [(i, j, str(direct[i][j] - best[i][j])) for i, j in pairs
                 if best[i][j] < direct[i][j]]
    need(not shortcuts, f"inactive original workbook stations shorten an active arc: {shortcuts[:5]}")
    return {"inactive_source_nodes": excluded, "active_ordered_pairs_checked": len(pairs),
            "strict_shortcuts": 0, "service_charged_at_inactive_station": True}


def literal_json(value):
    """Keep XLSX numerics exact when emitting the JSON input accepted by the solvers."""
    if type(value) is F:
        with localcontext() as context:
            context.prec = len(str(abs(value.numerator))) + len(str(value.denominator)) + 2
            number = Decimal(value.numerator) / Decimal(value.denominator)
        need(F(number) == value, "workbook cell did not round-trip as an exact decimal")
        return str(number)
    if type(value) is list:
        return "[" + ",".join(map(literal_json, value)) + "]"
    if type(value) is dict:
        return "{" + ",".join(f"{json.dumps(key)}:{literal_json(item)}" for key, item in value.items()) + "}"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def numeric(value):
    return list(map(numeric, value)) if type(value) is list else F(value)


def differences(canonical, literal, active):
    changed = []
    for key in FIELDS:
        a, b = numeric(canonical[key]), numeric(literal[key])
        if a == b:
            continue
        need(key in ("service", "population", "supply"), f"unexpected workbook difference in {key}")
        changed.extend({"field": key, "source_node": node, "canonical": str(x), "workbook": str(y)}
                       for node, x, y in zip(active, a, b) if x != y)
    return changed


def prepare(out):
    checked_payload(ARCHIVE.read_bytes(), ARCHIVE_SHA, "author RAR")
    latet = subprocess.check_output(["bsdtar", "-xOf", ARCHIVE, "Latet.rar"])
    unseal = decode((PROJECT / "runs/mv-v31b-formal/unseal.json").read_bytes())
    selected = {row["case"]: row for row in unseal["cases"] if row["case"] in CASES}
    need(set(selected) == set(CASES), "four author workbooks must come from the public case list")
    inputs = out / "inputs"
    sources = out / "workbooks"
    inputs.mkdir(parents=True, exist_ok=False)
    sources.mkdir()
    records = []
    for name, member in CASES.items():
        row = selected[name]
        canonical = decode(checked_payload((PROJECT / row["path"]).read_bytes(),
                                           row["sha256"], "unsealed canonical graph"))
        raw = subprocess.check_output(["bsdtar", "-xOf", "-", member], input=latet)
        checked_payload(raw, canonical["source"]["workbook_sha256"], member)
        (sources / member).write_bytes(raw)
        model, active = read_model(raw, row["vehicles"])
        need(active == canonical["source_nodes"], "workbook station filter differs from canonical graph")
        reported_differences = differences(canonical, model, active)
        original = inputs / f"{name}-workbook.json"
        original.write_text(literal_json({
            **model, "id": name, "family_id": "latet", "raw_id": canonical["raw_id"],
            "source_nodes": active, "source": canonical["source"],
        }) + "\n", encoding="utf-8")
        anonymous = inputs / f"{uuid4().hex}.json"
        anonymous.write_text(literal_json(model) + "\n", encoding="utf-8")
        named_data, anonymous_data = decode(original.read_bytes()), decode(anonymous.read_bytes())
        need(set(anonymous_data) == set(FIELDS) and all(
            numeric(named_data[key]) == numeric(anonymous_data[key]) for key in FIELDS),
             "anonymization changed the model or retained identifying metadata")
        records.append({
            "case": name, "workbook": member, "workbook_sha256": digest(raw),
            "canonical_sha256": row["sha256"], "differences": reported_differences,
            "named": original.name, "named_sha256": digest(original.read_bytes()),
            "anonymous": anonymous.name, "anonymous_sha256": digest(anonymous.read_bytes()),
            "endpoint_shortcut_failures": endpoint_shortcuts(oracle().Model.read(model)),
            "inactive_transits": inactive_transits(raw, active),
        })
    return records


def verify_inactive(prepared):
    records = decode((prepared / "workbooks.json").read_bytes())
    need(len(records) == 4 and {record["case"] for record in records} == set(CASES),
         "expected the four authorized author workbooks")
    result = []
    for record in records:
        member = CASES[record["case"]]
        raw = checked_payload((prepared / "workbooks" / member).read_bytes(),
                              record["workbook_sha256"], "staged author workbook")
        named = decode(checked_payload((prepared / "inputs" / record["named"]).read_bytes(),
                                        record["named_sha256"], "literal model"))
        result.append({"case": record["case"], "workbook_sha256": record["workbook_sha256"],
                       **inactive_transits(raw, named["source_nodes"])})
    return {"workbooks": len(result), "cases": result}


if __name__ == "__main__":
    stdlib_only()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(encode(verify_inactive(args.prepared)), encoding="utf-8")
