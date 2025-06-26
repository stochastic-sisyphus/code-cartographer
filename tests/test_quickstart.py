from pathlib import Path
import json

from code_cartographer.quickstart import quick_analyze


def test_quick_analyze(tmp_path: Path) -> None:
    sample_file = tmp_path / "sample.py"
    sample_file.write_text("def foo():\n    return 42\n")

    output = quick_analyze(tmp_path, tmp_path / "result.json")
    assert output.exists()

    data = json.loads(output.read_text())
    assert data.get("files")

