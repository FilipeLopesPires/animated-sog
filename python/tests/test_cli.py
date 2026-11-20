from asog import container
from asog.cli import main


def test_encode_info_check(tmp_path, synthetic_source, capsys):
    out = tmp_path / "cli.asog"
    main(["-q", "encode", str(synthetic_source), str(out), "--qp", "1", "--prune", "0"])
    man, _, _ = container.read(out)
    assert man["motion"]["qp"] == 1.0 and man["count"] == 3400
    main(["info", str(out)])
    assert "splats 3400 (400 static, 3000 dynamic)" in capsys.readouterr().out
    main(["check", str(synthetic_source), str(out)])
    report = capsys.readouterr().out
    assert report.startswith("splats 3400") and "centre error over life" in report and "SH view colour error" in report


def test_flipbook_encode_is_encode(tmp_path, synthetic_source):
    a, b = tmp_path / "a.asog", tmp_path / "b.asog"
    main(["-q", "encode", str(synthetic_source), str(a)])
    main(["-q", "flipbook", "encode", str(synthetic_source), str(b)])
    _, sa, _ = container.read(a)
    _, sb, _ = container.read(b)
    assert sa.keys() == sb.keys()
