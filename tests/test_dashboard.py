"""Dashboard renders without exceptions and with honest labels (skipped if streamlit isn't installed)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")


def test_dashboard_renders(built, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    import ranker
    from optimizer import optimize_front_page

    ranked = ranker.predict(str(built["artifacts"]), str(built["processed"]), "test",
                            str(tmp_path / "ranked.parquet"), "testv1")
    pool = ranked[ranked.group_id == ranked.group_id.iloc[0]]
    optimize_front_page(pool).to_parquet(tmp_path / "manifest.parquet", index=False)
    monkeypatch.setenv("NEWSREC_DASHBOARD_MANIFEST", str(tmp_path / "manifest.parquet"))
    monkeypatch.setenv("NEWSREC_DASHBOARD_RANKED", str(tmp_path / "ranked.parquet"))
    monkeypatch.setenv("NEWSREC_ARTIFACTS_DIR", str(built["artifacts"]))
    (built["artifacts"] / "CURRENT").write_text("testv1")
    at = AppTest.from_file(str(Path(__file__).parents[1] / "dashboard.py"), default_timeout=120).run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Model Confidence" not in text and "raw ranking score" in text
    assert any("SYNTHETIC" in w.value for w in at.warning)
