# tests/test_memory_store.py

from __future__ import annotations

from pathlib import Path

from rune_companion.memory.store import MemoryStore


def test_memory_add_query_update_delete(tmp_path: Path) -> None:
    db = tmp_path / "memory.sqlite3"
    store = MemoryStore(db)

    mem_id = store.add_memory(
        subject_type="user",
        subject_id="u1",
        text="User likes tea",
        tags=["pref", "food"],
        importance=0.9,
        source="test",
        person_ref=None,
    )
    assert mem_id > 0

    items = store.query_memory(subject_type="user", subject_id="u1", limit=10)
    assert len(items) == 1
    assert items[0].text == "User likes tea"
    assert set(items[0].tags) == {"pref", "food"}

    store.update_memory(mem_id, text="User likes green tea", tags=["pref"], importance=0.8)
    items2 = store.query_memory(subject_type="user", subject_id="u1", limit=10)
    assert items2[0].text == "User likes green tea"
    assert items2[0].tags == ["pref"]
    assert abs(items2[0].importance - 0.8) < 1e-6

    store.delete_memory(mem_id)
    items3 = store.query_memory(subject_type="user", subject_id="u1", limit=10)
    assert items3 == []
