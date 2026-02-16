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


def test_facts_upsert_and_conflict_policy(tmp_path: Path) -> None:
    db = tmp_path / "memory.sqlite3"
    store = MemoryStore(db)

    fid1 = store.upsert_fact(
        subject_type="user",
        subject_id="u1",
        key="preferred_name",
        value="Alfredo",
        confidence=0.9,
        source="explicit",
        evidence="запомни, меня зовут Альфредо",
        tags=["identity"],
        person_ref="user:u1",
    )
    assert fid1 > 0

    # same value -> should keep same row (id stable)
    fid2 = store.upsert_fact(
        subject_type="user",
        subject_id="u1",
        key="preferred_name",
        value="Alfredo",
        confidence=0.95,
        source="auto",
        evidence="(repeat)",
        tags=["identity", "repeat"],
        person_ref="user:u1",
    )
    assert fid2 == fid1

    f = store.get_fact(subject_type="user", subject_id="u1", key="preferred_name")
    assert f is not None
    assert f.value == "Alfredo"
    assert f.confidence >= 0.95
    assert "identity" in f.tags

    # lower priority overwrite should be blocked
    store.upsert_fact(
        subject_type="user",
        subject_id="u1",
        key="preferred_name",
        value="NotAlfredo",
        confidence=0.99,
        source="auto",
        evidence="call me NotAlfredo",
    )
    f2 = store.get_fact(subject_type="user", subject_id="u1", key="preferred_name")
    assert f2 is not None
    assert f2.value == "Alfredo"

    # higher/equal priority overwrite allowed
    store.upsert_fact(
        subject_type="user",
        subject_id="u1",
        key="preferred_name",
        value="Alfredo The Second",
        confidence=0.9,
        source="explicit",
        evidence="зови меня Alfredo The Second",
    )
    f3 = store.get_fact(subject_type="user", subject_id="u1", key="preferred_name")
    assert f3 is not None
    assert f3.value == "Alfredo The Second"


def test_facts_set_add_remove_values(tmp_path: Path) -> None:
    db = tmp_path / "memory.sqlite3"
    store = MemoryStore(db)

    store.add_fact_value(
        subject_type="user",
        subject_id="u1",
        key="likes",
        value="tea",
        source="explicit",
        evidence="запомни: я люблю чай",
    )
    store.add_fact_value(
        subject_type="user",
        subject_id="u1",
        key="likes",
        value="coffee",
        source="explicit",
        evidence="и кофе тоже",
    )

    f = store.get_fact(subject_type="user", subject_id="u1", key="likes")
    assert f is not None
    assert isinstance(f.value, list)
    assert set(f.value) == {"tea", "coffee"}

    store.remove_fact_value(
        subject_type="user",
        subject_id="u1",
        key="likes",
        value="tea",
        source="explicit",
        evidence="я больше не люблю чай",
    )
    f2 = store.get_fact(subject_type="user", subject_id="u1", key="likes")
    assert f2 is not None
    assert set(f2.value) == {"coffee"}

    store.remove_fact_value(
        subject_type="user",
        subject_id="u1",
        key="likes",
        value="coffee",
        source="explicit",
        evidence="и кофе тоже больше не люблю",
    )
    f3 = store.get_fact(subject_type="user", subject_id="u1", key="likes")
    assert f3 is None
