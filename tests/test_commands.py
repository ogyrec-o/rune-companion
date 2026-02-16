# tests/test_commands.py

from __future__ import annotations

from rune_companion.cli.commands import CommandRegistry


def test_command_registry_routes_4_and_5_params(state) -> None:
    reg = CommandRegistry()
    called = {"h4": 0, "h5": 0}

    def h4(state, args, user_id, room_id):
        called["h4"] += 1
        return "h4"

    def h5(state, args, user_id, room_id, emit):
        called["h5"] += 1
        if emit is not None:
            emit("note")
        return "h5"

    reg.register("a", h4, "a")
    reg.register("b", h5, "b")

    assert reg.handle(state, "/a x", user_id="u", room_id="r") == "h4"
    assert reg.handle(state, "/b y", user_id="u", room_id="r", emit=lambda _: None) == "h5"
    assert called["h4"] == 1
    assert called["h5"] == 1


def test_command_registry_unknown_and_non_command(state) -> None:
    reg = CommandRegistry()
    assert reg.handle(state, "hello") is None
    assert "Unknown command" in (reg.handle(state, "/nope") or "")
