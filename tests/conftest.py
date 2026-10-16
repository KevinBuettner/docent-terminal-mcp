import pytest

from docent_terminal_mcp import server


@pytest.fixture(autouse=True)
def _second_guesses(request, monkeypatch):
    """Second guesses (a line refused once, run when sent again) are off
    but in the tests marked second_guess: the rest type their lines once."""
    monkeypatch.setattr(server, "SECOND_GUESS_ON", "second_guess" in request.keywords)
    yield
    server.GUESS_PENDING.clear()
    server.GUESS_DECLINED.clear()
