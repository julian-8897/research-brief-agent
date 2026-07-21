import pytest

from src.settings import _bool_env, _csv_env, _float_env, _int_env, _optional_env


def test_int_env_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("X_INT", raising=False)
    assert _int_env("X_INT", 7) == 7


def test_int_env_parses_value(monkeypatch):
    monkeypatch.setenv("X_INT", "42")
    assert _int_env("X_INT", 7) == 42


def test_int_env_rejects_non_integer(monkeypatch):
    monkeypatch.setenv("X_INT", "abc")
    with pytest.raises(ValueError, match="X_INT must be an integer"):
        _int_env("X_INT", 7)


def test_float_env_parses_value_and_rejects_garbage(monkeypatch):
    monkeypatch.setenv("X_FLOAT", "0.25")
    assert _float_env("X_FLOAT", 1.0) == 0.25
    monkeypatch.setenv("X_FLOAT", "nan-ish")
    with pytest.raises(ValueError, match="X_FLOAT must be a float"):
        _float_env("X_FLOAT", 1.0)


def test_bool_env_defaults_and_truthy_spellings(monkeypatch):
    monkeypatch.delenv("X_BOOL", raising=False)
    assert _bool_env("X_BOOL", True) is True
    assert _bool_env("X_BOOL", False) is False
    for spelling in ("1", "true", "TRUE", " yes ", "on"):
        monkeypatch.setenv("X_BOOL", spelling)
        assert _bool_env("X_BOOL", False) is True
    for spelling in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("X_BOOL", spelling)
        assert _bool_env("X_BOOL", True) is False


def test_csv_env_parses_and_strips(monkeypatch):
    monkeypatch.delenv("X_CSV", raising=False)
    assert _csv_env("X_CSV") == ()
    monkeypatch.setenv("X_CSV", " a , b,, c ")
    assert _csv_env("X_CSV") == ("a", "b", "c")


def test_optional_env_normalizes_blank_to_none(monkeypatch):
    monkeypatch.delenv("X_OPT", raising=False)
    assert _optional_env("X_OPT", "default") == "default"
    monkeypatch.setenv("X_OPT", "   ")
    assert _optional_env("X_OPT", "default") is None
    monkeypatch.setenv("X_OPT", " value ")
    assert _optional_env("X_OPT") == "value"
