"""Tests for state management and initialization."""

import pytest
from typing import Dict, Any
from main import State

def test_state_initialization():
    """Test State class initialization with valid inputs."""
    state = State(
        messages=[],
        user_input="test query",
        google_results=None,
        bing_results=None,
        yandex_results=None,
        reddit_results=None,
        selected_reddit_urls=None,
        reddit_post_data=None,
        google_analysis=None,
        bing_analysis=None,
        yandex_analysis=None,
        reddit_analysis=None,
        final_answer=None
    )
    assert state["user_input"] == "test query"
    assert state["messages"] == []
    assert state["google_results"] is None

def test_state_get_method(mock_state: Dict[str, Any]):
    """Test state dictionary get method."""
    state = State(**mock_state)
    assert state.get("user_input") == "test query"
    assert state.get("non_existent_key", "default") == "default"

def test_state_required_fields():
    """Test State class requires all mandatory fields."""
    with pytest.raises(Exception):
        State(messages=[])  # Missing required fields

def test_state_type_validation():
    """Test State class type validation."""
    with pytest.raises(Exception):
        State(
            messages="not a list",  # Wrong type
            user_input=123,  # Wrong type
            google_results=None,
            bing_results=None,
            yandex_results=None,
            reddit_results=None,
            selected_reddit_urls=None,
            reddit_post_data=None,
            google_analysis=None,
            bing_analysis=None,
            yandex_analysis=None,
            reddit_analysis=None,
            final_answer=None
        )
