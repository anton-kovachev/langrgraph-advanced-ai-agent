"""Tests for final answer synthesis."""

import pytest
from unittest.mock import patch, Mock
from main import synthesize_final_answer

def test_synthesize_full_results(mock_state, mock_analysis_results, mock_llm):
    """Test synthesis with all analysis results available."""
    mock_state.update(mock_analysis_results)
    
    with patch('main.get_synthesis_messages') as mock_get_messages:
        llm = mock_llm
            llm.invoke.return_value = Mock(content="final synthesis")
            with patch('main.llm', llm):
                result = synthesize_final_answer(mock_state)            mock_get_messages.assert_called_once()
            assert result["final_answer"] == "final synthesis"

def test_synthesize_partial_results(mock_state, mock_analysis_results, mock_llm):
    """Test synthesis with partial analysis results."""
    partial_results = {k: v for k, v in mock_analysis_results.items() if k != "reddit_analysis"}
    mock_state.update(partial_results)
    
    with patch('main.get_synthesis_messages') as mock_get_messages:
        llm = mock_llm
            llm.invoke.return_value = Mock(content="partial synthesis")
            with patch('main.llm', llm):
                result = synthesize_final_answer(mock_state)            mock_get_messages.assert_called_once()
            assert result["final_answer"] == "partial synthesis"

def test_synthesize_empty_results(mock_state):
    """Test synthesis with no analysis results."""
    result = synthesize_final_answer(mock_state)
    assert result["final_answer"] == ""

def test_synthesize_invalid_inputs(mock_state, mock_llm):
    """Test synthesis with invalid input types."""
    mock_state.update({
        "google_analysis": None,
        "bing_analysis": 123,  # Invalid type
        "yandex_analysis": ["invalid"],  # Invalid type
        "reddit_analysis": None
    })
    
    with patch('main.get_synthesis_messages') as mock_get_messages:
        llm = mock_llm
        llm.complete.return_value = Mock(content="synthesis")
        with patch('main.llm', llm):
            result = synthesize_final_answer(mock_state)
            
            assert isinstance(result["final_answer"], str)

def test_synthesis_error_handling(mock_state, mock_analysis_results, mock_llm):
    """Test error handling in synthesis."""
    mock_state.update(mock_analysis_results)
    
    llm = mock_llm
    llm.complete.side_effect = Exception("Synthesis error")
    with patch('main.llm', llm):
        result = synthesize_final_answer(mock_state)
        assert result["final_answer"] == ""

@pytest.mark.parametrize("analysis_results,expected_empty", [
    ({}, True),
    ({"google_analysis": "result"}, False),
    ({"google_analysis": "", "bing_analysis": ""}, True),
    ({"google_analysis": None, "bing_analysis": None}, True),
])
def test_synthesis_empty_detection(mock_state, analysis_results, expected_empty):
    """Test detection of empty analysis results."""
    mock_state.update(analysis_results)
    result = synthesize_final_answer(mock_state)
    assert (result["final_answer"] == "") == expected_empty
