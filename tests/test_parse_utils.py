from src.utils.think_utils import strip_think_content


def test_strip_complete_think_block():
    text = "<think>private reasoning</think>\n\nVisible answer."
    assert strip_think_content(text) == "Visible answer."


def test_strip_reasoning_with_unmatched_standalone_closing_tag():
    text = "private reasoning\nacross lines\n</think>\n\nVisible answer."
    assert strip_think_content(text) == "Visible answer."


def test_last_unmatched_closing_tag_is_final_answer_boundary():
    text = "old reasoning\n</think>\nmore reasoning\n</think>\nFinal answer."
    assert strip_think_content(text) == "Final answer."


def test_preserve_plain_answer_and_inline_literal_tag_discussion():
    plain = "A normal answer without serialized reasoning."
    literal = "The literal token `</think>` closes a reasoning block."
    assert strip_think_content(plain) == plain
    assert strip_think_content(literal) == literal


def test_remove_residual_standalone_opening_tag():
    text = "<think>\nVisible text without a closing marker."
    assert strip_think_content(text) == "Visible text without a closing marker."
