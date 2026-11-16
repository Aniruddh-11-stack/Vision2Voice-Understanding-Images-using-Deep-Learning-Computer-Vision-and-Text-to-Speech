"""Unit tests for vision2voice.text_polish (cleans raw VGG16 + LSTM captions)."""

from vision2voice.text_polish import clean_caption, compose


def test_adds_missing_articles():
    assert clean_caption("little boy in red shorts kicks soccer ball in grassy field") == (
        "a little boy in red shorts kicks a soccer ball in a grassy field"
    )
    assert clean_caption("two dogs are playing in snow") == "two dogs are playing in snow"


def test_removes_repeats_and_dangling_words():
    assert clean_caption("dog is sitting on bench on bench with") == "a dog is sitting on a bench"


def test_compose_filters_off_topic_closeups():
    r = compose(
        "black dog and brown dog play in grassy field",
        ["horse", "horse"],
        [
            {"label": "horse", "caption": "black and white dog is running", "score": -0.4},
            {"label": "horse", "caption": "brown horse is running through field", "score": -0.7},
        ],
    )
    assert r["narration"] == (
        "A black dog and brown dog play in a grassy field. "
        "Up close, a brown horse is running through a field."
    )
