import os

os.environ["ENABLE_GRAMMAR_MODEL"] = "0"
from text_polish import clean_caption, compose, fix_articles, pluralize  # noqa: E402

# Real outputs of modelConcat_1_89.h5 (VGG16 + LSTM) on test images, and the polished result.
REAL = {
    "people are standing on the side of bus": "people are standing on the side of a bus",
    "man is wearing plaid paint and holding an electronic in his hand":
        "a man is wearing plaid paint and holding an electronic in his hand",
    "man in blue shirt is standing next to several basket poles":
        "a man in a blue shirt is standing next to several basket poles",
    "man in black shirt is standing next to woman with black hair":
        "a man in a black shirt is standing next to a woman with black hair",
    "man is standing on the edge of staircase with his eye in the air":
        "a man is standing on the edge of a staircase with his eye in the air",
}


def test_real_model_captions():
    for raw, want in REAL.items():
        assert clean_caption(raw) == want


def test_articles_after_prepositions_but_not_mass_nouns():
    assert clean_caption("boy in red shirt is jumping into water") == "a boy in a red shirt is jumping into water"
    assert clean_caption("two dogs are playing in snow") == "two dogs are playing in snow"
    assert clean_caption("dog is running through the grass") == "a dog is running through the grass"
    assert clean_caption("girl in front of house") == "a girl in front of a house"


def test_stutters_dangling_and_tokens():
    assert clean_caption("man in red and white helmet and helmet riding bike") == "a man in a red and white helmet riding a bike"
    assert clean_caption("black dog dog is running on on beach end") == "a black dog is running on a beach"
    assert clean_caption("dog is sitting on bench on bench with") == "a dog is sitting on a bench"


def test_a_an_and_plurals():
    assert fix_articles("a apple, an banana, a hour, an university") == "an apple, a banana, an hour, a university"
    assert pluralize("person") == "people" and pluralize("bus") == "buses" and pluralize("party") == "parties"


def test_compose_narration():
    r = compose("people are standing on the side of bus", ["bus", "person", "person", "dog"],
                ["man in blue shirt is standing next to several basket poles",
                 "people are standing on the side of bus"])
    assert r["caption"] == "People are standing on the side of a bus."
    assert r["closeups"] == ["A man in a blue shirt is standing next to several basket poles."]
    assert r["narration"] == ("People are standing on the side of a bus. I can also see a dog. "
                              "Up close, a man in a blue shirt is standing next to several basket poles.")


def test_live_mode_skips_closeups():
    r = compose("dog is running on beach", ["dog"], ["dog on beach"], mode="live")
    assert r["narration"] == "A dog is running on a beach."
    assert compose("dog on beach", [])["narration"] == "A dog on a beach."
