import os
os.environ["ENABLE_GRAMMAR_MODEL"] = "0"
from text_polish import clean_caption, compose, fix_articles, answer_sentence, pluralize


def test_artifacts_and_prompt_echo():
    assert clean_caption("a detailed picture of arafed dog sitting on a bench") == "a dog sitting on a bench"
    assert clean_caption("elephant walking in the grass") == "an elephant walking in the grass"
    assert clean_caption("dogs playing in the park") == "dogs playing in the park"
    assert clean_caption("people are walking on a street") == "people are walking on a street"
    assert clean_caption("there is a man riding a horse on the beach") == "a man riding a horse on the beach"


def test_repeats_and_dangling():
    assert clean_caption("a dog a dog sitting on a bench on a bench with a") == "a dog sitting on a bench"
    assert clean_caption("a cat on the the sofa and") == "a cat on the sofa"


def test_articles():
    assert fix_articles("a apple and an banana, a hour, an university, a umbrella") == \
        "an apple and a banana, an hour, a university, an umbrella"


def test_plurals():
    assert pluralize("person") == "people" and pluralize("bus") == "buses"
    assert pluralize("wine glass") == "wine glasses" and pluralize("knife") == "knives"
    assert pluralize("dog") == "dogs" and pluralize("party") == "parties"


def test_answers():
    assert answer_sentence("How many people are there?", "3") == "There are three people."
    assert answer_sentence("How many dogs are there?", "1") == "There is one dog."
    assert answer_sentence("Is it raining?", "no") == "No."
    assert answer_sentence("What colour is the bicycle?", "red") == "It looks like red."


def test_compose_full():
    r = compose("a brown dog sitting on a bench next to a bicycle bicycle",
                ["dog", "bicycle", "person", "person", "umbrella"], ["PARK"],
                "How many people are there?", "2")
    assert r["caption"] == "A brown dog sitting on a bench next to a bicycle."
    assert r["narration"] == ("I can see a brown dog sitting on a bench next to a bicycle. "
                              "I can also see two people and an umbrella. "
                              "There is text that reads “PARK”. There are two people.")


def test_read_mode():
    r = compose("a box of medicine on a table", [], ["Paracetamol", "500 mg"], mode="read")
    assert r["narration"].startswith("The text reads: “Paracetamol 500 mg”.")
