# -*- coding: utf-8 -*-
from src.kg.entries import Entry
from src.kg.mentions import Mention
from src.kg.relations import find_stated_relations

MARRIED = [{"rel": "married_to", "source": "jonathan", "target": "mina", "cues": ["husband", "wife"]}]


def _entry(text: str, narrator: str = "Mina Murray") -> Entry:
    return Entry(1, 8, "Journal", narrator, "1 Sept", "1897-09-01", "journal", text, 1)


def _found(text: str, spans: list[tuple[str, str, str]], narrator: str = "Mina Murray") -> list[dict]:
    """``spans`` are (surface, entity id or "", class); each surface is found in order."""
    mentions, at = [], 0
    for surface, eid, kind in spans:
        start = text.index(surface, at)
        at = start + len(surface)
        mentions.append(Mention(f"m{start}", 1, 8, 0, start, at, surface, "PER", kind, "test", eid or None))
    return find_stated_relations([_entry(text, narrator)], mentions, MARRIED, {"Mina Murray": "mina"})


def test_cue_next_to_both_names_is_kept():
    text = "Then Mina said she would sail with Jonathan, her husband, at dawn. "
    edges = _found(text, [("Mina", "mina", "PROP"), ("Jonathan", "jonathan", "PROP")])
    assert [(e["type"], e["source"], e["target"]) for e in edges] == [("married_to", "jonathan", "mina")]
    assert edges[0]["quote"].startswith("Then Mina")


def test_cue_far_from_one_name_is_dropped():
    text = ("Mina was tired after the long and dusty road from the coast and the inn, "
            "and Jonathan came in later with the innkeeper's husband. ")
    assert _found(text, [("Mina", "mina", "PROP"), ("Jonathan", "jonathan", "PROP")]) == []


def test_first_person_stands_for_the_narrator():
    text = "I must keep it to say to Jonathan, my husband. "
    edges = _found(text, [("I", "", "PRON"), ("Jonathan", "jonathan", "PROP"), ("my", "", "PRON")])
    assert len(edges) == 1


def test_first_person_in_dialogue_is_not_the_narrator():
    text = '"I must keep it to say to Jonathan, my husband," said Lucy. '
    edges = _found(text, [("I", "", "PRON"), ("Jonathan", "jonathan", "PROP"), ("my", "", "PRON")])
    assert edges == []


def test_section_naming_the_narrator_by_title_is_someone_else():
    text = "Miss Murray was out. I must keep it to say to Jonathan, my husband. "
    edges = _found(text, [("Miss Murray", "mina", "PROP"), ("I", "", "PRON"),
                          ("Jonathan", "jonathan", "PROP"), ("my", "", "PRON")])
    assert edges == []


def test_quote_parts_must_appear_in_order():
    import pytest
    from src.kg.relations import find_quoted_relations
    entries = [_entry("It was _very_ late. Then Mina said she would sail with Jonathan, her husband, at dawn. ")]
    row = {"rel": "married_to", "source": "jonathan", "target": "mina", "cues": []}
    edges = find_quoted_relations(entries, [dict(row, quote="It was very late. ... Jonathan, her husband")])
    assert edges[0]["method"] == "quoted" and edges[0]["entry_index"] == 1
    with pytest.raises(ValueError):
        find_quoted_relations(entries, [dict(row, quote="Jonathan, her husband ... It was very late.")])


def test_every_quote_in_relations_yml_is_in_the_book():
    from pathlib import Path
    from src.kg.build import documents
    from src.kg.config import load_config, load_yaml
    from src.kg.relations import find_quoted_relations
    root = Path(__file__).resolve().parents[1]
    rows = load_yaml(root / "data" / "gold" / "relations.yml")["relations"]
    edges = find_quoted_relations(documents(root, load_config(root / "configs" / "kg.yml")), rows)
    assert len(edges) == sum(1 for r in rows if r.get("quote"))
