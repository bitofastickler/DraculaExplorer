# -*- coding: utf-8 -*-
from src.kg.entries import Entry
from src.kg.mentions import Mention
from src.kg.resolve import parse_name, resolve_mentions

CFG = {
    "auto_merge_threshold": 0.85,
    "review_threshold": 0.60,
    "jaro_block": 0.92,
    "rare_token_max_surfaces": 30,
    "weights": {"jaro": 0.4, "surname": 0.3, "given": 0.2, "honorific": 0.1},
    "male_givens": ["jonathan", "arthur"],
    "female_givens": ["mina", "lucy"],
}


def _entry(text: str, idx: int = 1) -> Entry:
    return Entry(idx, 1, "Journal", "Jonathan Harker", "3 May", "1897-05-03", "journal", text, 1)


def _mention(surface, start, end, etype="PER", kind="PROP", entry=1, sent=0) -> Mention:
    return Mention(
        mention_id=f"m-{start}",
        entry_index=entry,
        chapter_number=1,
        sent_id=sent,
        start=start,
        end=end,
        surface=surface,
        entity_type=etype,
        mention_class=kind,
        extractor="test",
    )


def test_honorific_gender_blocks_a_merge():
    text = "Mrs. Westenra met Mr. Westenra there."
    mentions = [
        _mention("Mrs. Westenra", 0, 13),
        _mention("Mr. Westenra", 18, 30),
    ]
    result = resolve_mentions(mentions, [_entry(text)], CFG)
    assert result.mentions[0].entity_id != result.mentions[1].entity_id


def test_comma_apposition_links_holmwood_and_godalming():
    text = "Arthur Holmwood, Lord Godalming arrived."
    mentions = [
        _mention("Arthur Holmwood", 0, 15),
        _mention("Lord Godalming", 17, 31),
    ]
    result = resolve_mentions(mentions, [_entry(text)], CFG)
    assert result.mentions[0].entity_id == result.mentions[1].entity_id


def test_castle_and_count_stay_split():
    text = "Count Dracula left Castle Dracula."
    mentions = [
        _mention("Count Dracula", 0, 13, "PER"),
        _mention("Castle Dracula", 19, 33, "FAC"),
    ]
    result = resolve_mentions(mentions, [_entry(text)], CFG)
    assert result.mentions[0].entity_id != result.mentions[1].entity_id


def test_given_name_subset_links_jonathan():
    text = "Jonathan Harker signed. Jonathan waited."
    mentions = [
        _mention("Jonathan Harker", 0, 15),
        _mention("Jonathan", 24, 32),
    ]
    result = resolve_mentions(mentions, [_entry(text)], CFG)
    assert result.mentions[0].entity_id == result.mentions[1].entity_id


def test_false_appositive_does_not_merge_lucy_and_mrs_harker():
    text = "Lucy Westenra, Mrs. Harker arrived."
    mentions = [
        _mention("Lucy Westenra", 0, 13),
        _mention("Mrs. Harker", 15, 26),
    ]
    result = resolve_mentions(mentions, [_entry(text)], CFG)
    assert result.mentions[0].entity_id != result.mentions[1].entity_id


def test_married_title_plus_one_token_is_a_surname():
    mrs = parse_name("Mrs. Harker")
    assert mrs.honorific == "mrs"
    assert mrs.surname == "harker"
    assert mrs.given is None
    count = parse_name("Count Dracula")
    assert count.honorific == "count"
    assert count.surname is None
    assert count.given == "dracula"
    madam = parse_name("Madam Mina")
    assert madam.honorific == "madam"
    assert madam.surname is None
    assert madam.given == "mina"
    bare = parse_name("Harker")
    assert bare.surname is None
    assert bare.given == "harker"


def test_mr_and_mrs_harker_join_different_people():
    text = "Jonathan Harker and Mina Murray, later Mina Harker. Mr. Harker met Mrs. Harker."
    mentions = [
        _mention("Jonathan Harker", 0, 15),
        _mention("Mina Murray", 20, 31),
        _mention("Mina Harker", 39, 50),
        _mention("Mr. Harker", 52, 62),
        _mention("Mrs. Harker", 67, 78),
    ]
    result = resolve_mentions(mentions, [_entry(text)], CFG)
    ids = [m.entity_id for m in result.mentions]
    assert ids[0] == ids[3]
    assert ids[1] == ids[2] == ids[4]
    assert ids[0] != ids[1]


def test_surname_change_links_mina():
    text = "Mina Murray wrote. Later Mina Harker wrote."
    mentions = [
        _mention("Mina Murray", 0, 11),
        _mention("Mina Harker", 25, 36),
    ]
    result = resolve_mentions(mentions, [_entry(text)], CFG)
    assert result.mentions[0].entity_id == result.mentions[1].entity_id
