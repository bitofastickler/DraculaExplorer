# -*- coding: utf-8 -*-
"""Classical entity resolution over proper-name mentions.

Every automatic merge is one of: exact surface, token subset, comma
apposition, a same-given surname change, a knowledge must-link, or
average-linkage above the threshold. Honorific gender and distinct
given names block merges. Title nominals such as "the Count" attach
only when a single cluster in that chapter already carries the title.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from rapidfuzz.distance import JaroWinkler

from src.kg.entries import Entry
from src.kg.mentions import Mention

HONORIFICS = {
    "mr": "mr", "mister": "mr", "mrs": "mrs", "miss": "miss", "ms": "ms",
    "lord": "lord", "sir": "sir", "count": "count", "countess": "countess",
    "dr": "dr", "doctor": "doctor", "professor": "professor", "prof": "professor",
    "captain": "captain", "madam": "madam",
}
MALE_HONOR = {"mr", "lord", "sir", "count"}
FEMALE_HONOR = {"mrs", "miss", "ms", "countess", "madam"}
PARTICLES = {"van", "von", "de", "di", "da", "du"}
TITLE_NOMINALS = {
    "the count": "count",
    "the countess": "countess",
    "the professor": "professor",
    "the doctor": "doctor",
    "the captain": "captain",
}
MALE_PRON = {"he", "him", "his", "himself"}
FEMALE_PRON = {"she", "her", "hers", "herself"}

_TOKEN = re.compile(r"[A-Za-z0-9']+")


@dataclass
class NameParts:
    honorific: str | None
    tokens: tuple[str, ...]
    given: str | None
    surname: str | None

    @property
    def token_str(self) -> str:
        return " ".join(self.tokens)

    @property
    def gender(self) -> str | None:
        if self.honorific in MALE_HONOR:
            return "m"
        if self.honorific in FEMALE_HONOR:
            return "f"
        return None


@dataclass
class ResolveResult:
    mentions: list[Mention]
    review: list[dict] = field(default_factory=list)


def parse_name(surface: str) -> NameParts:
    cleaned = surface.replace(".", " ").replace("’", "'")
    raw = []
    for token in _TOKEN.findall(cleaned):
        token = token.lower()
        if token.endswith("'s"):
            token = token[:-2]
        if token:
            raw.append(token)
    honorific = None
    if raw and raw[0] in HONORIFICS:
        honorific = HONORIFICS[raw[0]]
        raw = raw[1:]
    if raw and raw[0] == "the" and len(raw) > 1 and raw[1] in HONORIFICS:
        honorific = HONORIFICS[raw[1]]
        raw = raw[2:]
    elif raw and raw[0] == "the":
        raw = raw[1:]
    given = None
    surname = None
    if len(raw) >= 2 and raw[-2] in PARTICLES:
        surname = " ".join(raw[-2:])
        given = raw[0] if len(raw) > 2 else None
    elif len(raw) >= 2:
        surname = raw[-1]
        given = raw[0]
    elif len(raw) == 1 and honorific in {"mr", "mrs", "miss", "ms"}:
        # "Mrs. Harker" is a married-name title plus a surname. The
        # married-name rule reads that surname. "Count Dracula" and
        # "Madam Mina" keep the single token as a given name, so a
        # pet name is not recorded as a surname.
        surname = raw[0]
    elif len(raw) == 1:
        given = raw[0]
    return NameParts(honorific, tuple(raw), given, surname)


def _gender_clash(a: NameParts, b: NameParts, male: set[str], female: set[str]) -> bool:
    def side(parts: NameParts) -> str | None:
        if parts.gender:
            return parts.gender
        if parts.given in male:
            return "m"
        if parts.given in female:
            return "f"
        return None

    ga, gb = side(a), side(b)
    return ga is not None and gb is not None and ga != gb


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _slug(surface: str) -> str:
    parts = parse_name(surface)
    text = parts.token_str or surface.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return slug or "entity"


def _pair_score(a: NameParts, b: NameParts, weights: dict) -> float:
    left = a.token_str or ""
    right = b.token_str or ""
    if not left or not right:
        jaro = 0.0
    else:
        jaro = JaroWinkler.normalized_similarity(left, right)
    if a.surname and b.surname:
        surname = 1.0 if a.surname == b.surname else 0.0
    else:
        surname = 0.5
    if a.given and b.given and (a.surname or b.surname or len(a.tokens) > 1 or len(b.tokens) > 1):
        # Single-token names store the token as given. Compare those via tokens.
        given = 1.0 if a.given == b.given else 0.0
    elif a.tokens and b.tokens and a.tokens[0] == b.tokens[0] and (len(a.tokens) > 1 or len(b.tokens) > 1):
        given = 1.0
    else:
        given = 0.5 if not (len(a.tokens) > 1 and len(b.tokens) > 1) else 0.0
    if a.honorific and b.honorific:
        honor = 1.0 if a.honorific == b.honorific else 0.0
    else:
        honor = 0.5
    return (
        weights.get("jaro", 0.4) * jaro
        + weights.get("surname", 0.3) * surname
        + weights.get("given", 0.2) * given
        + weights.get("honorific", 0.1) * honor
    )


def _tokens_subset(a: NameParts, b: NameParts) -> bool:
    if not a.tokens or not b.tokens:
        return False
    return set(a.tokens) < set(b.tokens) or set(b.tokens) < set(a.tokens)


def resolve_mentions(
    mentions: list[Mention],
    entries: list[Entry],
    config: dict | None = None,
    extra_must: list[tuple[str, str]] | None = None,
    extra_cannot: list[tuple[str, str]] | None = None,
) -> ResolveResult:
    config = config or {}
    weights = config.get("weights") or {}
    threshold = float(config.get("auto_merge_threshold", 0.85))
    review_at = float(config.get("review_threshold", 0.60))
    jaro_block = float(config.get("jaro_block", 0.92))
    rare_max = int(config.get("rare_token_max_surfaces", 30))
    male = set(config.get("male_givens") or [])
    female = set(config.get("female_givens") or [])

    text_of = {e.entry_index: e.text for e in entries}
    props = [m for m in mentions if m.mention_class == "PROP"]
    parts = [parse_name(m.surface) for m in props]
    diminutives = {k.lower(): v.lower() for k, v in (config.get("diminutives") or {}).items()}
    for part in parts:
        if part.given in diminutives:
            part.given = diminutives[part.given]
    n = len(props)
    index = {m.mention_id: i for i, m in enumerate(props)}

    # Surname -> given names attested on full (given + surname) mentions.
    givens_for: dict[str, set[str]] = defaultdict(set)
    honor_for_surname: dict[str, set[str]] = defaultdict(set)
    for part in parts:
        if part.surname and part.given and part.given != part.surname and len(part.tokens) >= 2:
            givens_for[part.surname].add(part.given)
        if part.surname and part.honorific:
            honor_for_surname[part.surname].add(part.honorific)
    surnames_for: dict[str, set[str]] = defaultdict(set)
    for surname, givens in givens_for.items():
        for given in givens:
            surnames_for[given].add(surname)

    surface_forms: dict[str, set[str]] = defaultdict(set)
    for part in parts:
        for tok in part.tokens:
            surface_forms[tok].add(part.token_str)
    rare = {tok for tok, forms in surface_forms.items() if 0 < len(forms) < rare_max and tok not in {"the", "a"}}

    def ambiguous_surname(surname: str | None) -> bool:
        return bool(surname) and len(givens_for.get(surname, ())) >= 2

    def cannot(i: int, j: int) -> bool:
        a, b = parts[i], parts[j]
        if props[i].entity_type != props[j].entity_type:
            return True
        if _gender_clash(a, b, male, female):
            return True
        # Two full names with different given names are different people,
        # including a shared married surname (Jonathan Harker / Mina Harker).
        if (
            a.given and b.given and a.surname and b.surname
            and len(a.tokens) >= 2 and len(b.tokens) >= 2
            and a.given != b.given
        ):
            return True
        # Miss Westenra is not Mrs Westenra. A married-name honorific may
        # still join a cluster that acquired this surname via a same-given
        # surname change (Miss Murray -> Mrs Harker).
        if a.surname and a.surname == b.surname and a.honorific and b.honorific and a.honorific != b.honorific:
            if {a.honorific, b.honorific} <= {"miss", "mrs", "mr"}:
                return True
        return False

    def married_name_ok(short: NameParts, full: NameParts) -> bool:
        """Whether a bare honorific plus surname may join a full name.

        ``Mrs`` joins only a given name already attested with two surnames
        (Mina Murray / Mina Harker). That keeps Mrs Westenra off Lucy.
        ``Mr`` and ``Miss`` join the single gender-compatible given name
        that uses this surname.
        """
        if short.honorific not in {"mr", "mrs", "miss"}:
            return False
        if not short.surname or full.surname != short.surname or not full.given:
            return False
        if short.honorific in FEMALE_HONOR and full.given in male:
            return False
        if short.honorific in MALE_HONOR and full.given in female:
            return False
        changed = (
            len(surnames_for.get(full.given, ())) >= 2
            and short.surname in surnames_for[full.given]
        )
        if short.honorific == "mrs":
            return changed
        pool = givens_for.get(short.surname, ())
        if short.honorific == "mr":
            compatible = [g for g in pool if g not in female]
        else:
            compatible = [g for g in pool if g not in male]
        return len(compatible) == 1 and full.given in compatible

    uf = _UF(n)
    must_reason: dict[tuple[int, int], str] = {}

    def link(i: int, j: int, reason: str) -> None:
        if i == j or cannot(i, j):
            return
        a, b = (i, j) if i < j else (j, i)
        must_reason[(a, b)] = reason
        uf.union(i, j)

    # Exact token strings, honorific ignored, so "Dr Seward" meets "Seward"
    # only when the tokens match. Honorific-only differences still match.
    by_tokens: dict[tuple, list[int]] = defaultdict(list)
    for i, part in enumerate(parts):
        if part.tokens:
            by_tokens[part.tokens].append(i)
    for group in by_tokens.values():
        # Same tokens and the same honorific always merge. A bare surname
        # merges with an honorific form only when that surname belongs to
        # one given name. Otherwise "Harker" would glue Mrs. Harker to Mr. Harker.
        for left_at, i in enumerate(group):
            for j in group[left_at + 1:]:
                if parts[i].honorific != parts[j].honorific:
                    token = parts[i].surname or (parts[i].tokens[0] if len(parts[i].tokens) == 1 else None)
                    if ambiguous_surname(token):
                        continue
                link(i, j, "exact")

    # Token subset: "Jonathan" inside "Jonathan Harker", "Van Helsing"
    # inside "Abraham Van Helsing". A bare surname shared by two given
    # names is not a subset link.
    for i in range(n):
        for j in range(i + 1, n):
            a, b = parts[i], parts[j]
            if props[i].entity_type != props[j].entity_type:
                continue
            if not _tokens_subset(a, b):
                continue
            short, full = (a, b) if len(a.tokens) < len(b.tokens) else (b, a)
            if (
                len(short.tokens) == 1
                and ambiguous_surname(short.tokens[0])
                and short.tokens[0] == (short.surname or short.given)
                and not married_name_ok(short, full)
            ):
                continue
            if (
                short.honorific in {"mr", "mrs", "miss"}
                and full.given
                and short.surname
                and short.surname == full.surname
                and not _has_given(short)
            ):
                if not married_name_ok(short, full):
                    continue
            if _gender_clash(short, full, male, female):
                continue
            link(i, j, "subset")

    # Comma apposition: "Arthur Holmwood, Lord Godalming".
    by_sent: dict[tuple, list[int]] = defaultdict(list)
    for i, mention in enumerate(props):
        by_sent[(mention.entry_index, mention.sent_id)].append(i)
    for group in by_sent.values():
        ordered = sorted(group, key=lambda i: props[i].start)
        for left_i, right_i in zip(ordered, ordered[1:]):
            left, right = props[left_i], props[right_i]
            gap = text_of.get(left.entry_index, "")[left.end:right.start]
            if gap.strip() not in {",", ", and"}:
                continue
            lp, rp = parts[left_i], parts[right_i]
            honor_diff = bool(lp.honorific) != bool(rp.honorific)
            # "Mrs. Harker" next to another surname is a different person.
            # Rank titles (Lord Godalming) are the appositions we want.
            if {lp.honorific, rp.honorific} & {"mrs", "mr", "miss"} and lp.surname != rp.surname:
                continue
            if honor_diff or _tokens_subset(lp, rp):
                link(left_i, right_i, "apposition")

    # Same given name, two attested surnames: Mina Murray / Mina Harker.
    for given, surnames in surnames_for.items():
        if len(surnames) != 2:
            continue
        members = [
            i for i, part in enumerate(parts)
            if part.given == given and part.surname in surnames and len(part.tokens) >= 2
        ]
        for i in members[1:]:
            link(members[0], i, "surname_change")

    def surfaces_match(part: NameParts, needle: str) -> bool:
        needle = needle.lower().strip()
        return part.token_str == needle or (part.honorific and f"{part.honorific} {part.token_str}" == needle)

    for a_s, b_s in extra_must or []:
        left = [i for i, part in enumerate(parts) if surfaces_match(part, a_s)]
        right = [i for i, part in enumerate(parts) if surfaces_match(part, b_s)]
        for i in left:
            for j in right:
                link(i, j, "gazetteer_must")

    blocked_cannot: set[tuple[int, int]] = set()
    for a_s, b_s in extra_cannot or []:
        left = [i for i, part in enumerate(parts) if surfaces_match(part, a_s)]
        right = [i for i, part in enumerate(parts) if surfaces_match(part, b_s)]
        for i in left:
            for j in right:
                if i == j:
                    continue
                pair = (i, j) if i < j else (j, i)
                blocked_cannot.add(pair)

    def blocked(i: int, j: int) -> bool:
        a, b = parts[i], parts[j]
        if a.surname and a.surname == b.surname:
            return True
        if a.token_str and b.token_str and JaroWinkler.normalized_similarity(a.token_str, b.token_str) >= jaro_block:
            return True
        shared = set(a.tokens) & set(b.tokens) & rare
        return bool(shared)

    # Average-linkage over must-link components.
    members: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        members[uf.find(i)].append(i)

    def component_cannot(ca: list[int], cb: list[int]) -> bool:
        for i in ca:
            for j in cb:
                pair = (i, j) if i < j else (j, i)
                if pair in blocked_cannot or cannot(i, j):
                    return True
        return False

    def component_score(ca: list[int], cb: list[int]) -> tuple[float, bool]:
        scores = []
        support = False
        for i in ca:
            for j in cb:
                if not blocked(i, j):
                    scores.append(0.0)
                    continue
                score = _pair_score(parts[i], parts[j], weights)
                scores.append(score)
                a, b = parts[i], parts[j]
                if (a.surname and a.surname == b.surname) or (set(a.tokens) & set(b.tokens) & rare):
                    support = True
        if not scores:
            return 0.0, False
        return sum(scores) / len(scores), support

    review: list[dict] = []
    guard = 0
    while len(members) > 1 and guard < n * n:
        guard += 1
        roots = list(members)
        best = None
        best_key = None
        for ai, ra in enumerate(roots):
            for rb in roots[ai + 1:]:
                ca, cb = members[ra], members[rb]
                if component_cannot(ca, cb):
                    continue
                score, support = component_score(ca, cb)
                if support and score >= threshold and (best is None or score > best):
                    best = score
                    best_key = (ra, rb)
        if best_key is None:
            break
        ra, rb = best_key
        members[ra].extend(members.pop(rb))

    # Review queue: blocked cross-cluster pairs in the band, capped.
    roots = list(members)
    seen_review = set()
    for ai, ra in enumerate(roots):
        for rb in roots[ai + 1:]:
            ca, cb = members[ra], members[rb]
            if component_cannot(ca, cb):
                continue
            score, _support = component_score(ca, cb)
            if review_at <= score < threshold:
                i, j = ca[0], cb[0]
                key = (props[i].surface.lower(), props[j].surface.lower())
                if key in seen_review:
                    continue
                seen_review.add(key)
                review.append({
                    "surface_a": props[i].surface,
                    "surface_b": props[j].surface,
                    "entry_a": props[i].entry_index,
                    "entry_b": props[j].entry_index,
                    "score": round(score, 4),
                })
                if len(review) >= 200:
                    break
        if len(review) >= 200:
            break

    # Provisional ids from the most frequent surface in the component.
    prov_of: dict[int, str] = {}
    used = set()
    for root, group in members.items():
        counts: dict[str, int] = defaultdict(int)
        for i in group:
            counts[props[i].surface] += 1
        surface = max(counts, key=lambda s: (counts[s], len(s)))
        slug = _slug(surface)
        base = slug
        k = 2
        while slug in used:
            slug = f"{base}_{k}"
            k += 1
        used.add(slug)
        prov_of[root] = slug
        for i in group:
            mention = props[i]
            mention.entity_id = slug
            pair_reasons = [reason for (a, b), reason in must_reason.items() if uf.find(a) == root and (i in (a, b))]
            mention.link_method = pair_reasons[0] if pair_reasons else "singleton"
            mention.link_score = 1.0

    by_id: dict[str, list[int]] = defaultdict(list)
    for i, mention in enumerate(props):
        if mention.entity_id:
            by_id[mention.entity_id].append(i)

    # Title nominals: "the Count" attaches only to one cluster in the chapter.
    for mention in mentions:
        if mention.mention_class != "NOM" or mention.entity_type != "PER":
            continue
        key = re.sub(r"\s+", " ", mention.surface.lower()).strip()
        key = re.sub(r"'s$", "", key)
        honor = TITLE_NOMINALS.get(key)
        if honor is None:
            continue
        candidates = []
        for eid, group in by_id.items():
            if props[group[0]].entity_type != "PER":
                continue
            if not any(props[i].chapter_number == mention.chapter_number for i in group):
                continue
            if any(parts[i].honorific == honor for i in group):
                candidates.append(eid)
        candidates = sorted(set(candidates))
        if len(candidates) == 1:
            mention.entity_id = candidates[0]
            mention.link_method = "nominal"
            mention.link_score = 1.0

    # Pronouns attach inside a sentence that already has one compatible person.
    linked_by_sent: dict[tuple, list[Mention]] = defaultdict(list)
    for mention in mentions:
        if mention.entity_id and mention.mention_class in {"PROP", "NOM"} and mention.entity_type == "PER":
            linked_by_sent[(mention.entry_index, mention.sent_id)].append(mention)

    cluster_gender: dict[str, str | None] = {}
    for eid, group in by_id.items():
        genders = set()
        for i in group:
            part = parts[i]
            if part.gender:
                genders.add(part.gender)
            elif part.given in male:
                genders.add("m")
            elif part.given in female:
                genders.add("f")
        cluster_gender[eid] = genders.pop() if len(genders) == 1 else None

    for mention in mentions:
        if mention.mention_class != "PRON" or mention.entity_id:
            continue
        nearby = linked_by_sent.get((mention.entry_index, mention.sent_id), [])
        ids = sorted({m.entity_id for m in nearby if m.entity_id})
        if len(ids) != 1:
            continue
        pronoun = mention.surface.lower()
        want = "m" if pronoun in MALE_PRON else "f" if pronoun in FEMALE_PRON else None
        have = cluster_gender.get(ids[0])
        if want and have and want != have:
            continue
        mention.entity_id = ids[0]
        mention.link_method = "same_sentence_pronoun"
        mention.link_score = 1.0

    return ResolveResult(mentions, review)


def _has_given(part: NameParts) -> bool:
    """True when the name has a given name distinct from its only token."""
    return bool(part.given and part.surname and len(part.tokens) >= 2)


def apply_gold_labels(mentions: list[Mention], gold_entities: list[dict]) -> None:
    """Rename a cluster to a gold slug when its mentions match one gold entity.

    Pairwise scores use cluster membership, not these slugs. The slug is
    the stable id queries and narrator edges use.
    """
    prepared = []
    for entity in gold_entities:
        aliases = {a.lower().strip() for a in entity.get("aliases") or []}
        prepared.append((entity["id"], entity.get("entity_type"), aliases, entity.get("canonical_name")))

    def matches(mention: Mention, aliases: set[str], etype: str | None) -> bool:
        if etype and mention.entity_type != etype:
            return False
        part = parse_name(mention.surface)
        forms = {part.token_str, f"{part.honorific} {part.token_str}".strip() if part.honorific else part.token_str}
        forms.add(re.sub(r"\s+", " ", mention.surface.lower()).strip())
        return bool(forms & aliases)

    by_cluster: dict[str, list[Mention]] = defaultdict(list)
    for mention in mentions:
        if mention.entity_id and mention.mention_class == "PROP":
            by_cluster[mention.entity_id].append(mention)

    rename: dict[str, str] = {}
    claimed: dict[str, tuple[int, str]] = {}
    for cid, group in by_cluster.items():
        votes: dict[str, int] = defaultdict(int)
        for mention in group:
            hits = [eid for eid, etype, aliases, _name in prepared if matches(mention, aliases, etype)]
            if len(hits) == 1:
                votes[hits[0]] += 1
        if not votes:
            continue
        winner = max(votes, key=lambda k: votes[k])
        if votes[winner] == 0:
            continue
        prev = claimed.get(winner)
        if prev is None or votes[winner] > prev[0]:
            if prev is not None:
                rename.pop(prev[1], None)
            claimed[winner] = (votes[winner], cid)
            rename[cid] = winner

    if not rename:
        return
    for mention in mentions:
        if mention.entity_id in rename:
            mention.entity_id = rename[mention.entity_id]
