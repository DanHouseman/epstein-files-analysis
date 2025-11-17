#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser as dateparser
from pypdf import PdfReader

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ------------------------- optional spaCy NER -------------------------

try:
    import spacy

    _NLP = None
    for model_name in ("en_core_web_trf", "en_core_web_md", "en_core_web_sm"):
        try:
            _NLP = spacy.load(model_name)
            print(f"[+] Loaded spaCy model: {model_name}")
            break
        except Exception:
            continue
    if _NLP is None:
        print("[!] spaCy installed but no English model found; using regex fallback only.")
except Exception:
    _NLP = None
    print("[!] spaCy not available; using regex fallback only.")

# ---------------------- optional sklearn topics -----------------------

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF

    _SKLEARN_AVAILABLE = True
    print("[+] sklearn available for topic modelling.")
except Exception:
    _SKLEARN_AVAILABLE = False
    print("[!] sklearn not available; topics will be empty.")

# --------------------------- regex & config ---------------------------

DATE_REGEX = re.compile(
    r"\b("
    r"\d{1,2}/\d{1,2}/\d{2,4}"
    r"|"
    r"[A-Z][a-z]{2,8}\s+\d{1,2},\s+\d{4}"
    r")\b"
)

HEADER_LINE_RE = re.compile(r"^\s*(From|To|Cc|Bcc|Subject)\s*:\s*(.*)$", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")

BASE_STOPWORDS = {
    "from", "sent", "subject", "re", "fw", "fwd",
    "http", "https", "www", "com",
    "gmail", "email", "message",
    "house", "oversight", "copyright",
    "please", "note", "information", "contained", "including",
    "attachments", "intended", "addressee", "property",
    "unlawful", "immediately", "destroy", "reserved", "rights",
    "july", "june", "august", "september", "october", "november",
    "december", "february", "january", "march", "april", "may",
}

# --------------------------- data classes ----------------------------

@dataclass
class Document:
    id: int
    path: str
    dates: List[str]
    subject: Optional[str]
    participants: List[str]
    token_count: int
    unique_token_count: int
    top_words: List[Dict[str, Any]]
    topic_ids: List[int]
    topic_labels: List[str]


@dataclass
class EntityRecord:
    id: int
    name: str
    label: str
    mentions: List[int]
    timeseries: List[Dict[str, Any]]


@dataclass
class Mention:
    id: int
    entity_id: int
    entity_name: str
    doc_id: int
    file_path: str
    date: Optional[str]
    sentence: str
    char_start: int
    char_end: int
    topic_id: Optional[int]
    topic_label: Optional[str]


# ------------------------------ IO -----------------------------------

def walk_files(root: Path) -> List[Path]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in (".txt", ".pdf"):
                out.append(p)
    out.sort()
    return out


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        text_chunks = []
        try:
            reader = PdfReader(str(path))
            for page in reader.pages:
                try:
                    text_chunks.append(page.extract_text() or "")
                except Exception:
                    continue
        except Exception as e:
            print(f"[!] Failed to read PDF {path}: {e}")
        return "\n".join(text_chunks)
    else:
        return path.read_text(encoding="utf-8", errors="ignore")


# --------------------------- NLP helpers -----------------------------

def extract_dates(text: str) -> List[str]:
    candidates = DATE_REGEX.findall(text)
    out = []
    for raw in candidates:
        try:
            dt = dateparser.parse(raw, fuzzy=True, dayfirst=False)
            if dt:
                out.append(dt.date().isoformat())
        except Exception:
            pass
    return sorted(set(out))


def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def extract_subject_participants(text: str) -> Tuple[Optional[str], List[str]]:
    subject = None
    participants = set()
    for line in text.splitlines():
        m = HEADER_LINE_RE.match(line)
        if not m:
            continue
        header, val = m.group(1).lower(), m.group(2)
        if header == "subject" and subject is None:
            subject = val.strip()
        if header in ("from", "to", "cc", "bcc"):
            for e in EMAIL_RE.findall(val):
                participants.add(e.strip())
    return subject, sorted(participants)


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\?\!])\s+", text)
    return [s.strip() for s in parts if s.strip()]


def ner_with_spacy(text: str):
    """Return list of (entity_text, label, start_char, end_char, sentence)."""
    doc = _NLP(text)
    ents = []
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "LOC"):
            ents.append(
                (
                    ent.text.strip(),
                    ent.label_,
                    int(ent.start_char),
                    int(ent.end_char),
                    ent.sent.text.strip(),
                )
            )
    return ents


def ner_fallback(text: str):
    ents = []
    # emails as entities
    for m in EMAIL_RE.finditer(text):
        ents.append(
            (
                m.group(),
                "EMAIL",
                m.start(),
                m.end(),
                text[max(0, m.start() - 80) : m.end() + 80].strip(),
            )
        )
    # capitalized phrases as rough names
    for sent in split_sentences(text):
        idx = 0
        while idx < len(sent):
            m = re.search(
                r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\b", sent[idx:]
            )
            if not m:
                break
            start = idx + m.start()
            end = idx + m.end()
            phrase = m.group(1)
            ents.append((phrase, "CAP", start, end, sent))
            idx = end
    return ents


def extract_entities(text: str):
    if _NLP is not None:
        return ner_with_spacy(text)
    return ner_fallback(text)


# --------------------- entity canonicalization & clustering ---------------------

def normalize_person_name(name: str) -> str:
    """Lowercase, strip punctuation, and keep last token as 'last name'."""
    name = re.sub(r"[^\w\s]", "", name.lower()).strip()
    parts = name.split()
    if not parts:
        return name
    return parts[-1]


# Hand-crafted alias groups for major PERSON / ORG-like entities.
# These are applied at mention time; comparisons are on lowercased, cleaned form.
ALIAS_GROUPS = {
    # Epstein
    "jeffrey epstein": {
        "jeffrey epstein",
        "jeff epstein",
        "jeffrey e",
        "jeff e",
        "j epstein",
        "j. epstein",
        "epstein",
    },
    # Trump
    "donald trump": {
        "donald trump",
        "donald j trump",
        "donald j. trump",
        "d trump",
        "d. trump",
        "trump",
    },
    # Clinton
    "bill clinton": {
        "bill clinton",
        "william clinton",
        "w j clinton",
        "w. j. clinton",
        "w clinton",
        "w. clinton",
        "bubba"
    },
    # CIA
    "cia": {
        "cia",
        "c i a",
        "c. i. a",
        "c.i.a",
        "c.i.a.",
    },
}

# Third-pass synonym groups for ORG/GPE/LOC-like entities
# We map cleaned variants -> canonical key.
def _clean_orgloc(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    s = " ".join(s.split())
    # strip leading "the"
    if s.startswith("the "):
        s = s[4:]
    return s

_ORG_LOC_SYNONYM_RAW = {
    "department of justice": [
        "doj",
        "d. o. j.",
        "d.o.j",
        "d.o.j.",
        "department of justice",
        "the department of justice",
        "u.s. department of justice",
        "us department of justice",
        "united states department of justice",
    ],
    "white house": [
        "white house",
        "the white house",
        "whitehouse",
    ],
    "new york times": [
        "new york times",
        "the new york times",
        "ny times",
        "nyt",
        "the times",
    ],
    "florida": [
        "florida",
        "state of florida",
        "the state of florida",
    ],
    "mercedes benz fashion week": [
        "mercedes benz fashion week",
        "mercedes-benz fashion week",
        "nyfw",
        "mbfw",
        "fashion week",
    ],
}

ORG_LOC_SYNONYM_LOOKUP: Dict[str, str] = {}
for canonical, variants in _ORG_LOC_SYNONYM_RAW.items():
    ckey = _clean_orgloc(canonical)
    for v in variants:
        ORG_LOC_SYNONYM_LOOKUP[_clean_orgloc(v)] = ckey


def canonical_entity_name(raw_name: str, label: str) -> str:
    """
    Compute a canonical key for an entity:
    - aggressively strip underscores (signature lines etc.)
    - case-insensitive, punctuation-stripped
    - for PERSON: keep full name, do NOT collapse to last name
    - for known alias groups (Epstein/Trump/CIA etc.), map to a shared canonical
    """
    if not raw_name:
        return raw_name

    # 1) Strip signature-line / formatting underscores anywhere in the span
    #    "Ruemmler_________________" -> "Ruemmler"
    #    "Wolf____ Jr"             -> "Wolf Jr"
    raw_name = re.sub(r"_+", " ", raw_name)
    raw_name = raw_name.strip()

    # 2) Lowercase, drop punctuation
    cleaned = re.sub(r"[^\w\s]", " ", raw_name).lower()
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return cleaned

    # 3) Hand-crafted alias groups (Epstein, Trump, CIA, etc.)
    #    Note: ALIAS_GROUPS should contain lowercase, space-normalized variants.
    for canonical, variants in ALIAS_GROUPS.items():
        if cleaned in variants:
            return canonical

    # 4) For PERSON: keep the full cleaned name.
    #    We no longer collapse to last name here; any last-name mapping is done
    #    later when we know if that last name is unique.
    #    For non-PERSON, we also just keep the full cleaned string.
    return cleaned


def is_similar_name(name1: str, label1: str, name2: str, label2: str) -> bool:
    """
    Conservative similarity:
    - same label
    - same canonical cleaned form (after alias groups)
    We *don't* do substring / fuzzy matching here; last-name vs full-name
    mapping is handled in a dedicated pass where we know whether the last name
    is unique.
    """
    if label1 != label2:
        return False

    c1 = canonical_entity_name(name1, label1)
    c2 = canonical_entity_name(name2, label2)
    return bool(c1 and c2 and c1 == c2)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1


# ------------------------------- core --------------------------------

def analyze(root: Path, out_path: Path):
    files = walk_files(root)
    print(f"[+] Found {len(files)} files under {root}")

    documents: List[Document] = []
    all_mentions: List[Mention] = []

    entities_map: Dict[str, int] = {}              # canonical_name -> entity id
    entities_meta: Dict[int, Dict[str, Any]] = {}  # eid -> meta
    global_counts: Counter = Counter()

    doc_texts_for_topics: List[str] = []
    doc_id_to_idx: Dict[int, int] = {}

    mention_id = 0
    next_entity_id = 0

    # --------------------- first pass: docs & mentions ---------------------

    if tqdm is not None:
        iterator = enumerate(tqdm(files, desc="Analyzing files", unit="file"))
    else:
        iterator = enumerate(files)

    for doc_id, path in iterator:
        text = read_text(path)
        if not text.strip():
            continue

        dates = extract_dates(text)
        primary_date = dates[0] if dates else None
        subject, participants = extract_subject_participants(text)
        tokens = tokenize(text)
        token_count = len(tokens)
        unique_token_count = len(set(tokens))

        # global word counts
        filtered_tokens = [t for t in tokens if t not in BASE_STOPWORDS]
        for t in filtered_tokens:
            global_counts[t] += 1

        word_counts = Counter(filtered_tokens)
        doc_top_words = [
            {"word": w, "count": int(c)} for w, c in word_counts.most_common(50)
        ]

        # text for topic modelling
        doc_text_for_topic = " ".join(filtered_tokens)
        doc_texts_for_topics.append(doc_text_for_topic)
        doc_id_to_idx[doc_id] = len(doc_texts_for_topics) - 1

        documents.append(
            Document(
                id=doc_id,
                path=str(path),
                dates=dates,
                subject=subject,
                participants=participants,
                token_count=token_count,
                unique_token_count=unique_token_count,
                top_words=doc_top_words,
                topic_ids=[],
                topic_labels=[],
            )
        )

        # NER
        ents_raw = extract_entities(text)

        for ent_text, label, start_char, end_char, sentence in ents_raw:
            raw_name = ent_text.strip()
            if not raw_name:
                continue

            name_key = canonical_entity_name(raw_name, label)
            if not name_key:
                continue

            if name_key not in entities_map:
                eid = next_entity_id
                next_entity_id += 1
                entities_map[name_key] = eid
                entities_meta[eid] = {
                    "id": eid,
                    "name": name_key,
                    "label": label,
                    "mentions": [],
                    "aliases": set([raw_name]),
                }
            else:
                eid = entities_map[name_key]
                entities_meta[eid].setdefault("aliases", set()).add(raw_name)

            m = Mention(
                id=mention_id,
                entity_id=eid,
                entity_name=name_key,
                doc_id=doc_id,
                file_path=str(path),
                date=primary_date,
                sentence=sentence,
                char_start=start_char,
                char_end=end_char,
                topic_id=None,
                topic_label=None,
            )
            all_mentions.append(m)
            entities_meta[eid]["mentions"].append(mention_id)
            mention_id += 1

    # -------------------------- topic modelling ----------------------------

    topics_summary: List[Dict[str, Any]] = []

    if _SKLEARN_AVAILABLE and doc_texts_for_topics:
        n_docs = len(doc_texts_for_topics)
        # heuristic: up to 12 topics, at least 2, around n_docs/10
        n_topics = min(12, max(2, n_docs // 10 or 2))

        print(f"[+] Topic modelling with NMF: {n_topics} topics on {n_docs} docs.")

        vectorizer = TfidfVectorizer(
            max_df=0.95,
            min_df=2,
            max_features=5000,
            stop_words="english",
        )
        tfidf = vectorizer.fit_transform(doc_texts_for_topics)
        nmf = NMF(
            n_components=n_topics,
            random_state=0,
            init="nndsvda",
        )
        W = nmf.fit_transform(tfidf)
        H = nmf.components_
        feature_names = vectorizer.get_feature_names_out()

        # topics with top words
        for topic_idx, topic_vec in enumerate(H):
            top_indices = topic_vec.argsort()[-8:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            label = ", ".join(top_words[:4])
            topics_summary.append(
                {"id": topic_idx, "label": label, "top_words": top_words}
            )

        # per-document dominant topics (top 2)
        for doc in documents:
            idx = doc_id_to_idx.get(doc.id)
            if idx is None:
                continue
            dist = W[idx]
            topic_ids = dist.argsort()[::-1][:2].tolist()
            topic_labels = [topics_summary[i]["label"] for i in topic_ids]
            doc.topic_ids = topic_ids
            doc.topic_labels = topic_labels

        # per-mention topic = doc's primary topic
        doc_to_topics = {d.id: d.topic_ids for d in documents}
        for m in all_mentions:
            tids = doc_to_topics.get(m.doc_id) or []
            if tids:
                m.topic_id = tids[0]
                m.topic_label = topics_summary[tids[0]]["label"]
    else:
        print("[!] Skipping topic modelling; sklearn unavailable or no docs.")

    # ----------------------- entity clustering pass -------------------------

    num_entities = len(entities_meta)
    uf = UnionFind(num_entities)

    # --- Last-name → full-name mapping for PERSON entities (global) ---

    # We want:
    # - "peter thiel" and "thiel" -> same entity, IF there is only one full-name
    #   entity with last name "thiel" in the entire corpus.
    # - If there are multiple "thiel" full names, we *do not* force them together.
    # - We *do not* try to merge first-name-only mentions ("mike") because
    #   they're ambiguous by design.

    last_name_to_full_eids: Dict[str, set] = defaultdict(set)
    single_last_eids: Dict[str, set] = defaultdict(set)

    for eid, meta in entities_meta.items():
        if meta["label"] != "PERSON":
            continue
        name = meta["name"]  # canonical cleaned name
        tokens = name.split()
        if not tokens:
            continue
        last = tokens[-1]
        if len(tokens) >= 2:
            # full name with a last token: e.g., "peter thiel" -> last="thiel"
            last_name_to_full_eids[last].add(eid)
        else:
            # single-token PERSON name; treat as bare last name candidate
            single_last_eids[last].add(eid)

    # For each last name, if there is exactly ONE full-name entity, merge all
    # "bare last name" entities into that one. This is where "Thiel" gets
    # mapped to "Peter Thiel" when Peter is unique in the corpus.
    for last, single_eid_set in single_last_eids.items():
        full_eids = last_name_to_full_eids.get(last, set())
        if len(full_eids) == 1:
            full_eid = next(iter(full_eids))
            for eid in single_eid_set:
                uf.union(eid, full_eid)

    # doc -> entity_ids for that doc
    doc_to_entity_ids: Dict[int, List[int]] = defaultdict(list)
    for m in all_mentions:
        doc_to_entity_ids[m.doc_id].append(m.entity_id)

    for doc_id, eids in doc_to_entity_ids.items():
        uniq_eids = list(set(eids))
        for i in range(len(uniq_eids)):
            for j in range(i + 1, len(uniq_eids)):
                eid1, eid2 = uniq_eids[i], uniq_eids[j]
                meta1 = entities_meta[eid1]
                meta2 = entities_meta[eid2]
                if is_similar_name(
                    meta1["name"], meta1["label"], meta2["name"], meta2["label"]
                ):
                    uf.union(eid1, eid2)

    # old eid -> new cluster id (0..K-1) after union-find (second pass)
    root_to_newid: Dict[int, int] = {}
    eid_to_new: Dict[int, int] = {}
    merged_entities_meta: Dict[int, Dict[str, Any]] = {}
    next_new_eid = 0

    for old_eid, meta in entities_meta.items():
        root = uf.find(old_eid)
        if root not in root_to_newid:
            root_to_newid[root] = next_new_eid
            merged_entities_meta[next_new_eid] = {
                "id": next_new_eid,
                "names": [meta["name"]],
                "label": meta["label"],
                "mentions": list(meta["mentions"]),
            }
            next_new_eid += 1
        else:
            nid = root_to_newid[root]
            merged_entities_meta[nid]["names"].append(meta["name"])
            merged_entities_meta[nid]["mentions"].extend(meta["mentions"])
        eid_to_new[old_eid] = root_to_newid[root]

    # canonical name per merged cluster = longest variant
    for nid, meta in merged_entities_meta.items():
        names = list(set(meta["names"]))
        names.sort(key=len, reverse=True)
        canonical = names[0]
        meta["name"] = canonical

    # update mentions with second-pass merged entity ids and names
    for m in all_mentions:
        new_id = eid_to_new.get(m.entity_id, m.entity_id)
        m.entity_id = new_id
        meta = merged_entities_meta[new_id]
        m.entity_name = meta["name"]

    # ------------------------- third-pass synonyms -------------------------

    # Now merge ORG/GPE/LOC-style entities that match our synonym table.
    final_id_map: Dict[int, int] = {}
    for nid in merged_entities_meta.keys():
        final_id_map[nid] = nid

    # Group entity cluster IDs by synonym canonical key
    syn_groups: Dict[str, List[int]] = defaultdict(list)
    for nid, meta in merged_entities_meta.items():
        label = meta["label"]
        if label not in ("ORG", "GPE", "LOC", "FAC"):
            continue
        cleaned_name = _clean_orgloc(meta["name"])
        canonical = ORG_LOC_SYNONYM_LOOKUP.get(cleaned_name)
        if canonical:
            syn_groups[canonical].append(nid)

    # For each synonym group, choose smallest id as representative
    for canonical, ids in syn_groups.items():
        if len(ids) < 2:
            continue
        rep = min(ids)
        for nid in ids:
            final_id_map[nid] = rep

    # Build final entity meta from final_id_map
    final_entities_meta: Dict[int, Dict[str, Any]] = {}
    for nid, meta in merged_entities_meta.items():
        rep = final_id_map[nid]
        if rep not in final_entities_meta:
            final_entities_meta[rep] = {
                "id": rep,
                "names": [],
                "label": meta["label"],
                "mentions": [],
            }
        final_entities_meta[rep]["names"].append(meta["name"])
        final_entities_meta[rep]["mentions"].extend(meta["mentions"])

    # Final canonical name per representative
    for rep, meta in final_entities_meta.items():
        names = list(set(meta["names"]))
        names.sort(key=len, reverse=True)
        meta["name"] = names[0]

    # Update mentions one more time with third-pass mapping
    for m in all_mentions:
        new_id = final_id_map.get(m.entity_id, m.entity_id)
        m.entity_id = new_id
        meta = final_entities_meta[new_id]
        m.entity_name = meta["name"]

    # ------------------ time-series per entity after merge ------------------

    entity_date_counts: Dict[int, Counter] = defaultdict(Counter)
    for m in all_mentions:
        if m.date:
            entity_date_counts[m.entity_id][m.date] += 1

    entities: List[EntityRecord] = []
    for nid, meta in final_entities_meta.items():
        counts = entity_date_counts[nid]
        ts = [{"date": d, "count": int(c)} for d, c in sorted(counts.items())]
        entities.append(
            EntityRecord(
                id=nid,
                name=meta["name"],
                label=meta["label"],
                mentions=sorted(set(meta["mentions"])),
                timeseries=ts,
            )
        )

    # ----------------------- graph from co-occurrence -----------------------

    graph_edges: Dict[Tuple[int, int], Dict[str, Any]] = {}
    sent_map: Dict[Tuple[int, str], List[Mention]] = defaultdict(list)

    for m in all_mentions:
        sent_key = (m.doc_id, m.sentence)
        sent_map[sent_key].append(m)

    # build quick doc lookup
    doc_by_id = {d.id: d for d in documents}

    for (doc_id, sentence), ms in sent_map.items():
        eids = sorted({m.entity_id for m in ms})
        if len(eids) < 2:
            continue
        # choose date: first mention date, else doc date
        doc = doc_by_id.get(doc_id)
        doc_dates = doc.dates if doc else []
        date_candidates = [m.date for m in ms if m.date] or doc_dates
        date = date_candidates[0] if date_candidates else None

        for a, b in combinations(eids, 2):
            if a > b:
                a, b = b, a
            key = (a, b)
            e = graph_edges.get(key)
            if not e:
                graph_edges[key] = {
                    "source": a,
                    "target": b,
                    "weight": 1,
                    "first_date": date,
                    "last_date": date,
                    "mention_ids": [mmm.id for mmm in ms],
                }
            else:
                e["weight"] += 1
                e["mention_ids"].extend(mmm.id for mmm in ms)
                if date:
                    if e["first_date"] is None or (
                        e["first_date"] and date < e["first_date"]
                    ):
                        e["first_date"] = date
                    if e["last_date"] is None or (
                        e["last_date"] and date > e["last_date"]
                    ):
                        e["last_date"] = date

    # degrees
    degrees: Counter = Counter()
    for (a, b), edge in graph_edges.items():
        degrees[a] += 1
        degrees[b] += 1

    graph_nodes = [
        {"id": e.id, "name": e.name, "type": e.label, "degree": int(degrees[e.id])}
        for e in entities
    ]
    graph_edges_arr = []
    for (a, b), e in graph_edges.items():
        graph_edges_arr.append(
            {
                "source": a,
                "target": b,
                "weight": int(e["weight"]),
                "first_date": e["first_date"],
                "last_date": e["last_date"],
                "mention_ids": sorted(set(e["mention_ids"])),
            }
        )

    # ----------------------------- output -----------------------------------

    mentions_out = [asdict(m) for m in all_mentions]
    documents_out = [asdict(d) for d in documents]
    entities_out = [asdict(e) for e in entities]

    analytics = {
        "summary": {
            "num_files": len(documents),
            "total_tokens": int(sum(d.token_count for d in documents)),
            "unique_tokens": int(len(global_counts)),
            "total_entities": int(len(entities)),
        },
        "documents": documents_out,
        "entities": entities_out,
        "mentions": mentions_out,
        "top_words": [
            {"word": w, "count": int(c)} for w, c in global_counts.most_common(300)
        ],
        "topics": {
            "topics": topics_summary,
        },
        "graph": {
            "nodes": graph_nodes,
            "edges": graph_edges_arr,
        },
    }

    out_path.write_text(json.dumps(analytics, indent=2), encoding="utf-8")
    print(f"[+] Wrote analytics JSON to {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Recursive text/PDF analysis -> analytics.json for D3 dashboard"
    )
    ap.add_argument("root", help="Root directory containing TXT/PDF files")
    ap.add_argument("--out", default="analytics.json", help="Output JSON path")
    args = ap.parse_args()

    analyze(Path(args.root), Path(args.out))


if __name__ == "__main__":
    main()