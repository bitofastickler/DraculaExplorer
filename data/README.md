# Data folder

Expected inputs (put at least one here):

- `dracula_corrected_entries.json` with structure:
  {"entries": [
    {"chapter_number": 1, "narrator": "Jonathan Harker", "date_iso": "1897-05-03", "text": "..."},
    ...
  ]}

- OR `dracula_ascii_rag.json` (array of chunk objects with at least: entry_index, chapter_number, narrator, text).

`gold/` holds alias lists, must-links, and query checks for the entity-resolution graph.
`kg/` holds the built graph (`graph.json`, `entities.jsonl`, `mentions.jsonl`, `eval.json`). Mention offsets refer to de-overlapped, whitespace-normalized entry text, not to the raw overlapping chunks.

The corpus JSON is committed. `kg/review_queue.csv` is a local tuning dump and can be deleted.
