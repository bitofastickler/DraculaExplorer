# Data folder

Expected inputs (put at least one here):

- `dracula_corrected_entries.json` with structure:
  {"entries": [
    {"chapter_number": 1, "narrator": "Jonathan Harker", "date_iso": "1897-05-03", "text": "..."},
    ...
  ]}

- OR `dracula_ascii_rag.json` (array of chunk objects with at least: entry_index, chapter_number, narrator, text).

Files are not committed by default.
