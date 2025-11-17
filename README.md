# Epstein Email Text Analysis Dashboard
This project is a small pipeline for exploring a corpus of text/PDF documents (e.g., the House Oversight Committee’s Jeffrey Epstein email release). 

It has two main parts:

1. `analyze.py` – a Python script that recursively analyzes a directory of `.txt` and `.pdf` files and produces a single `analytics.json` file.
2. `index.html` – a D3-powered dashboard that visualizes `analytics.json` (timeline, entities, topics, word clouds, co-occurrence heatmaps, etc.).
---
## 1. Getting the Data

1. Open the Google Drive folder:
```
https://drive.google.com/drive/folders/1ldncvdqIf6miiskDp_EDuGSDAaI_fJx8

```
2. In the Drive UI:
- Either click the dropdown next to the folder name and choose **“Download”** to get a ZIP of the entire folder,   or
- Manually select the relevant files/folders and choose **“Download”**.

3. Unzip the contents locally, e.g.:
```text
project-root/
  data/
    epstein_corpus/   # <- the extracted folder from Google Drive
      ... .txt/.pdf files ...
  analyze.py
  index.html
```

4.  Ensure that the directory you pass to analyze.py contains (recursively) all the .txt and .pdf files you want analyzed.
    
----------

## **2. Environment Setup**

  

### **Python Version**

-   Python **3.9+** recommended.
     

### **Install Python Dependencies**

  

Create and activate a virtual environment (recommended):

```
cd /path/to/project-root

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Install the core dependencies:

```
pip install python-dateutil pypdf tqdm spacy scikit-learn
```

Install a spaCy English model (at least one; the script will try en_core_web_trf, then en_core_web_md, then en_core_web_sm):

```
python -m spacy download en_core_web_sm
```

If spaCy or scikit-learn are missing, the script still runs but:

-   NER falls back to regex-only heuristics.
    
-   Topic modeling is disabled (topics list will be empty).
    

----------

## **3. What**

### **analyze.py**

### **Does**

  

At a high level, analyze.py walks the corpus, extracts structured information, and writes an analytics.json file that the dashboard consumes.

  

### **3.1 File Discovery and Reading**

-   Recursively walks a root folder and collects all *.txt and *.pdf files.
    
-   For each file:
    
    -   If it’s a PDF:
        
        -   Uses pypdf.PdfReader to extract text from each page, concatenating into a single string.
            
        
    -   If it’s a text file:
        
        -   Reads the file using UTF-8, ignoring errors.
            
        
    

  

### **3.2 Email Header Parsing**

  

For each document’s text:

-   Parses email-style header lines using a regex for:
    
    -   From:
        
    -   To:
        
    -   Cc:
        
    -   Bcc:
        
    -   Subject:
        
    
-   Extracts:
    
    -   subject: first Subject: line encountered.
        
    -   participants: set of email addresses found in From / To / Cc / Bcc headers.
        
    

  

### **3.3 Date Extraction**

-   Uses a date regex plus python-dateutil to identify date-like expressions in the text (e.g., 12/05/2019, December 5, 2019).
    
-   Normalizes them into YYYY-MM-DD ISO strings.
    
-   Stores per-document sorted unique date list.
    

  

### **3.4 Tokenization and Per-Document Stats**

-   Tokenizes text into words using a regex that:
    
    -   Normalizes to lowercase.
        
    -   Ignores extremely short or non-alphabetic tokens.
        
    
-   Maintains:
    
    -   token_count: total tokens in the document.
        
    -   unique_token_count: count of unique tokens.
        
    -   Per-document token frequency.
        
    
-   Applies a custom stopword list to drop boilerplate items like:
    
    -   Email disclaimers, generic legal fluff, month names, etc.
        
    
-   Produces top_words per document:
    
    -   An array of {"word": <str>, "count": <int>} for the most frequent non-stopword tokens.
        
    

  

### **3.5 Sentence Splitting**

-   Splits text into sentences using a simple rule:
    
    -   Split on ., ?, or ! followed by whitespace.
        
    
-   Stores the sentence string associated with each entity mention for later graph construction.
    

  

### **3.6 Entity Extraction (NER)**

  

Two paths:

1.  **spaCy NER (if available)**
    
    -   Uses a spaCy model (en_core_web_*) to extract entities.
        
    -   Keeps entities with labels: PERSON, ORG, GPE, LOC.
        
    -   For each entity: records entity_text, label, start_char, end_char, and the full sentence text.
        
    
2.  **Regex-only fallback (if spaCy unavailable)**
    
    -   Uses heuristics (capitalization, patterns, etc.) as a crude backup.
        
    -   Functionally, you still get entity mentions, but with less precision.
        
    

  

### **3.7 Canonicalization, Aliases, and Synonyms**

-   Canonicalizes entity names:
    
    -   Lowercases.
        
    -   Strips punctuation and extra whitespace.
        
    -   Normalizes certain patterns (e.g., middle initials, honorifics).
        
    
-   Applies hand-crafted alias groups for major players (Epstein, Maxwell, Clinton, Trump, key agencies like DOJ/FBI/CIA, etc.):
    
    -   Maps variants like Jeff E, J. Epstein, Epstein, etc. to a canonical form jeffrey epstein.
        
    -   Similarly for major organizations/geopolitical entities via a separate synonym mapping.
        
    
-   Builds a map:
    
    -   canonical_name -> entity_id.
        
    
-   Tracks per-entity metadata:
    
    -   Canonical name.
        
    -   Original label (PERSON/ORG/GPE/LOC).
        
    -   List of mention IDs.
        
    

  

### **3.8 Mentions and Union–Find Entity Merging**

-   Each entity mention is a Mention dataclass:
    
    -   Includes mention ID, entity ID, document ID, character offsets, sentence, and date(s).
        
    
-   Uses a Union–Find (disjoint-set) structure:
    
    -   To merge entities that are likely duplicates based on:
        
        -   Alias groups.
            
        -   Name similarity rules.
            
        
    -   Produces a final set of merged entity clusters with:
        
        -   Canonical name.
            
        -   Label.
            
        -   All mention IDs associated with that cluster.
            
        
    

  

### **3.9 Per-Entity Time Series**

-   For each final entity:
    
    -   Aggregates counts by date (from document date extraction).
        
    -   Stores a timeseries:
        
        -   [{ "date": "YYYY-MM-DD", "count": N }, ...].
            
        
    

  

### **3.10 Topic Modeling (Optional)**

-   If scikit-learn is available:
    
    -   Collects (most of) the document texts as a corpus.
        
    -   Uses TfidfVectorizer with:
        
        -   max_df=0.95, min_df=2, max_features=5000, English stopwords.
            
        
    -   Fits an NMF topic model with:
        
        -   Number of topics chosen as min(12, max(2, n_docs // 10 or 2)).
            
        
    -   For each topic:
        
        -   Extracts top words.
            
        -   Builds a human-readable label from the top ~4 words.
            
        
    
-   Each document stores:
    
    -   topic_ids: list of topic indices.
        
    -   topic_labels: the labels derived from NMF.
        
    

  

### **3.11 Entity Co-Occurrence Graph**

-   Builds a co-occurrence graph where:
    
    -   Nodes are entities (after merging).
        
    -   Edges connect entities that co-occur in:
        
        -   The same sentence, and/or
            
        -   The same document.
            
        
    
-   For each edge (source, target):
    
    -   Tracks:
        
        -   weight (co-occurrence count).
            
        -   first_date / last_date of shared mentions.
            
        -   Set of mention_ids contributing to this edge.
            
        
    

  

### **3.12 Global Summary and Output Structure**

  

The script produces an analytics.json with this structure (simplified):

```
{
  "summary": {
    "num_files": <int>,
    "total_tokens": <int>,
    "unique_tokens": <int>,
    "total_entities": <int>
  },
  "documents": [ { ... Document ... }, ... ],
  "entities": [ { ... EntityRecord ... }, ... ],
  "mentions": [ { ... Mention ... }, ... ],
  "top_words": [
    { "word": "<token>", "count": <int> }, ...
  ],
  "topics": {
    "topics": [
      { "id": <int>, "label": "<topword1, topword2,...>", "top_words": [ ... ] },
      ...
    ]
  },
  "graph": {
    "nodes": [ { "id": <int>, "label": "<name>", ... }, ... ],
    "edges": [ { "source": <id>, "target": <id>, "weight": <int>, ... }, ... ]
  }
}
```

This file is what the dashboard (index.html) loads.

----------

## **4. Running**

### **analyze.py**

  

From the project root (where analyze.py lives):

```
# Basic usage
python3 analyze.py data/epstein_corpus --out analytics.json
```

-   data/epstein_corpus should be replaced with the path to your extracted dataset (from Google Drive).
    
-   --out analytics.json sets the output file name/path. Common patterns:
    
    -   Write next to the HTML dashboard:
        
    

```
python3 analyze.py data/epstein_corpus --out analytics.json
# ensure index.html and analytics.json are in the same directory
```

- Or write into a subfolder and then copy or move it:
        
    

```
python3 analyze.py data/epstein_corpus --out dashboard/analytics.json
```

  

  

- When the script completes, you should see a message like:

```
[+] Wrote analytics JSON to analytics.json
```

----------

## **5. Running the Dashboard**

  

The dashboard is a static HTML/JS page that fetches analytics.json via fetch('analytics.json'), so **it must be served over HTTP** (not via file://).

  

### **5.1 Place Files**

  

Recommended layout:

```
project-root/
  index.html
  analytics.json
  (optional: assets/, css/, js/ if you break things out)
```

Ensure that analytics.json is in the **same directory** as index.html.

  

### **5.2 Run a Simple Local Web Server**

  

From the directory containing index.html and analytics.json:

```
# Python 3
python3 -m http.server 8000
```

Then open a browser to:

```
http://localhost:8000/index.html
```

You should see:

-   Summary stats (total documents, tokens, unique tokens, entities).
    
-   Filters for date range, topics, and possibly participants.
    
-   Visualizations such as:
    
    -   Timeline of document counts & entity activity.
        
    -   Word cloud of top tokens.
        
    -   Bar charts of top words.
        
    -   Topic co-occurrence heatmap.
        
    -   Entity network / graph views.
        
    

  

(These visualizations are all driven off the analytics.json fields described above.)

----------

## **6. Suggested Extensions and Future Work**

  

Here are some directions to extend both the analysis pipeline and the dashboard.

  

### **6.1 Analysis Pipeline Extensions**

1.  **Richer NER and Coreference:**
    
    -   Use a transformer-based NER model and/or coreference resolution to better group pronouns and aliases.
        
    -   Integrate Hugging Face models for more robust entity linking to external KBs (e.g., Wikidata).
        
    
2.  **Sentiment and Stance Analysis:**
    
    -   Per-sentence or per-document sentiment scores (e.g., neutral/positive/negative).
        
    -   Aggregate sentiment over time, by participant, or by topic.
        
    
3.  **Document Classification:**
    
    -   Cluster or classify documents as:
        
        -   Legal correspondence, scheduling, financial, social, etc.
            
        
    -   Add labels to Document objects and filter by “type” in the UI.
        
    
4.  **Thread/Conversation Reconstruction:**
    
    -   Use Subject and heuristics on quoted text to reconstruct email threads.
        
    -   Add a thread_id field to documents and visualize conversation trees.
        
    
5.  **Keyword/Pattern Detection:**
    
    -   Add rule-based extractors for specific patterns:
        
        -   Monetary amounts.
            
        -   Locations (beyond NER).
            
        -   Flight references, hotel bookings, etc.
            
        
    
6.  **Performance and Caching:**
    
    -   Cache intermediate representations (per-file JSON) and build analytics.json from them.
        
    -   Parallelize file processing with multiprocessing or concurrent.futures.
        
    

  

### **6.2 Dashboard Extensions**

1.  **Document-Level Detail View:**
    
    -   Click on a point in the timeline or a node in the graph to open a side panel/modal with:
        
        -   Full (or partial) text of the underlying document.
            
        -   Highlighted entities.
            
        -   Links to related documents.
            
        
    
2.  **Advanced Filtering and Search:**
    
    -   Free-text search across documents, subjects, and participants.
        
    -   Filter by:
        
        -   Specific entity/entities.
            
        -   Topic labels.
            
        -   Participant email domains (e.g., .gov, .org).
            
        
    
3.  **Entity-Centric Timeline Views:**
    
    -   Focused timeline per entity:
        
        -   Show activity spikes and co-occurring entities per time window.
            
        
    -   Export views as PNG/SVG.
        
    
4.  **Interactive Topic Exploration:**
    
    -   Allow switching topics on/off for a given view.
        
    -   Sliders to adjust number of topics or number of top words shown.
        
    
5.  **Export and Sharing:**
    
    -   Buttons to export filtered data slices (e.g., selected date range + topic) as CSV or JSON.
        
    -   Export graphs (e.g., co-occurrence network) as GraphML or GEXF.
        
    
6.  **LLM Integration:**
    
    -   Add a “summarize current selection” button that:
        
        -   Sends filtered documents/entities to an LLM backend.
            
        -   Returns a short narrative summary, hypotheses, or questions to investigate.
            
        
    

  

### **6.3 Engineering/DevOps**

1.  **Packaging:**
    
    -   Add a requirements.txt or pyproject.toml.
        
    -   Wrap analyze.py as a CLI tool (console_scripts entry point).
        
    
2.  **Testing:**
    
    -   Write unit tests for:
        
        -   Date extraction.
            
        -   Header parsing.
            
        -   Entity alias resolution.
            
        -   Graph construction.
            
        
    
3.  **Deployment:**
    
    -   Host the static dashboard on GitHub Pages or S3 + CloudFront.
        
    -   Run analyze.py in a scheduled job (e.g., GitHub Actions, cron) when new documents are added.
        
    

----------

## **7. Quickstart Summary**

1.  **Download data** from the Google Drive link and unpack it under data/.
    
2.  **Create a virtualenv** and pip install the dependencies.
    
3.  **Run the analyzer**:
    

```
python3 analyze.py data/epstein_corpus --out analytics.json
```


    
4.  **Serve the dashboard**:
    

```
python3 -m http.server 8000
# then go to http://localhost:8000/index.html
``` 
    
5.  **Explore** the emails via the interactive visualizations and iterate on the pipeline and UI as needed.
