# edit-pathways

Code behind *NewsEdits*, a dataset of news article revision histories. News articles are often rewritten after publication: facts are added, sentences deleted, paragraphs moved. We collect article versions from two public trackers of online news updates (NewsSniffer and the DiffEngine Twitter bots), align sentences across consecutive versions, and label each sentence with an edit action (addition, deletion, edit, refactor). We then ask whether edits are predictable: given one version of an article, can a model tell which sentences will be added, removed or moved in the next? The corpus covers 22 English- and French-language outlets, 2006-2021.

## Related paper

*NewsEdits: A News Article Revision Dataset and a Document-Level Reasoning Challenge* (NAACL 2022). Dataset and modeling code: https://github.com/isi-nlp/NewsEdits. LaTeX source is in `presentations/naacl2022/` (gitignored).

## Layout

- `scraping/` -- Scrapy project that fetches article versions from NewsSniffer.
- `data/diffengine-diffs/` -- dump script for the DiffEngine archive on archive.org.
- `sequential_parsing/` -- turns per-outlet sqlite dumps into sentence diffs (Dockerized, run on GCE).
- `spark_processing_scripts/` -- PySpark + Spark NLP pipeline that splits sentences and matches them across versions by embedding similarity. Bundles two Spark NLP model zips (~70 MB).
- `util/` -- NewsSniffer/DiffEngine parsers, refactor detection, labeling, S3/HDFS access.
- `modeling/` -- PyTorch Lightning models (RoBERTa sentence encoder, optional context layer); `runner.py` is the entry point, `scripts/` has launch configs.
- `evaluation/` -- Flask app and MTurk templates for the human annotation study.
- `gcs_sqlite_to_bq/` -- Cloud Function loading sqlite into BigQuery.
- `notebooks/` -- dated EDA, annotation-prep and error-analysis notebooks (2020-2021).

## How to run

- `pip install -r requirements.txt && pip install -e .`
- Sentence matching: `python -m spark_processing_scripts.runner --db_name <outlet> --num_files 500 --env local`
- Modeling: `python -m modeling.runner --experiment sentence --do_addition ...` (flags in `modeling/utils_parser.py`)
- Annotation app: `cd evaluation && python main.py`

## Data

Not included. The pipeline expects per-outlet sqlite databases (`nyt`, `wp`, `ap`, `guardian`, `bbc`, `reuters`, `cnn`, ...) built from NewsSniffer scrapes and the DiffEngine archive; `data/`, `*.db`, `*.csv` and `*.json` are gitignored. Use the released dataset at the link above. Source licenses: NewsSniffer AGPL-3.0, DiffEngine CC BY-ND 4.0.

## Status

Main development 2020-2021; small edits to the evaluation app and notebooks in mid-2024. Not actively maintained.
