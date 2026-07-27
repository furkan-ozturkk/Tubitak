"""Unit tests over the logic that does not need a corpus, a database or a model.

Run with:
  python3 -m unittest discover -s tests -t .

What is covered here is the reasoning a wrong answer would come from: split
assignment, evidence-id construction, official-set selection, the CLI's own
validations, and the validator's answer checks against a fake corpus repository.

What is deliberately *not* covered here is the corpus-to-database equality that
``validate.py::check_corpus_matches_database`` establishes. Faking the database in
that test would reproduce the exact problem the check exists to catch -- generation,
validation and the manual SQL all reading one table and agreeing on it -- so the only
meaningful version of that test runs against a real loaded Postgres, which means it
belongs in an integration suite against the running compose stack rather than here.
That suite does not exist yet; ``tests/test_validate_answers.py`` documents the gap
where the fake repository stands in for it.
"""
