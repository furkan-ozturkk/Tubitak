"""The corpus itself: what it contains, how it is hashed, how it is partitioned.

``corpus_loader`` – the raw ``*_2k.log`` bytes, plus the hashing and dev/test
                    split rules every tier shares.
``data_factory``  – ``corpus_provider()``, the single place a log file is opened
                    during generation; hands a tier one ``CorpusView``.
``dataset_specs`` – the curated per-dataset literals and grouping rules
                    (Section 7.1).

The database client is deliberately not here. ``src.utils.helper_postgres`` is
the transport that reaches loghub's ``lines`` table over ``datasetgen-net``, and
it lives beside ``src.utils.helper_vllm`` because both are clients for a service
this package does not itself own. This package is about the corpus as data.

Which of the two a tier reads from is not a style choice. Anything whose gold
value a reader must be able to reproduce — the easy tier's counts and lookups,
and the validator's re-derivation of them — goes through SQL. The medium and hard
tiers read the file, because they need contiguous evidence windows and
regex-grouped events rather than aggregates, and their answers are model-drafted
and human-reviewed rather than claimed to be machine-reproducible.
"""
