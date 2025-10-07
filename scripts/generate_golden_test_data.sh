#!/usr/bin/env bash

# Generates golden test data for RGZ regression tests.
bazel-bin/rgz/main subjects \
    --in=rgz/testdata/radio_subjects_test_subset.json \
    --out=rgz/testdata/radio_subjects_test_subset_processed.json \
    --cache=rgz/testdata/first
bazel-bin/rgz/main classifications \
    --in=rgz/testdata/radio_classifications_test_subset.json \
    --out=rgz/testdata/radio_classifications_test_subset_processed.json \
    --cache=rgz/testdata/first \
    --subjects=rgz/testdata/radio_subjects_test_subset_processed.json
bazel-bin/rgz/main host-lookup \
    --classifications=rgz/testdata/radio_classifications_test_subset_processed.json \
    --out=rgz/testdata/radio_classifications_test_subset_matched.json
bazel-bin/rgz/main aggregate \
    --subjects=rgz/testdata/radio_subjects_test_subset_processed.json \
    --classifications=rgz/testdata/radio_classifications_test_subset_matched.json \
    --out=rgz/testdata/consensus.json
