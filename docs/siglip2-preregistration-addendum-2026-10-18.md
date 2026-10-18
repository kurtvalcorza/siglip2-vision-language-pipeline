# Addendum to the SigLIP 2 pre-registration (2026-10-18, after the evaluation of record)

This addendum changes no setting. It records two status facts that commit `449edfc` had first written into
`docs/siglip2-preregistration.md` §5 in place. That file has been restored byte for byte to its registered text at
`1b5f790`.

1. **Evaluation of record.** The first hosted pre-registered run is the Colab CLI T4 run of 2026-10-18 (`6df20c7`, blob
   `1f274277`), recorded in `docs/release-verification.md` and carried in `RECORDED_EVALUATION`. Every later identical
   run is labelled `reproduction of the pre-registered evaluation`. The `e9ff7f9` run reproduced it exactly.
2. **The in-place edit by `449edfc`.** In §5 it replaced "Until then `RECORDED_EVALUATION = None`." with the pointer in
   item 1. No setting, rule or threshold was touched.

Post-hoc additions after the record are labelled as such wherever they appear. All of them are display-only:

- per-image predictions;
- an exact McNemar test and a paired bootstrap interval;
- tie-aware AP for the majority floor;
- the model id and revision in the pre-registered check (the values the record used).

No primary endpoint or success criterion was pre-registered. The fresh-test comparison is descriptive.
