# Reporting provenance

`analysis_data.json` stores hashes of source artifacts and the numerical coordinates underlying the analyses. Names in this internal export retain source identifiers so results can be traced without relabeling checkpoints.

The paper compares four measured pipelines: Gemma generation, MedGemma generation, MedGemma candidate-label likelihood, and Laya-Med direct decisions. Generation confidence is the saved emitted-answer confidence. Decision-distribution reliability is compared only between MedGemma label likelihood and Laya-Med, which use the same candidate support.

No MedGemma-Direct evaluation is present. Full-cohort Laya-Med results are an independent export; their fitted pipeline setting differs from the pilot. The study does not establish a controlled architectural speedup or a causal advantage from individual components.
