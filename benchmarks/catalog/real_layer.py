"""`EA13`-`EA16`: the questions answered entirely on the real layer.

ONNX Runtime kernel registrations and MLPerf Tiny v1.2 submissions, so their
answers are checkable against the upstream sources rather than against the
generator.
"""
from __future__ import annotations

REAL_LAYER: list[dict] = [
    {
        "id": "EA13",
        "title": "REAL: operators the CUDA provider does not implement",
        "question": ("Which ai.onnx operators does ONNX Runtime implement on "
                     "CPU but NOT on CUDA, so a GPU graph would break or fall "
                     "back?"),
        "why_graph": ("A real coverage gap between two execution providers, "
                      "expressed as an anti-join over the same operator node. "
                      "Answer is verifiable against onnxruntime's "
                      "docs/OperatorKernels.md."),
        "cypher": """
MATCH (k:Kernel)-[:IMPLEMENTS]->(op:Operator)
WHERE k.execution_provider = "CPUExecutionProvider" AND op.domain = "ai.onnx"
OPTIONAL MATCH (k2:Kernel)-[:IMPLEMENTS]->(op)
WHERE k2.execution_provider = "CUDAExecutionProvider"
WITH op.name AS operator, op.category AS category, count(k2) AS cuda_kernels
WHERE cuda_kernels = 0
RETURN operator, category, cuda_kernels
ORDER BY category
LIMIT 20
""",
    },
    {
        "id": "EA14",
        "title": "REAL: MLPerf Tiny throughput leaders",
        "question": ("On the real MLPerf Tiny v1.2 submissions, which board "
                     "posted the highest throughput for each benchmark task?"),
        "why_graph": ("Joins measured submissions to the board, the reference "
                      "model and the benchmark task in one linear path."),
        "cypher": """
MATCH (b:Board)<-[:ON_BOARD]-(d:Deployment)-[:MEASURES]->(m:Model)
      -[:SOLVES]->(t:BenchmarkTask)
WHERE d.provenance = "real"
WITH t.name AS task, b.name AS board, d.throughput_inf_s AS throughput_inf_s,
     d.accuracy AS accuracy
RETURN task, board, throughput_inf_s, accuracy
ORDER BY throughput_inf_s DESC
LIMIT 12
""",
    },
    {
        "id": "EA15",
        "title": "REAL: operators available on only one execution provider",
        "question": ("Which operators are registered on exactly one ONNX "
                     "Runtime execution provider -- i.e. using them pins you "
                     "to that backend?"),
        "why_graph": ("Portability risk as a degree count over the real kernel "
                      "registration graph."),
        "cypher": """
MATCH (k:Kernel)-[:IMPLEMENTS]->(op:Operator)
WHERE k.provenance = "real"
WITH op.name AS operator, op.domain AS domain,
     count(DISTINCT k.execution_provider) AS providers
WHERE providers = 1
RETURN operator, domain, providers
ORDER BY operator
LIMIT 20
""",
    },
    {
        "id": "EA16",
        "title": "REAL vs SYNTHETIC: what is measured and what is generated",
        "question": ("How much of this graph is real public data versus the "
                     "generated fleet, per source?"),
        "why_graph": ("Provenance is a first-class property, so the split is "
                      "one aggregation -- not a README claim you have to "
                      "trust."),
        "cypher": """
MATCH (k:Kernel)
WITH k.provenance AS provenance, k.source AS source, count(k.id) AS kernels
RETURN provenance, source, kernels
ORDER BY kernels DESC
""",
    },
]
