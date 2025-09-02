ART4=artifacts/ch4_4x4
ART6=artifacts/ch4_6x6

.PHONY: ch4-artifacts ch4-tables
ch4-artifacts:
	python examples/generate_artifacts.py --env 4x4 --outdir $(ART4)
	python examples/generate_artifacts.py --env 6x6 --outdir $(ART6)

ch4-tables: ch4-artifacts
	python examples/csv_to_latex.py $(ART4) --outdir artifacts/latex --no-wrap --round 0
