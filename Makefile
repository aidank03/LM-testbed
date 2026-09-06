.PHONY: install test demo evidence design
install:
	python -m pip install -e .
test:
	python -m unittest discover -s tests -v
demo:
	liner-stability run --config configs/demo.json --out outputs/demo
evidence:
	liner-stability evidence index --files docs/PROJECT_SECTION.md docs/MODEL_CARD.md --out outputs/evidence.json
design:
	liner-stability design --config configs/design_example.json --out outputs/design_ranking.json
