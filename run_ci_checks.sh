#!/bin/bash
./run_autoformat.sh
mypy .
pytest tests/ --pylint -m pylint --pylint-rcfile=.pylintrc
pytest tests/
