.PHONY: setup lint type test train dashboard build_features

setup:
    pip install -r requirements.txt
    pre-commit install

lint:
    ruff check .
    black --check .

type:
    mypy src

test:
    pytest -v --cov=src --cov-report=term-missing

train:
    python -m src.models.train_all

dashboard:
    streamlit run src/app/dashboard.py

build_features:
    python -m src.data.build_features
