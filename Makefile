#!make
-include .env
export


up:
	docker-compose up -d


down:
	docker-compose down


connect:
	psql -h $(PGHOST) --port $(PGPORT) --username $(PGUSER) --dbname $(PGDATABASE)


check: up
	. ./.venv/bin/activate && \
	python setup.py sdist && \
	twine check dist/*.tar.gz && \
	tox


deploy:
	python setup.py sdist
	twine upload dist/*tar.gz


venv:
	python3 -m venv .venv && make reqs


reqs: 
	. ./.venv/bin/activate && pip install -r requirements-dev.txt
	python -m spacy download $(SPACY_MODEL_VERSION)


integration:
	. ./.venv/bin/activate & python ./integration.py


test: up
	pytest -vv


test-focus:
	pytest -s -vv -m focus


fmt:
	black .
	flake8 ./pyspark_tooling
	flake8 ./tests
