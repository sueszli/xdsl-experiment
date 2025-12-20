# 
# local
# 

.PHONY: venv
venv:
	test -f requirements.txt || (uvx pipreqs . --mode no-pin --encoding utf-8 --ignore .venv && mv requirements.txt requirements.in && uv pip compile requirements.in -o requirements.txt)
	uv venv .venv --python 3.11
	uv pip install -r requirements.txt
	@echo "activate venv with: \033[1;33msource .venv/bin/activate\033[0m"

.PHONY: lock
lock:
	uv pip freeze > requirements.in
	uv pip compile requirements.in -o requirements.txt

.PHONY: fmt
fmt:
	uvx isort .
	uvx autoflake --remove-all-unused-imports --recursive --in-place .
	uvx black --line-length 5000 .

# 
# docker
# 

DOCKER_RUN = docker run --rm -v $(PWD):/workspace aziz-lang sh -c

.PHONY: build-image
build-image:
	docker build -t aziz-lang .

.PHONY: run
run: build-image
	$(DOCKER_RUN) 'uv run aziz-lang/main.py examples/optimize.aziz --mlir'

.PHONY: clean
clean:
	docker rmi aziz-lang 2>/dev/null || true
