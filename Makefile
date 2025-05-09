BUILDKIT_PROGRESS ?= plain

export

all: c-up

c-%:
	docker compose $* $(A)

up:
	$(MAKE) c-up A=--build

rebuild:
	$(MAKE) c-build A=--no-cache

sh:
	$(MAKE) c-run A='warpfactory bash'

test:
	$(MAKE) c-run A='warpfactory python -m pytest .'

poetry.lock: pyproject.toml
	poetry lock  

requirements-exp.txt: poetry.lock
	poetry export --without-hashes --without-urls --dev > $@
