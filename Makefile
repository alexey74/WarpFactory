clean:
	find . -path ./.venv -prune \
		-o -name '*.c' -print -o \
		-name '*.so' -print \
		-name __pycache__ -print | xargs rm -rf