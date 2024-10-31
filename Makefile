-include .env
export

all: up

up:
	$(MAKE) dc-up ARGS='--build --remove-orphans'

dc-%:
	docker compose $* $(ARGS)

gw:
	jupyter kernelgateway -y --log-level=DEBUG

test:
	octave-cli "$$PWD/Examples/4 Warp Shell/W1_Warp_Shell.m"

sh:
	make dc-run ARGS='app bash'
