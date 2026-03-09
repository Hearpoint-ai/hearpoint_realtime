.PHONY: eval-fast eval-live

PYTHON ?= /opt/miniconda3/envs/hearpoint-realtime/bin/python

eval-fast:
	mkdir -p reports/eval
	$(PYTHON) src/realtime/realtime_inference.py \
	  --test-file media/mixtures/derron_mubashir.wav \
	  --embedding media/enrollments/f75d8385-bedf-4082-babf-1825963c7e69.npy \
	  --report reports/eval/$(shell date +%Y%m%dT%H%M%S).json \
	  --threshold-profile dev \
	  --warmup-chunks 10

eval-live:
	mkdir -p reports/eval
	$(PYTHON) src/realtime/realtime_inference.py \
	  --embedding media/enrollments/f75d8385-bedf-4082-babf-1825963c7e69.npy \
	  --report reports/eval/$(shell date +%Y%m%dT%H%M%S)_live.json \
	  --threshold-profile dev \
	  --duration 30 \
	  --warmup-chunks 10
