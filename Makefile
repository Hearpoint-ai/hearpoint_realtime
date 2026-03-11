.PHONY: eval-fast eval-live make-fixture enroll enroll-record demo

PYTHON ?= /opt/homebrew/anaconda3/envs/hearpoint-realtime/bin/python
NAME ?= User
AUDIO ?= media/enrollments/sample.wav
DURATION ?= 10
EMBEDDING_MODEL ?= resemblyzer

# Generate the SI-SDR fixture (run once, or whenever speakers change)
make-fixture:
	$(PYTHON) scripts/make_fixture.py \
	  --target      data/our_speech_pool/Chris.wav \
	  --interferers data/our_speech_pool/Himanshu.wav data/our_speech_pool/Sanna.wav \
	  --noise        data/wham_noise/011a0101_0.061105_401c020r_-0.061105.wav \
	  --output-dir  media/si_sdr_fixture \
	  --embedding-model "$(EMBEDDING_MODEL)"

eval-fast:
	mkdir -p reports/eval
	$(PYTHON) src/realtime/realtime_inference.py \
	  --test-file      media/si_sdr_fixture/mixture.wav \
	  --embedding      media/si_sdr_fixture/enrollment.npy \
	  --reference-file media/si_sdr_fixture/reference.wav \
	  --report         reports/eval/$(shell date +%Y%m%dT%H%M%S).json \
	  --threshold-profile dev \
	  --warmup-chunks 10

eval-live:
	mkdir -p reports/eval
	$(PYTHON) src/realtime/realtime_inference.py \
	  --embedding media/enrollments/Derron_enrolment.npy \
	  --report reports/eval/$(shell date +%Y%m%dT%H%M%S)_live.json \
	  --threshold-profile dev \
	  --duration 30 \
	  --warmup-chunks 10

# Enroll a user from an existing stereo WAV file.
# Example:
#   make enroll NAME=Hady AUDIO=/path/to/enrollment.wav EMBEDDING_MODEL=resemblyzer
enroll:
	$(PYTHON) scripts/enroll.py --name "$(NAME)" --audio "$(AUDIO)" --embedding-model "$(EMBEDDING_MODEL)"

# Enroll a user by recording from microphone.
# Example:
#   make enroll-record NAME=Hady DURATION=5 EMBEDDING_MODEL=tfgridnet
enroll-record:
	$(PYTHON) scripts/enroll.py --name "$(NAME)" --record --duration "$(DURATION)" --embedding-model "$(EMBEDDING_MODEL)"

demo:
	$(PYTHON) scripts/demo.py
