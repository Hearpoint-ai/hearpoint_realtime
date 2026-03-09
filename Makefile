.PHONY: eval-fast eval-live make-fixture enroll enroll-record

PYTHON ?= /opt/miniconda3/envs/hearpoint-realtime/bin/python
NAME ?= User
AUDIO ?= media/enrollments/sample.wav
DURATION ?= 10

# Generate the SI-SDR fixture (run once, or whenever speakers change)
make-fixture:
	$(PYTHON) scripts/make_fixture.py \
	  --target      data/our_speech_pool/Chris.wav \
	  --interferers data/our_speech_pool/Himanshu.wav data/our_speech_pool/Sanna.wav \
	  --noise        data/wham_noise/011a0101_0.061105_401c020r_-0.061105.wav \
	  --output-dir  media/si_sdr_fixture

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
	  --embedding media/enrollments/f75d8385-bedf-4082-babf-1825963c7e69.npy \
	  --report reports/eval/$(shell date +%Y%m%dT%H%M%S)_live.json \
	  --threshold-profile dev \
	  --duration 30 \
	  --warmup-chunks 10

# Enroll a user from an existing stereo WAV file.
# Example:
#   make enroll NAME=Hady AUDIO=/path/to/enrollment.wav
enroll:
	$(PYTHON) scripts/enroll.py --name "$(NAME)" --audio "$(AUDIO)"

# Enroll a user by recording from microphone.
# Example:
#   make enroll-record NAME=Hady DURATION=5
enroll-record:
	$(PYTHON) scripts/enroll.py --name "$(NAME)" --record --duration "$(DURATION)"
