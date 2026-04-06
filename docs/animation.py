from manimlib import *
import numpy as np

IMG_INPUT  = "videos/Spectrogram_input.png"
IMG_OUTPUT = "videos/Spectrogram_output.png"

# ── helpers ──────────────────────────────────────────────────────────────────

def make_wave(func, x_min=-5, x_max=5, color=BLUE, stroke_width=2.5, **kw):
    return FunctionGraph(func, x_range=(x_min, x_max, 0.03),
                         color=color, stroke_width=stroke_width, **kw)


def spec_image(path, width=5.5):
    img = ImageMobject(path)
    img.set_width(width)
    return img


# ── main scene ───────────────────────────────────────────────────────────────

class HearPointExplainer(Scene):

    def construct(self):
        self.scene_1_problem()
        self.scene_2_chunking()
        self.scene_3_stft()
        self.scene_4_tfgridnet()
        self.scene_5_istft()
        self.scene_6_statefulness()
        self.scene_7_latency()
        self.scene_8_recap()

    # ── utilities ────────────────────────────────────────────────────────────

    def clear(self, run_time=0.4):
        mobs = [m for m in self.mobjects]
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)

    def section_label(self, text, color=BLUE_C):
        lbl = Text(text, font_size=22, color=color)
        lbl.to_corner(UL, buff=0.3)
        return lbl

    # ── Scene 1 · The Problem  (~18 s) ───────────────────────────────────────

    def scene_1_problem(self):
        title = Text("HearPoint", font_size=68, color=BLUE_C)
        sub   = Text("Real-Time Speaker Isolation", font_size=28, color=WHITE)
        sub.next_to(title, DOWN, buff=0.3)

        self.play(Write(title), run_time=1.0)
        self.play(FadeIn(sub),  run_time=0.6)
        self.wait(1.2)
        self.play(FadeOut(title), FadeOut(sub), run_time=0.5)

        # --- three overlapping waves ---
        sec_lbl = self.section_label("The Problem")
        self.play(FadeIn(sec_lbl), run_time=0.3)

        y_off = 1.6
        w1 = make_wave(lambda x: 0.55 * np.sin(2.0 * x),           color=BLUE)
        w2 = make_wave(lambda x: 0.40 * np.sin(3.7 * x + 1.1),     color=RED)
        w3 = make_wave(lambda x: 0.25 * np.sin(7.3 * x + 2.5),     color=GREEN)
        w1.shift(UP * y_off)
        w2.shift(UP * 0)
        w3.shift(DOWN * y_off)

        t1 = Text("Your Voice",  font_size=20, color=BLUE ).next_to(w1, LEFT, buff=0.15)
        t2 = Text("Background",  font_size=20, color=RED  ).next_to(w2, LEFT, buff=0.15)
        t3 = Text("Noise",       font_size=20, color=GREEN).next_to(w3, LEFT, buff=0.15)

        self.play(
            ShowCreation(w1), ShowCreation(w2), ShowCreation(w3),
            FadeIn(t1), FadeIn(t2), FadeIn(t3),
            run_time=1.2,
        )
        self.wait(0.8)

        # merge into messy combined wave
        mixed = make_wave(
            lambda x: 0.55*np.sin(2.0*x) + 0.40*np.sin(3.7*x+1.1) + 0.25*np.sin(7.3*x+2.5),
            color=YELLOW, stroke_width=2.0,
        )
        self.play(
            ReplacementTransform(VGroup(w1, w2, w3), mixed),
            FadeOut(t1), FadeOut(t2), FadeOut(t3),
            run_time=1.0,
        )
        self.wait(0.5)

        # shift wave up, show real input spectrogram below
        self.play(mixed.animate.shift(UP * 1.5), run_time=0.5)

        sp_img = spec_image(IMG_INPUT, width=9.5)
        sp_img.next_to(mixed, DOWN, buff=0.25)

        sp_note = Text("What the mic captures — all voices mixed",
                       font_size=20, color=YELLOW)
        sp_note.to_edge(DOWN, buff=0.35)

        self.play(FadeIn(sp_img), run_time=0.8)
        self.play(Write(sp_note), run_time=0.7)
        self.wait(1.8)
        self.clear()

    # ── Scene 2 · Chunking  (~13 s) ──────────────────────────────────────────

    def scene_2_chunking(self):
        lbl = self.section_label("Step 1 · Chunking")
        self.play(FadeIn(lbl), run_time=0.3)

        stream = Rectangle(width=10, height=0.55,
                           fill_color=BLUE_E, fill_opacity=0.7,
                           stroke_color=WHITE, stroke_width=1.5)
        stream.move_to(UP * 0.6)
        s_lbl = Text("Audio stream", font_size=20, color=WHITE).next_to(stream, UP, buff=0.15)

        self.play(ShowCreation(stream), FadeIn(s_lbl), run_time=0.7)

        cuts = VGroup()
        chunk_w = 10 / 10
        for i in range(1, 10):
            x = -5 + i * chunk_w
            cut = Line(stream.get_top() + RIGHT * (x + 5 - 5),
                       stream.get_bottom() + RIGHT * (x + 5 - 5),
                       color=WHITE, stroke_width=1.2)
            cut.move_to(RIGHT * x + UP * 0.6)
            cuts.add(cut)

        chunk_labels = VGroup()
        for i in range(10):
            x = -5 + (i + 0.5) * chunk_w
            cl = Text("8ms", font_size=12, color=YELLOW_B)
            cl.move_to(RIGHT * x + UP * 0.6)
            chunk_labels.add(cl)

        self.play(LaggedStart(*[ShowCreation(c) for c in cuts], lag_ratio=0.08), run_time=0.9)
        self.play(LaggedStart(*[FadeIn(c) for c in chunk_labels], lag_ratio=0.06), run_time=0.6)

        chunk_hl = Rectangle(width=chunk_w, height=0.55,
                             fill_color=YELLOW, fill_opacity=0.6,
                             stroke_color=YELLOW, stroke_width=2)
        chunk_hl.move_to(RIGHT * (-5 + 0.5 * chunk_w) + UP * 0.6)

        proc_box = RoundedRectangle(width=2.2, height=1.1, corner_radius=0.15,
                                    fill_color=GREY_D, fill_opacity=0.9,
                                    stroke_color=WHITE, stroke_width=1.5)
        proc_box.move_to(DOWN * 1.4)
        proc_lbl = Text("Process", font_size=20, color=WHITE).move_to(proc_box)

        self.play(FadeIn(chunk_hl), run_time=0.4)
        self.play(ShowCreation(proc_box), FadeIn(proc_lbl), run_time=0.5)
        self.play(chunk_hl.animate.move_to(proc_box.get_center()), run_time=0.6)

        info = Text("128 samples  ·  8 ms per chunk  ·  GPU processes in ~2 ms",
                    font_size=21, color=WHITE)
        info.to_edge(DOWN, buff=0.4)
        self.play(Write(info), run_time=0.8)
        self.wait(1.5)
        self.clear()

    # ── Scene 3 · STFT  (~20 s) ──────────────────────────────────────────────

    def scene_3_stft(self):
        lbl = self.section_label("Step 2 · STFT")
        self.play(FadeIn(lbl), run_time=0.3)

        # raw waveform on the left
        wave = make_wave(lambda x: 0.6 * np.sin(2*x) + 0.3 * np.sin(5*x+1),
                         x_min=-2.8, x_max=2.8, color=BLUE_B, stroke_width=2.5)
        wave.move_to(LEFT * 3.2)
        wl = Text("Chunk\n(time domain)", font_size=18, color=BLUE_B)
        wl.next_to(wave, DOWN, buff=0.3)

        self.play(ShowCreation(wave), FadeIn(wl), run_time=0.8)

        arr = Arrow(LEFT * 0.9, RIGHT * 0.9, color=WHITE, buff=0.05)
        stft_txt = Text("STFT", font_size=18, color=YELLOW)
        stft_txt.next_to(arr, UP, buff=0.1)
        self.play(GrowArrow(arr), FadeIn(stft_txt), run_time=0.5)

        # real input spectrogram on the right
        spec = spec_image(IMG_INPUT, width=4.8)
        spec.move_to(RIGHT * 3.1)

        freq_lbl = Text("Frequency", font_size=16, color=GREY_A)
        freq_lbl.next_to(spec, LEFT, buff=0.18)
        freq_lbl.rotate(PI / 2)
        time_lbl = Text("Time", font_size=16, color=GREY_A)
        time_lbl.next_to(spec, DOWN, buff=0.12)

        self.play(FadeIn(spec), run_time=0.8)
        self.play(FadeIn(freq_lbl), FadeIn(time_lbl), run_time=0.4)

        hann = Text("Hann window  →  tapers edges\nPhase preserved from input",
                    font_size=19, color=YELLOW_B)
        hann.to_edge(DOWN, buff=0.45)
        self.play(Write(hann), run_time=0.8)
        self.wait(1.2)

        eq = Text("X(f,t) = \u03a3 x(n) \u00b7 w(n-t) \u00b7 e^(-i2\u03c0fn/N)",
                  font_size=24)
        eq.set_color(WHITE)
        eq.move_to(DOWN * 2.2)
        self.play(FadeOut(hann), Write(eq), run_time=1.0)
        self.wait(1.5)
        self.clear()

    # ── Scene 4 · TFGridNet  (~32 s) ─────────────────────────────────────────

    def scene_4_tfgridnet(self):
        lbl = self.section_label("Step 3 · TFGridNet")
        self.play(FadeIn(lbl), run_time=0.3)

        IMG_W = 6.5
        spec = spec_image(IMG_INPUT, width=IMG_W)
        spec.move_to(ORIGIN + DOWN * 0.15)
        self.play(FadeIn(spec), run_time=0.6)

        iw = spec.get_width()
        ih = spec.get_height()
        cx = spec.get_center()[0]
        top_y  = spec.get_top()[1]
        left_x = spec.get_left()[0]

        # ── Intra LSTM (time sweep) ──
        N_ROWS = 9
        row_h = ih / N_ROWS
        intra_lbl = Text("Intra-LSTM  (time →)", font_size=20, color=BLUE_B)
        intra_lbl.to_edge(UP, buff=0.7)
        self.play(FadeIn(intra_lbl), run_time=0.3)

        bar_h = Rectangle(width=iw, height=row_h,
                          fill_color=BLUE_B, fill_opacity=0.40,
                          stroke_width=0)
        bar_h.move_to([cx, top_y - row_h / 2, 0])
        self.play(FadeIn(bar_h), run_time=0.2)
        for r in range(1, N_ROWS):
            y = top_y - row_h * r - row_h / 2
            self.play(bar_h.animate.move_to([cx, y, 0]), run_time=0.16)
        self.play(FadeOut(bar_h), run_time=0.2)

        # ── Inter LSTM (frequency sweep) ──
        N_COLS = 12
        col_w = iw / N_COLS
        inter_lbl = Text("Inter-LSTM  (frequency \u2191)", font_size=20, color=GREEN_B)
        inter_lbl.to_edge(UP, buff=0.7)
        self.play(ReplacementTransform(intra_lbl, inter_lbl), run_time=0.3)

        bar_v = Rectangle(width=col_w, height=ih,
                          fill_color=GREEN_B, fill_opacity=0.40,
                          stroke_width=0)
        cy = spec.get_center()[1]
        bar_v.move_to([left_x + col_w / 2, cy, 0])
        self.play(FadeIn(bar_v), run_time=0.2)
        for c in range(1, N_COLS):
            x = left_x + col_w * c + col_w / 2
            self.play(bar_v.animate.move_to([x, cy, 0]), run_time=0.13)
        self.play(FadeOut(bar_v), FadeOut(inter_lbl), run_time=0.2)

        # ── Speaker embedding injection ──
        emb_lbl = Text("Speaker Embedding (FiLM)", font_size=20, color=ORANGE)
        emb_lbl.to_edge(UP, buff=0.7)
        self.play(FadeIn(emb_lbl), run_time=0.3)

        enroll_wave = make_wave(lambda x: 0.3 * np.sin(3*x),
                                x_min=-1.2, x_max=1.2, color=ORANGE, stroke_width=2)
        enroll_wave.move_to(LEFT * 5.2 + DOWN * 2.0)
        enroll_lbl = Text("Enrollment\nclip", font_size=14, color=ORANGE)
        enroll_lbl.next_to(enroll_wave, DOWN, buff=0.1)

        vec_rects = VGroup(*[
            Rectangle(width=0.12, height=interpolate(0.15, 0.65, np.random.rand()),
                      fill_color=ORANGE, fill_opacity=0.9, stroke_width=0)
            for _ in range(8)
        ])
        vec_rects.arrange(RIGHT, buff=0.06)
        vec_rects.move_to(LEFT * 3.0 + DOWN * 2.0)
        vec_lbl = Text("256-d\nembedding", font_size=14, color=ORANGE)
        vec_lbl.next_to(vec_rects, DOWN, buff=0.1)

        arr_emb = Arrow(LEFT * 4.3 + DOWN * 2.0, LEFT * 3.6 + DOWN * 2.0,
                        color=ORANGE, buff=0.05, stroke_width=2)
        arr_inject = Arrow(LEFT * 2.3 + DOWN * 2.0, LEFT * 1.5 + DOWN * 1.0,
                           color=ORANGE, buff=0.05, stroke_width=2)
        times = Text("\u00d7", font_size=26, color=ORANGE)
        times.move_to(LEFT * 1.5 + DOWN * 1.0)

        self.play(ShowCreation(enroll_wave), FadeIn(enroll_lbl), run_time=0.5)
        self.play(GrowArrow(arr_emb), FadeIn(vec_rects), FadeIn(vec_lbl), run_time=0.5)
        self.play(GrowArrow(arr_inject), FadeIn(times), run_time=0.4)
        self.wait(0.5)
        self.play(FadeOut(VGroup(enroll_wave, enroll_lbl, arr_emb,
                                  vec_rects, vec_lbl, arr_inject, times,
                                  emb_lbl)), run_time=0.3)

        # ── Mask: transition input → output spectrogram ──
        mask_lbl = Text("Mask applied  (silence = suppressed)", font_size=20, color=YELLOW)
        mask_lbl.to_edge(UP, buff=0.7)
        self.play(FadeIn(mask_lbl), run_time=0.3)

        masked_spec = spec_image(IMG_OUTPUT, width=IMG_W)
        masked_spec.move_to(ORIGIN + DOWN * 0.15)
        self.play(ReplacementTransform(spec, masked_spec), run_time=1.2)

        eq = Text("Output(f,t)  =  Mask(f,t)  \u00d7  Input(f,t)",
                  font_size=24)
        eq.set_color(WHITE)
        eq.to_edge(DOWN, buff=0.4)
        self.play(Write(eq), run_time=0.8)
        self.wait(1.5)
        self.clear()

    # ── Scene 5 · ISTFT  (~10 s) ─────────────────────────────────────────────

    def scene_5_istft(self):
        lbl = self.section_label("Step 4 · ISTFT")
        self.play(FadeIn(lbl), run_time=0.3)

        # real output spectrogram on the left
        spec = spec_image(IMG_OUTPUT, width=4.8)
        spec.move_to(LEFT * 3.0)
        sp_lbl = Text("Isolated\nspectrogram", font_size=18, color=GREY_A)
        sp_lbl.next_to(spec, DOWN, buff=0.2)

        self.play(FadeIn(spec), FadeIn(sp_lbl), run_time=0.6)

        arr = Arrow(LEFT * 0.7, RIGHT * 0.7, color=WHITE, buff=0.05)
        istft_t = Text("ISTFT", font_size=18, color=YELLOW)
        istft_t.next_to(arr, UP, buff=0.1)
        self.play(GrowArrow(arr), FadeIn(istft_t), run_time=0.5)

        clean_wave = make_wave(lambda x: 0.6 * np.sin(2*x),
                               x_min=-2.5, x_max=2.5, color=GREEN_B, stroke_width=3)
        clean_wave.move_to(RIGHT * 3.2)
        cl_lbl = Text("Clean audio\n(target voice only)", font_size=18, color=GREEN_B)
        cl_lbl.next_to(clean_wave, DOWN, buff=0.2)

        self.play(ShowCreation(clean_wave), FadeIn(cl_lbl), run_time=0.9)

        note = Text("Overlap-add reassembles chunks  ·  phase carried unchanged",
                    font_size=19, color=WHITE)
        note.to_edge(DOWN, buff=0.45)
        self.play(Write(note), run_time=0.7)
        self.wait(1.8)
        self.clear()

    # ── Scene 6 · Statefulness  (~10 s) ──────────────────────────────────────

    def scene_6_statefulness(self):
        lbl = self.section_label("Statefulness")
        self.play(FadeIn(lbl), run_time=0.3)

        title = Text("LSTMs carry memory across chunks", font_size=26, color=WHITE)
        title.shift(UP * 2.5)
        self.play(Write(title), run_time=0.6)

        chunk_boxes = VGroup()
        for i in range(3):
            box = RoundedRectangle(width=1.6, height=0.9, corner_radius=0.1,
                                   fill_color=BLUE_E, fill_opacity=0.85,
                                   stroke_color=WHITE, stroke_width=1.5)
            box.move_to(RIGHT * (i * 3.2 - 3.2))
            lbl_c = Text(f"Chunk {i+1}", font_size=18, color=WHITE).move_to(box)
            chunk_boxes.add(VGroup(box, lbl_c))

        self.play(LaggedStart(*[FadeIn(b) for b in chunk_boxes], lag_ratio=0.2), run_time=0.7)

        state_arrows = VGroup()
        state_labels = VGroup()
        for i in range(2):
            a = Arrow(chunk_boxes[i].get_right(), chunk_boxes[i+1].get_left(),
                      color=ORANGE, buff=0.1, stroke_width=2.5)
            sl = Text("h, c", font_size=16, color=ORANGE)
            sl.next_to(a, UP, buff=0.08)
            state_arrows.add(a)
            state_labels.add(sl)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in state_arrows], lag_ratio=0.3),
            run_time=0.6,
        )
        self.play(FadeIn(state_labels), run_time=0.3)

        note = Text("Context accumulates across seconds of audio",
                    font_size=21, color=GREY_A)
        note.to_edge(DOWN, buff=0.6)
        self.play(Write(note), run_time=0.6)
        self.wait(1.8)
        self.clear()

    # ── Scene 7 · Latency  (~10 s) ───────────────────────────────────────────

    def scene_7_latency(self):
        lbl = self.section_label("Latency")
        self.play(FadeIn(lbl), run_time=0.3)

        title = Text("End-to-end latency breakdown", font_size=26, color=WHITE)
        title.shift(UP * 2.6)
        self.play(Write(title), run_time=0.5)

        specs = [
            ("Capture\n8 ms",    8,  BLUE_C),
            ("Lookahead\n4 ms",  4,  ORANGE),
            ("Model\n2 ms",      2,  GREEN_C),
        ]
        total_ms = sum(s[1] for s in specs)
        scale = 7.0 / total_ms

        bar_group = VGroup()
        x_cursor = -3.5
        for txt, ms, col in specs:
            w = ms * scale
            rect = Rectangle(width=w, height=0.7,
                             fill_color=col, fill_opacity=0.9,
                             stroke_color=WHITE, stroke_width=1)
            rect.move_to(RIGHT * (x_cursor + w / 2))
            lbl_b = Text(txt, font_size=15, color=WHITE).move_to(rect)
            bar_group.add(VGroup(rect, lbl_b))
            x_cursor += w

        self.play(LaggedStart(*[FadeIn(b) for b in bar_group], lag_ratio=0.3),
                  run_time=0.8)

        total_lbl = Text("\u2248 14 ms total  (< 20 ms end-to-end)",
                         font_size=24, color=YELLOW)
        total_lbl.to_edge(DOWN, buff=0.9)
        self.play(Write(total_lbl), run_time=0.6)

        note = Text("Comparable to a typical phone call",
                    font_size=20, color=GREY_A)
        note.next_to(total_lbl, DOWN, buff=0.25)
        self.play(FadeIn(note), run_time=0.4)
        self.wait(1.8)
        self.clear()

    # ── Scene 8 · Pipeline Recap  (~10 s) ────────────────────────────────────

    def scene_8_recap(self):
        title = Text("Full Pipeline", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=0.5)

        stages = ["Mic", "Chunks\n8 ms", "STFT", "TFGridNet\n(masked)", "ISTFT", "Speaker"]
        colors = [GREY_B, BLUE_C, TEAL_C, PURPLE_B, TEAL_C, GREEN_C]
        boxes, labels = [], []
        x_positions = np.linspace(-5.0, 5.0, len(stages))

        for i, (s, c, x) in enumerate(zip(stages, colors, x_positions)):
            box = RoundedRectangle(width=1.55, height=0.75, corner_radius=0.12,
                                   fill_color=c, fill_opacity=0.85,
                                   stroke_color=WHITE, stroke_width=1.2)
            box.move_to(RIGHT * x)
            lbl = Text(s, font_size=14, color=WHITE).move_to(box)
            boxes.append(box)
            labels.append(lbl)

        arrows = []
        for i in range(len(stages) - 1):
            a = Arrow(boxes[i].get_right(), boxes[i+1].get_left(),
                      color=WHITE, buff=0.05, stroke_width=1.8)
            arrows.append(a)

        self.play(
            LaggedStart(*[FadeIn(VGroup(b, l)) for b, l in zip(boxes, labels)],
                        lag_ratio=0.12),
            run_time=1.0,
        )
        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.12),
            run_time=0.7,
        )

        final = Text("One voice, isolated, in real time.",
                     font_size=32, color=YELLOW)
        final.to_edge(DOWN, buff=1.0)
        self.play(Write(final), run_time=1.0)
        self.wait(2.0)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
