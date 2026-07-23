# Score update visualizer

Build the self-contained visualizer with the repository's virtual environment:

```bash
.venv/bin/python scripts/score_visualizer/build_visualizer.py
```

Then open `scripts/score_visualizer/score_visualizer.html` in a browser. The
generated file embeds the source heatmaps, extracted features, and ten rounds of
both scoring modes, so it does not need a server.

Useful build options:

```bash
.venv/bin/python scripts/score_visualizer/build_visualizer.py \
  --fields 87 96.2 176 --iterations 15 --n-hood 3 --retain 0.5 --tau 3
```

- **Original anchor:** support is recalculated from the previous round, while
  every new score is blended with the original confidence.
- **Recursive:** support is recalculated from the previous round, and every new
  score is blended with the previous round's score.

The generated visualizer recomputes scores in the browser when you change:

- the temperature-spacing spread `tau`;
- the number of neighboring linecuts inspected on each side;
- the optional sigmoid support multiplier;
- the sigmoid center and width in
  `support * sigmoid((support - center) / width)`.

The defaults are `tau=3`, three linecuts per side, sigmoid center `0.6`, and
sigmoid width `0.1`. The sigmoid multiplier starts disabled.
