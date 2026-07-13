# LEO / CLEO figure

`leo_cleo_transformer.py` generates the LEO-vs-CLEO architecture figure in the
style of the original Transformer diagram. It is built on
[sane_tikz](https://github.com/negrinho/sane_tikz) by Renato Negrinho (MIT,
vendored under `sane_tikz/`), adapted from that repo's `examples/transformer.py`.

## Regenerate

```bash
cd assets/tikz
PYTHONPATH=. python3 leo_cleo_transformer.py   # -> both_stz.tex
pdflatex both_stz.tex                           # -> both_stz.pdf
pdftocairo -svg both_stz.pdf both_stz.svg       # -> both_stz.svg
python3 postprocess.py both_stz.svg             # -> ../../_includes/leo_cleo_transformer.svg
```

The post inlines `_includes/leo_cleo_transformer.svg` via `{% include %}` so the
figure is **theme-adaptive**: `postprocess.py` turns the ink into `currentColor`
(follows the page text colour) while the pastel block fills and the dark
on-block text stay fixed. currentColor only inherits when the SVG is inlined, so
it must be an include, not an `<img>`.

Edit `leo_cleo_transformer.py` (labels, colours, spacing) and rerun the four
steps above to update the figure.
