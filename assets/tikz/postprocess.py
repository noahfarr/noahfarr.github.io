"""Make the pdftocairo SVG theme-adaptive and inline-ready.

Turns the ink (rgb 0,0,0) into `currentColor` so it follows the page text
colour, while on-block text (a distinct dark) and the pastel block fills stay
fixed. Writes the result to ../../_includes/leo_cleo_transformer.svg, which the
blog post inlines via {% include %} (currentColor only inherits when inlined).

Usage:
    python3 postprocess.py both_stz.svg
"""
import os
import re
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "both_stz.svg"
out = os.path.join(os.path.dirname(__file__), "..", "..", "_includes",
                   "leo_cleo_transformer.svg")

s = open(src).read()
s = s[s.index("<svg"):]                                   # drop xml declaration
s = re.sub(r'<rect [^>]*fill="rgb\(100%, 100%, 100%\)"[^>]*/>\s*', '', s)  # no bg

s = s.replace("rgb(0%, 0%, 0%)", "currentColor")          # ink -> theme colour
s = s.replace('fill="rgb(95.292664%, 95.292664%, 95.684814%)" fill-opacity="1"',
              'fill="currentColor" fill-opacity="0.05"')   # N x box
s = s.replace('stroke="rgb(81.999207%, 81.999207%, 81.999207%)" stroke-opacity="1"',
              'stroke="currentColor" stroke-opacity="0.16"')  # divider

s = re.sub(r'(<svg\b[^>]*?)\s+width="[\d.]+"\s+height="[\d.]+"',
           r'\1 style="width:100%;height:auto;max-width:860px;display:block;margin:0 auto"',
           s, count=1)
if 'role="img"' not in s:
    s = s.replace('<svg ', '<svg role="img" aria-label="LEO and CLEO in the '
                  'transformer style." ', 1)

open(out, "w").write(s)
print("wrote", os.path.normpath(out), "(%d bytes)" % len(s))
