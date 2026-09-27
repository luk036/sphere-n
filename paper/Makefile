# Build n-sphere.pdf from n-sphere.md. Run from paper/.
#
# Flag rationale (md-to-latex skill, so the knowledge is not tribal):
#  - pandoc 3.11 no longer reads a positional *.yaml as metadata; it falls back
#    to markdown and leaks the YAML into the body. Hence --metadata-file.
#  - The source uses \(...\) math, so +tex_math_single_backslash is required.
#  - pdflatex cannot embed SVG; `make figs` pre-renders the *.svg twins and
#    svg2pdf.lua points pandoc at them.
#  - IEEEtran is the document class (10pt, journal mode): the canonical IEEE
#    two-column style. IEEEtran is two-column by default, and the wide displays
#    are reflowed in the source so they fit the column.
#  - --csl=ieee.csl vendors the IEEE numeric style locally (self-contained, no
#    independent-parent) so citeproc never reaches zotero.org: the build stays
#    offline and reproducible.

PANDOC := pandoc
FLAGS := -f markdown+tex_math_single_backslash \
         -F pandoc-crossref --citeproc -s -t latex -N --reference-links \
         --csl=ieee.csl \
         --pdf-engine=pdflatex -V documentclass=IEEEtran -V classoption=journal \
         --lua-filter=svg2pdf.lua \
         --metadata-file=latex.yaml --metadata-file=crossref.yaml

SVGS := vdcorput circle halton halton3d sphere disk

.PHONY: paper figs talk clean

paper: figs
	$(PANDOC) $(FLAGS) n-sphere.md -o n-sphere.pdf

figs: $(addsuffix .pdf,$(SVGS))

%.pdf: %.svg
	rsvg-convert -f pdf -o $@ $<

# Beamer talk (md-to-beamer skill): the preamble is loaded from the CLI
# because pandoc Markdown-mangles a YAML header-includes, and pandoc 3.11
# ignores positional *.yaml, so metadata goes through --metadata-file.
# --csl=ieee.csl keeps the deck's citations numeric, matching the paper.
TALKFLAGS := -F pandoc-crossref --citeproc -t beamer --pdf-engine=lualatex \
             --csl=ieee.csl \
             --include-in-header=preamble.tex --metadata-file=n-sphere-talk.yaml

talk:
	$(PANDOC) $(TALKFLAGS) n-sphere-talk.md -o n-sphere-talk.pdf

clean:
	$(RM) $(addsuffix .pdf,$(SVGS)) n-sphere.pdf n-sphere-talk.pdf
