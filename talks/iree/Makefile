# Talk deck: index.qmd (IREE community meeting)
#
#   make          build the diagrams and render the deck
#   make figures  only rebuild the diagrams (TikZ -> SVG)
#   make clean    remove rendered output and built diagrams

QMD  := index.qmd
FIGS := stack.svg pipeline.svg dispatcher.svg graph-break.svg timeline.svg frontends.svg

.PHONY: all render figures clean

all: render

render: index.html

figures: $(FIGS)

index.html: $(QMD) $(FIGS) styles.css step-numbers.html
	quarto render $(QMD)

%.svg: %.tex
	pdflatex -interaction=nonstopmode -halt-on-error $< > /dev/null
	pdf2svg $*.pdf $@
	rm -f $*.aux $*.log $*.pdf

clean:
	rm -rf index.html index_files $(FIGS)
