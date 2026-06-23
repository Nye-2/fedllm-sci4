LATEXMK ?= latexmk
MAIN ?= main.tex

.PHONY: all clean watch

all:
	$(LATEXMK) -pdf $(MAIN)

watch:
	$(LATEXMK) -pdf -pvc $(MAIN)

clean:
	$(LATEXMK) -C $(MAIN)
