CC=pdflatex
BIB=bibtex
PDF=project
REF=project

statapp.pdf: $(PDF).tex
	$(CC) $(PDF).tex
	$(BIB) $(REF).aux
	$(CC) $(PDF).tex
	$(CC) $(PDF).tex

clean:
	rm $(PDF).aux $(PDF).out $(PDF).pdf
