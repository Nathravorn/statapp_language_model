CC=pdflatex
BIB=bibtex
PDF=statapp
REF=statapp

statapp.pdf: $(PDF).tex
	$(CC) $(PDF).tex
	$(BIB) $(REF).aux
	$(CC) $(PDF).tex
	$(CC) $(PDF).tex

clean:
	rm $(PDF).aux $(PDF).out $(PDF).pdf
