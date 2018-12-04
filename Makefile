clean:
	nbstripout notebooks/*.ipynb
	python resources/yapf_nbformat/yapf_nbformat.py notebooks/*.ipynb
	yapf -r -i codenames/
