# Surface-Fuel Classifications Using Tessera Embeddings

## Development

1. Create the environment (or `source` if already created).
	```bash
	python -m venv .venv
	source .venv/bin/activate
	pip install -r requirements.txt
	```
1. When running Jupyter notebooks, I recommend starting a kernel in tmux then attaching VSCode to it:
	- `tmux new -s co-fire` or `tmux a -t co-fire`
	- Then, in tmux:
		```bash
		jupyter lab --no-browser
		```
		and let that run as long as you want. The URL can be used to select this kernel in VSCode.
