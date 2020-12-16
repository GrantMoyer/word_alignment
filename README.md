## Environment ##

Download dictionary and embeddings from the [releases page]. The file *numberbatch-en-##.##.zip* contains both the dictionary and the embeddings. Exctract it to the project directory. The project dependencies are listed in *requirements.txt*, and can be installed in a python virtual environment with:

    > python -mvenv .venv
    > . ./.venv/bin/activate
    > pip install -r requirements.txt

[releases page]: https://github.com/GrantMoyer/word_alignment/releases/new

## Usage ##

```
usage: word_alignment.py [-h] [-d file] [-e file] [-n int] word [word ...]

Given an arbitrary english word, generates an alignment chart of similar
words.

positional arguments:
  word                  The word(s) to generate alignment charts for.

optional arguments:
  -h, --help            show this help message and exit
  -d file, --dictionary file
                        A data file containing the dictionary. By default
                        numberbatch-en-19.08-dictionary.npy.lz4.
  -e file, --embeddings file
                        A data file containing word embeddings of the
                        dictionary. By default numberbatch-
                        en-19.08-embeddings.npy.lz4.
  -n int, --num-neighbors int
                        The number of neighbor words to check for most good,
                        evil, lawful, and chaotic. By default 32.
```
