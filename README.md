# Aspect Based Sentiment Analysis for identifying preferred characteristics of Brazilian beers

This repository contains the methodology tasks of the paper.

Methods:

- Step 1: Data collection from the site "brejas.com.br"
- Step 2: Data preprocessing
- Step 3: Aspect-Based Sentiment Analysis of Beer Characteristics (CC)
- Step 4: General sentiment analysis of comments
- Step 5: Selection of the best and worst reviews in terms of overall rating
- Step 6: Identification of CC and their categories, obtained in Step 3, most referenced in the resulting base of Step 5
- Results generation

TODO: update after text release

### Installing

~~~
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
~~~

Now, clone PORTFOLIO_py_openai_api and use it
~~~
git clone https://github.com/deniseiras/PORTFOLIO_py_openai_api.git
pip install -r ../PORTFOLIO_py_openai_api/requirements.txt
~~~

Configure the paths:
~~~
export PYTHONPATH=./:../PORTFOLIO_py_openai_api/:../PORTFOLIO_py_maritaca_api/
~~~

Create a file named '.env' in ../PORTFOLIO_py_openai_api/ setting your license. i.e.:

~~~bash
OPENAI_API_KEY=sk-........................
~~~

Create a file named '.env' in ../PORTFOLIO_py_maritaca_api/ setting your license. i.e.:

~~~bash
MARITACAAI_API_KEY=123........................
~~~

### Running

Each step has its own processing function called "run", which generates a base called "step_<number>.csv". The following steps uses the bases of the previous step.

~~~
python absa_beer.py 
~~~
