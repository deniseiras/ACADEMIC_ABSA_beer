# Aspect Based Sentiment Analysis for identifying preferred characteristics of Brazilian beers

This repository contains the methodology tasks of the paper.

ASBA techniques were used in beer evaluation texts to identify the feeling of each CC. AS techniques were applied to identify the overall sentiment of the review. The following sections present the Material and
methods, as seen in Figure 1: Step 1 and Step 2 described the tasks of data collection, analysis and pre-processing. Step 3 created the Main Base by selection of valid assessments. Step 4 identified and performed ASBA and Step 5 performed the general AS of the assessment in the Main Base. Step 6 generated Final bases through joining of the Main Base and the bases resulting from ABSA and AS and Step 7 described the processes for generating results using these bases.

![Fig1  Fluxograma](https://github.com/user-attachments/assets/7596b35a-f8a8-4214-a70b-813ae33bbb99)

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
