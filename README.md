# Unsupervised Aspect-Based Sentiment Analysis through LLM: A Case Study of an Unlabeled Portuguese Beer Database

This repository contains the methodology tasks of the paper above.


### ABSTRACT 

Aspect-Based Sentiment Analysis (ABSA) enables the identification of user preferences toward specific entities in text. One of the motivations for this study is the lack of research applying ABSA with Large Language Models (LLMs) to unlabeled datasets, particularly in Portuguese. This work applies ABSA and general Sentiment Analysis (SA) to Portuguese-language beer reviews from one Brazilian website, focusing on Beer Characteristics (BC) such as aroma, flavor, and visual attributes. Due to the unlabeled nature of the dataset, an unsupervised LLM approach was adopted, testing Sabiá-3 and GPT-4 mini with zero-shot, one-shot, and few-shot prompting. A representative 22-review subset was used to evaluate configurations, with the optimal setup (Sabiá-3, one review example, all BC examples) achieving 69.09% precision, 63.33% recall, and an F1-score of 66.08%. This configuration was applied to the full dataset with 467,431 BC records. Results revealed caramel, fruity, refreshing, and high drinkability as the most positively associated BC, while watery and low drinkability were the most negative. Temporal analysis identified growth in IPA and Russian Imperial Stout styles. Findings confirm that LLM-based ABSA in Portuguese can capture nuanced consumer preferences, offering actionable insights for the Brazilian craft beer market despite dataset and reproducibility limitations.


### Methodology 

ASBA techniques were used in beer evaluation texts to identify the feeling of each CC. AS techniques were applied to identify the overall sentiment of the review. The following figure presents the overall workflow: 
- Step 1 and Step 2 performs the tasks of data collection, analysis and pre-processing. 
- Step 3 creates the Main Base by selection of valid assessments. 
- Step 4 identifies and performs the BC identification, classification and sentiment analysis of each BC review (ASBA tasks).
- Step 5 performes the general SA of the reviews in the Main Base. 
- Step 6 generates the Final bases through joining of the Main Base and the bases resulting from ABSA and SA.
- Step 7 generates results using these bases.

![Fig1  Fluxograma](https://github.com/user-attachments/assets/7596b35a-f8a8-4214-a70b-813ae33bbb99)


## General instructions

For running all the Steps, you will need to get API keys from the LLMs OpenAI and Maritaca. 
The Steps that need the LMMs are Step 3, Step 4 and Step 5.

To get you keys please visit the following sites:
- https://platform.openai.com/docs/quickstart
- https://github.com/maritaca-ai/maritalk-api


Follow the instructions in the next sections for Installing, Configuring the system, Downloading the dataset and Running.
After these steps you can check the results as showed in the Results section.


## Installing

Before going ahead, install a python 3.11 version. 
The following steps will create a python virtual environment to not affect your system.

Get the code from this repository and install the python requirements.
~~~
cd ~/
git clone https://github.com/deniseiras/ACADEMIC_ABSA_beer
cd ~/ACADEMIC_ABSA_beer
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
~~~

Now, clone openai_api and maritaca_api and install them:
~~~
cd ~/
git clone https://github.com/deniseiras/PORTFOLIO_py_openai_api.git
cd ~/PORTFOLIO_py_openai_api
pip install -r requirements.txt
cd ~/
git clone https://github.com/deniseiras/PORTFOLIO_py_maritaca_api.git
cd ~/PORTFOLIO_py_maritaca_api
pip install -r requirements.txt
~~~


## Configuring the system

Configure the Python paths:
~~~bash
cd ~/ACADEMIC_ABSA_beer
export PYTHONPATH=./:../PORTFOLIO_py_openai_api/:../PORTFOLIO_py_maritaca_api/
~~~

Create a env file named .env in your ~/ACADEMIC_ABSA_beer and and set your working path.
~~~bash
mkdir ~/ABSA_beer__workdir
cd ~/ACADEMIC_ABSA_beer
touch .env
~~~

Edit the file .env to the following content:
~~~
WORK_DIR="~/ABSA_beer__workdir"
~~~

Create a file named '.env' in ~/PORTFOLIO_py_openai_api/ to set a Open AI license. i.e.:
~~~bash
OPENAI_API_KEY=sk-........................
~~~

Create a file named '.env' in ~/PORTFOLIO_py_maritaca_api/ setting your license. i.e.:

~~~bash
MARITACAAI_API_KEY=123........................
~~~


## Downloading the dataset 

Download your data set at the same directory defined in the WORK_DIR parameter in .env file.

**TODO PUT THE DATASET IN THE HUGGING FACE**

~~~bash
cd ~/ABSA_beer__workdir
wget XXX 
~~~

## Running

The main program is based at absa_beer/absa_beer.py , which calls from Step1 to Step 7.
The Step 1 collects data from the site brejas.com.br, Step 2 does the pre-processing and the Step 3 generates the Main Base.

However, the base could be obtained by downloading it from :

**TODO DOWNLOAD BASE SITE**

Each Step [number] has its own processing function called "run", which generates a dataset called "step_[number].csv". Each step uses the bases of the previous step.

All the datasets was previous available as shown in the "Downloading the datasets" section, so 
you may not need to run all the steps. I. e. the Steps from Step 1 to Step 3 are the most time demanding, and you shouldn't change this methods, otherwise you will have a differente dataset, because the site brejas.com.br might be updated with new reviews. 

If you want to run from the Step 4 (ABSA), you need to comment the calling of the Steps 1 to 3 in absa_beer.py file

To run all the Steps (except the commented calls):
~~~
python absa_beer/absa_beer.py 
~~~

If you need further assistance, please do no hesitate to contact me.


## Results

Each step generates the file step_[number].csv, containing the dataset resulting from each step.

The Steps 6 and 7 generates output texts and figures that demonstrate the final results.


