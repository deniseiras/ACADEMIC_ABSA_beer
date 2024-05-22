"""
Step 1: Data collection from the site "brejas.com.br"

:author: Denis Eiras

Functions:
    - 
"""

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
import dotenv
import os
from datetime import datetime
import pandas as pd
from step import Step


class Step_1(Step):

    def __init__(self) -> None:
        super().__init__()
        
        self.dic_decode = {}
        self.dic_decode['Aroma'] = 'review_aroma'
        self.dic_decode['Aparência'] = 'review_visual'
        self.dic_decode['Sabor'] = 'review_flavor'
        self.dic_decode['Sensação'] = 'review_sensation'
        self.dic_decode['Conjunto'] = 'review_general_set'
        self.dic_decode['Avaliação Geral'] = 'review_general_rate'
        
        self.max_page_reviews = 999999
        self.max_beer_page = 2000        
        
        
    # Function to get beer details
    def get_beer_details(self, beer_url):
        # Make a request to the beer page
        beer_response = requests.get(beer_url)
        beer_soup = BeautifulSoup(beer_response.content, 'html.parser')

        dic_beer_details = {}
        # Beer
        field_val = beer_soup.find('h1', {'class': 'contentheading'})
        if field_val is None:
            return
        dic_beer_details['beer_name'] = field_val.find('span', {'itemprop': 'name'}).text.strip()
        
        # Brewery
        field_row = beer_soup.find('div', {'class': 'jrCervejaria jrFieldRow'})
        if field_row is not None:
            field_val = field_row.find('div', {'class': 'jrFieldValue'})
            brewery_name = field_val.find('span', {'itemprop': 'brand'}).text.strip()
            try:
                brewery_url = field_val.find('a')['href']
            except:
                try:
                    brewery_url = beer_soup.find('a', string='Visite o website')['href']
                except:
                    brewery_url = ""
        else:
            brewery_name = ""
            brewery_url = ""
        dic_beer_details['beer_brewery_name'] = brewery_name
        dic_beer_details['beer_brewery_url'] = brewery_url
            
        # Style
        field_row = beer_soup.find('div', {'class': 'jrEstilo jrFieldRow'})
        if field_row is None: # some bug on page
            dic_beer_details = {}
            return dic_beer_details
        
        field_val = field_row.find('div', {'class': 'jrFieldValue'})
        dic_beer_details['beer_style'] = field_val.find('a').text.strip()

        # Alcohol
        field_row = beer_soup.find('div', {'class': 'jrAlcool jrFieldRow'})
        field_val = field_row.find('div', {'class': 'jrFieldValue'})
        dic_beer_details['beer_alcohol'] = field_val.text.strip()

        # Active (SIM/NAO)
        field_row = beer_soup.find('div', {'class': 'jrAtiva jrFieldRow'})
        field_val = field_row.find('div', {'class': 'jrFieldValue'})
        dic_beer_details['beer_is_active'] = field_val.find('a').text.strip()

        # Sazonal
        field_row = beer_soup.find('div', {'class': 'jrSazonal jrFieldRow'})
        field_val = field_row.find('div', {'class': 'jrFieldValue'})
        dic_beer_details['beer_is_sazonal'] = field_val.find('a').text.strip()
        
        # Cor SRM
        field_row = beer_soup.find('div', {'class': 'jrSrm jrFieldRow'})
        if field_row is not None:
            field_val = field_row.find('div', {'class': 'jrFieldValue'})
            dic_beer_details['beer_srm'] = field_val.text.strip()
        else:
            dic_beer_details['beer_srm'] = ""
        
        # IBU
        field_row = beer_soup.find('div', {'class': 'jrIbu jrFieldRow'})
        if field_row is not None:
            field_val = field_row.find('div', {'class': 'jrFieldValue'})
            dic_beer_details['beer_ibu'] = field_val.text.strip()
        else:
            dic_beer_details['beer_ibu'] = ""
            
        # Ingredients
        field_row = beer_soup.find('div', {'class': 'jrIngredientes jrFieldRow'})
        if field_row is not None:
            field_val = field_row.find('div', {'class': 'jrFieldValue'})
            dic_beer_details['beer_ingredients'] = field_val.text.strip()
        else:
            dic_beer_details['beer_ingredients'] = ""    
        
        return dic_beer_details


    # Function to get user reviews
    def get_beer_reviews(self, beer_url): 
        
        review_page = 1
        beer_reviews = []
        
        full_url = ''
        while review_page < self.max_page_reviews:
            
            # if looping for some reason
            if f'{beer_url}/avaliacoes?page={review_page}' == full_url:
                review_page +=1

            full_url = f'{beer_url}/avaliacoes?page={review_page}'
            print(f'Loading {full_url} ...')
            try:
                # initialize the web scrapper with the current page number
                response = requests.get(full_url)
                soup = BeautifulSoup(response.content, 'html.parser')        
            except:
                print(f'No more pages. Last = {review_page-1}')
                break

            # Iterate over all reviwes on page
            review_items = soup.find_all('div', {'class': 'jr-layout-outer jrRoundedPanel'})
            print(f'Review itens = {len(review_items)}')
            if len(review_items) == 0:
                print(f'No reviews found')
                break
            
            # Iterate over all reviews
            for item in review_items:
                dic_a_beer_review = {}
                
                # review_user
                field_row_user = item.find('div', {'class': 'jrUserInfoText'})
                user = field_row_user.find('a')
                if user is None:
                    continue
                
                user = field_row_user.find('a').text.strip()
                dic_a_beer_review["review_user"] = user
                
                # review_num_reviews
                field_reviews = field_row_user.find('span', {'class': 'jrReviewerReviews'})
                num_reviews_str = field_reviews.find('a').text.strip()
                num_reviews = int(num_reviews_str.split()[0])
                dic_a_beer_review["review_num_reviews"] = num_reviews
                
                # review content div        
                field_row_review_content = item.find('div', {'class': 'jrReviewLayoutRight jrReviewContent'})
                
                # review_datetime
                time_tag = field_row_review_content.find('time', {'class': 'jrReviewCreated'})
                datetime_str = time_tag['datetime']
                review_datetime = datetime.fromisoformat(datetime_str)
                dic_a_beer_review["review_datetime"] = review_datetime
                
                # review_aroma, review_visual, review_flavor, review_sensation, review_general_set, review_general_rate
                field_row_review_rating = item.find('div', {'class': 'jrRatingTable fwd-table'})
                if field_row_review_rating is None:
                    continue

                rating_rows = field_row_review_rating.find_all('div', class_='fwd-table-row')
                for row in rating_rows:
                    label = row.find('div', class_='jrRatingLabel').text.strip()
                    label = self.dic_decode[label]
                    value = row.find('div', class_='jrRatingValue').text.strip()
                    dic_a_beer_review[label] = value
            
                # review_comment
                field_row_review_comment = field_row_review_content.find('div', {'class': 'description jrReviewComment'})
                if field_row_review_comment is not None:
                    field_row_review_comment_div = field_row_review_comment.find('div')
                    review_comment = field_row_review_comment_div.text.strip()
                    review_comment = review_comment.replace("\n", " ")
                    review_comment = review_comment.replace("\r", " ")
                    review_comment = review_comment.replace("\t", " ")
                    review_comment = review_comment.replace("  ", " ")
                else:
                    review_comment = ""
                dic_a_beer_review["review_comment"] = review_comment

                beer_reviews.append(dic_a_beer_review)
                if item == review_items[-1]:
                    review_page +=1
                
                
        return beer_reviews


    # Main function
    def run(self):
        print("Starting ...")

        pais = 'brasil'
        root_url = 'https://www.brejas.com.br'
        country_url = f'{root_url}/cerveja/{pais}'
        order_type = 'rdate'  # most recent
        # order_type = 'reviews'  # most reviews
        
        self.df = None
        initial_beer_page = 1
        
        # now = datetime.now()
        # now_str = now.strftime("%Y%m%d-%H%M%S")
        with open(f'{self.work_dir}/step_1.csv', 'a') as f:
            while initial_beer_page <= self.max_beer_page:
                full_url = f'{country_url}/?page={initial_beer_page}&order={order_type}'
                print(f'Loading {full_url} ...')
                try:
                    # initialize the web scrapper with the current page number
                    response = requests.get(full_url)
                    soup = BeautifulSoup(response.content, 'html.parser')        
                except:
                    print(f'No more pages. Last = {initial_beer_page-1}')
                    break

                # Iterate over all beers on page
                beer_items = soup.find_all('div', {'class': 'jrListingTitle'})
                if len(beer_items) == 0:
                    print(f'No more beers found')
                    break
                # loop through beer_items
                for item in beer_items:
                    # finf beer url
                    beer_url = root_url
                    beer_url += item.find('a')['href']
                    
                    beer_details = [self.get_beer_details(beer_url)]
                    if len(beer_details) == 0:
                        print(f'No details found for {beer_url}')
                        continue

                    beer_reviews = self.get_beer_reviews(beer_url)
                    
                    df_beer_details = pd.DataFrame(beer_details)
                        
                    for review in beer_reviews:
                        df_item = pd.DataFrame()
                        for key, value in review.items():
                            df_item[key] = [value]
                        df_beer_rev = pd.concat([df_beer_details.reset_index(drop=True), df_item.reset_index(drop=True)], axis=1)
                        if self.df is None:
                            self.df = df_beer_rev
                        else: 
                            self.df = pd.concat([self.df, df_beer_rev], ignore_index=True)

                
                    if item == beer_items[-1]:
                        initial_beer_page +=1
                    
                self.df.to_csv(f, header=f.tell()==0, index=False)
                self.df = None
            
        print("Finished running!")
