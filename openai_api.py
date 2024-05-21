import openai
import dotenv
import os

# Whether your API call works at all, as total tokens must be below the model’s maximum limit (4097 tokens for gpt-3.5-turbo)

# Every response will include a finish_reason. The possible values for finish_reason are:

# stop: API returned complete message, or a message terminated by one of the stop sequences provided via the stop parameter
# length: Incomplete model output due to max_tokens parameter or token limit
# function_call: The model decided to call a function
# content_filter: Omitted content due to a flag from our content filters
# null: API response still in progress or incomplete
# Depending on input parameters, the model response may include different information.



def get_completion(prompt, model="gpt-3.5-turbo"):
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        # response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


# create main function
def main():
    
    # read openai.api_key from openai.env file

    dotenv.load_dotenv('./openai.env')
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    # set the prompt
    prompt = "What is the capital of France?"
    print(get_completion(prompt))

if __name__ == "__main__":
    main()
