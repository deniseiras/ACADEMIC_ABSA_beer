import time

from src.openai_api import get_completion as get_completion_openai
from src.maritacaai_api import get_completion as get_completion_maritacaai


class Prompt_AI:

    LLM_MIN_INTERVAL = 0.2
    MAX_TRIES = 3
    last_call_ts = 0.0

    def __init__(self, model: str, prompt: str):
        self.model = model
        self.prompt = prompt
        self.response = None
        self.finish_reason = None

    def _rate_limit_sleep(self, tries):
        now = time.time()

        if tries == 1:
            sleep_time = self.LLM_MIN_INTERVAL - (now - self.last_call_ts)
            if sleep_time > 0:
                print(f'Sleeping for {sleep_time} seconds before prompting AI...')
                time.sleep(sleep_time)
        else:
            sleep_time = 3 ** tries
            print(f'Prompt failed. Try #{tries} - Sleeping for {sleep_time} seconds before retrying...')
            time.sleep(sleep_time)
                
        self.last_call_ts = time.time()

    def get_completion(self):
        # Throttle antes da chamada

        if self.model not in ['gpt-3.5-turbo-0125', 'gpt-4o-mini', 'gpt-4', 'sabia-2-small', 'sabia-3', 'sabiazinho-4']:
            raise ValueError(f'Unsupported model: {self.model}')
        else:
            tries = 1
            while tries < self.MAX_TRIES:
                self._rate_limit_sleep(tries)
                if self.model in ['gpt-3.5-turbo-0125', 'gpt-4o-mini', 'gpt-4']:
                    self.response, self.finish_reason = get_completion_openai(
                        self.prompt,
                        model=self.model
                    )

                elif self.model in ['sabia-2-small', 'sabia-3', 'sabiazinho-4']:
                    self.response, self.finish_reason = get_completion_maritacaai(
                        self.prompt,
                        model_name=self.model
                    )
                    
                if self.finish_reason != "stop":
                    print(f'Finish reason not expected: {self.finish_reason}')
                    print(self.response)
                    tries += 1            
                    continue
                else:
                    break
                
        return self.response, self.finish_reason
