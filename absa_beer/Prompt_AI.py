from src.openai_api import get_completion as get_completion_openai
from src.maritacaai_api import get_completion as get_completion_maritacaai


class Prompt_AI:
    
    def __init__(self, model: str, prompt: str):
        self.model = model
        self.prompt = prompt
        self.response, self.finish_reason = None, None
        
        
    def get_completion(self):
        if self.model in ['gpt-3.5-turbo-0125', 'gpt-4o-mini', 'gpt-4']:
            self.response, self.finish_reason = get_completion_openai(self.prompt, model=self.model)
        elif self.model in ['sabia-2-small', 'sabia-3']:
            self.response, self.finish_reason = get_completion_maritacaai(self.prompt, model_name=self.model)
        else:
            raise ValueError(f'Unsupported model: {self.model}')
        
        return self.response, self.finish_reason
    
    