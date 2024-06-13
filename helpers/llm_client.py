from loguru import logger
from openai import OpenAI
from langchain_groq import ChatGroq
import pandas as pd
import os
import shutil
import random
import string
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential
from ratelimit import limits
from helpers.models import Models
import nirjas

class LLMClient():
    """
    This class provides a unified interface for interacting with various Large Language Models (LLMs)
    from different providers (Groq, NVIDIA, Together AI). It handles rate limiting, retries,
    and error logging for reliable inference.

    Supported Models:
        - Llama 3 8b (Groq)
        - Mistral 7b (Together AI)
        - Phi 3 mini, small (NVIDIA)
        - Gemma 1 7b (Groq)
    """

    def __init__(self):
        """
        Initializes clients for all supported LLM models. Requires the API keys to be environment variables
        """
        
        # Groq Doesn't support dynamic models (as far as I'm aware?) you have to provide model in the ChatGroq function
        # So I'll create separate clients for each GroqModel
        self.lamma3_8b_client = ChatGroq(
            groq_api_key = os.getenv('GROQ_API_KEY'),
            model =  Models.LLAMA_3_8b.value
        )

        self.gemma_7b_client = ChatGroq(
            groq_api_key = os.getenv('GROQ_API_KEY'),
            model = Models.GEMMA_7b.value 
        )

        self.nvidiaClient = self.model = OpenAI(
            base_url = "https://integrate.api.nvidia.com/v1",
            api_key= os.getenv('NVIDIA_API_KEY'),
        )

        self.togetherClient = self.client = OpenAI(
            base_url='https://api.together.xyz/v1',
            api_key=os.getenv('TOGETHER_API_KEY'),
        )

        self.random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
        self.error_log_file_name = os.path.join('logs', f"logs_{self.random_id}.log")
        logger.add(self.error_log_file_name)
        self.logger = logger

    def _get_rate_limit(self, model: Models):
        """
        Gets the rate limit (requests per minute) for a given model.

        Args:
            model: The LLM model.

        Returns:
            The rate limit as an integer.
        """
        if model.name == Models.LLAMA_3_8b.name or model.name == Models.GEMMA_7b.name:
            return 28
        elif model.name == Models.MISTRAL_7b.name:
            return 58
        elif model.name == Models.PHI_3_MINI.name or model.name == Models.PHI_3_SMALL.name:
            return 55
        else:
            raise Exception(f'Unrecognized model: {model.name}')
    
    def _extract_comments(self, data):
        """Extracts and concatenates all comments from keys containing "comment"."""
        all_comments = []

        for key, values in data.items():
            if "comment" in key:  # Check if the key has "comment" in it
                # Handle both list and single entry cases
                if isinstance(values, list):
                    for entry in values:
                        all_comments.append(entry["comment"])
                else:
                    all_comments.append(values)

        return "\n".join(all_comments)
            
    def _infer(self, model : Models, prompt : str, temperature = 0):
        """
        Performs inference with the specified LLM model, handling retries and rate limiting.

        Args:
            model: The LLM model to use.
            prompt: The text prompt for the model.
            temperature: The sampling temperature (0 for deterministic output).

        Returns:
            The model's response text.
        """

        rate_limit = self._get_rate_limit(model)

        @retry(reraise=True, wait=wait_random_exponential(min=60, max=120), stop=stop_after_attempt(5))
        @limits(calls=rate_limit, period=60)
        def _internal_func():
            if model.name == Models.LLAMA_3_8b.name:
                self.lamma3_8b_client.temperature = temperature
                response = self.lamma3_8b_client.invoke(prompt).content
            elif model.name == Models.GEMMA_7b.name:
                self.gemma_7b_client.temperature = temperature
                response = self.gemma_7b_client.invoke(prompt).content
            elif model.name == Models.MISTRAL_7b.name:
                chat_completion = self.togetherClient.chat.completions.create(
                    model=Models.MISTRAL_7b.value,
                    messages=[
                        {'role': 'user', 'content': prompt}
                    ],
                    n=1,
                    temperature=temperature,
                )
                response = chat_completion.choices[0].message.content
            elif model.name == Models.PHI_3_MINI.name or model.name == Models.PHI_3_SMALL.name:
                chat_completion = self.nvidiaClient.chat.completions.create(
                    model=model.value,
                    messages=[
                        {'role': 'user', 'content': prompt}
                    ],
                    n=1,
                    temperature=temperature,
                )
                response = chat_completion.choices[0].message.content
            else:
                raise Exception(f'Unrecognized model: {model.name}')

            return response
        
        return _internal_func()

    # TODO: Remove later
    def temp_function(self, df: pd.DataFrame):
        for index, row in df.iterrows():
            try: 
                comments = nirjas.extract(os.path.join('extras', row['file path']))
                comments = self._extract_comments(comments)
            except:
                with open(os.path.join('extras', row['file path']), "r") as f:
                    comments = f.read()
            df.loc[index, 'file_comments'] = comments
        return df

    
    def process_dataset(self, df : pd.DataFrame, df_path : str, model : Models, prompt_function,
                        parser, temperature = 0, output_path='results', log_every=0,
                        output_name=None, retry_fails=True, extra_file_path='extras'):
        
        """
        Processes a dataset of texts, generating responses using the specified LLM model.

        Args:
            ... (See original code for detailed parameter descriptions)

        Returns:
            A tuple containing two DataFrames:
                - The original DataFrame with 'response' column added.
                - The DataFrame with parsed responses (if parser is provided).
        """

        for index, row in df.iterrows():

            try: 
                comments = nirjas.extract(os.path.join(extra_file_path, row['file path']))
                prompt = prompt_function(self._extract_comments(comments))
            except:
                with open(os.path.join(extra_file_path, row['file path']), "r") as f:
                    comments = f.read()
                prompt = prompt_function(comments)

            if log_every > 0:
                if index % log_every == 0:
                    self.logger.info(f"Processing index: {index}")  
            try:
                df.loc[index, 'response'] = self._infer(model, prompt, temperature)
            except Exception as e:
                self.logger.error(f"Unhandled exception at index: {index}, Exception: {e}")
        
        error_indices = df[df['response'].isna()].index
        
        if retry_fails:
            idx = 0
            while len(error_indices) != 0:
                idx += 1
                if idx == 5:
                    break
                for index in error_indices:
                    
                    try: 
                        comments = nirjas.extract(os.path.join(extra_file_path, df.loc[index, 'file path']))
                        prompt = prompt_function(self._extract_comments(comments))
                    except:
                        with open(os.path.join(extra_file_path, df.loc[index, 'file path']), "r") as f:
                            comments = f.read()
                        prompt = prompt_function(comments)
                    for attempt in range(5):
                        try:
                            df.loc[index, 'response'] = self._infer(model, prompt, temperature)
                            self.logger.debug(f"Exception at index: {index} was retried successfully")
                            break
                        except:
                            time.sleep(0.5)


        modelName = model.name
        

        if output_name:
            df.to_csv(os.path.join(output_path, f'{output_name}.csv'))
            shutil.copyfile(''+self.error_log_file, f'{output_name}.log')
        else:
            output_name = df_path.split('.csv')[0] + f'-{modelName}.csv'
            df.to_csv(os.path.join(output_path, output_name))
            shutil.copyfile(''+self.error_log_file, f'{output_name}.log')
        
        open(''+self.error_log_file, 'w').close()

        df_remaining = df.copy()
        df_remaining = df_remaining[df_remaining['response'].notna()]

        # for index, row in df_remaining.iterrows():
        #     df_remaining.loc[index, 'response_parsed'] = parser(row['response'])

        # #calculate metrics & confusion matrix

        # report, matrix = calculate_metrics(dataset,
        #     response_converted= df_remaining['response_converted'], ground_truth=df_remaining['label'], gt_uncleaned=df['label'])

        return df