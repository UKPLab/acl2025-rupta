from typing import List, Union, Optional, Literal
import dataclasses

from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
import openai
import os
from openai import OpenAI
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI
import credentials as credentials

MessageRole = Literal["system", "user", "assistant"]


@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])


# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def gpt_completion(
        model: str,
        prompt: str,
        max_tokens: int = 1024,
        stop_strs: Optional[List[str]] = None,
        temperature: float = 0.0,
        num_comps=1,
) -> Union[List[str], str]:
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        # max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
        # n=num_comps,
    )
    if num_comps == 1:
        return response.choices[0].text  # type: ignore

    return [choice.text for choice in response.choices]  # type: ignore


# @retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def gpt_chat(
    client,
    model: str,
    messages: List[Message],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    num_comps=1,
    logprobs=False
):

    response = client.chat.completions.create(
        model=model,
        messages=[dataclasses.asdict(message) for message in messages],
        # max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        logprobs=logprobs
        # response_format={"type": "json_object"}
        # n=num_comps,
    )
    if num_comps == 1:
        return response.choices[0].message.content, response.usage, response.choices[0].finish_reason  # type: ignore

    return [choice.message.content for choice in response.choices]  # type: ignore


class ModelBase():
    def __init__(self, name: str):
        self.name = name
        self.is_chat = False
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def __repr__(self) -> str:
        return f'{self.name}'

    def generate_chat(self, messages: List[Message], format_instructions: str, parser, max_tokens: int = 1024, temperature: float = 0.2,
                      num_comps: int = 1, logprobs: bool = False) -> dict:
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None, temperature: float = 0.0, num_comps=1) -> Union[List[str], str]:
        raise NotImplementedError

    def get_langchain_model(self, temperature: float = 0.0):
        raise NotImplementedError

    def print_usage(self):
        raise NotImplementedError


class GPTChat(ModelBase):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.name = model_name
        self.is_chat = True
        self.client = None

    def generate_chat(self, messages: List[Message], format_instructions, parser, max_tokens: int = 1024, temperature: float = 0.0,
                      num_comps: int = 1, logprobs: bool = False):
        response = self.client.chat.completions.create(
            model=self.name,
            messages=[dataclasses.asdict(message) for message in messages],
            # max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logprobs=logprobs
            # response_format={"type": "json_object"}
            # n=num_comps,
        )
        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        output_finish_reason = response.choices[0].finish_reason
        output_text = response.choices[0].message.content
        try:
            output_dict = parser.invoke(output_text)
            output_dict['parse_success'] = True
        except Exception as e:
            messages.extend(
                [
                    Message(
                        role="assistant",
                        content=output_text,
                    ),
                    Message(
                        role="user",
                        content=format_instructions + f"\n\nWhen I parse your output, I got this error: {e}"
                    )
                ]
            )
            response_2 = self.client.chat.completions.create(
                model=self.name,
                messages=[dataclasses.asdict(message) for message in messages],
                # max_tokens=max_tokens,
                temperature=temperature,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                logprobs=logprobs
                # response_format={"type": "json_object"}
                # n=num_comps,
            )
            self.prompt_tokens += response_2.usage.prompt_tokens
            self.completion_tokens += response_2.usage.completion_tokens
            output_retry_finish_reason = response_2.choices[0].finish_reason
            output_retry_text = response_2.choices[0].message.content
            try:
                output_dict = parser.invoke(output_retry_text)
                output_dict['parse_success'] = True
            except:
                output_dict = {'parse_success': False}
            output_dict['retry_finish_reason'] = output_retry_finish_reason
            output_dict['raw_response'] = output_retry_text
        output_dict['finish_reason'] = output_finish_reason
        if 'raw_response' not in output_dict.keys():
            output_dict['raw_response'] = output_text

        return output_dict


class GPT4(GPTChat):
    def __init__(self, name):
        super().__init__(name)
        if name == "gpt-4":
            self.endpoint = credentials.gpt4_endpoint
            self.api_key = credentials.gpt4_api_key
            self.api_version = credentials.gpt4_api_version
        else:
            assert name == "gpt4-turbo-128k" or name == "gpt-4-turbo-preview"
            self.endpoint = credentials.gpt4_tb_endpoint
            self.api_key = credentials.gpt4_tb_api_key
            self.api_version = credentials.gpt4_tb_api_version
        self.client = OpenAI(
            api_key=self.api_key
        )
        
        # self.client = OpenAI(
        #     api_key=self.api_key,
        #     base_url=self.endpoint
        # )

    def get_langchain_model(self, temperature: float = 0.0):
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            deployment_name=self.name,
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_type="azure",
            temperature=temperature
        )

    def print_usage(self):
        if self.name == "gpt-4":
            print(
                f"*******{self.name}*******\nPrompt tokens number: {self.prompt_tokens}\n"
                f"Completion tokens number: {self.completion_tokens}. "
                f"Full price: {self.prompt_tokens / 1000 * 0.03 + self.completion_tokens / 1000 * 0.06}\n\n")
        else:
            assert self.name == "gpt4-turbo-128k" or self.name == "gpt-4-turbo-preview"
            print(
                f"*******{self.name}*******\nPrompt tokens number: {self.prompt_tokens}\n"
                f"Completion tokens number: {self.completion_tokens}. "
                f"Full price: {self.prompt_tokens / 1000 * 0.01 + self.completion_tokens / 1000 * 0.03}\n\n")


class GPT35(GPTChat):
    def __init__(self, name):
        super().__init__(name)
        self.endpoint = credentials.gpt35_endpoint
        self.api_key = credentials.gpt35_api_key
        self.api_version = credentials.gpt35_api_version
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )

    def get_langchain_model(self, temperature: float = 0.0):
        return AzureChatOpenAI(
            azure_endpoint=self.endpoint,
            openai_api_version=self.api_version,
            deployment_name=self.name,
            openai_api_key=self.api_key,
            openai_api_type="azure",
            temperature=temperature
        )

    def print_usage(self):
        print(
            f"*******{self.name}*******\nPrompt tokens number: {self.prompt_tokens}\n"
            f"Completion tokens number: {self.completion_tokens}. "
            f"Full price: {self.prompt_tokens / 1000000 * 0.5 + self.completion_tokens / 1000000 * 1.5}\n\n")


class OpenChat(GPTChat):
    def __init__(self, name):
        super().__init__(name)
        self.endpoint = credentials.openllm_endpoint
        self.api_key = credentials.openllm_api_key
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.endpoint
        )

    def get_langchain_model(self, temperature: float = 0.0):
        return ChatOpenAI(
            openai_api_base=self.endpoint,
            model=self.name,
            openai_api_key=self.api_key,
            temperature=temperature
        )

    def print_usage(self):
        print(
            f"*******{self.name}*******\nPrompt tokens number: {self.prompt_tokens}\n"
            f"Completion tokens number: {self.completion_tokens}. ")


class GPTDavinci(ModelBase):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.name = model_name

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None, temperature: float = 0, num_comps=1) -> Union[List[str], str]:
        return gpt_completion(self.name, prompt, max_tokens, stop_strs, temperature, num_comps)


class HFModelBase(ModelBase):
    """
    Base for huggingface chat models
    """

    def __init__(self, model_name: str, model, tokenizer, eos_token_id=None):
        super().__init__(model_name)
        self.name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        self.is_chat = True

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        # NOTE: HF does not like temp of 0.0.
        if temperature < 0.0001:
            temperature = 0.0001

        prompt = self.prepare_prompt(messages)

        outputs = self.model.generate(
            prompt,
            max_new_tokens=min(
                max_tokens, self.model.config.max_position_embeddings),
            use_cache=True,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            eos_token_id=self.eos_token_id,
            num_return_sequences=num_comps,
        )

        outs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        assert isinstance(outs, list)
        for i, out in enumerate(outs):
            assert isinstance(out, str)
            outs[i] = self.extract_output(out)

        if len(outs) == 1:
            return outs[0]  # type: ignore
        else:
            return outs  # type: ignore

    def prepare_prompt(self, messages: List[Message]):
        raise NotImplementedError

    def extract_output(self, output: str) -> str:
        raise NotImplementedError


class StarChat(HFModelBase):
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceH4/starchat-beta",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceH4/starchat-beta",
        )
        super().__init__("starchat", model, tokenizer, eos_token_id=49155)

    def prepare_prompt(self, messages: List[Message]):
        prompt = ""
        for i, message in enumerate(messages):
            prompt += f"<|{message.role}|>\n{message.content}\n<|end|>\n"
            if i == len(messages) - 1:
                prompt += "<|assistant|>\n"

        return self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

    def extract_output(self, output: str) -> str:
        out = output.split("<|assistant|>")[1]
        if out.endswith("<|end|>"):
            out = out[:-len("<|end|>")]

        return out


class CodeLlama(HFModelBase):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    def __init__(self, version: Literal["34b", "13b", "7b"] = "34b"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            f"codellama/CodeLlama-{version}-Instruct-hf",
            add_eos_token=True,
            add_bos_token=True,
            padding_side='left'
        )
        model = AutoModelForCausalLM.from_pretrained(
            f"codellama/CodeLlama-{version}-Instruct-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        super().__init__("codellama", model, tokenizer)

    def prepare_prompt(self, messages: List[Message]):
        if messages[0].role != "system":
            messages = [
                Message(role="system", content=self.DEFAULT_SYSTEM_PROMPT)
            ] + messages
        messages = [
            Message(role=messages[1].role, content=self.B_SYS +
                    messages[0].content + self.E_SYS + messages[1].content)
        ] + messages[2:]
        assert all([msg.role == "user" for msg in messages[::2]]) and all(
            [msg.role == "assistant" for msg in messages[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        messages_tokens: List[int] = sum(
            [
                self.tokenizer.encode(
                    f"{self.B_INST} {(prompt.content).strip()} {self.E_INST} {(answer.content).strip()} ",
                )
                for prompt, answer in zip(
                    messages[::2],
                    messages[1::2],
                )
            ],
            [],
        )
        assert messages[-1].role == "user", f"Last message must be from user, got {messages[-1].role}"
        messages_tokens += self.tokenizer.encode(
            f"{self.B_INST} {(messages[-1].content).strip()} {self.E_INST}",
        )
        # remove eos token from last message
        messages_tokens = messages_tokens[:-1]
        import torch
        return torch.tensor([messages_tokens]).to(self.model.device)

    def extract_output(self, output: str) -> str:
        out = output.split("[/INST]")[-1].split("</s>")[0].strip()
        return out
