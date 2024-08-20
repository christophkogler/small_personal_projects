THIS FILE IS MACHINE GENERATED. IT IS NOT GAURANTEED TO BE CORRECT, ONLY LIKELY TO BE.

# small_personal_projects\simple_readme_generator
## Requirements
- `pip install requests`
- An OpenAI API-compatible text generation backend at localhost:5000 (I recommend [this one](https://github.com/oobabooga/text-generation-webui) ).
- A Llama 3/3.1 model.
## Description
This directory contains a simple script to generate README.md files for directories and their subdirectories using a text generation API. The project requires the requests and tkinter libraries for functionality.

# small_personal_projects\simple_readme_generator\test
## Description
This directory contains a single empty folder named 'test'.

# completions_docs.txt
## Description
This file contains the API documentation for the Oobabooga API, specifically the completions endpoint. It outlines the parameters and options available for generating completions using the API.

## Parameters
The API accepts a variety of parameters to customize the generation of completions. These include:

- `model`: The ID of the model to use for generation.
- `prompt`: The prompt(s) to generate completions for, which can be a string, array of strings, array of tokens, or array of token arrays.
- `best_of`: The number of candidate completions to generate, with the best one returned based on log probability.
- `echo`: Whether to echo back the prompt in addition to the completion.
- `frequency_penalty`: A penalty value between -2.0 and 2.0 to discourage repetition.
- `logit_bias`: A JSON object mapping token IDs to bias values to modify the likelihood of certain tokens.
- `logprobs`: The number of log probabilities to include in the response.
- `max_tokens`: The maximum number of tokens to generate.
- `n`: The number of completions to generate for each prompt.
- `presence_penalty`: A penalty value between -2.0 and 2.0 to encourage new topics.
- `seed`: A seed value for deterministic sampling.
- `stop`: Sequences to stop generating tokens at.
- `stream`: Whether to stream back partial progress.
- `suffix`: A suffix to append to the completion.
- `temperature`: The sampling temperature between 0 and 2.
- `top_p`: The top probability mass to consider for nucleus sampling.
- `user`: A unique identifier for the end-user.

## Sampler Options
The API also supports various sampler options, including:

- `sampler_priority`: An array of sampler application order.
- `preset`: An Oobabooga preset to load.
- `dynamic_temperature`: Dynamic temperature settings.
- `repetition_penalty`: A repetition penalty value.
- `epsilon_cutoff`: An epsilon cutoff value.
- `eta_cutoff`: An eta cutoff value.
- `guidance_scale`: A guidance scale value.
- `negative_prompt`: A negative prompt.
- `penalty_alpha`: A penalty alpha value.
- `mirostat_mode`: A mirostat mode value.
- `mirostat_tau`: A mirostat tau value.
- `mirostat_eta`: A mirostat eta value.
- `temperature_last`: Whether to apply temperature after samplers.
- `do_sample`: Whether to perform sampling.
- `seed`: An inference seed.
- `encoder_repetition_penalty`: An encoder repetition penalty value.
- `no_repeat_ngram_size`: A no repeat ngram size value.
- `dry_multiplier`: A dry multiplier value.
- `dry_base`: A dry base value.
- `dry_allowed_length`: A dry allowed length value.
- `dry_sequence_breakers`: Dry sequence breakers.
- `truncation_length`: A truncation length value.
- `max_tokens_second`: A maximum streaming tokens per second value.
- `prompt_lookup_num_tokens`: A prompt lookup number of tokens value.
- `custom_token_bans`: Custom token bans.
- `auto_max_new_tokens`: Whether to automatically set a maximum number of new tokens.
- `ban_eos_token`: Whether to prevent the model from outputting the end of response token.
- `add_bos_token`: Whether to add the beginning of sequence token.
- `skip_special_tokens`: Whether to skip special tokens.
- `grammar_string`: A grammar string.

# readme_generator_v2.py
## Description
A simple script to generate a README.md for a directory and all its subdirectories. It uses a text generation API to generate the descriptions.

# requirements.txt
## Description
This file contains a list of dependencies required to run the Simple Readme Generator project. The listed packages are:

- requests: A Python library for making HTTP requests.
- tkinter: A Python library for creating graphical user interfaces.

These dependencies are necessary for the project's functionality and can be installed using pip.