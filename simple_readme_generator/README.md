## Requirements
- `pip install requests`
- An OpenAI API-compatible text generation backend at localhost:5000 (I recommend [this one](https://github.com/oobabooga/text-generation-webui) ).
- A Llama 3/3.1 model.

# small_personal_projects\simple_readme_generator
## Description
This directory contains a collection of files and subdirectories related to a simple README generator project. The project includes a documentation file for the completions API of the Obooga model, a language model developed by OpenAI, which allows users to generate text based on a prompt with various parameters. Additionally, there is a Python script, 'readme_generator_v2.py', that generates a README for a directory and its subdirectories. The script can be run from anywhere and can be controlled via command line arguments. The project also includes a test directory, 'test', which currently contains an empty folder named 'test'.

# small_personal_projects\simple_readme_generator\test
## Description
This directory contains a test folder named 'test', which is currently empty.

# completions_docs.txt
## Description
This file contains the documentation for the completions API endpoint of the Obooga model, a language model developed by OpenAI. The API allows users to generate text completions based on a given prompt, with various parameters to control the output, such as the model to use, prompt, and the number of completions to generate. It also includes options for controlling the model's behavior, such as frequency and presence penalties, and the ability to stream the output. The documentation provides detailed explanations of each parameter and its effects on the output, as well as examples of how to use the API effectively.

# readme_generator_v2.py
## Description
This script generates a README for a directory, as well as all its subdirectories. It can be run from anywhere, with the directory to be processed selected via a file picker GUI via tkinter. Command line arguments can be used to filter directories and files from being processed.

## Requirements
- pip install requests
- An OpenAI API-compatible text generation backend at localhost:5000 (I recommend [this one](https://github.com/oobabooga/text-generation-webui) ).
- A Llama 3/3.1 model

## Usage
1. Run the script from anywhere.
2. Select the directory to be processed using the file picker GUI.
3. The script will generate a README for the selected directory and all its subdirectories. The README will be saved in the selected directory as 'README.md'.