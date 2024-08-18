#This script generates a readme for a directory, as well as all subdirectories it has.
#It can be run from anywhere. The directory to be processed will be selected using a file picker gui via tkinter.
#Command line arguments can be used to filter directories and files from being processed: python readme_generator_v2.py arg1 arg2 arg3 ...

import sys
import os
import json
import http
import requests
import tkinter as tk
from tkinter import filedialog

#the settings you will most likely have to change are endpoint url, or the inference parameters in payload

#default is set up for Oobabooga WebUI at localhost. if local generation is not possible, this script should be OpenAI Completions API compatible and relatively easy to change endpoint to any similar API.
def api_call(prompt, endpoint_url = "http://localhost:5000/v1/completions"):

	headers = {
		"Content-Type": "application/json",
		#"Authorization: Bearer $OPENAI_API_KEY"
	}
	
	payload = json.dumps({
		"prompt": prompt,						#what is provided to the model as context. must be formatted depending on model.
		"temperature": .1,					#temperature. higher temperature approaches completely random response. 0 deterministic/nearly, .5-1.5 still reasonable, 2+ usually gibberish
		"max_tokens": 1024,					#max tokens that will generate
		"best_of": 2, 							#generate multiple and return the "best" (the one with the highest log probability per token)
		#"mirostat_mode": 1,
		#"mirostat_tau": 4,
		#"mirostat_eta": 0.1,
	})
	
	response = requests.post(endpoint_url, data = payload, headers = headers) 	#make a post to the endpoint url, providing it our header(s) and inference arguments.
	responseJson = response.json() 																#turn the response into something processable
	response = responseJson["choices"][0]["text"] 												#from choices in response, from the first item (only one by default), get the text
	#print(response) #for testing
	return response

def get_rel_dir_path(path, base_dir_path):
	return os.path.relpath(path, os.path.dirname(base_dir_path))
	
# for a given file_path, generate a readme.txt for the file. base_directory_path is an optional path to a 'base directory' the file is somewhere within.
def generate_readme_for_file(filepath, base_directory_path = None):
	if not base_directory_path:
		base_directory_path = filepath
	relative_path = get_rel_dir_path(filepath, base_directory_path)
	file_name = get_rel_dir_path(filepath, filepath)
	readme_prep_string = f"# {file_name}\n## Description"
	#print(file_name)
	with open(filepath, 'r', encoding="utf8") as file:
		content = file.read()
	if len(content) >= 20000:
		content = content[0:10000] + "Some text omitted to fit context length." + content[-10000:]
	#Llama 3.1 format convention:
	if not content:
		content = "This file is empty."
	system_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n<|eot_id|>"
	user_input = f"<|start_header_id|>user<|end_header_id|>\nWrite a concise description for the following file (located at {relative_path}). Avoid simply repeating the files contents. \n\n{content}\n<|eot_id|>"
	assistant_prep = f"<|start_header_id|>assistant<|end_header_id|>\n{readme_prep_string}"
	prompt = system_prompt + user_input + assistant_prep
	response = api_call(prompt)
	response = readme_prep_string + response
	return response

# provided a list of summaries, create a readme for a given directory. base_directory_path is an optional path to a 'base directory' the directory is somewhere within.
def generate_readme_for_directory(summaries, current_dir_path, base_directory_path = None):
	if not base_directory_path:
		base_directory_path = current_dir_path
	dir_path = get_rel_dir_path(current_dir_path, base_directory_path)
	dir_name = get_rel_dir_path(current_dir_path, current_dir_path)
	#Llama 3.1 format convention:
	summaries = "\n\n".join(summaries)
	readme_prep_string = f"# {dir_path}\n## Description"
	#prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\nWrite a readme for the following folder (located at {dir_path}) based on its contents. The folder contains files and subfolders which are described by the following:\n{summaries}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{readme_prep_string}"
	system_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|>"
	user_input = f"<|start_header_id|>user<|end_header_id|>\n\nProvided below are the summaries of all files and subdirectories in a directory (located at {dir_path}).\n\n{summaries}\n\nBased on the provided summaries, write a description for the directory '{dir_name}'.\n\nDo not simply repeat the provided information. Attempt to summarize and simplify.<|eot_id|>"
	assistant_prep = f"<|start_header_id|>assistant<|end_header_id|>\n\n{readme_prep_string}"
	#if we are in the 'base path', refer to the directory instead as a project
	if (dir_path == get_rel_dir_path(base_directory_path, base_directory_path)):
		system_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|>"
		user_input = f"<|start_header_id|>user<|end_header_id|>\n\nProvided below are the summaries of all files and subdirectories in a project (located at {dir_path}).\n\n{summaries}\n\nBased on the provided summaries, write a description for the directory '{dir_name}'.\n\nDo not simply repeat the provided information. Attempt to summarize and simplify.<|eot_id|>"
		assistant_prep = f"<|start_header_id|>assistant<|end_header_id|>\n\n{readme_prep_string}"
	prompt = system_prompt + user_input + assistant_prep
	response = api_call(prompt)
	if (get_rel_dir_path(current_dir_path, current_dir_path) == "simple_readme_generator"): # if we are IN simple_readme_generator
		simple_readme_generator_requirements = "## Requirements\n- `pip install requests`\n- An OpenAI API-compatible text generation backend at localhost:5000 (I recommend [this one](https://github.com/oobabooga/text-generation-webui) ).\n- A Llama 3/3.1 model.\n"
		readme_prep_string = f"# {dir_path}\n" + simple_readme_generator_requirements + f"## Description"
	return readme_prep_string + response

# generate readme's for a given directory. base_dir is an optional path to a directory the processed directory is somewhere within.
def process_directory_tree(dir_path, base_dir = None):
	if not base_dir:
		base_dir = dir_path
	diritems = os.listdir(dir_path)
	#filter additional folder/file names as necessary for your environment.
	#ignore readmes for now, .git folder, pycache(s)
	blacklist = ['readme.txt', 'readme.md', '.git', '__pycache__'] 
	diritems = [item for item in diritems if item.lower() not in blacklist] #filter for common things we DONT want (or can't process)
	diritems = [item for item in diritems if item not in sys.argv[1:]] #accept command line arguments to filter arbitrary additional amount of directory/file names from processing.
	
	if not diritems:
		return f"#{get_rel_dir_path(dir_path, base_dir)}\n##Description\nThis folder is empty."
		
	clean_summaries = []
	
	dir_summaries = []
	subdirectories = [d for d in diritems if os.path.isdir(os.path.join(dir_path, d))]
	for subdirectory in subdirectories:
		print(f"Processing subdirectory {subdirectory}...")
		subdirectory_path = os.path.join(dir_path, subdirectory)
		relative_path = get_rel_dir_path(subdirectory_path, base_dir)
		result = process_directory_tree(subdirectory_path, base_dir)
		dir_summaries.append(f"{relative_path}:\n{result}")
		clean_summaries.append(result)
	
	print(f"Processing files in {get_rel_dir_path(dir_path, base_dir)}...")
	
	directory_readme = ""
	readme_path_lower = os.path.join(dir_path, 'readme.md')
	readme_path_upper = os.path.join(dir_path, 'README.md')
	# if there is already a readme in the directory, just use it. Possibly some wasted effort in lower directories, but human written readme > ai slop almost all the time
	if os.path.isfile(readme_path_lower): # if there is a readme already in the file
		with open(readme_path_lower, 'r', encoding="utf-8") as dir_file:
			directory_readme = dir_file.read()
	elif os.path.isfile(readme_path_upper): # uppercase README
		with open(readme_path_upper, 'r', encoding="utf-8") as dir_file:
			directory_readme = dir_file.read()

	# if there is not a readme in the directory or it is machine generated, summarize any files, make a readme, and save it.
	if directory_readme == "" or directory_readme.startswith("THIS FILE IS MACHINE GENERATED."): 
		file_summaries = []
		files = [f for f in diritems if os.path.isfile(os.path.join(dir_path, f))]
		for file in files:
			print(f"Processing file {file}...")
			file_path = os.path.join(dir_path, file)
			result = generate_readme_for_file(file_path, base_dir)
			relative_path = get_rel_dir_path(file_path, base_dir)
			file_summaries.append(f"{relative_path}:\n{result}")
			clean_summaries.append(result)

		relativepath_summaries = dir_summaries + file_summaries
	
		directory_readme = ""
		readme_path = os.path.join(dir_path, 'README.md')
		directory_readme = generate_readme_for_directory(relativepath_summaries, dir_path, base_dir)
		with open(readme_path, 'w') as readme_file:
			directory_write_readme = "THIS FILE IS MACHINE GENERATED. IT IS NOT GAURANTEED TO BE CORRECT, ONLY LIKELY TO BE.\n\n"+directory_readme
			readme_file.write(directory_write_readme + "\n\n" + "\n\n".join(clean_summaries))
    
	return directory_readme

def main():
	root = tk.Tk()
	root.withdraw()
	directory_path = filedialog.askdirectory()
	print(f"Location of selected folder: {directory_path}")
	#print(os.listdir(directory_path))
	print(process_directory_tree(directory_path))

if __name__ == "__main__":
	main()