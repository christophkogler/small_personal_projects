#A small gradio webui application to provide an interface for rolling powers from the Celestial Forge.

import csv
import random
import gradio as gr
import math

# this should work on ANY csv file with the approximate organization: column 2 = cost (digits, digits & letters), column 4 = domain (string)
# this can probably(?) function on a mostly-empty 6 column csv... more or less jank, though.

# output will be more or less jank depending on how well it fits the organization:
# column 0 = ID, 1 = Name, 2 = Cost, 3 = Source, 4 = Domain, 5 = Description, 6 = notes

#MY MAGIC NUMBERS!
weighting_power = 2 # 0 - inf; weighting towards/from high point cost items. 0 = no weighting.

def load_csv_to_list(file_name):
    data = []
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print("File not found")
    return data

all_rows = load_csv_to_list("CelestialForgeV3-ALL-fixed.csv")

# for each row: 0 = ID, 1 = Name, 2 = Cost, 3 = Source, 4 = Domain, 5 = Description, 6 = notes
headers, data_rows = all_rows[0], all_rows[1:]
# data_rows = all_rows[1:]
roll_history = []  # Global list to keep track of roll history
domains = sorted(list(set(row[4] for row in data_rows)))  # get a sorted list of the domains

def power_row_stringifier(row):  # convert a power row into a string
    return f"{row[1]} - {row[4]}\n{row[2]}\n{row[5]}"

def load_previous_rolls():
    global roll_history
    try:
        with open("previous_rolls.csv", 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            roll_history = [row for row in reader]
    except FileNotFoundError:
        print("No previous roll history found.")
    return "\n".join([power_row_stringifier(roll) for roll in roll_history]), roll_history

def save_rolls():
    with open("previous_rolls.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(roll_history)

def roll_power(categorylist, points, fail_toggle):
	#
	viable_rolls = [row for row in data_rows if row[4] in categorylist and row not in roll_history]  # filter to ensure we don't use anything already in roll_history again
	if viable_rolls:
		if fail_toggle:
			#failable_rolls = [row for row in viable_rolls if row not in roll_history]  # filter to ensure we don't use anything already in roll_history again
			chosen_row = random.choice(viable_rolls)  # Pick completely randomly from all un-used rows
			if int(''.join(filter(str.isdigit, chosen_row[2]))) > points:
				roll_history.append(["N/A", "Failed roll.", "0" , "N/A", chosen_row[4], "N/A", "N/A"])
				return "Failed roll", points, "\n".join([power_row_stringifier(roll) for roll in roll_history]), roll_history
		else:
			viable_rolls = [row for row in data_rows if int(''.join(filter(str.isdigit, row[2]))) <= points]
			if weighted_toggle:
				# Extract point costs and calculate weights
				point_costs = [int(''.join(filter(str.isdigit, row[2]))) for row in viable_rolls]  # clean up the messy string values in row[2] to clean integers
				total_points = sum(point_costs)
				if total_points > 0:
					weights = [pow((cost / total_points) + 1, weighting_power) for cost in point_costs]  # Calculate weights (higher point costs will have significantly higher weights)
					chosen_row = random.choices(viable_rolls, weights=weights, k=1)[0]  # Choose a single weighted random item
				else:
					chosen_row = random.choice(viable_rolls)  # if rolling at 0, just pick at random.
			else:
				chosen_row = random.choice(viable_rolls)  # Pick randomly without weighting
		roll_history.append(chosen_row)  # Append the roll to the history
		power_string = power_row_stringifier(chosen_row)  # stringify for display
		updated_points = points - int(''.join(filter(str.isdigit, chosen_row[2])))  # Subtract the cost of the chosen roll from the points
		return power_string, updated_points, "\n".join([power_row_stringifier(roll) for roll in roll_history]), roll_history  # return roll string, update points, convert roll_history to a pretty string and update
	else:
		return "No viable roll found"

def process_word_count(word_count, ratio):
    return math.floor(word_count / ratio) * 100

def undo_last_roll(points):
	if roll_history:
		removed_roll = roll_history.pop()
		updated_points = points + int(''.join(filter(str.isdigit, removed_roll[2])))  # Add the cost of the removed roll back to the points
		updated_roll_history_string = "\n".join([power_row_stringifier(roll) for roll in roll_history])
		return updated_roll_history_string, roll_history, updated_points
	else:
		return "", [], points

with gr.Blocks() as page:
	with gr.Row():
		with gr.Column(scale=1):
			words = gr.Number(label="Word count")
			ratio = gr.Number(label="N words:100 points")
			pointsbtn = gr.Button("Calculate points")
			points = gr.Number(label="Points")
		with gr.Column(scale=5):
			domain_selector = gr.Dropdown(choices=domains, label="Included domains", multiselect=True, value=domains)
			with gr.Row():
				with gr.Column(scale=4):
					roll_button = gr.Button("Roll")
					undo_button = gr.Button("Undo a Roll")
				with gr.Column(scale=1):
					fail_toggle = gr.Checkbox(label="Allow Failed Rolls", info="Randomly draw without filtering for point cost. This can 'fail'. If disabled, all rolls will succeed.")
					weighted_toggle = gr.Checkbox(label="Use Weighted Rolls", info="Weight rolls to tend towards higher point cost. 'Failed rolls' overrides this back to random draws (or else you'd get almost all failures).")
	with gr.Row():
		with gr.Column():
			roll_result = gr.Textbox(label="Roll Result", interactive=False)
			roll_history_box = gr.Textbox(label="Roll History", interactive=False)
	with gr.Row():
		roll_history_sheet = gr.Dataframe(roll_history, headers=headers, label="Roll History")
	with gr.Row():
		load_rolls_button = gr.Button("Load Previous Rolls")
		save_rolls_button = gr.Button("Save Current Rolls")
	# define what the buttons actually do
	pointsbtn.click(fn=process_word_count, inputs=[words, ratio], outputs=points)
	roll_button.click(fn=roll_power, inputs=[domain_selector, points, fail_toggle], outputs=[roll_result, points, roll_history_box, roll_history_sheet])
	undo_button.click(fn=undo_last_roll, inputs=[points], outputs=[roll_history_box, roll_history_sheet, points])
	load_rolls_button.click(fn=load_previous_rolls, inputs=[], outputs=[roll_history_box, roll_history_sheet])
	save_rolls_button.click(fn=save_rolls, inputs=[], outputs=[])

page.launch(server_port=9191)
