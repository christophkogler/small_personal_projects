#based on the tutorial video by Andrej Karpathy @ https://www.youtube.com/watch?v=kCc8FmEb1nY

import torch
import random
import torch.nn as nn
from torch.nn import functional as F
from grokfast.grokfast import gradfilter_ema

#--------------training / inference hyperparameters---------
maximum_context_length = 1024
parallel_batch_size = 16

embedding_dimensions = 384 #must be modulo-able by head_count
head_count = 6
transformer_layers = 6

if embedding_dimensions % head_count != 0:
	pause = raw_input("HEAD COUNT MUST CLEANLY DIVIDE EMBEDDING DIMIENSIONS.")
	raise SystemExit()

max_inference_tokens = 1024

max_training_iters = 5000
learning_rate = 3e-4
dropout = 0.2

#grokfast
grokfast_ema_alpha = 0.98 # Momentum hyperparmeter of the EMA.
grokfast_ema_lambda = 2.0	# Amplifying factor hyperparameter of the filter

training_evaluation_interval = 50	#how often to evaluate loss? low number = slower
loss_evaluation_iterations = 10	#how many evalutions to do? big number = slower, more precise loss evaluation
generate_evaluation_iters = 1000					#how often to generate to test

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#if you run into cuda errors, consider switching to cpu mode while debugging.
#this can help by exposing errors with more readable traces.
#device = 'cpu'  

torch.manual_seed(8181)
#----------------------------------------------------------------------------------


#get the dataset... 8:00
with open('tiny-shakespeare.txt', 'r', encoding ='utf-8') as file:
	text = file.read()

print(f"length of dataset in characters: {len(text)}")

unique_chars = sorted(list(set(text))) #get unique characters
vocabulary_size = len(unique_chars) #count to get vocab size
print(f"{vocabulary_size} unique characters in dataset:\n{''.join(unique_chars)}")

#we are making an extremely simple, character level tokenization method
	#simple, small vocab size, but long input sequences
#most real-world work uses more complex tokenizers with sub-word multi-character tokens.
	#this makes their vocabulary much larger, but makes token input sequences shorter
#10:00
chartointegers = {ch:i for i, ch in enumerate(unique_chars)}
encode = lambda string: [chartointegers[c] for c in string] # convert a list of characters (string) to a list of integers
integertochars = {i:ch for i, ch in enumerate(unique_chars)}
decode = lambda list: ''.join([integertochars[i] for i in list]) # convert a list of integers to a list of characters, then return as string
#encoded_hello = encode("hello world!")
#print(encoded_hello)
#print(decode(encoded_hello))

#validation and training sets
#13:30
train_percent_as_float = 0.9
text_as_tensor = torch.tensor(encode(text), dtype=torch.long)
training_index = int(train_percent_as_float*len(text_as_tensor))
training_data = text_as_tensor[:training_index]
validation_data = text_as_tensor[training_index:]

#15:00
learning_context = training_data[:maximum_context_length]
learning_target = training_data[1:maximum_context_length+1]

for time in range(maximum_context_length):
	context = learning_context[:time+1]
	target = learning_target[time]
	#print(f"when input is {context} the target is: {target}")

# data loading
def get_training_batch(split='train'):
	# generate a small batch of data of contexts and targets
	data = training_data if split == 'train' else validation_data
	batch_random_data_indexes = torch.randint(len(data) - maximum_context_length, (parallel_batch_size,)) # get parallel_batch_size amount of random ints from 0 to (length of dataset - context)
	context = torch.stack([data[i:i+maximum_context_length] for i in batch_random_data_indexes]) #get random chunks of dataset for context
	target = torch.stack([data[i+1:i+maximum_context_length+1] for i in batch_random_data_indexes]) #get random chunk of dataset for targets
	context, target = context.to(device), target.to(device)
	return context, target

#40:00
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(loss_evaluation_iterations)
        for iteration in range(loss_evaluation_iterations):
            context, targets = get_training_batch(split)
            logits, loss = model(context, targets)
            losses[iteration] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#32:00
def easy_generate(model):
	data_index = random.randint(0, len(validation_data) - maximum_context_length//8)
	context = validation_data[data_index:data_index+maximum_context_length//8]
	context = context.unsqueeze(0)  # Add batch dimension
	context = context.to(device)	# send to cuda
	#empty_context = torch.zeros((1,1), dtype = torch.long, device = device)
	inference = model.generate(context)
	response_as_string = decode(inference[0].tolist())
	return response_as_string

def easy_report(model):
	model.to(device)
	print(easy_generate(model))
	train_model(model)
	print(easy_generate(model))

#35:00
def train_model(model, training_iters = max_training_iters):
	grads = None
	optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate) #create optimizer for the model
	for iteration in range(training_iters):									# train for total_steps
		if iteration % training_evaluation_interval == 0:
			losses = estimate_loss(model)
			print(f"step {iteration}/{training_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
		if iteration % generate_evaluation_iters == 0 and iteration != 0:
			print(easy_generate(model))
		context, targets = get_training_batch() 					# get context and targets
		logits, loss = model(context, targets) 						# examine logits and loss when context and targets are provided to the model
		optimizer.zero_grad(set_to_none = True)				# ???
		loss.backward()													# ???
		grads = gradfilter_ema(model, grads=grads, alpha=grokfast_ema_alpha, lamb=grokfast_ema_lambda)
		optimizer.step() 													# ???
	print(loss.item())

inputs, targets = get_training_batch()

"""
print(f"inputs: \n{inputs.shape}\n{inputs}")
print(f"targets: \n{targets.shape}\n{targets}")
print('---------')
for batches in range(parallel_batch_size):
	for time in range(maximum_context_length):
		context = learning_context[:time+1]
		target = learning_target[time]
		print(f"when input is {context} the target is: {target}")

#22:30
# super simple bigram model

class BigramLanguageModelv1(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocabulary_size, vocabulary_size)
    def forward(self, index, targets=None):
        logits = self.token_embedding_table(index)
        return logits


model = BigramLanguageModelv1()
model = model.to(device)
output = model(inputs, targets)
print(output.shape)


#25:00

class BigramLanguageModelv2(nn.Module):
	def __init__(self):
		super().__init__()
		# each token directly reads off the logits for the next token from a lookup table
		self.token_embedding_table = nn.Embedding(vocabulary_size, vocabulary_size)
	def forward(self, context, targets=None):
		logits = self.token_embedding_table(context)
		if targets is None:
			loss = None
		else:
			BATCH, TOKENS, CHANNELS = logits.shape
			logits = logits.view(BATCH*TOKENS, CHANNELS)  #convert logits to 2-dimensional
			targets = targets.view(BATCH*TOKENS)  #convert targets to 1-dimensional
			loss = F.cross_entropy(logits, targets)
		return logits, loss
		
	def generate(self, context, max_new_tokens = max_inference_tokens):
		# idx is (B, T) array of indices in the current context
		for _ in range(max_new_tokens):
			logits, loss = self(context)																	# get the predictions
			logits = logits[:, -1, :] 												# becomes (B, C)	# focus only on the last time step
			probs = F.softmax(logits, dim=-1) 								# (B, C)					# apply softmax to get probabilities
			idx_next = torch.multinomial(probs, num_samples=1) 	# (B, 1)					# sample from the distribution
			context = torch.cat((context, idx_next), dim=1)		 	# (B, T+1)				# append sampled index to the running sequence
		return context

bigram_v2 = BigramLanguageModelv2()
bigram_v2 = bigram_v2.to(device)
#logits, loss = bigram_v2(inputs, targets)

print(easy_generate(bigram_v2))
train_model(bigram_v2)
print(easy_generate(bigram_v2))



#42:00
#matrix multiplication
if False: #keep this stuff local, toggle on/off
	BATCH, TIME, CHANNELS = 4, 8, 2
	context = torch.randn(BATCH, TIME, CHANNELS)
	print(context[0])
	print(context.shape)
	
	context_bagofwords = torch.zeros((BATCH,TIME,CHANNELS))
	#averaging past context w/ for loops; inefficient and weak
	for batch_count in range(BATCH):
		for time_step in range(TIME):
			context_prev = context[batch_count, :time_step+1]
			context_bagofwords[batch_count, time_step] = torch.mean(context_prev, 0)
	print(context_bagofwords[0])
	
	#52:00
	token_weights = torch.tril(torch.ones(TIME, TIME))
	token_weights = token_weights / token_weights.sum(1, keepdim=True)
	context_bagofwords_2 = token_weights @ context # (TIME,TIME) @ (BATCH,TIME,CHANNELS) -> pytorch automagic -> (BATCH, TIME, TIME) @ (BATCH, TIME, CHANNELS)
	print(context_bagofwords_2[0])
	print(f"averages are comparable: {torch.allclose(context_bagofwords, context_bagofwords_2)}")
	
	#54:00
	tril = torch.tril(torch.ones(TIME, TIME))
	token_weights2 = torch.zeros((TIME,TIME))
	token_weights2 = token_weights2.masked_fill(tril == 0, float('-inf'))
	token_weights2 = F.softmax(token_weights2, dim = -1)
	context_bagofwords_3 = token_weights2 @ context # (TIME,TIME) @ (BATCH,TIME,CHANNELS) -> pytorch automagic -> (BATCH, TIME, TIME) @ (BATCH, TIME, CHANNELS)
	print(context_bagofwords_3[0])
	print(f"averages are comparable: {torch.allclose(context_bagofwords, context_bagofwords_3)}, {torch.allclose(context_bagofwords_2, context_bagofwords_3)}")
	
	if True: #a way to toggle the prints on/off
		#how to make self-attention more efficient - the 'mathematical trick', matrix multiplication
		#47:00
		matrix_a_dims = (3,3)
		a = torch.ones(matrix_a_dims)
		matrix_b_dims = (3,2)
		b = torch.randint(0,10,matrix_b_dims).float()
		c = a @ b
		print('a=')
		print(a)
		print('b=')
		print(b)
		print('a @ b -> c=')
		print(c)
		a = torch.tril(a)
		print('tril a =')
		print(a)
		print('b=')
		print(b)
		print('a @ b -> c=')
		c = a @ b
		print(c)
		a = a / torch.sum(a, 1, keepdim=True)
		print('a / sum(tril a)=')
		print(a)
		print('b=')
		print(b)
		print('a @ b -> c=')
		c = a @ b
		print(c)
#--

class BigramLanguageModelv3(nn.Module):
	def __init__(self):
		super().__init__()
		# each token directly reads off the logits for the next token from a lookup table
		self.token_embedding_table = nn.Embedding(vocabulary_size, embedding_dimensions)
		
		self.position_embedding_table = nn.Embedding(maximum_context_length, embedding_dimensions)
		
		self.lm_head = nn.Linear(embedding_dimensions, vocabulary_size)
		
	def forward(self, context, targets=None):
		B, T = context.shape
		token_embed = self.token_embedding_table(context)	#(BATCH,TIME,CHANNELS)
		
		position_embed = self.position_embedding_table(torch.arange(T, device=device))	#(TIME,CHANNELS)
		weights = token_embed + position_embed
		logits = self.lm_head(weights)

		#logits = self.lm_head(token_embed)							#(BATCH,TIME,vocabulary_size)
		if targets is None:
			loss = None
		else:
			BATCH, TOKENS, CHANNELS = logits.shape
			logits = logits.view(BATCH*TOKENS, CHANNELS)  #convert logits to 2-dimensional
			targets = targets.view(BATCH*TOKENS)  #convert targets to 1-dimensional
			loss = F.cross_entropy(logits, targets)
		return logits, loss
		
	def generate(self, context, max_new_tokens = max_inference_tokens):
		# idx is (B, T) array of indices in the current context
		for _ in range(max_new_tokens):
			logits, loss = self(context)											# get the predictions
			logits = logits[:, -1, :] 							# becomes (B, C)	# focus only on the last time step
			probs = F.softmax(logits, dim=-1) 					# (B, C)			# apply softmax to get probabilities
			idx_next = torch.multinomial(probs, num_samples=1) 	# (B, 1)			# sample from the distribution
			context = torch.cat((context, idx_next), dim=1)		# (B, T+1)			# append sampled index to the running sequence
			context = context[:, -maximum_context_length:]  						# Keep only the most recent tokens
		return context

#print(nn.Flatten(BigramLanguageModelv3().token_embedding_table))
#print(nn.Flatten(BigramLanguageModelv3().position_embedding_table))
#print(nn.Flatten(BigramLanguageModelv3().lm_head))
bigram_v3 = BigramLanguageModelv3()
bigram_v3.to(device)
easy_report(bigram_v3)
"""

#1:19:00
class Head(nn.Module):
	def __init__(self, head_size):
		super().__init__()
		self.key = nn.Linear(embedding_dimensions, head_size, bias=False)
		self.query = nn.Linear(embedding_dimensions, head_size, bias=False)
		self.value = nn.Linear(embedding_dimensions, head_size, bias=False)
		self.register_buffer('tril', torch.tril(torch.ones(maximum_context_length, maximum_context_length)))
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		# input of size (batch, time-step, channels)
		# output of size (batch, time-step, head size)
		B,T,C = x.shape
		keys = self.key(x)   # (B,T,hs)
		queries = self.query(x) # (B,T,hs)
		# compute attention scores ("affinities")
		weights = queries @ keys.transpose(-2,-1) * keys.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
		weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
		weights = F.softmax(weights, dim=-1) # (B, T, T)
		weights = self.dropout(weights)
		# perform the weighted aggregation of the values
		values = self.value(x) # (B,T,hs)
		out = weights @ values # (B, T, T) @ (B, T, hs) -> (B, T, hs)
		return out
#--

"""
#1:20:00
class BigramLanguageModelv4(nn.Module):
	def __init__(self):
		super().__init__()
		# each token directly reads off the logits for the next token from a lookup table
		self.token_embedding_table = nn.Embedding(vocabulary_size, embedding_dimensions)
		self.position_embedding_table = nn.Embedding(maximum_context_length, embedding_dimensions)
		self.sa_head = Head(embedding_dimensions)
		self.lm_head = nn.Linear(embedding_dimensions, vocabulary_size)
		
	def forward(self, context, targets=None):
		B, T = context.shape
		
		token_embed = self.token_embedding_table(context)	#(BATCH,TIME,CHANNELS)
		position_embed = self.position_embedding_table(torch.arange(T, device=device))	#(TIME,CHANNELS)
		weights = token_embed + position_embed
		weights = self.sa_head(weights)
		logits = self.lm_head(weights)
		
		if targets is None:
			loss = None
		else:
			BATCH, TOKENS, CHANNELS = logits.shape
			logits = logits.view(BATCH*TOKENS, CHANNELS)  #convert logits to 2-dimensional
			targets = targets.view(BATCH*TOKENS)  #convert targets to 1-dimensional
			loss = F.cross_entropy(logits, targets)
		return logits, loss
		
	def generate(self, context, max_new_tokens = max_inference_tokens):
		# idx is (B, T) array of indices in the current context
		for _ in range(max_new_tokens):
			context_cropped = context[:, -maximum_context_length:]  						# Keep only the most recent tokens
			logits, loss = self(context_cropped)											# get the predictions
			logits = logits[:, -1, :] 							# becomes (B, C)	# focus only on the last time step
			probs = F.softmax(logits, dim=-1) 					# (B, C)			# apply softmax to get probabilities
			idx_next = torch.multinomial(probs, num_samples=1) 	# (B, 1)			# sample from the distribution
			context = torch.cat((context, idx_next), dim=1)		# (B, T+1)			# append sampled index to the running sequence
		return context

bigram_v4 = BigramLanguageModelv4()
easy_report(bigram_v4)
"""

#1:22:00
#1:38:00 - add dropout
class MultiHeadAttention(nn.Module):
	def __init__(self, num_heads, head_size):
		super().__init__()
		self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
		self.proj = nn.Linear(head_size * num_heads, embedding_dimensions)
		self.dropout = nn.Dropout(dropout)
	def forward(self, context):
		heads_tensor = torch.cat([head(context) for head in self.heads], dim=-1)
		out = self.dropout(self.proj(heads_tensor))
		return out
#--

"""
#1:23:00
class LanguageModelv5(nn.Module):
	def __init__(self):
		super().__init__()
		# each token directly reads off the logits for the next token from a lookup table
		self.token_embedding_table = nn.Embedding(vocabulary_size, embedding_dimensions)
		self.position_embedding_table = nn.Embedding(maximum_context_length, embedding_dimensions)
		self.sa_heads = MultiHeadAttention(head_count, embedding_dimensions//head_count)
		self.lm_head = nn.Linear(embedding_dimensions, vocabulary_size)
		
	def forward(self, context, targets=None):
		B, T = context.shape
		
		token_embed = self.token_embedding_table(context)	#(BATCH,TIME,CHANNELS)
		position_embed = self.position_embedding_table(torch.arange(T, device=device))	#(TIME,CHANNELS)
		weights = token_embed + position_embed
		weights = self.sa_heads(weights)
		logits = self.lm_head(weights)
		
		if targets is None:
			loss = None
		else:
			BATCH, TOKENS, CHANNELS = logits.shape
			logits = logits.view(BATCH*TOKENS, CHANNELS)  #convert logits to 2-dimensional
			targets = targets.view(BATCH*TOKENS)  #convert targets to 1-dimensional
			loss = F.cross_entropy(logits, targets)
		return logits, loss
		
	def generate(self, context, max_new_tokens = max_inference_tokens):
		# idx is (B, T) array of indices in the current context
		for _ in range(max_new_tokens):
			context_cropped = context[:, -maximum_context_length:]  						# Keep only the most recent tokens
			logits, loss = self(context_cropped)											# get the predictions
			logits = logits[:, -1, :] 							# becomes (B, C)	# focus only on the last time step
			probs = F.softmax(logits, dim=-1) 					# (B, C)			# apply softmax to get probabilities
			idx_next = torch.multinomial(probs, num_samples=1) 	# (B, 1)			# sample from the distribution
			context = torch.cat((context, idx_next), dim=1)		# (B, T+1)			# append sampled index to the running sequence
		return context

lm_v5 = LanguageModelv5()
easy_report(lm_v5)
"""

#1:25:00
#1:32:00 - 4x inner dim
#1:38:00 - add dropout
class FeedForward(nn.Module):
	def __init__(self, embedding_dimensions):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(embedding_dimensions, 4 * embedding_dimensions),
			nn.ReLU(),
			nn.Linear(4 * embedding_dimensions, embedding_dimensions),
			nn.Dropout(dropout),
		)

	def forward(self, context):
		return self.net(context)

"""
#1:25:00
class LanguageModelv6(nn.Module):
	def __init__(self):
		super().__init__()
		# each token directly reads off the logits for the next token from a lookup table
		self.token_embedding_table = nn.Embedding(vocabulary_size, embedding_dimensions)
		self.position_embedding_table = nn.Embedding(maximum_context_length, embedding_dimensions)
		self.sa_heads = MultiHeadAttention(head_count, embedding_dimensions//head_count)
		self.ffwd = FeedForward(embedding_dimensions)
		self.lm_head = nn.Linear(embedding_dimensions, vocabulary_size)
		
	def forward(self, context, targets=None):
		B, T = context.shape
		
		token_embed = self.token_embedding_table(context)	#(BATCH,TIME,CHANNELS)
		position_embed = self.position_embedding_table(torch.arange(T, device=device))	#(TIME,CHANNELS)
		weights = token_embed + position_embed
		weights = self.sa_heads(weights)
		weights = self.ffwd(weights)
		logits = self.lm_head(weights)
		
		if targets is None:
			loss = None
		else:
			BATCH, TOKENS, CHANNELS = logits.shape
			logits = logits.view(BATCH*TOKENS, CHANNELS)  #convert logits to 2-dimensional
			targets = targets.view(BATCH*TOKENS)  #convert targets to 1-dimensional
			loss = F.cross_entropy(logits, targets)
		return logits, loss
		
	def generate(self, context, max_new_tokens = max_inference_tokens):
		# idx is (B, T) array of indices in the current context
		for _ in range(max_new_tokens):
			context_cropped = context[:, -maximum_context_length:]  						# Keep only the most recent tokens
			logits, loss = self(context_cropped)											# get the predictions
			logits = logits[:, -1, :] 							# becomes (B, C)	# focus only on the last time step
			probs = F.softmax(logits, dim=-1) 					# (B, C)			# apply softmax to get probabilities
			idx_next = torch.multinomial(probs, num_samples=1) 	# (B, 1)			# sample from the distribution
			context = torch.cat((context, idx_next), dim=1)		# (B, T+1)			# append sampled index to the running sequence
		return context

lm_v6 = LanguageModelv6()
easy_report(lm_v6)
"""

#1:27:00
#1:30:00 - add residual connections
#1:36:00 - add layernorm
class Block(nn.Module):
	def __init__(self, embedding_dimensions, n_heads = head_count):
		super().__init__()
		head_size = embedding_dimensions//n_heads
		self.sa = MultiHeadAttention(n_heads, head_size)
		self.ffwd = FeedForward(embedding_dimensions)
		self.ln1 = nn.LayerNorm(embedding_dimensions)
		self.ln2 = nn.LayerNorm(embedding_dimensions)

	def forward(self, context):
		#weight = self.sa(context) 
		context = context + self.sa(self.ln1(context)) 
		#weight = self.ffwd(context)
		context = context + self.ffwd(self.ln2(context))
		return context

"""
#1:25:00
#1:37:00 - add layernorm
class LanguageModelv7(nn.Module):
	def __init__(self):
		super().__init__()
		# each token directly reads off the logits for the next token from a lookup table
		self.token_embedding_table = nn.Embedding(vocabulary_size, embedding_dimensions)
		self.position_embedding_table = nn.Embedding(maximum_context_length, embedding_dimensions)
		self.blocks = nn.Sequential(
			Block(embedding_dimensions, head_count),
			Block(embedding_dimensions, head_count),
			Block(embedding_dimensions, head_count),
			nn.LayerNorm(embedding_dimensions),
		)
		self.lm_head = nn.Linear(embedding_dimensions, vocabulary_size)
		
	def forward(self, context, targets=None):
		B, T = context.shape
		
		token_embed = self.token_embedding_table(context)	#(BATCH,TIME,CHANNELS)
		position_embed = self.position_embedding_table(torch.arange(T, device=device))	#(TIME,CHANNELS)
		weights = token_embed + position_embed
		weights = self.blocks(weights)
		logits = self.lm_head(weights)
		
		if targets is None:
			loss = None
		else:
			BATCH, TOKENS, CHANNELS = logits.shape
			logits = logits.view(BATCH*TOKENS, CHANNELS)  #convert logits to 2-dimensional
			targets = targets.view(BATCH*TOKENS)  #convert targets to 1-dimensional
			loss = F.cross_entropy(logits, targets)
		return logits, loss
		
	def generate(self, context, max_new_tokens = max_inference_tokens):
		# idx is (B, T) array of indices in the current context
		for _ in range(max_new_tokens):
			context_cropped = context[:, -maximum_context_length:]  						# Keep only the most recent tokens
			logits, loss = self(context_cropped)											# get the predictions
			logits = logits[:, -1, :] 							# becomes (B, C)	# focus only on the last time step
			probs = F.softmax(logits, dim=-1) 					# (B, C)			# apply softmax to get probabilities
			idx_next = torch.multinomial(probs, num_samples=1) 	# (B, 1)			# sample from the distribution
			context = torch.cat((context, idx_next), dim=1)		# (B, T+1)			# append sampled index to the running sequence
		return context

lm_v7 = LanguageModelv7()
easy_report(lm_v7)
"""

#1:38:00
class LanguageModelv8(nn.Module):
	def __init__(self):
		super().__init__()
		# each token directly reads off the logits for the next token from a lookup table
		self.token_embedding_table = nn.Embedding(vocabulary_size, embedding_dimensions)
		self.position_embedding_table = nn.Embedding(maximum_context_length, embedding_dimensions)
		self.blocks = nn.Sequential(*[Block(embedding_dimensions, head_count) for _ in range(transformer_layers)])
		self.ln_f = nn.LayerNorm(embedding_dimensions)
		self.lm_head = nn.Linear(embedding_dimensions, vocabulary_size)
		
		# better init, not covered in the original GPT video, but important, will cover in followup video
		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
		
	def forward(self, context, targets=None):
		B, T = context.shape
		
		token_embed = self.token_embedding_table(context)	#(BATCH,TIME,CHANNELS)
		position_embed = self.position_embedding_table(torch.arange(T, device=device))	#(TIME,CHANNELS)
		weights = token_embed + position_embed
		weights = self.blocks(weights)
		weights = self.ln_f(weights)
		logits = self.lm_head(weights)
		
		if targets is None:
			loss = None
		else:
			BATCH, TOKENS, CHANNELS = logits.shape
			logits = logits.view(BATCH*TOKENS, CHANNELS)  #convert logits to 2-dimensional
			targets = targets.view(BATCH*TOKENS)  #convert targets to 1-dimensional
			loss = F.cross_entropy(logits, targets)
		return logits, loss
		
	def generate(self, context, max_new_tokens = max_inference_tokens):
		# idx is (B, T) array of indices in the current context
		for _ in range(max_new_tokens):
			context_cropped = context[:, -maximum_context_length:]  						# Keep only the most recent tokens
			logits, loss = self(context_cropped)											# get the predictions
			logits = logits[:, -1, :] 							# becomes (B, C)	# focus only on the last time step
			probs = F.softmax(logits, dim=-1) 					# (B, C)			# apply softmax to get probabilities
			idx_next = torch.multinomial(probs, num_samples=1) 	# (B, 1)			# sample from the distribution
			context = torch.cat((context, idx_next), dim=1)		# (B, T+1)			# append sampled index to the running sequence
		return context

lm_v8 = LanguageModelv8()
lm_v8 = lm_v8.to(device)
print(sum(p.numel() for p in lm_v8.parameters())/1e6, 'M parameters')
easy_report(lm_v8)