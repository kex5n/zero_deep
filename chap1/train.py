from trainer import *
from optimizer import Adam
from simple_cbow import *
from util import *
import numpy as np

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = "you say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()
