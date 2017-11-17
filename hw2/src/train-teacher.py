
# coding: utf-8

# In[ ]:


from utils import *


# In[ ]:


captions_train_trimmed = []

for captions in captions_train_filtered:
    keep_captions = []
    for caption in captions:
        keep = True

        for word in caption.split(' '):
            if word not in output_lang.word2index:
                keep = False
                break

        # Remove if pair doesn't match input and output conditions
        if keep:
            keep_captions.append(caption)
    
    captions_train_trimmed.append(keep_captions)

L1 = sum([len(captions) for captions in captions_train_trimmed])
L2 = sum([len(captions) for captions in captions_train_filtered])
print("Trimmed from %d pairs to %d, %.4f of total" % (L2, L1, L1 / L2))


# In[ ]:


clip = 5.0

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio, max_length=MAX_LENGTH):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Get size of input and target sentences
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable)
    
    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:
        
        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di] # Next target is next input

    else:
        # Without teacher forcing: use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            
            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            
            decoder_input = Variable(torch.LongTensor([[ni]])) # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            if ni == EOS_token: break

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.data[0] / target_length


# In[ ]:


def evaluate_on_train(index, encoder, decoder, max_length=80):
    input_variable = X_train[index]
    input_variable = Variable(torch.FloatTensor(input_variable))
    if USE_CUDA: input_variable = input_variable.cuda()
    
    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable)
    
    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[SOS_token]])) # SOS
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()
    
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    
    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_context, decoder_hidden, encoder_outputs
        )
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
            
        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()
    
    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]


# In[ ]:


import math

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


# In[ ]:


attn_model = 'general'
hidden_size = 256
n_layers = 1
dropout_p = 0.05
MODEL_DIR = '../models/teacher-3/'

# Initialize models
encoder = EncoderRNN(num_features, hidden_size, n_layers)
decoder = AttnDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p)

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

# Initialize optimizers and criterion
learning_rate = 0.001
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()


# In[ ]:


# Configuring training
n_epochs = 50000
plot_every = 200
print_every = 100
save_every = 1000

# Keep track of time elapsed and running averages
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every


# In[ ]:


# Begin!
import time
import random

start = time.time()

for epoch in range(1, n_epochs + 1):
    teacher_forcing_ratio = 1 - epoch / n_epochs
    # Get training data for this cycle
    index = random.randrange(num_videos)
    input_variable = X_train[index]
    input_variable = Variable(torch.FloatTensor(input_variable))
    if USE_CUDA: input_variable = input_variable.cuda()
    
    captions = captions_train_trimmed[index]
    num_captions = len(captions)
    caption = captions[epoch % num_captions]
    target_variable = variable_from_sentence(output_lang, caption)
    
    # Run the train function
    loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio)

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss

    if epoch == 0: continue

    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
        print(print_summary)
        
        encoder.eval()
        decoder.eval()
        output_words, decoder_attn = evaluate_on_train(index, encoder, decoder)
        output_sentence = ' '.join(output_words[:-1])
        print('Truth:   ', captions[epoch % num_captions])
        print('Predict: ', output_sentence)
        encoder.train()
        decoder.train()
        with open('./log.txt', 'a') as log_f:
            log_f.write(print_summary + '\n')
            log_f.write('Truth:   ' + captions[epoch % num_captions] + '\n')
            log_f.write('Predict: ' + output_sentence + '\n')
        
    if epoch % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0
    
    if epoch % save_every == 0:
        encoder.eval()
        decoder.eval()
        torch.save(encoder, MODEL_DIR + 'encoder_epoch{}.sd'.format(str(epoch)))
        torch.save(decoder, MODEL_DIR + 'decoder_epoch{}.sd'.format(str(epoch)))
        encoder.train()
        decoder.train()


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

show_plot(plot_losses)


# In[ ]:


def evaluate(id_, max_length=80):
    index = filenames_test.index(id_ + '.npy')
    input_variable = X_test[index]
    input_variable = Variable(torch.FloatTensor(input_variable))
    if USE_CUDA: input_variable = input_variable.cuda()
    
    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[SOS_token]])) # SOS
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()
    
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    
    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_context, decoder_hidden, encoder_outputs
        )
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
            
        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()
    
    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]


# In[ ]:


def postprocess(words):
    for i in range(len(words) - 1, -1, -1):
        if words[i] == '.':
            continue
        if words[i] == 'a' and words[i-1] == 'a':
            continue
        break
    
    return words[:i+1]

print(postprocess(['a', 'boy', 'is', 'a', 'a', '.', '.']))


# In[ ]:


encoder.eval()
decoder.eval()

with open('output_5.csv', 'w') as f:
    for i in range(num_videos_test):
        id_ = filenames_test[i][:-4]
        output_words, decoder_attn = evaluate(id_)
        output_sentence = ' '.join(postprocess(output_words[:-1]))
        f.write(id_ + ',' + output_sentence + '\n')

encoder.train()
decoder.train()

