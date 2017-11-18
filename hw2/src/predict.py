
# coding: utf-8

# In[ ]:


from utils import *
import os
import sys
import numpy


# In[ ]:


MODEL_DIR = '../models/'
encoder = torch.load(os.path.join(MODEL_DIR, 'encoder_epoch50000.sd'))
decoder = torch.load(os.path.join(MODEL_DIR, 'decoder_epoch50000.sd'))

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

# Set to not-training mode to disable dropout
encoder.eval()
decoder.eval()


# In[ ]:


def evaluate(input_variable, max_length=80):
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


# In[ ]:


data_dirname, id_filename, data_subdirname, out_filename = sys.argv[1:]

ids = []
with open(os.path.join(data_dirname, id_filename)) as f:
    for line in f:
        line = line.strip()
        if len(line) > 4:
            ids.append(line)

print('Start predicting ...')
with open(out_filename, 'w') as f:
    for id_ in ids:
        filepath = os.path.join(data_dirname, data_subdirname, 'feat', id_ + '.npy')
        X = np.load(filepath)
        output_words, decoder_attn = evaluate(X)
        output_sentence = ' '.join(postprocess(output_words[:-1]))
        f.write(id_ + ',' + output_sentence + '\n')
print('Finish predicting.')

