
# coding: utf-8

# In[ ]:


from utils import *


# In[ ]:


# encoder = EncoderRNN(num_features, hidden_size, n_layers)
# decoder = AttnDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p)

# encoder.load_state_dict(torch.load(MODEL_DIR + 'encoder_epoch100000.sd'))
# decoder.load_state_dict(torch.load(MODEL_DIR + 'decoder_epoch100000.sd'))

MODEL_DIR = '../models/teacher-3/'
encoder = torch.load(MODEL_DIR + 'encoder_epoch100000.sd')
decoder = torch.load(MODEL_DIR + 'encoder_epoch100000.sd')

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

# Set to not-training mode to disable dropout
encoder.eval()
decoder.eval()


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


def evaluate_on_train(index, max_length=80):
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


# special mission

ids = [
    'klteYv1Uv9A_27_33.avi',
    '5YJaS2Eswg0_22_26.avi',
    'UbmZAe5u5FI_132_141.avi',
    'JntMAcTlOF0_50_70.avi',
    'tJHUH9tpqPg_113_118.avi',
]

for id_ in ids:
    output_words, decoder_attn = evaluate(id_)
    output_sentence = ' '.join(output_words[:-1])
    print(output_sentence)
