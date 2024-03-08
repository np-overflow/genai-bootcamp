import discord
import keras
import numpy as np
from discord.ext import commands

BOT_TOKEN = "your token here"
batch_sz = 64
epoch = 100
latent_dim = 256
num_samples = 10000
data_path = "fra.txt"

bot = commands.Bot(command_prefix = "/", intents = discord.Intents.all())

intents = discord.Intents.default()
intents.members = True

model = keras.models.load_model("s2s_model.keras")

input_texts, target_texts = [], []
input_chars, target_chars = set(), set()

with open(data_path, "r", encoding="utf-8") as file:
    lines = file.read().split("\n")

# populate working data
for line in lines[:min(num_samples, len(lines) - 1)]:
    # we use _ to indicate that a variable should be ignored
    # in this case, the 3rd column is the citation (something something), 
    # which is irrelevant to our task at hand
    input_text, target_text, _ = line.split("\t")

    target_text = "\t" + target_text + "\n"
    input_texts.append(input_text)
    target_texts.append(target_text)

    input_chars.update(input_text)
    target_chars.update(target_text)


input_chars, target_chars = sorted(list(input_chars)), sorted(list(target_chars))
num_enc_token, num_dec_token = len(input_chars), len(target_chars)
max_enc_seqlen, max_dec_seqlen = len(max(input_texts, key=len)), \
                                    len(max(target_texts, key=len))

print(f"Number of samples: {len(input_texts)}\n" +
      f"Number of unique input tokens: {num_enc_token}\n" +
      f"Number of unique output tokens: {num_dec_token}\n" +
      f"Max seqlen for input: {max_enc_seqlen}\n" + 
      f"Max seqlen for output: {max_dec_seqlen}")

input_token_index, target_token_index = dict([(char, i) for i, char in enumerate(input_chars)]), \
                                        dict([(char, i) for i, char in enumerate(target_chars)])

encoder_input_data = np.zeros((len(input_texts),
                            max_enc_seqlen, 
                            num_enc_token))

decoder_input_data = np.zeros((len(input_texts),
                                max_dec_seqlen,
                                num_dec_token))

decoder_target_data = np.zeros((len(input_texts),
                                max_dec_seqlen,
                                num_dec_token))

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    decoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0

encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = keras.Input(shape=(latent_dim,))
decoder_state_input_c = keras.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

def decode_sequence(input_seq):
    # encode the input as state vectors
    states_value = encoder_model.predict(input_seq, verbose=0)

    # generate empty target sequence of length 1
    target_seq = np.zeros((1, 1, num_dec_token))
    # populate the first character of target sequence with the start character
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    # sampling loop for a batch of sequences
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value, verbose=0
        )

        # sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # exit condition: either hit max length or find stop character
        if sampled_char == "\n" or len(decoded_sentence) > max_dec_seqlen:
            stop_condition = True

        # update target seq
        target_seq = np.zeros((1, 1, num_dec_token))
        target_seq[0, 0, sampled_token_index] = 1.0

        # update states
        states_value = [h, c]
    return decoded_sentence


@bot.event
async def on_ready():
    print("start")
    try: 
        x = await bot.tree.sync()
        print(f"synced {len(x)} commands")
    except Exception as n:
        print(n)

@bot.tree.command(description="translate english to chinese")
async def translate(interaction: discord.Interaction, content: str):
    input_seq = np.zeros((1, max_enc_seqlen, num_enc_token), dtype='float32')
    for t, char in enumerate(content):
        if char in input_token_index:
            input_seq[0, t, input_token_index[char]] = 1.0
        else:
            print(f"'{char}' not in training vocab")

    input_seq[0, t + 1 :, input_token_index[" "]] = 1.0

    decoded_sentence = decode_sequence(input_seq)
    
    await interaction.response.send_message(decoded_sentence)


bot.run(BOT_TOKEN)