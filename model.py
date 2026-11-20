"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    words = []
    for sentence in sentences:
        word_list = sentence.split(' ') 
        for word in word_list:
            words.append(word)

    unique_words = []
    whether_break = False
    for word in words:
        for u_word in unique_words:
            if word == u_word:
                whether_break = True
                break
        if not whether_break:
            unique_words.append(word)
        whether_break = False


    
    token = {}


    if specials == ('<pad>', '<bos>', '<eos>', '<unk>'):
        token = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
        for i, word in enumerate(unique_words):
            if word == '<pad>' or word == '<bos>' or word == '<eos>' or word == '<unk>':
                del unique_words[i]
    elif len(specials) == 5:
        token[specials] = 0

    else:
        for i, s_word in enumerate(specials):
            token[s_word] = i

 
    for i, u_word in enumerate(unique_words):
        if len(specials) == 5:
            token[u_word] = i+1
        else:
            token[u_word] = i+len(specials)

    return token

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    id_to_token = {}

    for ele in token_to_id:
        id_to_token[token_to_id[ele]] = ele

    return id_to_token

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    if sentence == '':
        return []
    
    words = sentence.split(' ')

    possible_unk = ["<unk>", "unk", 'UNK']

    id_list = []

    for word in words:
        unknown = False
        try:
            id = token_to_id[word]

        except KeyError:
            unknown = True

        if unknown:
            for case in possible_unk:
                try:
                    id_list.append(token_to_id[case])
                except KeyError:
                    continue
        else:
            id_list.append(token_to_id[word])

    return id_list

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    if ids == []:
        return []

    tokens = []

    for id in ids:
        tokens.append(id_to_token[id])

    return tokens

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    if max_len == 0:
        return []

    elif len(ids) == max_len:
        return ids

    elif len(ids) > max_len:
        return ids[:-(len(ids)-max_len)]

    else:
        for _ in range(max_len-len(ids)):
            ids.append(pad_id)
        return ids

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    stack_list = []
    for l in padded_sequences:
        stack_list.append(l)

    stack = torch.tensor(stack_list)

    return stack

# Step 7 - scale_embeddings_by_sqrt_d_model
import math

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    scale = math.sqrt(d_model)

    return embeddings * scale

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    result = []
    
    for i in range(d_model//2):
        result.append(float(10000**(-2*i/d_model)))

    return torch.tensor(result)

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    float_tensor = []
    for i in range(max_len):
        float_tensor.append([float(i)])
    return torch.tensor(float_tensor)

# Step 10 - fill_even_indices_with_sin (not yet solved)
# TODO: implement

# Step 11 - fill_odd_indices_with_cos (not yet solved)
# TODO: implement

# Step 12 - build_sinusoidal_positional_encoding (not yet solved)
# TODO: implement

# Step 13 - add_positional_encoding_to_embeddings (not yet solved)
# TODO: implement

# Step 14 - build_padding_mask (not yet solved)
# TODO: implement

# Step 15 - build_causal_mask (not yet solved)
# TODO: implement

# Step 16 - combine_padding_and_causal_masks (not yet solved)
# TODO: implement

# Step 17 - compute_raw_attention_scores (not yet solved)
# TODO: implement

# Step 18 - scale_attention_scores (not yet solved)
# TODO: implement

# Step 19 - mask_attention_scores_with_neg_inf (not yet solved)
# TODO: implement

# Step 20 - softmax_attention_weights (not yet solved)
# TODO: implement

# Step 21 - apply_attention_weights_to_values (not yet solved)
# TODO: implement

# Step 22 - scaled_dot_product_attention (not yet solved)
# TODO: implement

# Step 23 - split_last_dim_into_heads (not yet solved)
# TODO: implement

# Step 24 - transpose_heads_before_sequence (not yet solved)
# TODO: implement

# Step 25 - merge_heads_back_to_model_dim (not yet solved)
# TODO: implement

# Step 26 - apply_linear_projection (not yet solved)
# TODO: implement

# Step 27 - project_to_query_key_value (not yet solved)
# TODO: implement

# Step 28 - split_qkv_into_heads (not yet solved)
# TODO: implement

# Step 29 - multi_head_scaled_dot_product_attention (not yet solved)
# TODO: implement

# Step 30 - merge_heads_and_project_output (not yet solved)
# TODO: implement

# Step 31 - assemble_multi_head_attention_forward (not yet solved)
# TODO: implement

# Step 32 - apply_ffn_first_linear_and_relu (not yet solved)
# TODO: implement

# Step 33 - apply_ffn_second_linear (not yet solved)
# TODO: implement

# Step 34 - position_wise_feed_forward_network (not yet solved)
# TODO: implement

# Step 35 - compute_layer_norm_mean_and_variance (not yet solved)
# TODO: implement

# Step 36 - normalize_and_scale_with_gamma_beta (not yet solved)
# TODO: implement

# Step 37 - apply_residual_add_and_norm (not yet solved)
# TODO: implement

# Step 38 - apply_dropout_with_keep_mask (not yet solved)
# TODO: implement

# Step 39 - encoder_layer_self_attention_sublayer (not yet solved)
# TODO: implement

# Step 40 - encoder_layer_feed_forward_sublayer (not yet solved)
# TODO: implement

# Step 41 - assemble_encoder_layer (not yet solved)
# TODO: implement

# Step 42 - stack_encoder_layers (not yet solved)
# TODO: implement

# Step 43 - decoder_layer_masked_self_attention_sublayer (not yet solved)
# TODO: implement

# Step 44 - decoder_layer_cross_attention_sublayer (not yet solved)
# TODO: implement

# Step 45 - decoder_layer_feed_forward_sublayer (not yet solved)
# TODO: implement

# Step 46 - assemble_decoder_layer (not yet solved)
# TODO: implement

# Step 47 - stack_decoder_layers (not yet solved)
# TODO: implement

# Step 48 - apply_final_output_projection (not yet solved)
# TODO: implement

# Step 49 - tie_output_projection_to_token_embeddings (not yet solved)
# TODO: implement

# Step 50 - apply_log_softmax_over_vocab (not yet solved)
# TODO: implement

# Step 51 - run_transformer_forward (not yet solved)
# TODO: implement

# Step 52 - init_encoder_layer_parameters (not yet solved)
# TODO: implement

# Step 53 - init_decoder_layer_parameters (not yet solved)
# TODO: implement

# Step 54 - init_embedding_and_projection_parameters (not yet solved)
# TODO: implement

# Step 55 - collect_model_parameters_into_list (not yet solved)
# TODO: implement

# Step 56 - shift_targets_right_with_start_token (not yet solved)
# TODO: implement

# Step 57 - compute_noam_learning_rate (not yet solved)
# TODO: implement

# Step 58 - build_uniform_smoothing_distribution (not yet solved)
# TODO: implement

# Step 59 - set_confidence_on_gold_tokens (not yet solved)
# TODO: implement

# Step 60 - zero_pad_column_and_pad_token_rows (not yet solved)
# TODO: implement

# Step 61 - compute_label_smoothed_kl_loss (not yet solved)
# TODO: implement

# Step 62 - average_loss_over_non_pad_tokens (not yet solved)
# TODO: implement

# Step 63 - compute_token_accuracy_ignoring_pad (not yet solved)
# TODO: implement

# Step 64 - initialize_adam_optimizer_state (not yet solved)
# TODO: implement

# Step 65 - update_adam_first_moment (not yet solved)
# TODO: implement

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 68 - compute_adam_parameter_update (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

