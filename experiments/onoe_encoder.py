import numpy as np

def get_example(generator, glove_dict, batch_size, answer_num,
                eval_data=False, simple_mention=True,
                elmo=None, bert=None, bert_tokenizer=None, finetune_bert=False,
                data_config=None, is_labeler=False, type_elmo=None, all_types=None,
                use_type_definition=False):

  use_elmo_batch = True if elmo is not None else False ### use elmo batch
  #use_elmo_batch = True if not eval_data else False ### use elmo batch

  embed_dim = 300 if elmo is None else 1024
  cur_stream = [None] * batch_size
  no_more_data = False

  while True:
    bsz = batch_size
    seq_length = 25
    for i in range(batch_size):
      try:
        cur_stream[i] = list(next(generator))
      except StopIteration:
        no_more_data = True
        bsz = i
        break

    max_seq_length = min(50, max([len(elem[1]) + len(elem[2]) + len(elem[3]) for elem in cur_stream if elem]))
    token_embed = np.zeros([bsz, max_seq_length, embed_dim], np.float32)
    token_seq_length = np.zeros([bsz], np.float32)
    token_bio = np.zeros([bsz, max_seq_length, 4], np.float32)
    mention_start_ind = np.zeros([bsz, 1], np.int64)
    mention_end_ind = np.zeros([bsz, 1], np.int64)
    max_mention_length = min(20, max([len(elem[3]) for elem in cur_stream if elem]))
    max_span_chars = min(25, max(max([len(elem[5]) for elem in cur_stream if elem]), 5))
    max_n_target = max([len(elem[4][:]) for elem in cur_stream if elem])
    annot_ids = np.zeros([bsz], np.object)
    span_chars = np.zeros([bsz, max_span_chars], np.int64)
    mention_embed = np.zeros([bsz, max_mention_length, embed_dim], np.float32)
    targets = np.zeros([bsz, answer_num], np.float32)

    mention_headword_embed = np.zeros([bsz, embed_dim], np.float32)
    mention_span_length = np.zeros([bsz], np.float32)

    if elmo is not None:
      token_embed = np.zeros([bsz, 3, max_seq_length, embed_dim], np.float32)
      mention_embed = np.zeros([bsz, 3, max_mention_length, embed_dim], np.float32)
      mention_headword_embed = np.zeros([bsz, 3, embed_dim], np.float32)
      elmo_mention_first = np.zeros([bsz, 3, embed_dim], np.float32)
      elmo_mention_last = np.zeros([bsz, 3, embed_dim], np.float32)
    if glove_dict is not None and elmo is not None and not is_labeler:
      token_embed = np.zeros([bsz, 4, max_seq_length, embed_dim], np.float32)
      mention_embed = np.zeros([bsz, 4, max_mention_length, embed_dim], np.float32)
      mention_headword_embed = np.zeros([bsz, 4, embed_dim], np.float32)
      elmo_mention_first = np.zeros([bsz, 4, embed_dim], np.float32)
      elmo_mention_last = np.zeros([bsz, 4, embed_dim], np.float32)
    
    # Only Train: batch to ELMo embeddings
    # Will get CUDA memory error if batch size is large
    if use_elmo_batch:
      token_seqs = []
      keys = []
      for i in range(bsz):
        left_seq = cur_stream[i][1]
        if len(left_seq) > seq_length:
          left_seq = left_seq[-seq_length:]
        mention_seq = cur_stream[i][3]
        right_seq = cur_stream[i][2]
        token_seqs.append(left_seq + mention_seq + right_seq)
        keys.append(cur_stream[i][0])

      try:
        elmo_emb_batch = get_elmo_vec_batch(token_seqs, elmo) # (batch, 3, len, dim)
      except:
        print('ERROR:', bsz, token_seqs, cur_stream[i])
        raise

    for i in range(bsz):
      left_seq = cur_stream[i][1]
      if len(left_seq) > seq_length:
        left_seq = left_seq[-seq_length:]
      mention_seq = cur_stream[i][3]
      annot_ids[i] = cur_stream[i][0]
      right_seq = cur_stream[i][2]
      mention_headword = cur_stream[i][6]

      token_seq = left_seq + mention_seq + right_seq
    #   context_seq = left_seq + right_seq
      mention_start_ind[i] = min(seq_length, len(left_seq))
      mention_end_ind[i] = min(49, len(left_seq) + len(mention_seq) - 1)
      mention_start_actual = len(left_seq)
      mention_end_actual = len(left_seq) + len(mention_seq) - 1
      if elmo is None and bert is None: # GLoVe or BERT
        if not finetune_bert: # GLoVe
          for j, word in enumerate(token_seq):
            if j < max_seq_length:
              token_embed[i, j, :embed_dim] = get_word_vec(word, glove_dict)
      elif elmo is not None and bert is None: # ELMo
        # sentence
        if use_elmo_batch: # Train
          elmo_emb = elmo_emb_batch[i] # (3, len, dim)
        else: # Eval
          elmo_emb = get_elmo_vec(token_seq, elmo)
        n_layers, seq_len, elmo_dim = elmo_emb.shape
        assert n_layers == 3, n_layers
        assert seq_len == len(token_seq), (seq_len, len(token_seq), token_seq, elmo_emb.shape)
        assert elmo_dim == embed_dim, (elmo_dim, embed_dim)
        token_embed = get_elmo_vec(context_seq, elmo)
        if seq_len <= max_seq_length:
        # if len(context_seq) <= max_seq_length:
          token_embed[i, :n_layers, :seq_len, :] = elmo_emb
        #   token_embed[i, :n_layers, :len(context_seq), :] = this_token_embed
        else:
          token_embed[i, :n_layers, :, :] = elmo_emb[:, :max_seq_length, :] 
        #   token_embed[i, :n_layers, :, :] = this_token_embed[:, :max_seq_length, :] 
        # mention span
        start_ind = len(left_seq)
        end_ind = len(left_seq) + len(mention_seq) - 1
        elmo_mention = elmo_emb[:, start_ind:end_ind+1, :]
        mention_len = end_ind - start_ind + 1
        assert mention_len == elmo_mention.shape[1] == len(mention_seq),(mention_len, elmo_mention.shape[1], len(mention_seq), mention_seq, elmo_mention.shape, token_seq, elmo_emb.shape) # (mention_len, elmo_mention.shape[0], len(mention_seq))
        if mention_len < max_mention_length: 
          mention_embed[i, :n_layers, :mention_len, :] = elmo_mention 
        else:
          mention_embed[i, :n_layers, :mention_len, :] = elmo_mention[:, :max_mention_length, :]
        # mention first & last words
        elmo_mention_first[i, :n_layers, :] = elmo_mention[:, 0, :]
        elmo_mention_last[i, :n_layers, :] = elmo_mention[:, -1, :]
        # headword
        try:
          headword_location = mention_seq.index(mention_headword)
        except:
          #print('WARNING: ' + mention_headword + ' / ' + ' '.join(mention_seq))
          # find the headword
          headword_location = 0
          headword_candidates = [i for i, word in enumerate(mention_seq) if mention_headword in word]
          if headword_candidates:
            headword_location = headword_candidates[0]
        mention_headword_embed[i, :n_layers, :] = elmo_mention[:, headword_location, :]
        # add 300d-GLoVe
        if glove_dict is not None and not is_labeler:
          # sentence
          for j, word in enumerate(token_seq):
            if j < max_seq_length:
              token_embed[i, 3, j, :300] = get_word_vec(word, glove_dict)
          # mention span
          for j, mention_word in enumerate(mention_seq):
            if j < max_mention_length:
              if simple_mention:
                mention_embed[i, 3, j, :300] = [k / len(cur_stream[i][3]) for k in
                                            get_word_vec(mention_word, glove_dict)]
              else:
                mention_embed[i, 3, j, :300] = get_word_vec(mention_word, glove_dict)
          # mention first & last words
          elmo_mention_first[i, 3, :300] = get_word_vec(mention_seq[0], glove_dict)
          elmo_mention_last[i, 3, :300] = get_word_vec(mention_seq[-1], glove_dict)
          # headword
          mention_headword_embed[i, 3, :300] = get_word_vec(mention_headword, glove_dict)
      for j, _ in enumerate(left_seq):
        token_bio[i, min(j, 49), 0] = 1.0  # token bio: 0(left) start(1) inside(2)  3(after)
      for j, _ in enumerate(right_seq):
        token_bio[i, min(j + len(mention_seq) + len(left_seq), 49), 3] = 1.0
      for j, _ in enumerate(mention_seq):
        if j == 0 and len(mention_seq) == 1:
          token_bio[i, min(j + len(left_seq), 49), 1] = 1.0
        else:
          token_bio[i, min(j + len(left_seq), 49), 2] = 1.0
      token_seq_length[i] = min(50, len(token_seq))

      span_chars[i, :] = pad_slice(cur_stream[i][5], max_span_chars, pad_token=0)
      for answer_ind in cur_stream[i][4]:
        targets[i, answer_ind] = 1.0
      mention_span_length[i] = min(len(mention_seq), 20)

    feed_dict = {"annot_id": annot_ids,
                 "mention_embed": mention_embed,
                 "span_chars": span_chars,
                 "y": targets,
                 "mention_headword_embed": mention_headword_embed,
                 "mention_span_length": mention_span_length}
    feed_dict["token_bio"] = token_bio
    feed_dict["token_embed"] = token_embed
    feed_dict["token_seq_length"] = token_seq_length
    feed_dict["mention_start_ind"] = mention_start_ind
    feed_dict["mention_end_ind"] = mention_end_ind
    if elmo is not None:
      feed_dict["mention_first"] = elmo_mention_first 
      feed_dict["mention_last"] = elmo_mention_last
    if no_more_data:
      if eval_data and bsz > 0:
        yield feed_dict
      break
    yield feed_dict