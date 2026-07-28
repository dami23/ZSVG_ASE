import torch
import torchvision.transforms as T
from PIL import Image
import clip
from clip.simple_tokenizer import SimpleTokenizer
import spacy
import numpy as np
import pdb

clip_model, _ = clip.load('RN50')

def text_process():
    nlp = spacy.load('en_core_web_lg')
    # query = 'a man with brown hair in a dark green shirt'
    query = 'faces'
    q_chosen = query.strip()
    # embed = nlp(q_chosen)

    qtmp = nlp(str(q_chosen))
    if len(qtmp) == 0:
        # logger.error('Empty string provided')
        raise NotImplementedError
    qlen = len(qtmp)
    q_chosen = q_chosen + ' PD'*(50 - qlen)
    q_chosen_emb = nlp(q_chosen)
    if not len(q_chosen_emb) == 50:
        q_chosen_emb = q_chosen_emb[:50]

    q_chosen_emb_vecs = np.array([q.vector for q in q_chosen_emb])

    sentence_token = clip.tokenize(query).cuda() # Tokenize
    sentence_feature = clip_model.encode_text(sentence_token)
    sentence_feature = sentence_feature / sentence_feature.norm(dim=1, keepdim=True)

    print(sentence_feature.shape)

    chunks = {}
    for chunk in qtmp.noun_chunks: # dependency parsing
        for i in range(chunk.start, chunk.end): # 문장에서 각 word가 어느 noun phrase에 속하지 확인 
            chunks[i] = chunk

    print('noun phrase: ', chunks, len(chunks))

    for token in q_chosen_emb:
        if token.head.i == token.i:
            root_word = token.head
            
    print('root word: ', root_word)
    if len(chunks) == 0:
        noun_phrase = root_word
    else:
        noun_phrase = chunks[root_word.i].text

    noun_phrase_token = clip.tokenize(noun_phrase).cuda() # Tokenize
    noun_phrase_feature = clip_model.encode_text(noun_phrase_token)
    noun_phrase_feature = noun_phrase_feature / noun_phrase_feature.norm(dim=1, keepdim=True) # normalize
    print('root word & noun phrase: ', noun_phrase)


    print(noun_phrase_feature.shape)
    pdb.set_trace()

    global_local_textual_feature = 0.5 * sentence_feature + (1 - 0.5) * noun_phrase_feature
    print(global_local_textual_feature.shape)

def visual_process():
    image_path = '/home/mi/projects/test_examples/test_codes/01.jpg'
    image = Image.open(image_path).convert("RGB")
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] 
    # width, height = image.size 

    transform = T.Compose([T.Resize(800), 
                        T.ToTensor(), 
                        T.Normalize(mean, std) 
                        ])

    resized_img = transform(image).unsqueeze(0)

    transform = T.Resize((300,300))
    input_img = transform(resized_img.cuda())

    feature_map = clip_model.encode_image(input_img)

    pdb.set_trace()

def improved_clustered_attention():
    from fast_transformers.attention.improved_clustered_attention import ImprovedClusteredAttention
    from fast_transformers.masking import LengthMask, FullMask

    self.clustered_attention = ImprovedClusteredAttention(clusters = 8, topk = 8)
    attn_mask = FullMask(N, device = qq.device)
    length_mask = LengthMask(qq.new_full((B,), N, dtype=torch.int64))
    src2_clustered = self.clustered_attention(qq_trans, kk_trans, vv_trans, attn_mask=attn_mask, query_lengths=length_mask, key_lengths=length_mask)

if __name__ == '__main__':
    # visual_process()
    text_process()


