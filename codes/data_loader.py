import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import functional as F
from torchvision import transforms

import ast
import PIL
import spacy
import pandas as pd
import numpy as np
from pathlib import Path
import pdb

from utils import DataWrap, DataWrap_RefCOCO
from extended_config import cfg as conf
import clip

nlp = spacy.load('en_core_web_lg')
# clip_model, _ = clip.load('RN50') #ViT-B/32 RN50

ATTRIBUTE_DEPS = {
    'amod', 'compound', 'nummod', 'poss', 'prep', 'pobj', 'appos',
    'acl', 'relcl', 'agent', 'advmod', 'npadvmod'
}
ATTRIBUTE_ADPOSITIONS = {
    'with', 'wearing', 'wear', 'in', 'on', 'of', 'holding', 'hold',
    'has', 'having', 'near', 'next', 'beside', 'behind', 'front',
    'under', 'over', 'above', 'below', 'between', 'from', 'at', 'inside',
    'outside'
}
SPATIAL_ATTRIBUTE_WORDS = {
    'left', 'right', 'top', 'bottom', 'upper', 'lower', 'middle',
    'center', 'centre', 'corner', 'front', 'back', 'behind', 'near',
    'next', 'side', 'edge', 'closest', 'farthest', 'first', 'second',
    'third', 'fourth', 'last'
}
SPATIAL_ADPOSITIONS = {
    'behind', 'near', 'beside', 'under', 'over', 'above', 'below',
    'between', 'inside', 'outside', 'front', 'back', 'next'
}
SPATIAL_CONNECTORS = {
    'the', 'a', 'an', 'of', 'from', 'to', 'at', 'in', 'on', 'by'
}
SPATIAL_PHRASE_ENDS = {
    'corner', 'side', 'edge', 'row', 'column', 'area', 'part', 'section',
    'front', 'back', 'middle', 'center', 'centre', 'top', 'bottom',
    'left', 'right'
}

def pil2tensor(image, dtype: np.dtype):
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim == 2:
        a = np.expand_dims(a, 2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    
    return torch.from_numpy(a.astype(dtype, copy=False))


def _clean_phrase(text):
    return ' '.join(text.split())


def _select_anchor_token(doc):
    root = next((token for token in doc if token.head.i == token.i), doc[0])
    if root.pos_ in {'NOUN', 'PROPN', 'PRON'}:
        return root

    for token in root.subtree:
        if token.pos_ in {'NOUN', 'PROPN', 'PRON'}:
            return token

    for token in doc:
        if token.dep_ in {'nsubj', 'nsubjpass', 'dobj', 'pobj', 'attr'} and token.pos_ in {'NOUN', 'PROPN', 'PRON'}:
            return token

    for token in doc:
        if token.pos_ in {'NOUN', 'PROPN', 'PRON'}:
            return token

    return root


def _noun_chunk_for_token(chunks, token):
    containing_chunks = [
        chunk for chunk in chunks
        if chunk.start <= token.i < chunk.end
    ]
    if containing_chunks:
        return max(containing_chunks, key=lambda chunk: (len(chunk), -chunk.start))
    return None


def _span_from_token_ids(doc, token_ids):
    token_ids = {idx for idx in token_ids if 0 <= idx < len(doc) and not doc[idx].is_punct}
    if not token_ids:
        return ''
    return _clean_phrase(doc[min(token_ids): max(token_ids) + 1].text)


def _is_spatial_token(token):
    text = token.text.lower()
    lemma = token.lemma_.lower()
    return (
        text in SPATIAL_ATTRIBUTE_WORDS
        or lemma in SPATIAL_ATTRIBUTE_WORDS
        or lemma in SPATIAL_ADPOSITIONS
    )


def _collect_spatial_phrase_token_ids(doc, chunks):
    token_ids = set()

    for token in doc:
        subtree_tokens = [t for t in token.subtree if not t.is_punct]
        subtree_has_spatial = any(_is_spatial_token(t) for t in subtree_tokens)
        if not (_is_spatial_token(token) or (token.dep_ == 'prep' and subtree_has_spatial)):
            continue

        token_ids.add(token.i)

        chunk = _noun_chunk_for_token(chunks, token)
        if chunk is not None:
            token_ids.update(range(chunk.start, chunk.end))

        subtree_ids = [t.i for t in subtree_tokens]
        if subtree_has_spatial and len(subtree_ids) <= 8:
            token_ids.update(subtree_ids)

        start = token.i
        while start > 0:
            prev = doc[start - 1]
            prev_text = prev.text.lower()
            prev_lemma = prev.lemma_.lower()
            if prev.is_punct:
                break
            if (
                _is_spatial_token(prev)
                or prev_text in SPATIAL_CONNECTORS
                or prev_lemma in SPATIAL_ADPOSITIONS
                or prev.pos_ in {'ADJ', 'NUM'}
            ):
                start -= 1
                continue
            break

        end = token.i + 1
        while end < len(doc):
            nxt = doc[end]
            nxt_text = nxt.text.lower()
            nxt_lemma = nxt.lemma_.lower()
            prev_text = doc[end - 1].text.lower()
            if nxt.is_punct:
                break
            if (
                _is_spatial_token(nxt)
                or nxt_text in SPATIAL_CONNECTORS
                or nxt_lemma in SPATIAL_ADPOSITIONS
                or nxt.pos_ in {'ADJ', 'NUM'}
                or nxt.dep_ in {'pobj', 'amod', 'compound', 'det'}
                or (prev_text in SPATIAL_PHRASE_ENDS and nxt.pos_ in {'NOUN', 'PROPN', 'PRON'})
            ):
                end += 1
                continue
            break

        token_ids.update(range(start, end))

    return token_ids


def select_attribute_phrase(doc):
    """Select the target phrase plus discriminative attribute/location context."""
    if len(doc) == 0:
        return ''

    non_punct_tokens = [token for token in doc if not token.is_punct]
    has_content_verb = any(token.pos_ in {'VERB', 'AUX'} for token in non_punct_tokens)
    if len(non_punct_tokens) <= 6 and not has_content_verb:
        return _clean_phrase(doc.text)

    chunks = list(doc.noun_chunks)
    if not chunks:
        return _clean_phrase(doc.text)

    anchor = _select_anchor_token(doc)
    base_chunk = _noun_chunk_for_token(chunks, anchor)
    token_ids = set(range(base_chunk.start, base_chunk.end)) if base_chunk is not None else {anchor.i}

    # Expand around the target noun with visual attributes and short relations.
    for _ in range(2):
        for token in doc:
            token_text = token.lemma_.lower()
            is_attribute_link = (
                token.dep_ in ATTRIBUTE_DEPS
                or token_text in ATTRIBUTE_ADPOSITIONS
                or token.text.lower() in SPATIAL_ATTRIBUTE_WORDS
            )
            if not is_attribute_link:
                continue

            if token.head.i in token_ids or token.i in token_ids:
                token_ids.update(t.i for t in token.subtree if not t.is_punct)

    for token in doc:
        subtree_ids = {t.i for t in token.subtree}
        if token.i > anchor.i and token.pos_ == 'VERB' and token.tag_ == 'VBG' and anchor.i in subtree_ids:
            token_ids.update(t.i for t in token.subtree if not t.is_punct)

    token_ids.update(_collect_spatial_phrase_token_ids(doc, chunks))

    phrase = _span_from_token_ids(doc, token_ids)
    if phrase:
        return phrase

    return _clean_phrase(base_chunk.text if base_chunk is not None else chunks[0].text)


def select_noun_phrase(doc):
    # Backward-compatible name: npvec now carries attribute-focused local text.
    return select_attribute_phrase(doc)


class NewDistributedSampler(DistributedSampler):
    """
    Same as default distributed sampler of pytorch
    Just has another argument for shuffle
    Allows distributed in validation/testing as well
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)


class ImgQuDataset(Dataset):
    """
    Any Grounding dataset.
    Args:
        train_file (string): CSV file with annotations
        The format should be: img_file, bbox, queries
        Can have same img_file on multiple lines
    """

    def __init__(self, cfg, csv_file, ds_name, split_type='train'):
        self.cfg = cfg
        self.ann_file = csv_file
        self.ds_name = ds_name
        self.split_type = split_type

        self.image_data = self._read_annotations(csv_file)
        # self.image_data = self.image_data.iloc[:200]
        self.img_dir = Path(self.cfg.ds_info[self.ds_name]['img_dir'])

        self.phrase_len = 20
        self.item_getter = getattr(self, 'simple_item_getter')

        self.device = torch.device(cfg.device)

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        return self.item_getter(idx)

    def simple_item_getter(self, idx):
        img_file, annot, query_orig = self.load_annotations(idx)
        img = PIL.Image.open(img_file).convert('RGB')

        h, w = img.height, img.width
        
        if self.cfg.ds_to_use == 'ocid':
            query = query_orig.strip()[:-1]
        else:
            query = query_orig.strip() 
       
        qtmp = nlp(query)
        if len(qtmp) == 0:
            # logger.error('Empty string provided')
            raise NotImplementedError
        
        qlen = len(qtmp)
        q_chosen = query   ## + ' PD'*(self.phrase_len - qlen)
        q_chosen_emb = nlp(q_chosen)
        if not len(q_chosen_emb) == self.phrase_len:
            q_chosen_emb = q_chosen_emb[:self.phrase_len]

        if self.cfg.mdl_to_use == 'retina':
            if not self.cfg.text_context:
                q_chosen_emb_vecs = np.array([q.vector for q in q_chosen_emb])
            else:
                query_emb_vecs = np.array([q.vector for q in q_chosen_emb])

                chunks = {}
                for chunk in qtmp.noun_chunks: # dependency parsing
                    for i in range(chunk.start, chunk.end): 
                        chunks[i] = chunk

                if len(chunks) == 0:
                    attribute_phrase_feature = query_emb_vecs
                    q_chosen_emb_vecs = query_emb_vecs
                else:
                    attribute_phrase = select_attribute_phrase(qtmp)

                    attribute_phrase_chosen = attribute_phrase + ' PD'*(self.phrase_len - len(nlp(attribute_phrase)))

                    attribute_phrase_emb = nlp(attribute_phrase_chosen)
                    if not len(attribute_phrase_emb) == self.phrase_len:
                        attribute_phrase_emb = attribute_phrase_emb[:self.phrase_len]
                    attribute_phrase_feature = np.array([np.vector for np in attribute_phrase_emb])
                    noun_phrase_emb_vecs = attribute_phrase_feature

                    # q_chosen_emb_vecs = np.concatenate((q_chosen_emb_vecs, noun_phrase_feature), axis = 0)
                    q_chosen_emb_vecs = (0.5 * query_emb_vecs + (1 - 0.5) * attribute_phrase_feature)

        elif self.cfg.mdl_to_use == 'clip':
            sentence_token = clip.tokenize(q_chosen)  # Tokenize torch.device('cuda')
            
            chunks = {}
            for chunk in qtmp.noun_chunks: # dependency parsing
                for i in range(chunk.start, chunk.end): 
                    chunks[i] = chunk
            
            if len(chunks) == 0:
                noun_phrase_token = sentence_token
                q_chosen_token = sentence_token
            else:
                attribute_phrase = select_attribute_phrase(qtmp)
                noun_phrase_token = clip.tokenize(attribute_phrase)

                q_chosen_token = sentence_token
                    
        # Annot is in x1y1x2y2 format
        target = np.array(annot)
        img = img.resize((self.cfg.resize_img[0], self.cfg.resize_img[1]))

        target = np.array([target[1], target[0], target[3], target[2]])
        target = np.array([
            target[0] / h, target[1] / w,
            target[2] / h, target[3] / w
        ])
        # Target in range -1 to 1
        target = 2 * target - 1

        img = pil2tensor(img, np.float_).float().div_(255)
        out = {
            'img': img,
            'idxs': torch.tensor(idx).long(),
            'qvec': q_chosen_token,
            'npvec': torch.tensor(noun_phrase_token).long(),
            'qlens': torch.tensor(qlen),
            'annot': torch.from_numpy(target).float(),
            'orig_annot': torch.tensor(annot).float(),
            'img_size': torch.tensor([h, w])
        }

        return out

    def load_annotations(self, idx):
        annotation_list = self.image_data.iloc[idx]
        img_file, x1, y1, x2, y2, queries = annotation_list
        
        if self.cfg.ds_to_use == 'refcoco' or self.cfg.ds_to_use == 'refcoco+' or self.cfg.ds_to_use == 'refcocog':
            file_name = 'COCO_train2014_' + img_file.zfill(16)
            img_file = self.img_dir / file_name
        else:
            img_file = self.img_dir / f'{img_file}'

        if isinstance(queries, list):
            query_chosen = np.random.choice(queries)
        else:
            assert isinstance(queries, str)
            query_chosen = queries
        if '_' in query_chosen:
            query_chosen = query_chosen.replace('_', ' ')
        # annotations = np.array([y1, x1, y2, x2])
        annotations = np.array([x1, y1, x2, y2])

        return img_file, annotations, query_chosen

    def _read_annotations(self, trn_file):
        trn_data = pd.read_csv(trn_file)
        trn_data['bbox'] = trn_data.bbox.apply(lambda x: ast.literal_eval(x))

        if self.ds_name == 'vg_2b' or self.ds_name == 'vg_2ub' or self.ds_name == 'vg_3b' or self.ds_name == 'vg_3ub':
            sample = trn_data['phrase'].iloc[0]
            if sample[0] == '[':
                trn_data['phrase'] = trn_data['phrase'].apply(lambda x: ast.literal_eval(x))

        else:
            sample = trn_data['query'].iloc[0]
            if sample[0] == '[':
                trn_data['query'] = trn_data['query'].apply(lambda x: ast.literal_eval(x))
                
        trn_data['x1'] = trn_data.bbox.apply(lambda x: x[0])
        trn_data['y1'] = trn_data.bbox.apply(lambda x: x[1]) 
        trn_data['x2'] = trn_data.bbox.apply(lambda x: x[2])
        trn_data['y2'] = trn_data.bbox.apply(lambda x: x[3])

        if self.ds_name == 'flickr30k' or self.ds_name == 'flickr30k_c0' or self.ds_name == 'flickr30k_c1':
            trn_data = trn_data.assign(image_fpath=trn_data.img_id.apply(lambda x: f'{x}.jpg'))
            trn_df = trn_data[['image_fpath', 'x1', 'y1', 'x2', 'y2', 'query']]
                               
        elif self.ds_name == 'refclef':
            trn_df = trn_data[['img_id', 'x1', 'y1', 'x2', 'y2', 'query']]
                               
        elif self.ds_name == 'ocid':
            trn_df = trn_data[['image_fpath', 'x1', 'y1', 'x2', 'y2', 'query']]
        
        elif self.ds_name == 'refcoco' or self.ds_name == 'refcoco+' or self.ds_name == 'refcocog':
            trn_df = trn_data[['img_id', 'x1', 'y1', 'x2', 'y2', 'query']]

        elif self.ds_name == 'vg_2b' or self.ds_name == 'vg_2ub' or self.ds_name == 'vg_3b' or self.ds_name == 'vg_3ub':
            # trn_df = trn_data[['x1', 'y1', 'x2', 'y2', 'img_id', 'matched_box_ind', 'obj_name', 'query', 'region_id', 'image_fpath']]
            trn_df = trn_data[['image_fpath', 'x1', 'y1', 'x2', 'y2', 'phrase']]

        return trn_df

def collater(batch):
    qlens = torch.Tensor([i['qlens'] for i in batch])
    max_qlen = int(qlens.max().item())

    # query_vecs = [torch.Tensor(i['query'][:max_qlen]) for i in batch]
    out_dict = {}
    for k in batch[0]:
        out_dict[k] = torch.stack([b[k] for b in batch]).float()
    out_dict['qvec'] = out_dict['qvec'][:, :max_qlen]

    return out_dict


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return NewDistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

def get_dataloader(cfg, dataset: Dataset, is_train: bool) -> DataLoader:
    is_distributed = cfg.do_dist
    images_per_gpu = cfg.bs
    if is_distributed:
        # DistributedDataParallel
        batch_size = images_per_gpu
        num_workers = cfg.nw
    else:
        # DataParallel
        batch_size = images_per_gpu * cfg.num_gpus
        num_workers = cfg.nw * cfg.num_gpus
    if is_train:
        shuffle = True
    else:
        shuffle = False if not is_distributed else True
    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    return DataLoader(dataset, batch_size=batch_size,
                      sampler=sampler, drop_last=is_train,
                      num_workers=num_workers, collate_fn=collater) # drop_last=is_train

def get_data(cfg):
    # Get which dataset to use
    ds_name = cfg.ds_to_use

    # Training file
    trn_csv_file = cfg.ds_info[ds_name]['trn_csv_file']
    trn_ds = ImgQuDataset(cfg=cfg, csv_file=trn_csv_file,
                          ds_name=ds_name, split_type='train')
    trn_dl = get_dataloader(cfg, trn_ds, is_train=True)
    
    # Validation file
    val_csv_file = cfg.ds_info[ds_name]['val_csv_file']
    val_ds = ImgQuDataset(cfg=cfg, csv_file=val_csv_file,
                          ds_name=ds_name, split_type='valid')
    val_dl = get_dataloader(cfg, val_ds, is_train=False)
       
    
    if cfg.ds_to_use == 'refcoco' or cfg.ds_to_use == 'refcoco+':
        testA_csv_file = cfg.ds_info[ds_name]['testA_csv_file']
        testA_ds = ImgQuDataset(cfg=cfg, csv_file=testA_csv_file,
                            ds_name=ds_name, split_type='valid')
        testA_dl = get_dataloader(cfg, testA_ds, is_train=False)

        testB_csv_file = cfg.ds_info[ds_name]['testB_csv_file']
        testB_ds = ImgQuDataset(cfg=cfg, csv_file=testB_csv_file,
                            ds_name=ds_name, split_type='valid')
        testB_dl = get_dataloader(cfg, testB_ds, is_train=False)

        data = DataWrap_RefCOCO(path=cfg.tmp_path, train_dl=trn_dl, valid_dl=val_dl,
                    testA_dl={'testA': testA_dl}, testB_dl={'testB': testB_dl})
        
    else:
        test_csv_file = cfg.ds_info[ds_name]['test_csv_file']
        test_ds = ImgQuDataset(cfg=cfg, csv_file=test_csv_file,
                            ds_name=ds_name, split_type='valid')
        test_dl = get_dataloader(cfg, test_ds, is_train=False)

        data = DataWrap(path=cfg.tmp_path, train_dl=trn_dl, valid_dl=val_dl,
                    test_dl={'test0': test_dl})
        
    return data

if __name__ == '__main__':
    cfg = conf
    data = get_data(cfg, ds_name='refclef')
