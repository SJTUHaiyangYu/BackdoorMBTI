from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import trange


class SimCalc:
    def __init__(self, batch_size):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.batch_size = batch_size
        print(f"fast sim calc initialized.")

    def calc_sim(self, ref_text_lst, eval_text_lst):
        all_score_lst = []
        assert len(ref_text_lst) == len(eval_text_lst)
        for start_idx in trange(0, len(ref_text_lst), self.batch_size):
            end_idx = start_idx + self.batch_size
            sent_lst_a = ref_text_lst[start_idx:end_idx]
            sent_lst_b = eval_text_lst[start_idx:end_idx]
            embeddings1 = self.model.encode(sent_lst_a, convert_to_tensor=True)
            embeddings2 = self.model.encode(sent_lst_b, convert_to_tensor=True)
            all_score_lst += cosine_similarity(embeddings1, embeddings2).tolist()
        return all_score_lst
