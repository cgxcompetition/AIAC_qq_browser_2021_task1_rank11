import scipy
from sklearn.metrics.pairwise import cosine_similarity

def test_spearmanr_cv(vid_embedding, annotation_file,val_idx):
    relevances, similarities = [], []
    with open(annotation_file, 'r') as f:
        for i,line in enumerate(f):
            if i not in val_idx:
                continue
            query, candidate, relevance = line.split()
            if query not in vid_embedding:
                raise Exception(f'ERROR: {query} NOT found')
            if candidate not in vid_embedding:
                raise Exception(f'ERROR: {candidate} NOT found')
            query_embedding = vid_embedding.get(query)
            candidate_embedding = vid_embedding.get(candidate)
            similarity = cosine_similarity([query_embedding], [candidate_embedding])[0][0]
            similarities.append(similarity)
            relevances.append(float(relevance))

    spearmanr = scipy.stats.spearmanr(similarities, relevances).correlation
    return spearmanr

def test_spearmanr(vid_embedding, annotation_file):
    relevances, similarities = [], []
    with open(annotation_file, 'r') as f:
        for line in f:
            query, candidate, relevance = line.split()
            if query not in vid_embedding:
                raise Exception(f'ERROR: {query} NOT found')
            if candidate not in vid_embedding:
                raise Exception(f'ERROR: {candidate} NOT found')

            query_embedding = vid_embedding.get(query)
            candidate_embedding = vid_embedding.get(candidate)
            similarity = cosine_similarity([query_embedding], [candidate_embedding])[0][0]
            similarities.append(similarity)
            relevances.append(float(relevance))

    spearmanr = scipy.stats.spearmanr(similarities, relevances).correlation
    return spearmanr
