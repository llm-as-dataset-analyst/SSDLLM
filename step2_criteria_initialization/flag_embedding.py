from FlagEmbedding import FlagModel
from tqdm import tqdm

model = FlagModel('BAAI/bge-large-en-v1.5', 
                    use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

def get_embedding(text):
    text = text.replace("\n", " ")
    return model.encode(text)

def cosine_similarity(embedding1, embedding2):
    return embedding1 @ embedding2.T

def find_most_similar_indices(list1, list2, return_second=False):
    embeddings_list1 = [get_embedding(text) for text in list1]
    embeddings_list2 = [get_embedding(text) for text in list2]

    most_similar_indices_list1 = []
    second_similar_indices_list1 = []
    most_similar_pairs_list1 = []
    second_similar_pairs_list1 = []

    for idx, embedding1 in tqdm(enumerate(embeddings_list1)):
        all_similar_scores = []

        for i, embedding2 in enumerate(embeddings_list2):
            similarity = cosine_similarity(embedding1, embedding2)
            all_similar_scores.append((similarity, i))

        # 按相似度从高到低排序
        all_similar_scores.sort(reverse=True, key=lambda x: x[0])

        # 获取最相似和第二相似的索引
        most_similar_indices_list1.append(all_similar_scores[0][1])
        second_similar_indices_list1.append(all_similar_scores[1][1])

    return most_similar_indices_list1, second_similar_indices_list1

if __name__ == "__main__":
    sentences_1 = ["mood", "action", "location", "joyful"]
    sentences_2 = ["activity", "environment", "emotion", "interaction"]

    most, second = find_most_similar_indices(sentences_1, sentences_2)
    print(most)
    print(second)

    for topic, most_idx in zip(sentences_1, most):
        print(topic, " - ", sentences_2[most_idx])
