import pandas as pd
import json
import numpy as np


def convert_to_integer(s):
    num_str = ''
    for char in s:
        if char.isdigit():
            num_str += char
        else:
            break
    return int(num_str)


def write_movie_table(file_path, write_path):
    movie_id_set = set()
    movie_info_list = []
    movieid2genres = {}
    genre_set = set()

    with open(file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    
    for key, value in data.items():
        meta = value.get('meta', {})
        movie_info = {
            'movie_id': int(meta.get('movie_idx', '')[1:]),  # remove the 'm' prefix
            'movie_name': meta.get('movie_name', ''),
            'release_year': convert_to_integer(meta.get('release_year', '')),
            'rating': float(meta.get('rating', '')),
            'votes': int(meta.get('votes', '')),
        }
        if (not movie_info['movie_id'] in movie_id_set):
            movie_id_set.add(movie_info['movie_id'])
            movie_info_list.append(movie_info)

            genre_str = meta.get('genre', '[]')
            genre_str = genre_str.strip('[')
            genre_str = genre_str.strip(']')
            if len(genre_str) > 0:
                genre_list = genre_str.split(',')
                movieid2genres[movie_info['movie_id']] = set()
                for genre in genre_list:
                    genre = genre.strip()
                    genre = genre.strip('\'')
                    genre = genre.strip()
                    genre_set.add(genre)
                    movieid2genres[movie_info['movie_id']].add(genre)


    df = pd.DataFrame(movie_info_list)


    df.to_csv(write_path, index=False)

    return movie_info_list, genre_set, movieid2genres

def write_genres_csv(genre_set, write_path):
    genre_list = list(genre_set)
    genre_list.sort()
    df = pd.DataFrame(enumerate(genre_list), columns=['genre_id', 'genre'])
    df.to_csv(write_path, index=False)
    genre2id = {genre: idx for idx, genre in enumerate(genre_list)}
    return genre2id


def write_movies_genres_csv(movieid2genres, genre2id, write_path):
    movie_id = []
    genre_id = []
    for movie, genres in movieid2genres.items():
        for genre in genres:
            movie_id.append(movie)
            genre_id.append(genre2id[genre])
    df = pd.DataFrame({'movie_id': movie_id, 'genre_id': genre_id})
    df.to_csv(write_path, index=False)


def write_speaker_table(file_path, write_path):
    speaker_info_list = []

    with open(file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    
    for speaker_id, value in data.items():
        meta = value.get('meta', {})
        speaker_info = {
            'speaker_id': int(speaker_id[1:]),  # remove the 'u' prefix
            'character_name': meta.get('character_name', ''),
            'movie_id': int(meta.get('movie_idx', '')[1:]),  # remove the 'm' prefix
            'gender': meta.get('gender', ''),
            'credit_pos': int(-1 if meta.get('credit_pos', '') == '?' else meta.get('credit_pos', '')),
        }
        speaker_info_list.append(speaker_info)
    

    df = pd.DataFrame(speaker_info_list)


    df.to_csv(write_path, index=False)


def write_conversations_csv(file_path, write_path):
    conversation_info_list = []

    with open(file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    
    for conversation_id, value in data.items():
        movie_id = int(value['meta']['movie_idx'][1:])  # remove the 'm' prefix
        conversation_id = int(conversation_id[1:])  # remove the 'L' prefix
        conversation_info = {
            'conversation_id': conversation_id,
            'movie_id': movie_id,
        }
        conversation_info_list.append(conversation_info)
    

    df = pd.DataFrame(conversation_info_list)


    df.to_csv(write_path, index=False)


def write_utterances_csv(file_path, write_path, train_idx : np.array):
    utterance_info_list = []

    train_idx = set(train_idx.tolist())

    with open(file_path, 'r', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx not in train_idx:
                continue
            data = json.loads(line)
            utterance_id = int(data['id'][1:])  # remove the 'L' prefix
            speaker_id = int(data['speaker'][1:])   # remove the 'u' prefix
            conversation_id = int(data['conversation_id'][1:])  # remove the 'L' prefix
            reply_to = int(-1 if data['reply-to'] == None else data['reply-to'][1:])  # remove the 'L' prefix

            utterance_info = {
                'utterance_id': utterance_id,
                'speaker_id': speaker_id,
                'conversation_id': conversation_id,
                'reply_to': reply_to,
            }
            utterance_info_list.append(utterance_info)
    

    df = pd.DataFrame(utterance_info_list)

    print(len(df), len(train_idx))
    assert len(df) == len(train_idx)


    df.to_csv(write_path, index=False)



if __name__ == "__main__":
    conversations_file_path = r"F:\datasets\NLP\movie-corpus\conversations.json"
    
    movie_info_list, genre_set, movieid2genres = write_movie_table(conversations_file_path, r"./metadata/movie/movies.csv")

    genre2id = write_genres_csv(genre_set, r"./metadata/movie/genres.csv")

    write_movies_genres_csv(movieid2genres, genre2id, r"./metadata/movie/movies_genres.csv")

    speakers_file_path = r"F:\datasets\NLP\movie-corpus\speakers.json"
    write_speaker_table(speakers_file_path, r"./metadata/movie/speakers.csv")

    write_conversations_csv(conversations_file_path, r"./metadata/movie/conversations.csv")

    utterances_file_path = r"F:\datasets\NLP\movie-corpus\utterances.jsonl"
    train_idx = np.load(r"./metadata/movie/train_idx.npy")
    write_utterances_csv(utterances_file_path, r"./metadata/movie/utterances.csv", train_idx)


