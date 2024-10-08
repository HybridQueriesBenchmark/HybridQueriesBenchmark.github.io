DROP DATABASE IF EXISTS mtmd;

CREATE DATABASE IF NOT EXISTS mtmd;

CREATE TABLE mtmd.movies (
    movie_id Int32,
    movie_name String,
    release_year Int32,
    rating Float32,
    votes Int32,
    PRIMARY KEY movie_id
) ENGINE = MergeTree() ORDER BY movie_id
SETTINGS index_granularity = 8192;

CREATE TABLE mtmd.genres (
    genre_id Int32,
    genre String,
    PRIMARY KEY genre_id
) ENGINE = MergeTree() ORDER BY genre_id
SETTINGS index_granularity = 8192;

CREATE TABLE mtmd.movies_genres (
    movie_id Int32,
    genre_id Int32,
    PRIMARY KEY (movie_id, genre_id)
) ENGINE = MergeTree() ORDER BY (movie_id, genre_id)
SETTINGS index_granularity = 8192;

CREATE TABLE mtmd.speakers (
    speaker_id Int32,
    character_name Nullable(String),
    movie_id Int32,
    gender Nullable(Char),
    credit_pos Int32,
    PRIMARY KEY speaker_id
) ENGINE = MergeTree() ORDER BY speaker_id
SETTINGS index_granularity = 8192;

CREATE TABLE mtmd.conversations (
    conversation_id Int32,
    movie_id Int32,
    PRIMARY KEY conversation_id
) ENGINE = MergeTree() ORDER BY conversation_id
SETTINGS index_granularity = 8192;

CREATE TABLE mtmd.utterances (
    utterance_id Int32,
    conversation_id Int32,
    speaker_id Int32,
    reply_to Int32,
    embedding Array(Float32),
    INDEX idx_utterances_embedding embedding TYPE vector_similarity('hnsw', 'L2Distance', 'f32', 16, 64, 100),
    CONSTRAINT constraint_name_1 CHECK length(embedding) = 768,
    PRIMARY KEY utterance_id
) ENGINE = MergeTree() ORDER BY utterance_id
SETTINGS index_granularity = 8192;
