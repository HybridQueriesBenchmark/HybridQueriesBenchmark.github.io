DROP DATABASE IF EXISTS stfd;
CREATE DATABASE stfd;

USE stfd;

DROP TABLE IF EXISTS fungis;
SET allow_experimental_vector_similarity_index = 1;
CREATE TABLE fungis (
    id Int32,
    year Nullable(Int32),
    month Nullable(Int32),
    day Nullable(Int32),
    countryCode String,
    scientificName String,
    Substrate String,
    Latitude Float32,
    Longitude Float32,
    Habitat String,
    poisonous UInt8,
    embedding Array(Float32),
    INDEX idx_fungis_embedding_gin embedding TYPE vector_similarity('hnsw', 'L2Distance', 'f32', 16, 64, 100),
    CONSTRAINT constraint_name_1 CHECK length(embedding) = 768,
    PRIMARY KEY id
) ENGINE = MergeTree
ORDER BY id;