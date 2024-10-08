CREATE INDEX idx_movies_movie_name ON movies (movie_name);
CREATE INDEX idx_movies_release_year ON movies (release_year);
CREATE INDEX idx_movies_rating ON movies (rating);
CREATE INDEX idx_movies_votes ON movies (votes);

CREATE INDEX idx_genres_genre ON genres (genre);

CREATE INDEX idx_speakers_character_name ON speakers (character_name);
CREATE INDEX idx_speakers_gender ON speakers (gender);
CREATE INDEX idx_speakers_credit_pos ON speakers (credit_pos);

CREATE INDEX idx_utterances_reply_to ON utterances (reply_to);
CREATE INDEX idx_utterances_embedding ON utterances USING hnsw (embedding vector_l2_ops);
