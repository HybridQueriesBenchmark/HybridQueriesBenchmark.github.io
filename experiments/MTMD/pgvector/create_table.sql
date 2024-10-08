DROP DATABASE IF EXISTS "mtmd";
CREATE DATABASE "mtmd";
\c mtmd;
CREATE EXTENSION vector;

DROP TABLE if exists movies;
CREATE TABLE movies (
    movie_id int,
    movie_name varchar(100),
    release_year int,
    rating float,
    votes int,
    primary key (movie_id)
);

DROP TABLE if exists genres;
CREATE TABLE genres (
    genre_id int,
    genre varchar(20),
    primary key (genre_id)
);

DROP TABLE if exists movies_genres;
CREATE TABLE movies_genres (
    movie_id int references movies(movie_id),
    genre_id int references genres(genre_id),
    foreign key (movie_id) references movies(movie_id),
    foreign key (genre_id) references genres(genre_id)
);

DROP TABLE if exists speakers;
CREATE TABLE speakers (
    speaker_id int,
    character_name varchar(100),
    movie_id int references movies(movie_id),
    gender char(1),
    credit_pos int,
    primary key (speaker_id),
    foreign key (movie_id) references movies(movie_id)
);

DROP TABLE if exists conversations;
CREATE TABLE conversations (
    conversation_id int,
    movie_id int references movies(movie_id),
    primary key (conversation_id),
    foreign key (movie_id) references movies(movie_id)
);

DROP TABLE if exists utterances;
CREATE TABLE utterances (
    utterance_id int,
    conversation_id int references conversations(conversation_id),
    speaker_id int references speakers(speaker_id),
    reply_to int,
    embedding vector(768),
    primary key (utterance_id),
    foreign key (conversation_id) references conversations(conversation_id),
    foreign key (speaker_id) references speakers(speaker_id)
);