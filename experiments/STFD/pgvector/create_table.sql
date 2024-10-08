DROP DATABASE IF EXISTS "stfd";
CREATE DATABASE "stfd";
\c stfd;
CREATE EXTENSION vector;

DROP TABLE if exists "fungis";
CREATE TABLE "fungis" (
    id int,
    year int,
    month int,
    day int,
    "countryCode" char(2),
    "scientificName" varchar(110),
    "Substrate" varchar(50),
    "Latitude" float,
    "Longitude" float,
    "Habitat" varchar(60),
    poisonous boolean,
    embedding vector(768),
    primary key (id)
);