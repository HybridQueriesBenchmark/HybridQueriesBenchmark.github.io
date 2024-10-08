ALTER TABLE fungis ADD INDEX idx_fungis_year (year) TYPE minmax GRANULARITY 1;  


ALTER TABLE fungis ADD INDEX idx_fungis_month (month) TYPE minmax GRANULARITY 1;


ALTER TABLE fungis ADD INDEX idx_fungis_day (day) TYPE minmax GRANULARITY 1;


ALTER TABLE fungis ADD INDEX idx_fungis_countryCode (countryCode) TYPE minmax GRANULARITY 1;



ALTER TABLE fungis ADD INDEX idx_fungis_scientificName (scientificName) TYPE minmax GRANULARITY 1;



ALTER TABLE fungis ADD INDEX idx_fungis_Substrate (Substrate) TYPE minmax GRANULARITY 1;



ALTER TABLE fungis ADD INDEX idx_fungis_Latitude (Latitude) TYPE minmax GRANULARITY 1;



ALTER TABLE fungis ADD INDEX idx_fungis_Longitude (Longitude) TYPE minmax GRANULARITY 1;



ALTER TABLE fungis ADD INDEX idx_fungis_Habitat (Habitat) TYPE minmax GRANULARITY 1;



ALTER TABLE fungis ADD INDEX idx_fungis_poisonous (poisonous) TYPE minmax GRANULARITY 1;



ALTER TABLE fungis ADD INDEX idx_fungis_embedding_gin embedding TYPE vector_similarity('hnsw', 'L2Distance', 'f32', 16, 64, 100) GRANULARITY 10000;
