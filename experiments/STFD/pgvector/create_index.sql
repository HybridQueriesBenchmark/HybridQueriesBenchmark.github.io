\timing
CREATE INDEX idx_fungis_year ON "fungis" (year);
CREATE INDEX idx_fungis_month ON "fungis" (month);
CREATE INDEX idx_fungis_day ON "fungis" (day);
CREATE INDEX idx_fungis_countryCode ON "fungis" ("countryCode");
CREATE INDEX idx_fungis_scientificName ON "fungis" ("scientificName");
CREATE INDEX idx_fungis_Substrate ON "fungis" ("Substrate");
CREATE INDEX idx_fungis_Latitude ON "fungis" ("Latitude");
CREATE INDEX idx_fungis_Longitude ON "fungis" ("Longitude");
CREATE INDEX idx_fungis_Habitat ON "fungis" ("Habitat");
CREATE INDEX idx_fungis_poisonous ON "fungis" (poisonous);
CREATE INDEX idx_fungis_embedding_gin ON "fungis" USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64);
