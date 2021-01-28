-- SQLite

PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;


-- Core Entities:

CREATE TABLE IF NOT EXISTS Media (
	media_id BLOB PRIMARY KEY,
	parent_id BLOB NULL REFERENCES Media, -- support nested media pieces (e.g. page in a document or crop of a photo)
	metadata TEXT NULL -- child media will have extraction information (e.g. page number or bounding box coordinates)
);

CREATE TABLE IF NOT EXISTS Embeddings (
    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dim SMALLINT NOT NULL,
    name TEXT NOT NULL UNIQUE,  -- short name
    version TEXT NOT NULL DEFAULT '',  -- model and version identifier
    ready BOOLEAN NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS Manifolds (
	manifold_id INTEGER PRIMARY KEY AUTOINCREMENT,
	embedding_id INTEGER NOT NULL REFERENCES Embeddings ON DELETE RESTRICT,
	building BOOLEAN NOT NULL DEFAULT 0,
	ready BOOLEAN NOT NULL DEFAULT 0,
	merged BOOLEAN NOT NULL DEFAULT 0,
	inactive BOOLEAN NOT NULL DEFAULT 0,
	metadata TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS Relations (
	relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
	metadata TEXT NOT NULL -- JSON
);


-- Extra variable data:

CREATE TABLE IF NOT EXISTS Thumbnails (
	-- A thumbnail depicts the contents of a media
	media_id BLOB NOT NULL PRIMARY KEY REFERENCES Media ON DELETE CASCADE,
	thumbnail BLOB NULL
);


-- Core Relations:

CREATE TABLE IF NOT EXISTS MediaLocations (
	url URL NOT NULL PRIMARY KEY, -- data urls can be used for sentences as media
	hash_function TEXT NULL DEFAULT NULL, -- used to produce the media_id
	media_id BLOB NULL REFERENCES Media ON DELETE CASCADE,
	ts_first_access DATETIME NOT NULL,
	ts_last_access DATETIME NOT NULL,
	ts_first_fail DATETIME DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS MediaLocations_media_id ON MediaLocations(media_id);
CREATE INDEX IF NOT EXISTS MediaLocations_ts_first_access ON MediaLocations (ts_first_access);


CREATE TABLE IF NOT EXISTS ManifoldItems (  -- vector computed and stored from the associated media embedding
	manifold_id INT NOT NULL REFERENCES Manifolds ON DELETE RESTRICT,
	item_i INT NOT NULL,  -- we will not need null items because it's impossible to retry building a manifold.
	media_id BLOB NOT NULL REFERENCES Media ON DELETE CASCADE,
	PRIMARY KEY (manifold_id, item_i),
	UNIQUE (manifold_id, media_id)
);

CREATE INDEX IF NOT EXISTS ManifoldItems_media_id ON ManifoldItems(media_id);

CREATE TABLE IF NOT EXISTS ManifoldHoles (
	manifold_id INT NOT NULL REFERENCES Manifolds ON DELETE CASCADE,
	media_id BLOB NOT NULL REFERENCES Media ON DELETE CASCADE,
	msg TEXT NULL,
	PRIMARY KEY (manifold_id, media_id)
);

CREATE INDEX IF NOT EXISTS ManifoldHoles_media_id ON ManifoldHoles(media_id);

CREATE TABLE IF NOT EXISTS MediaRelations (
	relation_id INT NOT NULL REFERENCES Relations ON DELETE CASCADE,
	media_id BLOB NOT NULL REFERENCES Media ON DELETE CASCADE,
	other_media_id BLOB NOT NULL REFERENCES Media ON DELETE CASCADE,
	PRIMARY KEY (relation_id, media_id, other_media_id)
);

CREATE INDEX IF NOT EXISTS MediaRelations_media_id ON MediaRelations(media_id);
CREATE INDEX IF NOT EXISTS MediaRelations_other_media_id ON MediaRelations(other_media_id);


-- Procrustean alignment

CREATE TABLE IF NOT EXISTS GeneralizedProcrustesAnalysis
(
    gpa_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dim SMALLINT NOT NULL,
    procrustes_distance FLOAT NOT NULL,
    metadata TEXT NOT NULL  -- may contain info such as the number of iterations and number of training samples.
    -- mean_manifold_id INTEGER NOT NULL REFERENCES Manifolds ON DELETE RESTRICT,
    -- a unified "procrustean" manifold could be built after training, but we will instead translate to the original one
);

CREATE TABLE IF NOT EXISTS OrthogonalProcrustesModel (
    gpa_id INT NOT NULL REFERENCES GeneralizedProcrustesAnalysis ON DELETE RESTRICT,
    embedding_id INT NOT NULL REFERENCES Embeddings ON DELETE RESTRICT,
    scale FLOAT NOT NULL,
    orthogonal_matrix BLOB NOT NULL,
    PRIMARY KEY (gpa_id, embedding_id)
);
