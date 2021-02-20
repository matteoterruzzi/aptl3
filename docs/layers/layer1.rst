
Layer 1 API (aptl3.db)
======================


.. currentmodule:: aptl3.db
.. py:currentmodule:: aptl3.db

.. autoapi-inheritance-diagram:: aptl3.db.Database
    :parts: 1


.. code-block:: python

    from tempfile import TemporaryDirectory
    from aptl3.db import Database
    # Use a temp directory to avoid leaving traces behind.
    with TemporaryDirectory() as _data_dir:
        db = Database(data_dir=_data_dir)
        e = db.get_embedding_id("sentence")
        m, _ = db.ingest_url("data:,A test sentence")
        db.ingest_url("data:,Another test sentence")
        db.build_new_manifold(embedding_id=e)
        r, n = db.search(m.hex())
        print(n, 'results')
        for row in db.execute("SELECT * FROM results.ResultsMedia WHERE results_id = ?", (r,)):
            print(row)

See the complete :doc:`/autoapi/index`.
The main classes and methods of `aptl3.db` and its submodules are unified in this page for convenience of the reader.

As the above code may suggest, the API is accessible by importing a single class:
`aptl3.db.Database` which is constructed via multiple inheritance from the superclasses listed below.


Schema
-----------------

.. autoapiclass:: aptl3.db.locations.SchemaDatabase
    :noindex:
    :members:
    :undoc-members:
    :show-inheritance:

Locations
-----------------

.. autoapiclass:: aptl3.db.locations.LocationsDatabase
    :noindex:
    :members:
    :undoc-members:
    :show-inheritance:

Embeddings
-----------------

.. autoapiclass:: aptl3.db.embeddings.EmbeddingsDatabase
    :noindex:
    :members:
    :undoc-members:
    :show-inheritance:

Manifolds
-----------------

.. autoapiclass:: aptl3.db.manifolds.ManifoldsDatabase
    :noindex:
    :members:
    :undoc-members:
    :show-inheritance:

Relations
-----------------

.. autoapiclass:: aptl3.db.relations.RelationsDatabase
    :noindex:
    :members:
    :undoc-members:
    :show-inheritance:

Procrustean
-----------------

.. autoapiclass:: aptl3.db.procrustean.ProcrusteanDatabase
    :noindex:
    :members:
    :undoc-members:
    :show-inheritance:

Search
-----------------

.. autoapiclass:: aptl3.db.search.SearchDatabase
    :noindex:
    :members:
    :undoc-members:
    :show-inheritance:

Thumbnails
-----------------

.. autoapiclass:: aptl3.db.thumbnails.ThumbnailsDatabase
    :noindex:
    :members:
    :undoc-members:
    :show-inheritance:

Full database API
----------------------------------

.. autoapiclass:: aptl3.db.Database
    :noindex:
    :members:
    :undoc-members:
    :show-inheritance:

See also the complete :doc:`/autoapi/index`.
