from .thumbnails import ThumbnailsDatabase
from .search import SearchDatabase


class Database(SearchDatabase, ThumbnailsDatabase):
    pass
