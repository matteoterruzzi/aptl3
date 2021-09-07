import logging
import os
from urllib.request import pathname2url, url2pathname

from PyQt5.QtCore import Qt, QSize, QModelIndex, QByteArray, QCoreApplication
from PyQt5.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent
from PyQt5.QtSql import QSqlQueryModel, QSqlQuery
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QFileDialog

from .media_view import MediaView
from ..db import Database


class HomeWidget(QWidget):

    def __init__(self, py_db: Database):
        super().__init__()

        self._py_db = py_db
        self._logger = logging.getLogger(self.__class__.__name__)

        self.setWindowTitle(self.__class__.__module__.rsplit('.', 1)[0])
        self.setContentsMargins(0, 0, 0, 0)
        self.setAcceptDrops(True)

        self.vbox = QVBoxLayout(self)
        self.vbox.setContentsMargins(0, 0, 0, 0)
        self.vbox.setSpacing(0)

        self.text = QLineEdit()
        self.text.setContentsMargins(10, 10, 10, 10)
        self.text.returnPressed.connect(self._start_query)
        self.vbox.addWidget(self.text)

        self.model = QSqlQueryModel()
        self.model.setQuery("SELECT media_id from Media;")
        # self.model.setQuery("SELECT DISTINCT media_id from ManifoldItems NATURAL JOIN Manifolds WHERE ready;")

        self.grid = MediaView()
        self.grid.setModel(self.model)
        self.grid.doubleClicked.connect(self._grid_dbl_clicked)
        self.vbox.addWidget(self.grid)

    def sizeHint(self) -> QSize:
        return QSize(1070, 700)

    def closeEvent(self, a0: QCloseEvent) -> None:
        # self.grid.wbig.close()
        pass

    def _grid_dbl_clicked(self, index: QModelIndex):
        _id: QByteArray = index.data()
        self.text.setText(_id.data().hex())
        self._start_query()

    def _start_query(self):
        self.setCursor(Qt.WaitCursor)
        QCoreApplication.processEvents()
        # noinspection PyBroadException
        try:
            self.model.clear()
            QCoreApplication.processEvents()

            line = self.text.text().strip()
            if not line:
                self.model.setQuery("SELECT media_id from Media;")
                # self.model.setQuery("SELECT DISTINCT media_id from ManifoldItems NATURAL JOIN Manifolds WHERE ready;")
                return

            if line == '@import':
                _source_dir = QFileDialog.getExistingDirectory(
                    self, "Select a directory to import", "",
                    QFileDialog.DontResolveSymlinks | QFileDialog.ReadOnly)
                _source_dir = os.path.abspath(_source_dir)
                assert os.path.isdir(_source_dir)
                self._logger.info(f"Starting ingestion of {_source_dir}")
                self._py_db.ingest_file_directory(_source_dir)  # NOTE: this will freeze the GUI for a long time!
                self._py_db.notify_bg_manifold_build()
                self._py_db.thumbs_load()
                self._logger.info(f"Ingestion finished. ")
                _source_dir_url_like = 'file:' + pathname2url(_source_dir) + '%'
                query = QSqlQuery()
                query.prepare('SELECT media_id FROM MediaLocations WHERE url LIKE ? ORDER BY url')
                query.bindValue(0, _source_dir_url_like)
                self.model.setQuery(query)
                return

            results_id, inserted = self._py_db.search(line, n=1000, search_k=-1)

            query_s = (
                f'SELECT media_id FROM results.ResultsMediaFiltered '
                f'WHERE results_id = {results_id:d} ORDER BY rank ASC')

            query = QSqlQuery()
            query.exec(query_s)
            if not query.isActive():
                self._logger.debug(f'results query: {query.lastQuery():s}')
                self._logger.error(f'results query error: {query.lastError().text():s}')
            self.model.setQuery(query)
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            self.setCursor(Qt.ArrowCursor)

    def dragEnterEvent(self, a0: QDragEnterEvent) -> None:
        data = a0.mimeData()
        if not data.hasFormat('text/uri-list'):
            a0.ignore()
            return
        a0.accept()
        a0.setDropAction(Qt.LinkAction)

    def dropEvent(self, a0: QDropEvent) -> None:
        data = a0.mimeData()
        action = a0.dropAction()
        if action == Qt.IgnoreAction:
            a0.ignore()
            return
        if not data.hasFormat('text/uri-list'):
            a0.ignore()
            return
        a0.accept()
        self.setCursor(Qt.WaitCursor)
        self.model.clear()
        QCoreApplication.processEvents()

        ul = data.data('text/uri-list').data().decode().splitlines()
        ul = map(str.strip, ul)
        ul = filter(lambda _uri: not _uri.startswith('#'), ul)
        ul = ['file:/'+_uri[len('file:///'):] if _uri.startswith('file:///') else _uri for _uri in ul]
        ul = list(ul)

        file_binds = []
        dir_binds = []

        # Ingest file and dirs from received uri-list
        for uri in ul:
            if uri.startswith('file:'):
                pathname = url2pathname(uri[len('file:'):])
                if os.path.isdir(pathname):
                    self._logger.info(f'RECURSIVELY ADDING {pathname}')
                    out = self._py_db.ingest_file_directory(pathname)
                    dir_binds.append(f'{uri}%')
                    self._logger.debug(f'ingest_file_directory({pathname}) -> {out}')
                    continue

            self._logger.info(f'ADDING {uri}')
            media_id, flags, ex = self._py_db.try_ingest_url(uri)
            if ex is not None:
                self._logger.exception(f'FAILING {uri}: {ex}')
            else:
                file_binds.append(uri)
                self._logger.debug(f'{media_id=} {flags=} {ex=}')

        self._py_db.notify_bg_manifold_build()
        self._py_db.thumbs_load()

        # Build WHERE expression with file and dir binds
        conditions = []
        file_placeholders = ', '.join('?'*len(file_binds))
        if file_placeholders:
            file_placeholders = f'url IN ({file_placeholders})'
            conditions.append(file_placeholders)
        conditions.extend(['url LIKE ?']*len(dir_binds))

        # Prepare query and bind values for file and dirs
        query = QSqlQuery()
        query.prepare(f'SELECT media_id FROM MediaLocations WHERE {" OR ".join(conditions)} ORDER BY url')
        i = -1
        for i, bind in enumerate(file_binds):
            query.bindValue(i, bind)
        for j, bind in zip(range(i + 1, i + 1 + len(dir_binds)), dir_binds):
            query.bindValue(j, bind)
        query.exec()

        if not query.isActive():
            self._logger.debug(f'results query: {query.lastQuery():s}')
            self._logger.error(f'results query error: {query.lastError().text():s}')

        self.model.setQuery(query)
        self.setCursor(Qt.ArrowCursor)
