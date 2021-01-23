import logging

from PyQt5.QtCore import Qt, QSize, QModelIndex, QByteArray, QCoreApplication
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtSql import QSqlQueryModel, QSqlQuery
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QDialog

from .media_view import MediaView
from ..db import Database


class HomeWidget(QWidget):

    def __init__(self, py_db: Database):
        super().__init__()

        self._py_db = py_db
        self._logger = logging.getLogger(self.__class__.__name__)

        self.setWindowTitle(self.__class__.__module__.rsplit('.', 1)[0])
        self.setContentsMargins(0, 0, 0, 0)

        self.vbox = QVBoxLayout(self)
        self.vbox.setContentsMargins(0, 0, 0, 0)
        self.vbox.setSpacing(0)

        self.text = QLineEdit()
        self.text.setContentsMargins(10, 10, 10, 10)
        self.text.returnPressed.connect(self._start_query)
        self.vbox.addWidget(self.text)

        self.model = QSqlQueryModel()
        # model.setQuery("Select Media.id from Media;")
        self.model.setQuery("SELECT DISTINCT media_id from ManifoldItems NATURAL JOIN Manifolds WHERE ready;")

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

            query = QSqlQuery()
            query.exec('DETACH results')
            if not query.isActive():
                print('detach query:', query.lastQuery())
                print('detach query error:', query.lastError().text())
                print('model query:', self.model.query().lastQuery())
                print('model query error:', self.model.lastError().text())
                return

            query = QSqlQuery()
            query.exec(Database.get_attach_results_db_query())
            if not query.isActive():
                print('attach query:', query.lastQuery())
                print('attach query error:', query.lastError().text())
                return

            line = self.text.text()
            if not line:
                self.model.setQuery("SELECT DISTINCT media_id from ManifoldItems NATURAL JOIN Manifolds WHERE ready;")
                return

            results_id, inserted = self._py_db.search(line, n=1000, search_k=-1)

            query_s = (
                f'WITH ResultsStats AS (SELECT MIN(rank)+0.55 AS min_rank, MAX(rank)-MIN(rank) AS rank_range '
                f'                      FROM results.ResultsMedia WHERE results_id = {results_id:d}) '
                f'SELECT media_id FROM ('
                f'   SELECT media_id, min(rank) AS mrank '
                f'   FROM results.ResultsMedia '
                f'   WHERE results_id = {results_id:d} '
                f'   GROUP BY media_id ORDER BY mrank) '
                f'WHERE mrank - (SELECT min_rank FROM ResultsStats) <= (SELECT rank_range / 2 FROM ResultsStats)')

            query = QSqlQuery()
            query.exec(query_s)
            if not query.isActive():
                self._logger.debug(f'results query: {query.lastQuery():s}')
                self._logger.error(f'results query error: {query.lastError().text():s}')
            self.model.setQuery(query_s)
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            self.setCursor(Qt.ArrowCursor)
