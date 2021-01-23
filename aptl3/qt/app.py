import argparse
import logging
from typing import List, Optional

from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtSql import QSqlDatabase
from PyQt5.QtWidgets import QApplication, QStyleFactory

from ..db import Database


class Application(QApplication):

    def __init__(self, argv: List[str]) -> None:
        super().__init__(argv)

        parser = argparse.ArgumentParser(description='Launch a Qt application to explore and search your media.')
        parser.add_argument('db', type=str, help='database data directory')
        parser.add_argument('--log', default='WARNING', type=str,
                            help='logging level (defaults to WARNING)',
                            choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'])
        args = parser.parse_args()
        logging.basicConfig(level=args.log.upper(),
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M:%S')

        self._db: Optional[QSqlDatabase] = None
        self.py_db: Database = Database(args.db)
        self.connect_db()

        self.setStyle(QStyleFactory.create("Fusion"))
        dp = QPalette()  # Dark palette
        dp.setColor(QPalette.Window, QColor(53, 53, 53))
        dp.setColor(QPalette.WindowText, Qt.white)
        dp.setColor(QPalette.Base, QColor(25, 25, 25))
        dp.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dp.setColor(QPalette.ToolTipBase, Qt.white)
        dp.setColor(QPalette.ToolTipText, Qt.white)
        dp.setColor(QPalette.Text, Qt.white)
        dp.setColor(QPalette.Button, QColor(53, 53, 53))
        dp.setColor(QPalette.ButtonText, Qt.white)
        dp.setColor(QPalette.BrightText, Qt.yellow)
        dp.setColor(QPalette.Link, QColor(42, 130, 218))

        dp.setColor(QPalette.Highlight, QColor(255, 210, 43))  # 42, 130, 218))
        dp.setColor(QPalette.HighlightedText, Qt.black)

        self.setPalette(dp)
        self.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; } ")

    def connect_db(self):
        self.disconnect_db()
        db = QSqlDatabase.addDatabase('QSQLITE')
        db.setConnectOptions('QSQLITE_OPEN_READONLY;QSQLITE_OPEN_URI;QSQLITE_ENABLE_SHARED_CACHE')
        db.setDatabaseName(self.py_db.get_db_file())
        if not db.open() or not db.exec(Database.get_attach_results_db_query()):
            raise RuntimeError(db.lastError().text())
        self._db = db

    def disconnect_db(self):
        if self._db is not None:
            self._db.close()
            QCoreApplication.processEvents()
