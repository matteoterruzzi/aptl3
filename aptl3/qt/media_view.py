import marshal
import zlib

from PyQt5.QtCore import QMargins, Qt, QSize, QModelIndex, QByteArray
from PyQt5.QtGui import QBrush, QPainter, QPixmap, QPixmapCache
from PyQt5.QtSql import QSqlQuery
from PyQt5.QtWidgets import QListView, QItemDelegate, QStyleOptionViewItem, QStyle

PIXMAP_CACHE_HITS = 0
PIXMAP_CACHE_MISS = 0


class MediaItemDelegate(QItemDelegate):

    def __init__(self) -> None:
        super().__init__()
        self.setClipping(True)
        self.size_hint = QSize(175, 175)

    def sizeHint(self, option: 'QStyleOptionViewItem', index: QModelIndex) -> QSize:
        return self.size_hint
    
    def _get_media_pixmap(self, _id, option: 'QStyleOptionViewItem', size: QSize) -> QPixmap:
        global PIXMAP_CACHE_MISS, PIXMAP_CACHE_HITS
        pix = QPixmapCache.find(str(_id))
        if pix is not None:
            PIXMAP_CACHE_HITS += 1
            return pix
        PIXMAP_CACHE_MISS += 1

        text_url_query = QSqlQuery()
        text_url_query.prepare('select url from MediaLocations where media_id = ? AND url LIKE "data:,%" LIMIT 1;')
        text_url_query.bindValue(0, _id)

        if not text_url_query.exec():
            raise RuntimeError
        if text_url_query.first():
            text = text_url_query.value(0).split('data:,', 1)[1]
            pix = QPixmap(size)
            painter = QPainter(pix)

            c = option.palette.base().color()
            painter.fillRect(pix.rect(), c)

            # painter.setFont( QFont("Arial") );
            c = option.palette.text().color()
            painter.setPen(c)
            painter.drawText(pix.rect().marginsRemoved(QMargins(5, 5, 5, 5)),
                             # Qt.TextWrapAnywhere | Qt.TextJustificationForced,
                             Qt.TextWordWrap,
                             text)
        else:
            thumb_query = QSqlQuery()
            thumb_query.prepare('select thumbnail from Thumbnails where media_id = ?;')
            thumb_query.bindValue(0, _id)

            if not thumb_query.exec():
                raise RuntimeError
            if not thumb_query.first():
                raise KeyError
            pxb = thumb_query.value(0)
            if len(pxb) < 10:
                raise ValueError(len(pxb))
            assert isinstance(pxb, QByteArray), type(pxb)
            # pxd = marshal.loads(zlib.decompress(pxb.data()))['jpeg']
            pxd = pxb
            pix = QPixmap()
            pix.loadFromData(pxd)
            pix = pix.scaled(size, Qt.KeepAspectRatioByExpanding)

            thumb_query.clear()

        QPixmapCache.insert(str(_id), pix)

        text_url_query.clear()

        # print(f'PIXMAP_CACHE_HITS/TOTAL = {PIXMAP_CACHE_HITS/(PIXMAP_CACHE_HITS+PIXMAP_CACHE_MISS):.0%}')
        return pix

    def paint(self, painter: QPainter, option: 'QStyleOptionViewItem', index: QModelIndex) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter_box_rect = option.rect.marginsRemoved(QMargins(1, 1, 1, 1))
        painter_pix_rect = option.rect.marginsRemoved(QMargins(10, 10, 10, 10))
        try:
            painter.save()
            try:
                painter.setClipRect(painter_box_rect, Qt.IntersectClip)
                if option.state & QStyle.State_Selected:
                    painter.fillRect(painter_box_rect, option.palette.highlight())
                    painter.setPen(option.palette.highlight().color())
                else:
                    painter.setPen(option.palette.alternateBase().color())
                painter.drawRect(painter_box_rect)
            finally:
                painter.restore()
            try:
                pix = self._get_media_pixmap(index.data(), option, painter_pix_rect.size())
            except Exception as ex:
                c = option.palette.text().color()
                c.setAlphaF(.15)
                painter.fillRect(painter_pix_rect, QBrush(c, Qt.DiagCrossPattern))

                if not isinstance(ex, KeyError):
                    import traceback
                    traceback.print_exc()
            else:
                painter.drawPixmap(painter_pix_rect, pix, pix.rect())
        except:
            import traceback
            traceback.print_exc()


class MediaView(QListView):
    def __init__(self):
        super().__init__()

        self.delegate = MediaItemDelegate()
        self.setItemDelegate(self.delegate)

        self.setGridSize(self.delegate.size_hint)
        self.setUniformItemSizes(True)

        self.setContentsMargins(0, 0, 0, 0)
        self.setResizeMode(QListView.Adjust)
        self.setViewMode(QListView.IconMode)
        self.setFlow(QListView.LeftToRight)

        self.setSelectionMode(QListView.ExtendedSelection)
        self.setEditTriggers(QListView.NoEditTriggers)

        self.setDragDropMode(QListView.NoDragDrop)

        self.setMinimumSize(
            self.delegate.size_hint.width() * 1 + self.verticalScrollBar().sizeHint().width() + 5, 300)
        self.setBaseSize(1070, 700)

    def sizeHint(self) -> QSize:
        return QSize(1070, 700)
