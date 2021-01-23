import sys

from .app import Application
from .home_widget import HomeWidget


def main():
    # Database.initialize_static_embeddings()

    # Qt
    app = Application(sys.argv)
    w = HomeWidget(app.py_db)
    w.show()

    # import cProfile
    # cProfile.run('app.exec()', sort='cumtime')

    app.exec()


if __name__ == '__main__':
    main()
