from sys import argv, addaudithook

commands = [
    "qt",
    "manifolds",
    "thumbs",
    "load_coco",
    "build_procrustes",
]


_LOGGED_EVENTS = [
    'os.chflags',
    'os.chmod',
    'os.chown',
    'os.link',
    'os.remove',
    'os.removexattr',
    'os.rename',
    'os.rmdir',
    'os.setxattr',
    'os.startfile',
    'os.symlink',
    'os.system',
    'os.truncate',
    'shutil',
]


def _audit_hook(name: str, args: tuple) -> None:
    log = False

    if name == 'open':
        file, mode, flags = args
        if mode is not None and ('w' in mode or '+' in mode):
            log = True
    if any(e in name for e in _LOGGED_EVENTS):
        log = True

    if log:
        print(f'Audited {name}({", ".join(map(repr, args))})')


def main_():
    addaudithook(_audit_hook)

    if len(argv) >= 2:
        cmd = argv.pop(1)
    else:
        cmd = ""
    argv[0] += ' ' + cmd

    if cmd == 'qt':
        from .qt.__main__ import main
        return main()
    if cmd == 'manifolds':
        from .scripts.manifolds import main
        return main()
    if cmd == 'thumbs':
        from .scripts.thumbs import main
        return main()
    if cmd == 'load_coco':
        from .scripts.load_coco import main
        return main()
    if cmd == 'build_procrustes':
        from .scripts.build_procrustes import main
        return main()
    if cmd == 'load_dir':
        from .scripts.load_dir import main
        return main()

    print("Unrecognized command.")
    print("Please choose one from " + ", ".join(commands) + ".")
    return 1


if __name__ == '__main__':
    exit(main_())
