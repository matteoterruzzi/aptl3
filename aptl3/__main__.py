from sys import argv

commands = [
    "qt",
    "manifolds",
    "make_thumbs",
    "load_coco",
    "build_procrustes",
]

def main_():
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
    if cmd == 'make_thumbs':
        from .scripts.make_thumbs import main
        return main()
    if cmd == 'load_coco':
        from .scripts.load_coco import main
        return main()
    if cmd == 'build_procrustes':
        from .scripts.build_procrustes import main
        return main()

    print("Unrecognized command.")
    print("Please choose one from " + ", ".join(commands) + ".")
    return 1

if __name__ == '__main__':
    exit(main_())


