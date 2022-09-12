from pathlib import Path

from .LFDJPG import LFDJPG

class LFDIMG():

    @staticmethod
    def load(filepath, loader_func=None):
        if filepath.suffix == '.jpg':
            return LFDJPG.load ( str(filepath), loader_func=loader_func )
        else:
            return None
