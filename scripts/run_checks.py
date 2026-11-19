"""Standalone synthetic-only verification; no Git or patient artifact required."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pytest

def main():
    original=Path.open
    def guarded(path,*args,**kwargs):
        if path.suffix=='.psv' and (ROOT/'data/raw') in path.resolve().parents:
            raise PermissionError('This release checks invented fixtures only')
        return original(path,*args,**kwargs)
    Path.open=guarded
    try:return pytest.main(['-q',str(ROOT/'tests')])
    finally:Path.open=original

if __name__=='__main__':raise SystemExit(main())
