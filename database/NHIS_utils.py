"""Import NHIS csv data to Postgresql database."""
import pandas as pd
from sqlalchemy import create_engine

from . import dbs


def to_sql():
    engine = create_engine('postgresql://alexandreperez@localhost:5432/nhis')
    NHIS = dbs['NHIS']

    for name, path in NHIS.frame_paths.items():
        print(f'\n{name}\n\tReading csv')
        df = pd.read_csv(path)
        print(f'\tLowering columns')
        df.columns = [c.lower() for c in df.columns]
        print(f'\tConverting to sql')
        df.to_sql(name, engine)


