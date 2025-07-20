# -*- encoding: utf-8 -*-



import unittest
import sqlalchemy as sa
from datetime import datetime

# TODO: use the data.py version
def create_table(engine, md, desc):
    """Create a table according to description `desc`. The description
    contains keys:
    * `name` – table name
    * `columns` – list of column names
    * `types` – list of column types. If not specified, then `string` is
      assumed
    * `data` – list of lists representing table rows

    Returns a SQLAlchemy `Table` object with lodaded data.
    """

    TYPES = {
            "integer": sa.Integer,
            "string": sa.String,
            "date": sa.Date,
            "id": sa.Integer,
    }
    table = sa.Table(desc["name"], md,
                     sa.Column("id", sa.Integer, primary_key=True))

    types = desc.get("types")
    if not types:
        types = ["string"] * len(desc["columns"])

    col_types = dict(list(zip(desc["columns"], desc["types"])))
    for name, type_ in list(col_types.items()):
        real_type = TYPES[type_]
        if type_ == 'id':
            col = sa.Column(name, real_type, primary_key=True)
        else:
            col = sa.Column(name, real_type)

        table.append_column(col)

    # Create tables and insert data with explicit connection (SQLAlchemy 2.x)
    with engine.begin() as conn:
        md.create_all(conn)
        
        buffer = []
        for row in desc["data"]:
            record = {}
            for key, value in zip(desc["columns"], row):
                if col_types[key] == "date":
                    value = datetime.strptime(value, "%Y-%m-%d")
                record[key] = value
            buffer.append(record)

        for row in buffer:
            conn.execute(table.insert().values(row))

    return table


class SQLTestCase(unittest.TestCase):
    """Class with helper SQL assertion functions."""

    def create_engine(self, connection=None):
        self.engine = sa.create_engine(connection or "sqlite://")
        self.metadata = sa.MetaData(self.engine)

    def assertColumnEqual(self, left, right):
        """Assert that the `left` and `right` columns have equal base columns
        depsite being labeled."""

        self.assertCountEqual(left.base_columns, right.base_columns)

    def table(self, name):
        """Return fully reflected table `name`"""
        return self.metadata.table(name, autoload=True)
