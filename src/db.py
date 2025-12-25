import psycopg2
import pandas as pd


DB_CONFIG = {
    "host": "localhost",
    "dbname": "rcmsys",
    "user": "postgres",
    "password": "123456",
    "port": 5432
}


def get_connection():
    return psycopg2.connect(DB_CONFIG)


def load_unprocessed_events():
    conn = psycopg2.connect(
        host="localhost",
        dbname="rcmsys",
        user="postgres",
        password="123456"
    )

    query = """
    SELECT
        "id",
        "userId",
        "movieId",
        "eventName",
        "createdAt"
    FROM event
    WHERE processed = false
    ORDER BY "createdAt" ASC
    """

    df = pd.read_sql(query, conn)
    conn.close()
    return df


def mark_events_processed(event_ids: list[int]):
    if not event_ids:
        return

    conn = psycopg2.connect(
        host="localhost",
        dbname="rcmsys",
        user="postgres",
        password="123456"
    )
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE event
        SET processed = TRUE
        WHERE id = ANY(%s)
        """,
        (event_ids,)
    )

    conn.commit()
    cursor.close()
    conn.close()