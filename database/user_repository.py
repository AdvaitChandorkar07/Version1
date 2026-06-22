
from database.db import get_connection
from models.user import User

class UserRepository:

    @staticmethod
    def create_user(user):

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO users
            VALUES (?, ?, ?, ?)
            """,
            (
                user.user_id,
                user.name,
                user.semantic_path,
                user.steering_path
            )
        )

        conn.commit()
        conn.close()

    @staticmethod
    def get_user(user_id):

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *
            FROM users
            WHERE user_id=?
            """,
            (user_id,)
        )

        row = cursor.fetchone()

        conn.close()

        if row is None:
            return None

        return User(*row)

    @staticmethod
    def update_vectors(
        user_id,
        semantic_path,
        steering_path
    ):

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE users
            SET semantic_path=?,
                steering_path=?
            WHERE user_id=?
            """,
            (
                semantic_path,
                steering_path,
                user_id
            )
        )

        conn.commit()
        conn.close()