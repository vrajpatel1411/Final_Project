import mysql.connector


class DatabaseConnection:

    def __init__(self):

        self.host = 'bpckxdhheipyolgzptzd-mysql.services.clever-cloud.com'
        self.user = 'uwrcopx1vlhfian5'
        self.password = 'gvAZNDPZR5uMAamqkg3m'
        self.port = 3306
        self.database = 'bpckxdhheipyolgzptzd'

        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                port=self.port,
                database=self.database
            )
            print("Database connection established successfully!")
        except mysql.connector.Error as err:
            print("Connection error:", err)




    def close(self):
        if self.connection:
            self.connection.close()
            print("Database connection closed.")

    def findOne(self, query):

        if not self.connection:
            print("Error: Database connection not established.")
            return None

        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            return result

        except mysql.connector.Error as err:
            print("Error executing query:", err)
            return None

    def fetch_all(self, query):

        if not self.connection:
            print("Error: Database connection not established.")
            return None

        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return result

        except mysql.connector.Error as err:
            print("Error executing query:", err)
            return None
