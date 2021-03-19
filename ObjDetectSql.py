import mysql.connector as mysql
from database import db
import json
cursor = db.cursor()


def InsertData(date,time, PATH ):
    sql = "INSERT INTO project (date, time, PATH) VALUES (%s, %s, %s)"
    val = (date, time, PATH)
    cursor.execute(sql, val)
    db.commit()

# def selectAll(date):
#     sql = "SELECT * FROM project".format(date)
#     cursor.execute(sql)
#     result = cursor.fetchall()
#     # print(type(result))
#     return result