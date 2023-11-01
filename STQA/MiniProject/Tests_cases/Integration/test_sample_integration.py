import unittest
import sys
sys.path.insert(1, '/home/freakcap/Desktop/STQA/Application')
from backend import Database
database = Database("../../books.db")

class Wrapper:

    def insert(self,title,author,year,isbn):
        database.insert(title,author,year,isbn)
        rows = database.search(title,author,year,isbn)
        database.delete(rows[0][0])
        if(len(rows)==1):
            return True
        return False

    def search(self,title="",author="",year="",isbn=""):
        database.insert(title,author,year,isbn)
        rows = database.search(title)
        database.delete(rows[0][0])
        for row in rows:
            if(row[1]==title):
                return True
        return False
    
    def update(self,updISBN,title,author,year,isbn):
        database.insert(title,author,year,isbn)
        rowsBefore = database.search(title)
        database.update(rowsBefore[0][0],title,author,year,updISBN)
        rowsAfter = database.search(title)
        database.delete(rowsBefore[0][0])
        for row in rowsAfter:
            if(row[0]==rowsBefore[0][0] and row[4]==updISBN):
                return True
        return False

    def delete(self,title,author,year,isbn):
        database.insert(title,author,year,isbn)
        rows = database.search(title,author,year,isbn)
        for row in rows:
            database.delete(row[0])
        rowsAfter = database.search(title,author,year,isbn)
        if(len(rowsAfter)==0):
            return True
        return False

class IntegrationTestsApp(unittest.TestCase):

    wrapper = Wrapper()
    title = "test"
    author = "test"
    year = "1999"
    isbn = "testISBN"

    def test_INSERT(self):
        flag = self.wrapper.insert(self.title,self.author,self.year,self.isbn)
        self.assertTrue(flag,"INSERT operation failed.")
        print("INSERT integration passed.")

    def test_SEARCH(self):
        flag = self.wrapper.search(self.title,self.author,self.year,self.isbn)
        self.assertTrue(flag,"SEARCH operation failed.")
        print("SEARCH integration passed.")

    def test_UPDATE(self):
        flag = self.wrapper.update("updatedISBN",self.title,self.author,self.year,self.isbn)
        self.assertTrue(flag,"UPDATE operation failed.")
        print("UPDATE integration passed.")

    def test_DELETE(self):
        flag = self.wrapper.delete(self.title,self.author,self.year,self.isbn)
        self.assertTrue(flag,"DELETE operation failed.")
        print("DELETE integration passed.")
    
if __name__ == '__main__':
    unittest.main()
    
