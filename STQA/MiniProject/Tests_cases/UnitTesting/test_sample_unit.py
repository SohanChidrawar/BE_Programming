import unittest
import sys
sys.path.insert(1, '/home/freakcap/Desktop/STQA/Application')
from backend import Database
database = Database("../../books.db")

class UnitTestsApp(unittest.TestCase):

    def test_nonNumericYear_INSERT(self):
        flag = database.insert("bookt","i am auth","abcd","IBN000df")
        self.assertFalse(flag,"Non Numeric year should not be allowed.")
    
    def test_floatYear_INSERT(self):
        flag = database.insert("bookt","i am auth","1967.45","IBN000df")
        self.assertFalse(flag,"Float year should not be allowed.")
    
    def test_emptyFields_INSERT(self):
        flag = database.insert("","",1920,"")
        self.assertFalse(flag,"Incomplete or blank arguments in insert function are not valid.")

    def test_shortTitle_INSERT(self):
        flag = database.insert("b","i am auth","1967","IBN000df")
        self.assertFalse(flag,"Title less than 2 letters should not be allowed.")
    
    def test_shortAuthorName_INSERT(self):
        flag = database.insert("bookt","ia","1967","IBN000df")
        self.assertFalse(flag,"Author name less than 3 letters should not be allowed")

    def test_VIEW(self):
        result = database.view()
        self.assertTrue(len(result)>0,"Database is empty !!")
    
    def test_SEARCH(self):
        result = database.search("","","","")
        self.assertFalse(result,"At least one search parameter should be present for success.")

if __name__ == '__main__':
    unittest.main()
    
