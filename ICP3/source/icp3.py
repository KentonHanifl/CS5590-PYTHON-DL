'''
ICP 3 CS 5590 UMKC
Kenton Hanifl
2/8/2019
'''

from bs4 import BeautifulSoup
import requests

# ======= EMPLOYEE AND FULLTIME EMPLOYEE CLASSES ======= 
class Employee:
    '''
    Represents an employee who can have any number of hours per day
    (As opposed to a FullTimeEmployee who must work 40 hours per week)
    '''

    EmployeeList = []
    #defaults: name = "no name", family = [], salary = 0, department = "no department", hours = 0
    #currently just assigned in the input loop
    def __init__(self, name, family, salary, department, hours):
        self.name = name
        self.family = family
        self.salary = salary
        self.department = department
        self.hours = hours
        Employee.EmployeeList.append(self) # for getting total employee count

    def __str__(self):
        return "employee with name: {0}, family: {1}, salary: {2}, department: {3}, hours: {4}".format(self.name, self.family, self.salary, self.department, self.hours)
    
    def getWeeklyPay(self):
            return self.salary * self.hours * 7

    # -- Start Static Functions --
    def getEmployeeCount():
        return len(Employee.EmployeeList)    

    def getAverageSalary():
        '''
        Gets the number of employees and salary for each employee
        returns the average salary for each employee
        '''
        Count = Employee.getEmployeeCount()
        
        # take care of divide by 0 error
        if Count == 0:
            return 0
        
        Total = 0
        for employee in Employee.EmployeeList:
            Total += employee.salary
        AVG = Total / Count
        return AVG

    def deleteAllEmployees():
        Employee.EmployeeList.clear()

class FullTimeEmployee(Employee):
    '''
    Represents a fulltime employee who must work 40 hours per week
    Inherits all methods from Employee
    '''

    def __init__(self, name, family, salary, department):
        super().__init__(name, family, salary, department, 40) # calls employee's init but sets hours to 40

    def __str__(self):
        return "full-time employee with name: {0}, family: {1}, salary: {2}, department: {3}, hours: {4}".format(self.name, self.family, self.salary, self.department, self.hours)
    
# ======= PARSE HTML PAGE AND PRINT =======
def ParsePage(file):
    '''
    Parses an HTML file using BeautifulSoup4
    Returns the title of the page and any links in the <a> tags in the page in a tuple
    '''
    #with open(filename, encoding = "utf8") as page:
    fileplain = file.text
    soup = BeautifulSoup(fileplain,"html.parser")
        
    title = soup.title.string
    
    links = []
    
    for link in soup.find_all('a'):
        href = link.get('href')
        text = link.string
        links.append((text,href)) #append to list in tuples
        
    return (title, links)

def PrintLinks(links, num = -1):
    '''
    Takes in links in the form:
    tuple(text, href)

    Prints them out in the form:
    text: <link text>
    link: <href>

    Optionally, you can specify a number of lines to print.
    By default (or if a negative number is entered) it will print all links.
    '''
    print("====== LINKS ======")
    i = num if num >= -1 else len(links)
    for text,href in links:
        if i == 0:
            break
        # Prints are on separate lines because if the href or text are None, they can't be concatenated. This way will still print None.
        print("text: ", end = "")
        print(text)
        print("link: ", end = "")
        print(href, end = "\n\n\n")
        i -= 1
    print("======= END =======")

# ======= INPUT LOOPS =======
def employeeLoop():
    '''
    A simple input loop for creating both types of employees, printing the full list and average salary, and deleting the employees.
    '''
    cont = True
    
    while cont:
        
        choice = input("1) Create an Employee\n2) Create a full-time employee\n3) Print a list of all employees\n4) Print the average salary of all employees\n5) Delete all employees\n6) Quit\n")
        #if no input, loop
        if len(choice) == 0:
            continue
        
        if choice[0] == "1":
            #get name, family, salary, department, hours and sanitize input
            nameRaw = input("Please input a name OR press enter to assign no name\n")
            name = nameRaw if nameRaw != "" else "no name"
            familyRaw = input("Please input names for family separated by spaces OR press enter to assign no family\n")
            familyList = familyRaw.split(" ")
            family = familyList if familyList != [] else ["no family"]
            salaryRaw = input("Please input a salary OR press enter to assign no salary\n")
            salary = int(salaryRaw) if salaryRaw.isdigit() else 0
            departmentRaw = input("Please input a department OR press enter to assign no department\n")
            department = departmentRaw if departmentRaw != "" else "no department"
            hoursRaw = input("Please input a number of hours OR press enter to assign no hours\n")
            hours = int(hoursRaw) if hoursRaw.isdigit() else 0             

            #create employee
            newEmployee = Employee(name, family, salary, department, hours)
            print("Created: ",end="")
            print(newEmployee)

        elif choice[0] == "2":
            #get name, family, salary, department and sanitize input
            nameRaw = input("Please input a name OR press enter to assign no name\n")
            name = nameRaw if nameRaw != "" else "no name"
            familyRaw = input("Please input names for family separated by spaces OR press enter to assign no family\n")
            familyList = familyRaw.split(" ")
            family = familyList if familyList != [] else ["no family"]
            salaryRaw = input("Please input a salary OR press enter to assign no salary\n")
            salary = int(salaryRaw) if salaryRaw.isdigit() else 0
            departmentRaw = input("Please input a department OR press enter to assign no department\n")
            department = departmentRaw if departmentRaw != "" else "no department"           

            #create full-time employee
            newFullEmployee = FullTimeEmployee(name, family, salary, department)
            print("Created: ",end="")
            print(newFullEmployee)
            
        elif choice[0] == "3":
            for employee in Employee.EmployeeList:
                print(employee)

        elif choice[0] == "4":
            print(Employee.getAverageSalary())

        elif choice[0] == "5":
            Employee.deleteAllEmployees()
            print("Deleted all employees")

        else:
            cont = False

def linksLoop():
    '''
    A simple input loop for reading in an HTML file from the web, printing the title
    '''
    cont = True
    links = []
    title = ""
    while cont:

        choice = input("1) Choose a website to parse\n2) Output the first (n) links in the last read file\n3) Print the title of the page\n4) Quit \n")
        #if no input, loop
        if len(choice) == 0:
            continue
        
        if choice[0] == "1":
            website = input("Please input a website name\n")
            try:
                print("please wait...")
                file = requests.get(website)
                (title,links) = ParsePage(file)
                print("done")
            except:
                print("There was a problem getting or parsing the HTML file.\n")

        elif choice[0] == "2":
            numLinksStr = input("How many links would you like to output? (type nothing to print all lines)\n")
            numLinks = int(numLinksStr) if numLinksStr.isdigit() else -1
            PrintLinks(links,numLinks)

        elif choice[0] == "3":
            print(title)

        else:
            cont = False

def main():
    employeeLoop()
    linksLoop()
    
main()
    

