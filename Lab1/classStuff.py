class Flight:
    def __init__(self,flightNum,planeCode,startLoc,endLoc,price,startTime,endTime):
        self.flightNum=flightNum
        self.planeCode=planeCode
        self.startLoc = startLoc
        self.endLoc = endLoc
        self.price = price
        self.startTime = startTime
        self.endTime = endTime
        self.passengers = []

    
        

class Person:
    def __init__(self, name="", age=0):
        self.name = name
        self.age = age
        self.flights = []
        
    def bookFlight(self,flightNum):
        self.flights.append(flightNum)
        
        

class Employee(Person):
    def __init__(self, name="", age=0,job=""):
        super().__init__(name,age)
        self.job = job
    

class Passenger(Person,Employee):
    def __init__(self,name="",age=0,flightNum=0)
