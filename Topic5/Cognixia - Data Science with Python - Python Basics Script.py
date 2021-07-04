# -*- coding: utf-8 -*-
"""
@author: Hemant Rathore
"""

### Introduction to Python - 
## 1. Environment Setup  & using Python in Interactive Mode


import os
import sys

# Know your python installation path


os.path.dirname(sys.executable)


## know & set your working directory


os.getcwd()

os.chdir("D:\Data Science Training\Data")

os.getcwd()


# help

help(os.getcwd)


# Interactive Python Mode - Use it as a calculator

1+2

2*4

a = 3/4

a

print(a)

b = a+2
 
print(b)


a,b = 100,200.5

print(a,b)

print("hello")


# printing variables along with text :

 
# %s - String 

# %d - Integers

# %f - Floats


print('first num is %d '%a)

name = "Python"

print('i am learning %s '%name)



##### 2. Data Structures in Python

#### 2.1 List - Array type structure, it supports different datatype in 1 single list

## to create list use []


a = [1,2,3,4]

print(a)

type(a)

b = ['a','b','c','d']


print(b)

# OR

b = ["a","b","c","d"]

print(b)


c = [1.2,2.3,3.4,4.5]

print(c)


# different datatypes

d = ['a',1,2.345,1+2.34j]


print(d)


## Accessing List elements

range(1,100)

l1 = list(range(1,100))

print(l1)

# indexes start with 0

l1[0]

l1[-1]

l1[98]

l1[1:5] # staring from start position upto (n-1)th position


l1[1:]

l1[:5]

l1[-5:-1] # upto second last

# List Concatenation

l2 = ['a','b','c','d']

l1

l1+l2

# Multipliction on list
 
l2*2


# updaing list element

len(l2)

l2[3] = 5

l2

# adding new element

l2
l2.append([6,7])

l2

l2.extend([8,9])

l2

l2 = l2 + [10,11]

l2

# inserting at specific index (index,value)

l2.insert(2,100)

l2

# deleting element

del(l2[4])

l2

# copying list - assignment is not copy, it creates reference to same object


l3 = l2


l3

l2[2] = 200

l2

l3

# correct way to copy

l4 = l2.copy()

l2[2] = 300

l2

l3

l4

# Creating and accessing nested List


l5 = [1,2,3,[4,5,[6,7]]]

l5

l5[3]

l5[3][0]

l5[3][2]

l5[3][2][0]



#### 2.2 tuple - immutable list - read only

## to create tuple use ()

t1 = (1,2,3,4)

print(t1)

type(t1)

t1[0]

t1[1:3]

t1[-1]

# assignment not possible

t1[3] = 4


# multiplication

t1*2

# tuple concatenation

t1 + (5,) 


#### 2.3 dictionary - hash table type structure, contains key-value pairs

## to create dictionary use {}


d1 = {"name1":"value1","name2":"value2"}

print(d1)

type(d1)

# accessing dict.

d1["name1"]

d1["name2"]


d1.keys()

d1.values()


# add new element


d1.update({"name3" :"value3"})

d1

# update existing element

d1.update({"name2" :"value20"})

d1

# deleting elements

del(d1["name2"])

d1



d1 = {1:10,"name2":"value2"}

d1[1]


### 3 Control Structures in Python

## 3.1 If Else Control Structure


var1 = 200

var2 = 50



if var1 >100 and var2 <= 100:
    print(var1)
        


    
# if else
    

if var1 >200 and var2 <= 100:
    print(var1)
    
else:
    print(var2)
    

    
# else-if
    
if var1 >200:
    print(var1)

elif var2< 50:
    print(var2)
else:
    print("nothing")
    

     

## 3.2 Loops

# for loop


num = list(range(100))

for i in num:
    print(num[i])
    
    
    

# using Continue to skip sequence
    
num = list(range(10))

for i in num:
    if i == 5:
        continue
    print(num[i])




# while loop
    
    
count = 10

while count > 1:
    print(count)
    count -= 1 # count = count -1
    

# using break
    
count = 10

while count > 1:
    if count==5:
        break
    print(count)
    count -= 1 
    


### 4 Creating Custom Functions
    
def my_fun(name = 'Ram'):
    print("Hello, My name is %s"%name)
     

my_fun()

my_fun('John')

# function with return
    
def my_fun(name = 'Ram'):
    
    return("Hello, My name is %s"%name)
    
out = my_fun()  

print(out)
    
## Exception Handling


def simple_fun(n):
    print(n)

l0 = [1, 2, 3, 4, 5]


for i in range(20):
    simple_fun(l0[i])
    
    
for i in range(20):
   try:
        simple_fun(l0[i])
   except IndexError: # Raised when accessing a non-existing index of a list
        simple_fun(0)
        print("Index Error-exception for i = %d"%i)
   finally:
       print("done!")
       
# Some exception examples are ArithmeticError, OverflowError, ZeroDivisionError etc.       
       
# For handling all type of exceptions
       
for i in range(20):
   try:
        simple_fun(l0[i])
   except Exception: # Raised in case of all the exceptions
        simple_fun(0)
        print("faced some Exceptiones for i = %d"%i)
   finally:
       print("Done!")
        

