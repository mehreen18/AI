#Write a function that takes a list of student scores and returns how many passed (score ≥ 60), how many failed, and the average score
scores = []

def student():
    for i in range(5):
        score = int(input(f"Enter score for student {i+1}: "))
        scores.append(score)   

        if score >= 60:
            print("This student has successfully passed the exam")
        else:
            print("This student should try again")

    average_score = sum(scores) / len(scores)
    print(f"Average student score is {average_score:.2f}")

student()
print(scores)

#Given a sentence as a string, count how many times each word appears and 
# store the result in a dictionary.
#  Print words that appear more than once.
string="do what you want want"
count=string.count("want")
print(count)
data={count}
print(data)


#Write a program that reads a list of filenames like 
# ['data.csv','notes.txt','model.py','test.csv'] and separates
#  them into groups by file extension using a dictionary.
filenames=["data.csv","notes.txt","model.py","test.csv"]
groupby_extentions={}

for filename in filenames:
    ext=filename.split('.')[-1]
    if ext not in groupby_extentions:
        groupby_extentions[ext]=[]

    groupby_extentions[ext].append(filename)   

print(groupby_extentions)


#Create a function that takes a list of numbers and returns 
# a new list with all duplicates removed but
#the original order kept. Do NOT use set() directly.
lst=[]
result=set()
def removeDuplicate():
    global result
    for i in range(6):
        n=int(input(f"enter some numbers{i+1}")) 
        lst.append(n)
    result=set(lst)
removeDuplicate()        
print(result)        

#Write a function flatten(nested) that takes a deeply nested
#  list like [1,[2,[3,4]],5] and returns [1,2,3,4,5]. Use recursion.

def flatten(nested):
    result=[]
    for item in nested:
        if isinstance(item,list):
            result.extend(flatten(item)) 
        else :
            result.append(item)
    return result        
print(flatten([2,[3,4],5,6]))  

#Build a simple contact book
#  using a dictionary. Support add, search, update,
#  and delete operations through a menu loop.
info = {}

def add():
    name = input("Enter student name: ").strip()
    email = input("Enter email: ").strip()
    phone_str = input("Enter phone number: ").strip()
    try:
        phone = int(phone_str)
    except ValueError:
        print("Invalid phone number. Contact not added.")
        return
    info[name] = {"email": email, "phone": phone}
    print(f"Contact {name} added.")


def update():
    name = input("Enter name to update: ").strip()
    if name in info:
        email = input("Enter new email (leave blank to keep): ").strip()
        phone_input = input("Enter new phone number (leave blank to keep): ").strip()
        if email:
            info[name]['email'] = email
        if phone_input:
            try:
                info[name]['phone'] = int(phone_input)
            except ValueError:
                print("Invalid phone number. Update aborted.")
                return
        print(f"Name {name} updated successfully")
    else:
        print("Name not found")


def delete():
    name = input("Enter name to delete: ").strip()
    if name in info:
        del info[name]
        print(f"{name} is deleted successfully")
    else:
        print("Name not found")


def show():
    if info:
        for name, details in info.items():
            print(f"{name}, {details.get('phone')}, {details.get('email')}")
    else:
        print("Empty list")


while True:
    print(" ----------menu-----------")
    print("1 add contact")
    print("2 update contact by name")
    print("3 delete contact by name")
    print("4 show all")
    print("5 exit")
    choice = input("Enter your choice: ").strip()
    if choice == "1":
        add()
    elif choice == "2":
        update()
    elif choice == "3":
        delete()
    elif choice == "4":
        show()
    elif choice == "5":
        print("good bye")
        break
    else:
        print("Invalid choice")


    


# **kwargs → keyword arguments
def show(**info):
 for key, val in info.items():
  print(f"{key}: {val}")
show(name="Ali", age=22, city="Lahore")



# Matrix multiplication

import numpy as np
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
print(np.dot(a, b))

# POST request (used for AI APIs)
import requests
url = "GROQ_API_KEY"
headers = {
 "Authorization": "Bearer YOUR_API_KEY",
 "Content-Type" : "application/json"
}
body = {
 "model" : "groq",
 "messages": [{"role": "user", "content": "Hello!"}]
}
response = requests.post(url, headers=headers, json=body)
print(response.json())








import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model    = "llama-3.3-70b-versatile",
    messages = [{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)



#####




import torch
a=torch.tensor([1,2,3,4,5,6])

print(a.dtype)
print(a.reshape(2,3))
c=torch.zeros((2,3))
print(c)


import torch
##creating tensor

a=torch.zeros((2,3))
print(a)
b=torch.ones(4,5)
print(b)
c=torch.rand((2,3))
print(c)
d=torch.arange(0,10,2)
print(d)
e=torch.linspace(0,10,10)
print(e)
f=torch.eye(2,3)
print(f)
g=torch.full((3,3),15)
print(g)

## tensor shapes
import torch
x=torch.tensor([1,2,3,4,5])
print(x)
print(x.shape)
print(torch.empty_like(x))
print(torch.zeros_like(x))
print(torch.ones_like(x))

##tensor datatypes
import torch
a=torch.tensor([1.2,2.0,3.0,4.0], dtype=torch.int32)
print(a)
b=torch.tensor([1,2,3,4],dtype=torch.float64)
print(b)
c=torch.tensor([1,2,3,4], dtype=torch.float32)
print(c)

##methamatical operations
import torch
x=torch.tensor([1,2,3,4,5,6])
print(x)
print(x+2)
print(x-2)
print(x/2)
print(x**2)

## element wise operations
import torch
a=torch.rand(2,3)
b=torch.rand(2,3)
print(a)
print(b)
print(a+b)
print(a-b)
print(a/b)
print(a**b)

c=torch.tensor([1,-2,3,-4])
r=torch.abs(c)
print(r)
s=torch.neg(c)
print(s)

