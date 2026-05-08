import matplotlib.pyplot as plt
import numpy as np

x=[1,2,3,4,5]
y=[10,14,12,18,20]

plt.plot(x,y,marker='o')
plt.title("Sample Line Chart")
plt.xlabel("Time")
plt.ylabel("Distance")
plt.grid(True)
plt.show()

students = ["A", "B", "C", "D", "E"]
marks = [85, 70, 90, 65, 95]
plt.bar(students, marks)
plt.title("Student Marks")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.show()

students = ["A", "B", "C", "D", "E"]
marks = [85, 70, 90, 65, 95]
plt.bar(students, marks, color=["red", "blue", "green", "orange", "purple"])
plt.title("Student Marks")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.show()

activities = ["Study", "Exercise", "Sleep", "Hobbies"]
hours = np.array([4, 1, 7, 2])
plt.pie(hours,labels=activities)
plt.show()