# 2MA918 / 2MA404 / 2MA62Ä | Optimization | 5 hp | 10/11/2025 - 18/01/2026

## Optimization and Optimization Methods

Welcome to this course in optimization for 2MA404, 2MA62Ä and 2MA918!

**Teacher and examiner**: Jonas Nordqvist ([jonas.nordqvist@lnu.se](mailto:jonas.nordqvist@lnu.se))
**Teaching assistant**: Jakob Dautovic ([jakob.dautovic@lnu.se](mailto:jakob.dautovic@lnu.se))

**Literature:**

* *Optimization* by J. Lundgren, M. Rönnqvist and P. Värbrand

* *Optimization EXERCISES* by M. Henningsson, J. Lundgren, M. Rönnqvist and P. Värbrand

Note: The book is also available in Swedish and is then called *Optimeringslära*.

**2MA404/2MA918/2MA62Ä:** The course is given in three different versions 2MA918 (5 hp) for CTMAT, 2MA404 (7.5 hp) for NGMAT and 2MA62Ä for teacher education students (lärarstudenter).

**Schedule:** We meet twice a week for lectures on Tuesdays and Thursdays at 10-12 (there are some exceptions e.g. the first week, see TimeEdit for details). *Lecture notes* will also be posted in advance under the section Lecture Notes. The notes will in some cases fill in gaps that may be found in the course literature. Although I still strongly recommend the book as your primary resource.

**Laborations:** We will have laborations in Python (see TimeEdit for specific schedule). The teachers (2MA62Ä) will have separate laborations.

**Demonstrations:** Once per week there will be a practice session (räkneövning) held by Jakob, where he will demonstrate exercises and answer questions.

**Different credits:** The different versions of the course are assigned different credit values. Course 2MA918 is a 5-credit course, while 2MA404 and 2MA62Ä are 7.5-credit courses. All versions include laborations and a written exam, but the latter two courses also have additional assignments to account for the additional credits.

## Study Guide

Current general plan. Last updated: 2025-11-04

**Course literature**

- [A]: Optimization
- [B]: Optimization, exercise book

### Week 45

Introduction to the topic, mathematical modelling, key terminology such as: local, global optimality, convexity, introduction to linear optimization, graphical solution.

**Reading instruction [A]:**
Sec: 1, 2.1–2.5, 3

**Recommended exercises [B]:**
1.1, 1.2, 1.3, 2.2, 2.3, 2.5, 2.6 

### Week 46

Transformations of problems, optimality in linear problems, feasbile basic solutions, algebraic solutions of linear problems, the simplex method

**Reading instruction [A]:**
Sec: 3, 4

**Recommended exercises [B]:**
3.1, 3.3, 4.1, 4.3, 4.5, 4.10, 4.11

### Week 47

The dual problem, relation between primal and dual, optimality based on duality, sensitivity analysis. 

**Reading instruction [A]:**
Sec: 5, 6

**Recommended exercises [B]:**
5.4, 5.6, 6.5, 6.7

### Week 48

Introduction to networks and network problems. We solve for instance the minimum spanning tree problem using Kruskal's and Prim's algorithm. If time admits we will also start on the minimum cost network flow problem.

* Thursday: This lecture we also plan to kickstart the assignment for 2MA404 and 2MA62Ä which you will work with simultaneously.
* Friday: Workshop and grading of the first laboration, Time: 8-12 (2MA404/2MA918).


**Reading instruction [A]:**
Sec: 8.1-8.3

**Recommended exercises [B]:**
8.12, 8.4

### Week 49

Minimum cost network flow, network simplex method, introduction to non-linear problems, more on convexity analysis

* Monday: Workshop and grading of the first laboration, Time: 8-12 (2MA62Ä).


**Reading instruction [A]:**
Sec: 8.6.1-8.6.3, 8.7, 9

**Recommended exercises [B]:**
8.29, 9.4, 9.5, 9.6, 9.7, 9.8 

### Week 50

Methods for unconstrained optimization such as steepest descent, and Newton's method. We will also look into some methods for one-dimensional optimization. We also consider optimality conditions for non-linear problems, the so-called Karush-Kuhn-Tucker conditions (KKT).

**Reading instruction [A]:**
Sec: 10.1-10.4, 11

**Recommended exercises [B]:**
10.1(c), 10.3, 10.4, 10.5, 11.1, 11.3, 11.4, 11.5

### Week 51

Methods for constrained optimization, Lagrange duality and Lagrange relaxation, subgradient optimization.

* Tuesday: Laboration II, Time: 13-17 (2MA62Ä)
* Friday:  Laboration II, Time: 8-12 (2MA404/2MA918)

**Reading instruction [A]:**
Sec: 12, 17

**Recommended exercises [B]:** 
12.1, 12.3, 12.10, 12.12, 12.14, 12.15, 17.1, 17.2, 17.3

[Solutions](./Solutions.zip)

## Laborations

Last updated: 2025-11-04

The laborations are U/G i.e., fail or pass, and will typically be examined during the laboration with some minor work thereafter. The instructions will be published here at least one week before the laborations, and all students are expected to have carefully read the instructions prior to the laboration.

**2MA404/2MA918**

* Laboration I, Friday 28/11
* Laboration II, Friday 19/12

**2MA62Ä**

* Laboration I, Monday 1/12
* aboration II, Tuesday 16/12

### Getting started with Python and SciPy

The computer assignment in this course is done in Python with the SciPy-package. 

Below are some links for setting up Python and the SciPy-package on your own computer. If you are having troubles, please let me know.

**Python:**

* Download installer at [https://www.python.org/](https://www.python.org/)
* Get comfortable with a package manager for your system, e.g., **pip** or **conda**.

**SciPy:**

* Have a look at the installation guide at [https://scipy.org/install/](https://scipy.org/install/) but read the next bullets before you install anything.
* In one of the exercises we will be using a deprecated 'simplex' method for educational purposes. At the time of writing, this still works in the latest scipy but with a deprecated warning and the functionality may be removed at a moments notice.
* So if possible try to install with version 1.8.1. For example **pip install scipy==1.8.1** with **pip** as the package manager. Or **conda install scipy=1.8.1**.

**NumPy:**

* To brush up on NumPy usage take a look at the beginner guide: [https://numpy.org/doc/stable/user/absolute_beginners.html](https://numpy.org/doc/stable/user/absolute_beginners.html)

[Laboration I](./lab1/Optimization_and_modelling.pdf)

[Laboration II](./lab2/Optimization_and_modelling.pdf)

## Old Exams

[Exam 2012](./exams/Exam_7-11-2012.pdf)

[Exam 2018](./exams/Exam_26-05-2018.pdf)

[Exam 2023](./exams/Exam_10-01-2023.pdf)

[Exam 2023 Solutions](./exams/Exam_Solutions_10-01-2023.pdf)

[Exam 2024](./exams/Exam_8-01-2024.pdf)
